"""Static verification of the widened-footprint pkgvar -- no decode, no cluster.

Checks the three things that would silently corrupt a widened run and that a decode
would NOT surface as an error:

  GATE A  zero-pad identity   with an empty pad table the variant must reproduce the
                              baseline HMM geometry AND pwm_emission bit-for-bit.
  GATE B  geometry            widened n_states / tf_lens / tf_starts / pwm_emission rows.
  GATE C  register            the fiber p-vector's motif columns must land exactly on the
                              motif states, forward and reverse. An off-by-one here shifts
                              every ABF1 site and shows up only as a quietly worse score.
  GATE D  sequence neutrality  pad rows of pwm_emission must equal the background state's
                              row, i.e. the widening really is Fiber-layer-only.

    conda activate robocop-2024
    python verify_wide.py                       # tests pkgvar/seq_maskoff_wide
    python verify_wide.py --variant seq_maskoff_widenull
"""
import argparse
import importlib
import os
import pickle
import sys
import tempfile

import numpy as np

BASE_TRAIN = "robocop_train_fiberonly"
SHIPPED = "inputs/all_TFs_1000pealVal_params_pseudo.pkl"


def build(variant, pads_path, pwm, nuc_emission):
    """Run the variant's geometry code on a synthetic shared dict."""
    for m in [k for k in list(sys.modules) if k == "robocop" or k.startswith("robocop.")]:
        del sys.modules[m]
    sys.path.insert(0, "pkgvar/%s/" % variant)
    R = importlib.import_module("robocop.robocop")
    sys.path.pop(0)

    tfs = np.array(sorted(k for k in pwm if k not in ("background",)))
    d = {"n_tfs": len(tfs), "tfs": tfs, "nucleotides": 1, "padding": 0,
         "nucleosome_prob": 1e-3, "timepoints": 1}
    d["tf_pads"] = R.read_tf_pads(tfs, pads_path)
    R.get_transition_matrix_info(d, pwm, 1)
    R.stack_pwms(d, pwm, nuc_emission)
    return R, d


def fail(msg):
    print("  FAIL: %s" % msg)
    fail.n += 1
fail.n = 0


def ok(msg):
    print("  ok: %s" % msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="seq_maskoff_wide")
    ap.add_argument("--pads", default="tf_pads.tsv")
    a = ap.parse_args()

    pwm = pickle.load(open(BASE_TRAIN + "/pwm.p", "rb"), encoding="latin1")
    nuc_emission = np.load(BASE_TRAIN + "/nuc_emission.npy")
    base = pickle.load(open(BASE_TRAIN + "/HMMconfig.pkl", "rb"), encoding="latin1")

    # ---------------------------------------------------------------- GATE A ----
    print("GATE A  zero-pad identity vs %s" % BASE_TRAIN)
    empty = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False)
    empty.write("# no pads\n")
    empty.close()
    _, z = build(a.variant, empty.name, pwm, nuc_emission)
    os.unlink(empty.name)
    for k in ("n_states", "silent_states_begin", "nuc_start", "nuc_len"):
        (ok if int(z[k]) == int(base[k]) else fail)("%s = %d" % (k, z[k]))
    (ok if np.array_equal(z["tf_lens"], base["tf_lens"]) else fail)("tf_lens identical")
    (ok if np.array_equal(z["tf_starts"], base["tf_starts"]) else fail)("tf_starts identical")
    (ok if np.array_equal(z["pwm_emission"], base["pwm_emission"]) else fail)(
        "pwm_emission bit-identical %s" % (z["pwm_emission"].shape,))

    # ---------------------------------------------------------------- GATE B ----
    print("\nGATE B  widened geometry (%s)" % a.pads)
    R, d = build(a.variant, a.pads, pwm, nuc_emission)
    pads = d["tf_pads"]
    tfs = list(d["tfs"])
    grew = [i for i in range(len(tfs)) if pads[i].sum() > 0]
    extra = int(sum(pads[i].sum() for i in grew))
    exp_states = int(base["n_states"]) + 2 * extra
    (ok if int(d["n_states"]) == exp_states else fail)(
        "n_states %d -> %d (expected +2*%d)" % (base["n_states"], d["n_states"], extra))
    (ok if int(d["tf_lens"].sum()) == int(base["tf_lens"].sum()) + extra else fail)(
        "sum(tf_lens) %d -> %d" % (base["tf_lens"].sum(), d["tf_lens"].sum()))
    for i in grew:
        L = pwm[tfs[i]].shape[1]
        pl, pr = int(pads[i][0]), int(pads[i][1])
        (ok if int(d["tf_lens"][i]) == L + pl + pr else fail)(
            "%s tf_lens %d -> %d (motif %d + %d/%d)" % (tfs[i], base["tf_lens"][i],
                                                        d["tf_lens"][i], L, pl, pr))
    starts_ok = all(d["tf_starts"][i] == d["tf_starts"][i-1] + 2*d["tf_lens"][i-1]
                    for i in range(1, len(tfs)))
    (ok if starts_ok else fail)("tf_starts tile without gap or overlap")
    (ok if d["pwm_emission"].shape[0] == d["silent_states_begin"] else fail)(
        "pwm_emission rows == silent_states_begin (%d)" % d["silent_states_begin"])

    # ---------------------------------------------------------------- GATE C ----
    print("\nGATE C  fiber p-vector register")
    shipped = pickle.load(open(SHIPPED, "rb"))["p"]
    varmod = sys.modules["robocop.robocop"]
    src = open(varmod.__file__).read()
    pklname = [l for l in src.split("\n") if "params_pseudo" in l and "open(" in l][0]
    pklname = pklname.split("'")[1]
    wide = pickle.load(open(pklname, "rb"))["p"]
    print("  variant reads %s" % pklname)
    for i in grew:
        tf = tfs[i]
        if tf not in wide:
            continue
        L = pwm[tf].shape[1]
        pl, pr = int(pads[i][0]), int(pads[i][1])
        s, e = int(d["tf_starts"][i]), int(d["tf_starts"][i]) + 2 * int(d["tf_lens"][i])
        for strand, other in (("watson_signal", "crick_signal"),
                              ("crick_signal", "watson_signal")):
            ps = np.zeros(d["silent_states_begin"])
            pf = np.asarray(wide[tf][strand]["A"], dtype=float)
            pr_v = np.asarray(wide[tf][other]["A"], dtype=float)[::-1]
            if len(pf) + len(pr_v) != e - s:
                fail("%s/%s block length %d != p length %d" % (tf, strand, e - s,
                                                               len(pf) + len(pr_v)))
                continue
            ps[s:e] = np.concatenate((pf, pr_v))
            core_f = ps[s + pl: s + pl + L]
            core_r = ps[s + d["tf_lens"][i] + pr: s + d["tf_lens"][i] + pr + L]
            want_f = np.asarray(shipped[tf][strand]["A"], dtype=float)
            want_r = np.asarray(shipped[tf][other]["A"], dtype=float)[::-1]
            (ok if np.array_equal(core_f, want_f) else fail)(
                "%s %s fwd core == shipped motif vector" % (tf, strand[:6]))
            (ok if np.array_equal(core_r, want_r) else fail)(
                "%s %s rev core == mirrored shipped vector" % (tf, strand[:6]))

    # ---------------------------------------------------------------- GATE D ----
    print("\nGATE D  sequence-layer neutrality of pad states")
    em = d["pwm_emission"]
    bg_row = np.transpose(pwm["background"])[0]
    bad = 0
    for i in grew:
        pl, pr = int(pads[i][0]), int(pads[i][1])
        s, tl = int(d["tf_starts"][i]), int(d["tf_lens"][i])
        rows = (list(range(s, s + pl)) + list(range(s + tl - pr, s + tl))
                + list(range(s + tl, s + tl + pr)) + list(range(s + 2*tl - pl, s + 2*tl)))
        for r in rows:
            if not np.array_equal(em[r], bg_row):
                bad += 1
    (ok if bad == 0 else fail)("all %d pad rows equal the background emission row"
                               % sum(int(pads[i].sum()) * 2 for i in grew))
    (ok if np.array_equal(em[0], bg_row) else fail)("state 0 (background) row unchanged")

    print("\n%s" % ("ALL GATES PASSED" if fail.n == 0 else "%d CHECK(S) FAILED" % fail.n))
    sys.exit(1 if fail.n else 0)


if __name__ == "__main__":
    main()
