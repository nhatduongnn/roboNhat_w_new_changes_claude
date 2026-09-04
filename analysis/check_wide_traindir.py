"""Gate a freshly built widened trainDir before anything decodes against it.

verify_wide.py checks the pkgvar's geometry code in isolation. This checks the ARTEFACT it
produced -- the HMMconfig.pkl a decode will actually load -- against the pad table and the
params pkl, and against the baseline trainDir it must stay comparable to.

The failure this is really guarding: HMMconfig.pkl freezes tf_lens, but the fiber p-vectors
are read at runtime from a pkl on disk. Rebuild one without the other (edit tf_pads.tsv,
re-run make_params_wide.py, forget to retrain) and the two disagree. robocop.py raises on
that mismatch now, but only after a job has queued, launched and loaded its data.

    python check_wide_traindir.py robocop_train_wideA --variant seq_maskoff_wide
"""
import argparse
import os
import pickle
import sys

import numpy as np

BASE = "robocop_train_fiberonly"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trainDir")
    ap.add_argument("--variant", default="seq_maskoff_wide")
    ap.add_argument("--pads", default="tf_pads.tsv")
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()

    bad = []
    def check(cond, msg):
        print(("  ok: " if cond else "  FAIL: ") + msg)
        if not cond:
            bad.append(msg)

    d = pickle.load(open(os.path.join(a.trainDir, "HMMconfig.pkl"), "rb"), encoding="latin1")
    base = pickle.load(open(os.path.join(a.base, "HMMconfig.pkl"), "rb"), encoding="latin1")
    pwm = pickle.load(open(os.path.join(a.base, "pwm.p"), "rb"), encoding="latin1")
    tfs = list(d["tfs"])

    print("trainDir %s  (variant %s)" % (a.trainDir, a.variant))
    check("tf_pads" in d, "HMMconfig carries dshared['tf_pads']")
    if "tf_pads" not in d:
        sys.exit(1)
    pads = np.asarray(d["tf_pads"], dtype=int)

    # --- pads match the table on disk --------------------------------------------
    spec = {}
    for raw in open(a.pads):
        line = raw.split("#")[0].strip()
        if line:
            f = line.split()
            spec[f[0]] = (int(f[1]), int(f[2]))
    want = np.array([spec.get(t, (0, 0)) for t in tfs], dtype=int)
    check(np.array_equal(pads, want),
          "frozen pads == %s  (%s)" % (a.pads, {t: spec[t] for t in sorted(spec)}))

    # --- geometry ------------------------------------------------------------------
    grew = [i for i in range(len(tfs)) if pads[i].sum() > 0]
    extra = int(sum(pads[i].sum() for i in grew))
    check(int(d["n_states"]) == int(base["n_states"]) + 2 * extra,
          "n_states %d = baseline %d + 2*%d" % (d["n_states"], base["n_states"], extra))
    for i in grew:
        L = pwm[tfs[i]].shape[1]
        check(int(d["tf_lens"][i]) == L + int(pads[i].sum()),
              "%s block %d bp = motif %d + %d/%d" % (tfs[i], d["tf_lens"][i], L,
                                                     pads[i][0], pads[i][1]))
    check(all(d["tf_starts"][i] == d["tf_starts"][i-1] + 2*d["tf_lens"][i-1]
              for i in range(1, len(tfs))), "tf_starts tile exactly")
    check(d["pwm_emission"].shape[0] == d["silent_states_begin"],
          "pwm_emission rows == silent_states_begin")

    # --- pads are sequence-neutral ---------------------------------------------------
    bg_row = np.transpose(pwm["background"])[0]
    em, nbad = d["pwm_emission"], 0
    for i in grew:
        pl, pr, s, tl = int(pads[i][0]), int(pads[i][1]), int(d["tf_starts"][i]), int(d["tf_lens"][i])
        for r in (list(range(s, s+pl)) + list(range(s+tl-pr, s+tl))
                  + list(range(s+tl, s+tl+pr)) + list(range(s+2*tl-pl, s+2*tl))):
            nbad += 0 if np.array_equal(em[r], bg_row) else 1
    check(nbad == 0, "all pad rows emit the background PWM column (sequence-neutral)")

    # --- the prior did NOT move ------------------------------------------------------
    # Widening must not change the transition prior, or the comparison stops being
    # one-variable. tf_prob comes from calculateKD(pwm) and the PWMs are untouched.
    check(np.array_equal(np.asarray(d["tf_prob"]), np.asarray(base["tf_prob"])),
          "tf_prob bit-identical to %s (prior unchanged)" % a.base)
    check(float(d["background_prob"]) == float(base["background_prob"])
          and float(d["nucleosome_prob"]) == float(base["nucleosome_prob"]),
          "background/nucleosome prob unchanged")

    # --- the params pkl the variant reads agrees with this geometry -------------------
    src = open("pkgvar/%s/robocop/robocop.py" % a.variant).read()
    pkl = [l for l in src.split("\n") if "params_pseudo" in l and "open(" in l][0].split("'")[1]
    p = pickle.load(open(pkl, "rb"))["p"]
    print("  variant reads %s" % pkl)
    for i in grew:
        tf = tfs[i]
        if tf in p:
            n = len(np.asarray(p[tf]["watson_signal"]["A"]))
            check(n == int(d["tf_lens"][i]),
                  "%s p-vector %d bp == block %d bp" % (tf, n, d["tf_lens"][i]))

    print("\n%s" % ("TRAINDIR OK" if not bad else "%d CHECK(S) FAILED" % len(bad)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
