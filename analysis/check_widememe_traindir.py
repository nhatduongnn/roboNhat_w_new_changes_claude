"""Gate a meme-file-widened trainDir before anything is decoded against it.

Widening moves silently when it goes wrong: an off-by-one in the pad slice shifts a TF's
whole Fiber-seq footprint relative to its motif and still produces a model that decodes
without error, just wrongly. A mis-sized `combined_low_count` slice is worse -- numpy
broadcasts a length-1 array to any width, so a TF that should carry a real pooled footprint
silently falls back to a flat scalar and nothing complains. These checks make both loud.

    conda activate robocop-2024
    python check_widememe_traindir.py robocop_train_widememe
    python check_widememe_traindir.py robocop_train_wide10all \\
        --pads tf_pads_wide10all.tsv \\
        --meme inputs/motifs_meme_wide10all.txt \\
        --params inputs/all_TFs_1000pealVal_params_pseudo_wide10all.pkl

Exits non-zero on the first failure, so it can be chained after the train job.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
from robocop.utils import parameterize as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MEME_SRC = os.path.join(HERE, "inputs", "motifs_meme.txt")
BASELINE = os.path.join(HERE, "robocop_train_fiberonly", "HMMconfig.pkl")
POOLED = "combined_low_count"

fails = []


def check(ok, msg, detail=""):
    print("%s %s%s" % ("ok:  " if ok else "FAIL:", msg, ("  " + detail) if detail else ""))
    if not ok:
        fails.append(msg)


def read_pads(path):
    pads = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if line:
            f = line.split()
            pads[f[0]] = (int(f[1]), int(f[2]))
    return {t: p for t, p in pads.items() if p != (0, 0)}


def expected_tf_prob(meme, cfg):
    """Reproduce parameterize.getDBFconc:80-90 from this meme file and the config's own
    background, in the config's TF order. Should match cfg['tf_prob'] bit-for-bit."""
    from robocop.utils import concentration_probability_conversion as C
    pwm = P.getMotifsMEME(meme)
    pwm["background"] = cfg["pwm_emission"][0].reshape(5, 1)
    pwm["unknown"] = P.computeUnknown(pwm["background"])
    conc = {k: P.calculateKD(pwm, k) for k in pwm}
    conc["background"], conc["unknown"], conc["nucleosome"] = 1.0, 0.1, 35
    prob = C.convert_to_prob(conc, pwm)
    s = sum(prob.values())
    return np.array([prob[t] / s for t in cfg["tfs"]])


def rc5(pwm):
    """reverse_complement, matching robocop.py exactly."""
    out = np.zeros((5, pwm.shape[1]))
    out[0], out[1], out[2], out[3] = pwm[3, ::-1], pwm[2, ::-1], pwm[1, ::-1], pwm[0, ::-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trainDir")
    ap.add_argument("--pads", default=os.path.join(HERE, "tf_pads.tsv"))
    ap.add_argument("--meme", default=os.path.join(HERE, "inputs", "motifs_meme_wide.txt"))
    ap.add_argument("--params", default=os.path.join(
        HERE, "inputs", "all_TFs_1000pealVal_params_pseudo_wide.pkl"))
    ap.add_argument("--reference", default=None,
                    help="another HMMconfig.pkl whose geometry must match exactly")
    a = ap.parse_args()

    cfg = pickle.load(open(os.path.join(a.trainDir, "HMMconfig.pkl"), "rb"))
    base = pickle.load(open(BASELINE, "rb"))
    pads = read_pads(a.pads)
    orig = P.getMotifsMEME(MEME_SRC)
    wide = P.getMotifsMEME(a.meme)
    tfs = list(cfg["tfs"])
    idx = {t: i for i, t in enumerate(tfs)}
    print("\ntrainDir %s\n  pads   %s (%d TFs)\n  meme   %s\n  params %s\n"
          % (a.trainDir, os.path.basename(a.pads), len(pads),
             os.path.basename(a.meme), os.path.basename(a.params)))

    # --- 1. meme identity -------------------------------------------------------
    bad = [tf for tf, (l, r) in pads.items()
           if not np.array_equal(wide[tf][:4, l:wide[tf].shape[1] - r], orig[tf][:4])]
    check(not bad, "every widened motif's core survives verbatim (%d TFs)" % len(pads),
          "" if not bad else "broken: %s" % bad[:5])
    untouched = [k for k in orig if k not in pads]
    check(all(np.array_equal(wide[k], orig[k]) for k in untouched),
          "the %d non-widened motifs are unchanged" % len(untouched))

    # --- 2. geometry ------------------------------------------------------------
    extra = sum(l + r for l, r in pads.values())
    check(cfg["n_states"] == base["n_states"] + 2 * extra,
          "n_states %d = baseline %d + 2*%d" % (cfg["n_states"], base["n_states"], extra))
    check(cfg["nuc_start"] == base["nuc_start"] + 2 * extra, "nuc_start shifted by 2*%d" % extra)
    check(int(np.sum(cfg["tf_lens"])) == int(np.sum(base["tf_lens"])) + extra,
          "sum(tf_lens) %d = baseline %d + %d" % (np.sum(cfg["tf_lens"]),
                                                  np.sum(base["tf_lens"]), extra))
    bad = [tf for tf, (l, r) in pads.items()
           if cfg["tf_lens"][idx[tf]] != orig[tf].shape[1] + l + r]
    check(not bad, "every widened block is motif + pads", "" if not bad else str(bad[:5]))
    if a.reference and os.path.exists(a.reference):
        ref = pickle.load(open(a.reference, "rb"))
        check(np.array_equal(cfg["tf_lens"], ref["tf_lens"])
              and np.array_equal(cfg["tf_starts"], ref["tf_starts"])
              and cfg["n_states"] == ref["n_states"],
              "geometry identical to %s" % a.reference)

    # --- 3. sequence register ---------------------------------------------------
    bad_f, bad_r = [], []
    for tf, (l, r) in pads.items():
        i = idx[tf]
        s, L = int(cfg["tf_starts"][i]), int(cfg["tf_lens"][i])
        if not np.array_equal(cfg["pwm_emission"][s + l:s + L - r][:, :4], orig[tf][:4].T):
            bad_f.append(tf)
        if not np.array_equal(cfg["pwm_emission"][s + L:s + 2 * L], rc5(wide[tf]).T):
            bad_r.append(tf)
    check(not bad_f, "forward blocks carry the shipped PWM at the right offset",
          "" if not bad_f else str(bad_f[:5]))
    check(not bad_r, "reverse blocks are the reverse complement of the full widened motif",
          "" if not bad_r else str(bad_r[:5]))

    # --- 4. fiber register ------------------------------------------------------
    p = pickle.load(open(a.params, "rb"))["p"]
    wrong, flat = [], []
    for tf in tfs:
        L = int(cfg["tf_lens"][idx[tf]])
        if tf not in p:
            # falls back to the length-1 combined_low_count, which numpy broadcasts.
            # Fine for an unwidened TF; for a widened one it means a flat scalar.
            if tf in pads:
                flat.append(tf)
            continue
        n = len(np.ravel(p[tf]["watson_signal"]["A"]))
        if n != L:
            wrong.append("%s(%d!=%d)" % (tf, n, L))
    check(not wrong, "every params entry matches its block length (%d TFs)" % len(tfs),
          "" if not wrong else str(wrong[:5]))
    check(not flat, "no widened TF silently falls back to the flat scalar",
          "" if not flat else "%d do: %s" % (len(flat), flat[:5]))
    fitted = [t for t in pads if t in p and
              len(np.ravel(p[t]["watson_signal"]["A"])) == cfg["tf_lens"][idx[t]]]
    off = [t for t in fitted
           if not (pads[t][0] <= int(np.argmin(np.ravel(p[t]["watson_signal"]["A"]).astype(float)))
                   < cfg["tf_lens"][idx[t]] - pads[t][1])]
    print("note: most-protected column falls inside the motif core for %d/%d widened TFs%s"
          % (len(fitted) - len(off), len(fitted),
             "" if not off else "; outside for %s" % off[:6]))

    # --- 5. prior ---------------------------------------------------------------
    ratios = {t: float(cfg["tf_prob"][idx[t]]) / float(base["tf_prob"][idx[t]]) for t in pads}
    if ratios:
        lo = min(ratios, key=ratios.get)
        hi = max(ratios, key=ratios.get)
        print("note: widened tf_prob vs baseline -- min %s %.4g, max %s %.4g, median %.4g"
              % (lo, ratios[lo], hi, ratios[hi], float(np.median(list(ratios.values())))))
    # A non-widened TF's prior does NOT stay numerically fixed, and expecting it to is
    # wrong: convert_to_prob solves one unbound root `p` over ALL motif lengths and then
    # renormalises, so widening any TF moves p and every prob follows as
    #     prob_new/prob_base = (p_new/p_base)^len * (S_base/S_new)
    # With 12 TFs widened by 20bp that is a ~1% shift on untouched TFs -- arithmetic, not a
    # model change. So check the two things that ARE invariant:
    #   (a) the built tf_prob equals a first-principles getDBFconc recomputation exactly;
    #   (b) once the shared root shift is divided out, every non-widened TF lands on one
    #       constant (measured residual spread 4.4e-14, i.e. machine precision).
    exp = expected_tf_prob(a.meme, cfg)
    check(np.array_equal(np.asarray(cfg["tf_prob"], dtype=float), exp),
          "tf_prob is bit-identical to a getDBFconc recomputation from this meme file")

    unmoved = [t for t in tfs if t not in pads and t != "unknown"]
    if unmoved:
        pb, pn = float(base["background_prob"]), float(cfg["background_prob"])
        L = np.array([int(base["tf_lens"][idx[t]]) for t in unmoved], dtype=float)
        raw = np.array([float(cfg["tf_prob"][idx[t]]) / float(base["tf_prob"][idx[t]])
                        for t in unmoved])
        # The invariant is that ONE shared root explains all 141 at once, i.e. that
        # log(raw) is exactly affine in tf_len. Fit the root rather than assuming it is
        # cfg["background_prob"]/base["background_prob"]: convert_to_prob solves the root
        # numerically, and the value it STORES converges less tightly than the tf_probs it
        # built with it -- 4.4e-7 relative at 320-column motifs versus 3e-15 at 40. Testing
        # the stored ratio therefore fails on motif width, not on anything about the model.
        # This form is strictly stronger: real scatter (a TF whose prior moved on its own)
        # inflates the residual under ANY root, so nothing that the old check caught is
        # let through, and the stored-vs-implied gap is now reported instead of conflated.
        coef = np.linalg.lstsq(np.vstack([np.ones_like(L), L]).T,
                               np.log(raw), rcond=None)[0]
        r_fit = float(np.exp(coef[1]))
        spread = float((raw / r_fit ** L).max() - (raw / r_fit ** L).min())
        check(spread < 1e-8,
              "the %d non-widened TFs moved ONLY via the shared unbound root "
              "(raw %.4f-%.4f, residual spread %.2g)"
              % (len(unmoved), raw.min(), raw.max(), spread))
        drift = r_fit / (pn / pb) - 1.0
        print("note: unbound root implied by tf_prob is %.4g relative from the stored "
              "background_prob (%.12f vs %.12f)%s"
              % (drift, r_fit, pn / pb,
                 "" if abs(drift) < 1e-5 else "  <-- LARGE, inspect convert_to_prob"))

    print()
    if fails:
        print("FAILED %d check(s): %s" % (len(fails), "; ".join(fails)))
        sys.exit(1)
    print("all gates passed")


if __name__ == "__main__":
    main()
