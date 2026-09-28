"""Gate a decoy trainDir (one extra motif `Zz_decoy_abf1`) against the trainDir it extends (2026-09-21).

Plan: ~/.claude/plans/okay-so-i-want-humming-hearth.md, Gate 2. Decoy-aware sibling of
check_widememe_traindir.py; R-free (robocop_kd.calculateKD, no robocop import).

    python check_decoy_traindir.py robocop_train_decoy14      --src robocop_train_fiberonly --width 14 \
        --n-states 3514 --params inputs/all_TFs_1000pealVal_params_pseudo_decoy14.pkl
    python check_decoy_traindir.py robocop_train_wide_decoy23 --src robocop_train_widememe  --width 23 \
        --n-states 3550 --params inputs/all_TFs_1000pealVal_params_pseudo_wide_decoy23.pkl

Checks
  geometry   n_states == expected == src + 2L + 1; tfs == sorted(src tfs + decoy); every real motif keeps
             its tf index, tf_len and tf_start; decoy starts where src `unknown` started; nuc_start and
             silent_states_begin shift by exactly 2L
  pwm.p      background row == src background row bit-for-bit (recomputed from SacCer3.fa at training);
             every real motif == src bit-for-bit; decoy == background column, bit-for-bit, L columns
  Kd         calculateKD(decoy) == 1.0 exactly; every real motif's Kd == src's exactly
  emission   every real motif's forward+reverse pwm_emission rows == src; the rows after the decoy block
             (nucleosome) == src rows shifted by 2L; decoy FORWARD block == background exactly; decoy
             REVERSE block == reverse_complement(background) (robocop.py stack_pwms), and its deviation
             from background is reported (background is not strand-symmetric, so it cannot be 0)
  weights    ln w_k = ln(tf_prob_k) - L_k ln(background_prob) equals src for every real motif, unknown and
             the nucleosome (|diff| <= 1e-9): the extra motif moves tf_prob only through the shared root
  transition every entry of the transition matrix equals src under the old->new state map, except the
             central silent row (ssb, the prior row), which make_w_trainDir rebuilds every round
  params     the decoy's Fiber-seq entry has length L on both strands and is flat
Exits non-zero on any failure.
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from robocop_kd import calculateKD  # noqa: E402

DECOY = "Zz_decoy_abf1"
fails = []


def check(ok, msg, detail=""):
    print("%s %s%s" % ("ok:  " if ok else "FAIL:", msg, ("  " + detail) if detail else ""))
    if not ok:
        fails.append(msg)


def rc5(p):
    out = np.zeros((5, p.shape[1]))
    out[0], out[1], out[2], out[3] = p[3, ::-1], p[2, ::-1], p[1, ::-1], p[0, ::-1]
    return out


def lnw(cfg):
    bg = float(cfg["background_prob"])
    out = {t: math.log(float(cfg["tf_prob"][i])) - int(cfg["tf_lens"][i]) * math.log(bg)
           for i, t in enumerate(cfg["tfs"])}
    out["nucleosome"] = math.log(float(cfg["nucleosome_prob"])) - 147 * math.log(bg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trainDir")
    ap.add_argument("--src", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--n-states", type=int, required=True)
    ap.add_argument("--params", required=True)
    a = ap.parse_args()
    L = a.width
    ld = lambda d, f: pickle.load(open(os.path.join(HERE, d, f), "rb"), encoding="latin1")
    cfg, src = ld(a.trainDir, "HMMconfig.pkl"), ld(a.src, "HMMconfig.pkl")
    pwm, pwm0 = ld(a.trainDir, "pwm.p"), ld(a.src, "pwm.p")
    print("\ntrainDir %s vs src %s (decoy %s, L %d)\n" % (a.trainDir, a.src, DECOY, L))

    # ---- geometry
    tfs, tfs0 = [str(x) for x in cfg["tfs"]], [str(x) for x in src["tfs"]]
    check(int(cfg["n_states"]) == a.n_states == int(src["n_states"]) + 2 * L + 1,
          "n_states %d == expected %d == src %d + 2*%d + 1" % (cfg["n_states"], a.n_states, src["n_states"], L))
    check(tfs == sorted(tfs0 + [DECOY]), "tfs == sorted(src tfs + %s) (%d -> %d)" % (DECOY, len(tfs0), len(tfs)))
    k = tfs.index(DECOY)
    check(k == tfs0.index("unknown") and tfs[k + 1] == "unknown" and tfs[:k] == tfs0[:k],
          "decoy at index %d, between %s and unknown; every real motif keeps its index" % (k, tfs[k - 1]))
    i0 = {t: i for i, t in enumerate(tfs0)}
    i1 = {t: i for i, t in enumerate(tfs)}
    real = [t for t in tfs0 if t != "unknown"]
    bad = [t for t in tfs0 if int(cfg["tf_lens"][i1[t]]) != int(src["tf_lens"][i0[t]])]
    check(not bad, "tf_lens of all %d src TFs unchanged" % len(tfs0), str(bad[:5]))
    bad = [t for t in real if int(cfg["tf_starts"][i1[t]]) != int(src["tf_starts"][i0[t]])]
    check(not bad, "tf_starts of all %d real motifs unchanged" % len(real), str(bad[:5]))
    ds = int(cfg["tf_starts"][k])
    check(int(cfg["tf_lens"][k]) == L and ds == int(src["tf_starts"][i0["unknown"]]),
          "decoy block: tf_len %d, tf_start %d (= src unknown's start)" % (cfg["tf_lens"][k], ds))
    check(int(cfg["tf_starts"][i1["unknown"]]) == int(src["tf_starts"][i0["unknown"]]) + 2 * L,
          "unknown's block shifts by 2L")
    for key in ("nuc_start", "silent_states_begin"):
        check(int(cfg[key]) == int(src[key]) + 2 * L, "%s %d == src %d + 2L" % (key, cfg[key], src[key]))
    for key in ("padding", "nucleotides", "n_vars", "timepoints", "nuc_len", "nuc_present"):
        check(cfg[key] == src[key], "%s unchanged (%r)" % (key, cfg[key]))
    check(int(cfg["n_tfs"]) == int(src["n_tfs"]) + 1, "n_tfs %d == src + 1" % cfg["n_tfs"])

    # ---- pwm.p
    bg, bg0 = np.asarray(pwm["background"], float), np.asarray(pwm0["background"], float)
    check(bg.shape == bg0.shape and np.array_equal(bg, bg0),
          "pwm.p background == src background bit-for-bit %s" % [repr(float(x)) for x in bg.ravel()])
    check(list(pwm) == list(pwm0)[:-2] + [DECOY] + list(pwm0)[-2:],
          "pwm.p keys == src keys + decoy (before background/unknown)")
    bad = [m for m in pwm0 if m != "background" and not np.array_equal(pwm[m], pwm0[m])]
    check(not bad, "every src pwm.p entry (%d motifs + unknown) unchanged bit-for-bit" % (len(pwm0) - 2), str(bad[:5]))
    dec = np.asarray(pwm[DECOY], float)
    check(dec.shape == (5, L) and np.array_equal(dec, np.tile(bg.reshape(5, 1), (1, L))),
          "decoy PWM (%s) == background column x %d, bit-for-bit" % (dec.shape, L))

    # ---- Kd
    kd = calculateKD(pwm, DECOY)
    check(kd == 1.0, "calculateKD(decoy) == 1.0 exactly (got %r)" % kd)
    bad = [m for m in real if calculateKD(pwm, m) != calculateKD(pwm0, m)]
    check(not bad, "calculateKD of all %d real motifs == src exactly" % len(real), str(bad[:5]))

    # ---- emission
    E, E0 = np.asarray(cfg["pwm_emission"]), np.asarray(src["pwm_emission"])
    check(E.shape[0] == E0.shape[0] + 2 * L, "pwm_emission rows %d == src %d + 2L" % (E.shape[0], E0.shape[0]))
    check(np.array_equal(E[:ds], E0[:ds]), "pwm_emission rows 0..%d (background + every real motif block) == src" % (ds - 1))
    check(np.array_equal(E[ds + 2 * L:], E0[ds:]), "pwm_emission rows after the decoy block == src rows shifted by 2L")
    fwd, rev = E[ds:ds + L], E[ds + L:ds + 2 * L]
    check(np.array_equal(fwd, np.tile(bg.reshape(1, 5), (L, 1))), "decoy FORWARD block == background exactly (seq LR 1)")
    check(np.array_equal(rev, rc5(dec).T), "decoy REVERSE block == reverse_complement(background) (stack_pwms)")
    lr = np.log(rev[0, :4] / bg.ravel()[:4])
    print("note: decoy REVERSE block per-base ln(LR vs background) A %+.5f C %+.5f G %+.5f T %+.5f; max |ln LR| "
          "over %d bp %.4f nats; expected ln LR per bp under background %.2e"
          % (lr[0], lr[1], lr[2], lr[3], L, L * np.abs(lr).max(), float(np.dot(bg.ravel()[:4], lr))))

    # ---- weights
    w, w0 = lnw(cfg), lnw(src)
    dw = {t: abs(w[t] - w0[t]) for t in tfs0 + ["nucleosome"]}
    worst = max(dw, key=dw.get)
    check(dw[worst] <= 1e-9, "ln w of every src TF, unknown and the nucleosome == src (max |diff| %.2e at %s)"
          % (dw[worst], worst))
    print("note: decoy ln w at training %.6g (w %.6g = Kd x 1; the tuner rebuilds it every round)"
          % (w[DECOY], math.exp(w[DECOY])))

    # ---- transition matrix under the old->new state map
    T, T0 = np.asarray(cfg["transition_matrix"]), np.asarray(src["transition_matrix"])
    n0, ssb0, ssb = int(src["n_states"]), int(src["silent_states_begin"]), int(cfg["silent_states_begin"])
    m = np.arange(n0)
    m[(m >= ds) & (m < ssb0)] += 2 * L                     # unknown + nucleosome states
    sil = np.arange(ssb0, n0)                              # silent: central first, then one per TF (in tfs order)
    j = sil - ssb0
    m[ssb0:] = ssb + j + (j - 1 >= i0["unknown"])          # the decoy's silent state is inserted before unknown's
    sub = T[np.ix_(m, m)]
    diff = sub != T0
    rows = sorted(set(np.argwhere(diff)[:, 0].tolist()))
    check(rows in ([], [ssb0]), "transition matrix == src under the state map except the prior row (differs in %s)" % rows)
    new_only = np.setdiff1d(np.arange(int(cfg["n_states"])), m)
    check(len(new_only) == 2 * L + 1, "new states not in the map: %d == 2L + 1 (decoy block + its silent state)" % len(new_only))
    for key in ("initial_probs", "end_probs"):
        a0, a1 = np.asarray(src[key]), np.asarray(cfg[key])
        ok = np.array_equal(a1[m][np.arange(n0) != ssb0], a0[np.arange(n0) != ssb0]) if key == "end_probs" else True
        if key == "end_probs":
            check(ok, "end_probs == src under the state map (prior row excluded)")

    # ---- params
    p = pickle.load(open(os.path.join(HERE, a.params), "rb"))["p"]
    for ch in ("watson_signal", "crick_signal"):
        v = np.asarray(p[DECOY][ch]["A"], float)
        check(v.shape == (L,) and np.all(v == v[0]), "params %s: %s A flat, length %d, p = %.6f" % (
            os.path.basename(a.params), ch, len(v), v[0]))

    print()
    if fails:
        print("FAILED %d check(s): %s" % (len(fails), "; ".join(fails)))
        sys.exit(1)
    print("all gates passed")


if __name__ == "__main__":
    main()
