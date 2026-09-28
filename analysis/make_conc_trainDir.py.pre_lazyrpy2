"""Build a trainDir whose HMM prior has ONE factor's concentration scaled by lambda.

Why this exists
---------------
The per-DBF "concentration" that becomes the HMM's transition prior is computed in
`parameterize.getDBFconc` (pkg/robocop/utils/parameterize.py:72-91):

    dbf_conc[tf]          = calculateKD(pwm, tf)     # :80  Kd of the optimal site
    dbf_conc['background']= 1.0                      # :82
    dbf_conc['unknown']   = 0.1                      # :83
    dbf_conc['nucleosome']= 35                       # :84
    dbf_conc = convert_to_prob(dbf_conc, pwm)        # :87  solve unbound root
    renormalise to sum 1                             # :88-90

That runs ONLY at train time (robocop_em.py:72); a decode reads the frozen
`transition_matrix` out of trainDir/HMMconfig.pkl (robocop_no_em.py:51). So the obvious way
to change a concentration is a ~20 min retrain per value.

It is not necessary. `robocop.set_transition` (robocop.py:815) writes ONLY row
`silent_states_begin` of the transition matrix:

    t_mat[ssb, 0]           = background_prob
    t_mat[ssb, nuc_start]   = nucleosome_prob
    t_mat[ssb, ssb + i + 1] = tf_prob[i]

Everything else -- motif-internal transitions, the nucleosome dinucleotide model,
`pwm_emission`, `tf_starts`/`tf_lens`/`n_states` -- is concentration-independent. And
because robocop_em.py:116 has `iterations = 0`, training is a pure config BUILD, not a fit.
So recomputing that one row is EXACTLY what a retrain would produce, in ~1 min.

Two gates make that claim checkable rather than merely plausible:
  1. FIDELITY -- recompute at lambda=1 and require it to reproduce the source
     HMMconfig's tf_prob / background_prob / nucleosome_prob bit-exactly.
  2. BLAST RADIUS -- after patching, require that only row `silent_states_begin` of
     transition_matrix changed.
Both abort the build on failure.

Usage
-----
    python make_conc_trainDir.py --lam 30
    python make_conc_trainDir.py --tf Abf1_murphy --lam 30 \
           --src robocop_train_fiberonly --out robocop_train_conc30
    python make_conc_trainDir.py --set unknown=0.01 --hold-nucleosome --out <dir>
"""
import argparse
import json
import math
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, '../pkg/')
from robocop import robocop as R                                    # noqa: E402
from robocop.utils.parameterize import calculateKD                  # noqa: E402
from robocop.utils import concentration_probability_conversion as C  # noqa: E402

# Files a trainDir carries besides HMMconfig.pkl. None depend on concentration, so they are
# copied straight across from the source trainDir.
SIDECARS = ("config.ini", "pwm.p", "nuc_emission.npy",
            "nuc_dinucleotide_model.txt", "negParamsMNase.pkl")

# Path the set_initial_probs CDLL needs, relative to analysis/.
CSHARED = "../pkg/robocop/librobocop.so"


def dbf_probs(pwm, lams):
    """Reproduce getDBFconc:80-90 exactly, with dbf_conc[tf] multiplied by lams[tf].

    `lams` is {tf: multiplier}; several factors move in ONE call because
    convert_to_prob solves a single unbound root over every motif length at once.
    Scaling them one at a time in separate trainDirs would give a different -- and
    wrong -- answer, since each solve would see the others unscaled.

    Returns ({name: probability} summing to 1, {tf: scaled concentration}).
    """
    conc = {k: calculateKD(pwm, k) for k in pwm.keys()}
    conc['background'] = 1.0
    conc['unknown'] = 0.1
    conc['nucleosome'] = 35
    for tf in lams:
        if tf not in conc:
            sys.exit("error: %s not in pwm (have %d motifs)" % (tf, len(pwm)))
    for tf, lam in lams.items():
        conc[tf] *= lam
    prob = C.convert_to_prob(conc, pwm)
    total = sum(prob.values())
    return {k: v / total for k, v in prob.items()}, {t: conc[t] for t in lams}


def solve_nuc_lam(pwm, lams, target, lo=1e-8, hi=1e3, rtol=1e-12, max_iter=200):
    """lambda_nucleosome that puts prob['nucleosome'] back at `target`, given `lams`.

    Moving any other concentration moves the shared unbound root, and the 147 bp nucleosome
    feels that as p^147 -- lambda_unknown = 0.01 alone raises nucleosome_prob ~21x. The
    compensating lambda therefore depends on the WHOLE vector, so it is re-solved per build
    rather than reused. prob['nucleosome'] is monotone increasing in its own lambda, so a
    bisection in log space is safe; each step is one dbf_probs call (no decode).
    """
    def nuc(lam_nuc):
        return dbf_probs(pwm, dict(lams, nucleosome=lam_nuc))[0]['nucleosome']

    a, b = math.log(lo), math.log(hi)
    if not nuc(lo) <= target <= nuc(hi):
        sys.exit("error: nucleosome_prob %.6e is not reachable with lambda_nucleosome in "
                 "[%g, %g]" % (target, lo, hi))
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        p = nuc(math.exp(m))
        if abs(p / target - 1.0) < rtol:
            break
        if p < target:
            a = m
        else:
            b = m
    return math.exp(m)


def apply_probs(cfg, prob):
    """Write `prob` into cfg's transition matrix / initial / end probs, in place."""
    tf_prob = np.array([prob[t] for t in list(cfg['tfs'])])
    R.set_transition(cfg, tf_prob, prob['background'], prob['nucleosome'])
    R.set_initial_probs(cfg)
    R.set_end_probs(cfg)
    return tf_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=None,
                    help="multiplier on --tf's concentration (single-factor form)")
    ap.add_argument("--tf", default=None, help="factor to scale (single-factor form)")
    ap.add_argument("--set", action="append", default=None, metavar="TF=LAM",
                    help="scale several factors in one trainDir, e.g. "
                         "--set Abf1_murphy=300 --set Reb1_badis=5. Repeatable. They must "
                         "be applied together: convert_to_prob solves one unbound root "
                         "across all motif lengths, so scaling them in separate trainDirs "
                         "would not compose.")
    ap.add_argument("--hold-nucleosome", action="store_true",
                    help="solve lambda_nucleosome so nucleosome_prob stays at the source value "
                         "despite the other --set factors")
    ap.add_argument("--src", default="robocop_train_fiberonly",
                    help="trainDir to inherit everything else from")
    ap.add_argument("--out", default=None,
                    help="destination trainDir (default robocop_train_conc<lam>)")
    args = ap.parse_args()

    if args.set:
        lams = {}
        for spec in args.set:
            if "=" not in spec:
                sys.exit("--set wants TF=LAM, got %r" % spec)
            k, v = spec.split("=", 1)
            lams[k] = float(v)
    elif args.tf is not None and args.lam is not None:
        lams = {args.tf: args.lam}
    else:
        sys.exit("give either --set TF=LAM (repeatable) or both --tf and --lam")
    if args.hold_nucleosome and "nucleosome" in lams:
        sys.exit("error: --hold-nucleosome solves lambda_nucleosome itself; don't also --set it")

    lam_tag = "_".join("%s%s" % (t.split("_")[0], ("%g" % l).replace(".", "p"))
                       for t, l in sorted(lams.items()))
    out = args.out or ("robocop_train_conc_%s" % lam_tag)
    if os.path.abspath(out) == os.path.abspath(args.src):
        sys.exit("error: --out must differ from --src (refusing to overwrite the source)")

    print("=== make_conc_trainDir ===")
    for t, l in sorted(lams.items()):
        print("scale: %-18s x %g" % (t, l))
    print("src  :", args.src)
    print("out  :", out)

    pwm = pickle.load(open(os.path.join(args.src, "pwm.p"), "rb"))
    cfg = pickle.load(open(os.path.join(args.src, "HMMconfig.pkl"), "rb"),
                      encoding="latin1")
    cfg['robocopC'] = CSHARED

    src_tf_prob = np.asarray(cfg['tf_prob']).copy()
    src_bg = float(cfg['background_prob'])
    src_nuc = float(cfg['nucleosome_prob'])
    src_tmat = cfg['transition_matrix'].copy()
    src_ip = np.asarray(cfg['initial_probs']).copy()
    src_ep = np.asarray(cfg['end_probs']).copy()
    ssb = int(cfg['silent_states_begin'])
    tfs = list(cfg['tfs'])
    nuc_lam_solved = None
    if args.hold_nucleosome:
        nuc_lam_solved = solve_nuc_lam(pwm, lams, src_nuc)
        lams["nucleosome"] = nuc_lam_solved
        print("hold : lambda_nucleosome solved = %.10g (keeps nucleosome_prob at %.6e)"
              % (nuc_lam_solved, src_nuc))
    # `nucleosome` is a legitimate concentration (dbf_probs sets it to 35 and applies the
    # multiplier like any other), but it is NOT a member of cfg['tfs'] -- that array holds
    # the 153 motifs plus `unknown`, and the nucleosome has its own 531-state block. Checking
    # only against cfg['tfs'] therefore rejected a knob the machinery already supports.
    #
    # It has to be settable, because `unknown` cannot be tuned without it: unknown carries ~5%
    # of the prior mass, so lowering it drives the shared unbound root up, and the 147 bp
    # nucleosome amplifies that as p^147 -- measured at +1085% to +2036% for lambda_unknown
    # 0.3 down to 0.01. Compensating the nucleosome in the SAME call is the only way to move
    # `unknown` without destroying the one part of the model that is already correct.
    # `background` stays unsettable on purpose: its concentration is pinned at 1.0 and defines
    # the scale the root is solved against, so scaling it just renames the units.
    settable = set(tfs) | {"nucleosome"}
    missing = [t for t in lams if t not in settable]
    if missing:
        sys.exit("error: not in this model: %s" % ", ".join(missing))
    # `nucleosome` has no slot in cfg['tfs'] / tf_prob -- it owns its own 531-state block and
    # its probability lives in cfg['nucleosome_prob']. Index only the real TF entries.
    tidx = {t: tfs.index(t) for t in lams if t in tfs}

    # ---- gate 1: fidelity. lambda=1 must reproduce the source config bit-exactly. ----
    p1, _ = dbf_probs(pwm, {t: 1.0 for t in lams})
    tfp1 = np.array([p1[t] for t in tfs])
    d_tf = float(np.max(np.abs(tfp1 / src_tf_prob - 1.0)))
    d_bg = abs(p1['background'] / src_bg - 1.0)
    d_nuc = abs(p1['nucleosome'] / src_nuc - 1.0)
    print("\n[gate 1] lambda=1 vs source HMMconfig:")
    print("   max|rel diff| tf_prob = %.3e   background = %.3e   nucleosome = %.3e"
          % (d_tf, d_bg, d_nuc))
    if max(d_tf, d_bg, d_nuc) > 0.0:
        sys.exit("FIDELITY GATE FAILED: recomputing dbf_conc at lambda=1 does not reproduce "
                 "%s/HMMconfig.pkl. The patch cannot be trusted -- retrain instead." % args.src)
    apply_probs(cfg, p1)
    n_changed = int((np.abs(cfg['transition_matrix'] - src_tmat) > 0).sum())
    n_ip = int((np.abs(np.asarray(cfg['initial_probs']) - src_ip) > 0).sum())
    n_ep = int((np.abs(np.asarray(cfg['end_probs']) - src_ep) > 0).sum())
    print("   re-running set_transition/set_initial_probs/set_end_probs at lambda=1 changed "
          "%d transition, %d initial, %d end entries" % (n_changed, n_ip, n_ep))
    if n_changed or n_ip or n_ep:
        sys.exit("FIDELITY GATE FAILED: the lambda=1 rebuild is not a no-op.")
    print("   OK -- the rebuild is exact.")

    # ---- the actual patch ----
    prob, conc_new = dbf_probs(pwm, lams)
    tf_prob = apply_probs(cfg, prob)

    # ---- gate 2: blast radius. Only row `ssb` of the transition matrix may move. ----
    diff = np.abs(cfg['transition_matrix'] - src_tmat)
    rows = sorted(set(np.argwhere(diff > 0)[:, 0].tolist()))
    print("\n[gate 2] transition_matrix rows changed: %s (silent_states_begin=%d)"
          % (rows if rows else "NONE", ssb))
    if any(l != 1.0 for l in lams.values()) and rows != [ssb]:
        sys.exit("BLAST RADIUS GATE FAILED: expected only row %d to change, got %s"
                 % (ssb, rows))
    print("   OK.")

    other = np.delete(np.arange(len(tfs)), sorted(tidx.values()))
    max_other = float(np.max(np.abs(tf_prob[other] / src_tf_prob[other] - 1.0)))
    print()
    for t in sorted(lams):
        if t in tidx:
            i = tidx[t]
            print("%-18s tf_prob: %.6e -> %.6e  (x%.2f, requested x%g)"
                  % (t, src_tf_prob[i], tf_prob[i], tf_prob[i] / src_tf_prob[i], lams[t]))
        else:   # nucleosome -- reported from nucleosome_prob, it has no tf_prob slot
            print("%-18s nuc_prob: %.6e -> %.6e  (x%.2f, requested x%g)"
                  % (t, src_nuc, prob['nucleosome'], prob['nucleosome'] / src_nuc, lams[t]))
    print("background_prob : %.9f -> %.9f" % (src_bg, prob['background']))
    print("nucleosome_prob : %.6e -> %.6e" % (src_nuc, prob['nucleosome']))
    print("max |rel change| over the other %d TFs: %.3e" % (len(other), max_other))

    # ---- write ----
    os.makedirs(out, exist_ok=True)
    for fn in SIDECARS:
        src_fn = os.path.join(args.src, fn)
        if os.path.isfile(src_fn):
            shutil.copy2(src_fn, os.path.join(out, fn))
        else:
            print("   note: %s absent from %s, skipped" % (fn, args.src))
    with open(os.path.join(out, "HMMconfig.pkl"), "wb") as f:
        pickle.dump(cfg, f)

    patch = dict(
        lams=lams, src=os.path.abspath(args.src), out=os.path.abspath(out),
        silent_states_begin=ssb,
        factors={t: (dict(index=tidx[t],
                          conc_before=float(calculateKD(pwm, t)),
                          conc_after=float(conc_new[t]),
                          tf_prob_before=float(src_tf_prob[tidx[t]]),
                          tf_prob_after=float(tf_prob[tidx[t]]),
                          tf_prob_ratio=float(tf_prob[tidx[t]] / src_tf_prob[tidx[t]]))
                     if t in tidx else
                     dict(index=None,                      # nucleosome: own block, own prob
                          conc_before=35.0, conc_after=float(conc_new[t]),
                          tf_prob_before=src_nuc,
                          tf_prob_after=float(prob['nucleosome']),
                          tf_prob_ratio=float(prob['nucleosome'] / src_nuc)))
                 for t in lams},
        background_prob_before=src_bg, background_prob_after=float(prob['background']),
        nucleosome_prob_before=src_nuc, nucleosome_prob_after=float(prob['nucleosome']),
        max_rel_change_other_tfs=max_other,
        transition_rows_changed=rows,
        nucleosome_lambda_solved=nuc_lam_solved,
    )
    with open(os.path.join(out, "conc_patch.json"), "w") as f:
        json.dump(patch, f, indent=2)

    print("\nwrote %s/HMMconfig.pkl + conc_patch.json" % out)
    print("=== done ===")


if __name__ == "__main__":
    main()
