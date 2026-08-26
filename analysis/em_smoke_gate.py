"""Pass/fail gate on a short EM run, so a bad fit never spends a full training job.

`update_transition_probs` has never executed in this codebase (`robocop_em.py` ships
`iterations = 0`), so the first real question is simply whether the Baum-Welch step runs
and behaves. This asserts the things that must hold after any correct EM run, and exits
non-zero otherwise -- which is what lets the full job be chained behind it with
`sbatch --dependency=afterok`.

Checks:
  1. one em_trace/iter{i}.npz per iteration, plus iter0
  2. likelihood.txt has iterations+1 entries and is monotonically non-decreasing
  3. iter0's prior is bit-identical to the unfitted baseline trainDir
  4. the prior actually moved -- if nothing changed, EM silently did nothing
  5. every prior stays finite, non-negative, and sums with background+nucleosome to 1

Usage
-----
    python em_smoke_gate.py robocop_train_em10_smoke --iters 2
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np

FAILURES = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        FAILURES.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trainDir")
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--baseline", default="robocop_train_fiberonly")
    args = ap.parse_args()

    print("EM smoke gate on %s (expecting %d iterations)" % (args.trainDir, args.iters))

    files = sorted(glob.glob(os.path.join(args.trainDir, "em_trace", "iter*.npz")),
                   key=lambda p: int(re.search(r"iter(\d+)\.npz$", p).group(1)))
    check(len(files) == args.iters + 1,
          "%d trace files (expected %d)" % (len(files), args.iters + 1))
    if not files:
        sys.exit("no trace at all -- EM did not reach the loop")

    mats, bgs, nucs = [], [], []
    tfs = None
    for p in files:
        z = np.load(p)
        if tfs is None:
            tfs = [str(x) for x in z["tfs"]]
        mats.append(np.asarray(z["tf_prob"], dtype=float))
        bgs.append(float(z["background_prob"]))
        nucs.append(float(z["nucleosome_prob"]))
    mats = np.array(mats)

    lltxt = os.path.join(args.trainDir, "likelihood.txt")
    lls = [float(x) for x in open(lltxt)] if os.path.isfile(lltxt) else []
    check(len(lls) == args.iters + 1,
          "likelihood.txt has %d entries (expected %d)" % (len(lls), args.iters + 1))
    if len(lls) > 1:
        d = np.diff(lls)
        check(d.min() >= 0,
              "log-likelihood non-decreasing (%.4f -> %.4f, worst step %+.4f)"
              % (lls[0], lls[-1], d.min()))

    if os.path.isdir(args.baseline):
        base = pickle.load(open(os.path.join(args.baseline, "HMMconfig.pkl"), "rb"),
                           encoding="latin1")
        bp = np.ravel(base["tf_prob"]).astype(float)
        btfs = [str(x) for x in base["tfs"]]
        check(btfs == tfs, "state list matches %s" % args.baseline)
        if btfs == tfs:
            check(np.array_equal(mats[0], bp),
                  "iter0 prior bit-identical to the unfitted %s" % args.baseline)
            i = tfs.index("Abf1_murphy") if "Abf1_murphy" in tfs else None
            if i is not None:
                check(mats[0][i] == 1.79736716e-07 or abs(mats[0][i] / 1.79736716e-07 - 1) < 1e-9,
                      "iter0 Abf1_murphy = %.10g (unpatched calculateKD value)" % mats[0][i])

    moved = int((mats[-1] != mats[0]).sum())
    check(moved > 0, "%d of %d priors changed (EM actually updated something)"
          % (moved, len(tfs)))

    check(np.isfinite(mats).all() and (mats >= 0).all(),
          "all priors finite and non-negative")
    tot = mats[-1].sum() + bgs[-1] + nucs[-1]
    check(abs(tot - 1.0) < 1e-6,
          "final tf_prob + background + nucleosome sums to %.10f" % tot)

    if "Nhp6a_zhu" in tfs and "Abf1_murphy" in tfs:
        na, ab = tfs.index("Nhp6a_zhu"), tfs.index("Abf1_murphy")
        r0 = mats[0][na] / mats[0][ab]
        r1 = mats[-1][na] / mats[-1][ab] if mats[-1][ab] > 0 else float("inf")
        print("\n  Nhp6a/Abf1 prior ratio: %.1fx -> %.1fx   (Abf1 %.4e -> %.4e)"
              % (r0, r1, mats[0][ab], mats[-1][ab]))

    print()
    if FAILURES:
        print("SMOKE GATE FAILED (%d):" % len(FAILURES))
        for m in FAILURES:
            print("  - " + m)
        sys.exit(1)
    print("SMOKE GATE PASSED")


if __name__ == "__main__":
    main()
