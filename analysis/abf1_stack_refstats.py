#!/usr/bin/env python
"""Reference recall/precision per RUN (not per set) + Fisher tests on the set agreements.

Recall here is score_robocop.match_peaks (greedy one-to-one, tol 30) of each run's whole
posterior>=0.5 call list against each ABF1 reference, pooled over chrII+chrXIV+chrIV.
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_robocop as S
import abf1_stack_sets as SETS
import abf1_stack_report as R

CHROMS = ["chrII", "chrXIV", "chrIV"]
TOL = 30

stack, callstsv = sys.argv[1], sys.argv[2]
extra = R.load_extra()
ref = {k: {} for k in ("macisaac_slice", "rossi_pwm", "macisaac_full", "rossi_cx")}
import abf1_stack_profile as AP
for c in CHROMS:
    m, ro = AP.ref_centers(c)
    ref["macisaac_slice"][c] = m
    ref["rossi_pwm"][c] = ro
    ref["macisaac_full"][c] = extra["macisaac_full"].get(c, np.array([]))
    ref["rossi_cx"][c] = extra["rossi_cx"].get(c, np.array([]))

print("\n=== PER-RUN CALLS AND REFERENCE MATCHES (posterior>=0.5, tol %d bp, 3 chromosomes) ===" % TOL)
hdr = "%-6s %-7s" % ("run", "thr") + "".join("%-26s" % k for k in ref)
print(hdr)
for thr in (0.5, 0.10):
    for tag in ("bo09", "bu01"):
        calls = {c: [x["center"] for x in SETS.calls_for(stack, tag, c, thr)[0]] for c in CHROMS}
        n = sum(len(v) for v in calls.values())
        cells = []
        for k in ref:
            tp = nr = 0
            for c in CHROMS:
                a = S.match_peaks(calls[c], list(ref[k][c]), TOL)
                tp += a["tp"]; nr += a["n_ref"]
            cells.append("%-26s" % ("tp %3d /ref %3d  R %.3f P %.3f" % (tp, nr, tp / nr, tp / n)))
        print("%-6s %-7s" % (tag, thr) + "".join(cells) + "   (calls %d)" % n)

df = pd.read_csv(callstsv, sep="\t")
print("\n=== FISHER EXACT on 'call has a reference site within %d bp', only_bo09 vs only_bu01 ===" % TOL)
for k, col in [("macisaac_slice", "d_macisaac"), ("rossi_pwm", "d_rossi"),
               ("macisaac_full", "macisaac_full"), ("rossi_cx", "rossi_cx")]:
    a = df[df["set"] == "only_bo09"][col] <= TOL
    b = df[df["set"] == "only_bu01"][col] <= TOL
    t = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
    print("  %-15s bo09 %2d/%2d vs bu01 %2d/%2d   Fisher p = %.3f"
          % (k, t[0][0], len(a), t[1][0], len(b), stats.fisher_exact(t)[1]))

print("\n=== NUMERICAL HEALTH: were the OTHER run's posteriors valid at these positions? ===")
for chrom in CHROMS:
    for tag, other in (("bo09", "bu01"), ("bu01", "bo09")):
        z = np.load(os.path.join(stack, "%s_%s.npz" % (other, chrom)), allow_pickle=True)
        ok, pos = z["ok"], z["pos"]
        sub = df[(df["chrom"] == chrom) & (df["set"] == "only_%s" % tag)]
        bad = int((~ok[sub["center"].values - pos[0]]).sum())
        print("  only_%s on %-7s n=%2d : %d sit where %s's posterior is INVALID"
              % (tag, chrom, len(sub), bad, other))
