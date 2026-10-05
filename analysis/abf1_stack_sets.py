#!/usr/bin/env python
"""Partition ABF1 posterior>=0.5 calls of two runs into both / only-A / only-B.

Calls use score_robocop.call_abf1 (the repo's single call-caller, `track >= threshold`,
score_robocop.py:322-338/361-372) on the INVALID-position-dropped track, exactly as
count_calls.py:150-153 does -- only the threshold differs (0.5 here, FIXED_THRESHOLD=0.10
there).

Matching is SYMMETRIC global-greedy: all cross-run pairs within MATCH_TOL (=30, the repo
convention, count_calls.py:65) sorted by distance, taken one-to-one closest-first. Unlike
score_robocop.match_peaks (which walks the reference list in coordinate order) this does not
depend on which run is called "ref"; the match counts from both are printed for cross-check.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_robocop as S

MATCH_TOL = 30
THRESH = 0.5
CHROMS = ["chrII", "chrXIV", "chrIV"]


def calls_for(stackdir, tag, chrom, thresh=THRESH):
    z = np.load(os.path.join(stackdir, "%s_%s.npz" % (tag, chrom)), allow_pickle=True)
    ok = z["ok"]
    tr, p = z["abf1"][ok], z["pos"][ok]
    runs = S.call_abf1(tr, p, thresh)
    for r in runs:
        i = int(np.searchsorted(p, r["center"]))
        r["peak"] = float(tr[max(0, i - 25):i + 26].max())
        r["width"] = r["end"] - r["start"] + 1
    return runs, int(z["n_invalid"]), len(p)


def pair(a, b, tol=MATCH_TOL):
    """-> (pairs, only_a_idx, only_b_idx). Closest-first one-to-one."""
    cand = []
    bc = np.array([x["center"] for x in b])
    for i, x in enumerate(a):
        j0, j1 = np.searchsorted(bc, [x["center"] - tol, x["center"] + tol + 1])
        for j in range(j0, j1):
            cand.append((abs(x["center"] - b[j]["center"]), i, j))
    cand.sort()
    ua, ub, pairs = set(), set(), []
    for d, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); pairs.append((i, j, d))
    return pairs, [i for i in range(len(a)) if i not in ua], [j for j in range(len(b)) if j not in ub]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--a", default="bo09")
    ap.add_argument("--b", default="bu01")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    print("%-7s %-8s %8s %8s %8s %8s %8s %8s" %
          ("chrom", "", "calls", "invalid", "valid_bp", "both", "only_A", "only_B"))
    for chrom in CHROMS:
        ca, na, va = calls_for(args.stack, args.a, chrom)
        cb, nb, vb = calls_for(args.stack, args.b, chrom)
        ca.sort(key=lambda r: r["center"]); cb.sort(key=lambda r: r["center"])
        pairs, oa, ob = pair(ca, cb)
        print("%-7s %-8s %8d %8d %8d" % (chrom, args.a, len(ca), na, va))
        print("%-7s %-8s %8d %8d %8d %8d %8d %8d"
              % (chrom, args.b, len(cb), nb, vb, len(pairs), len(oa), len(ob)))
        # cross-check against score_robocop.match_peaks (asymmetric, both directions)
        m1 = S.match_peaks([x["center"] for x in ca], [x["center"] for x in cb], MATCH_TOL)
        m2 = S.match_peaks([x["center"] for x in cb], [x["center"] for x in ca], MATCH_TOL)
        print("        match_peaks tp: A-as-pred %d | B-as-pred %d | symmetric pairs %d"
              % (m1["tp"], m2["tp"], len(pairs)))
        for i, j, d in pairs:
            rows.append(dict(set="both", chrom=chrom, center_a=ca[i]["center"], center_b=cb[j]["center"],
                             center=ca[i]["center"], d=d, peak_a=ca[i]["peak"], peak_b=cb[j]["peak"],
                             width_a=ca[i]["width"], width_b=cb[j]["width"]))
        for i in oa:
            rows.append(dict(set="only_%s" % args.a, chrom=chrom, center_a=ca[i]["center"], center_b=-1,
                             center=ca[i]["center"], d=-1, peak_a=ca[i]["peak"], peak_b=float("nan"),
                             width_a=ca[i]["width"], width_b=-1))
        for j in ob:
            rows.append(dict(set="only_%s" % args.b, chrom=chrom, center_a=-1, center_b=cb[j]["center"],
                             center=cb[j]["center"], d=-1, peak_a=float("nan"), peak_b=cb[j]["peak"],
                             width_a=-1, width_b=cb[j]["width"]))
    cols = ["set", "chrom", "center", "center_a", "center_b", "d",
            "peak_a", "peak_b", "width_a", "width_b"]
    with open(args.out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(("%.4f" % r[c]) if isinstance(r[c], float) else str(r[c])
                               for c in cols) + "\n")
    from collections import Counter
    print("\nTOTALS", Counter(r["set"] for r in rows))
    print("wrote %s (%d rows)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
