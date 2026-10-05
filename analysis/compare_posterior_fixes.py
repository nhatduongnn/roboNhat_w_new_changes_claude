"""Three-way comparison of the posterior-normalisation fixes, on one window.

A  read-side normalisation in score_robocop.region_optable; decode is the UNCHANGED binary,
   so the fix is applied when the stored posterior is read.
B  linear normalisation inside posterior_decoding (librobocop_norm.so).
C  log-space with log-sum-exp inside posterior_decoding (librobocop_logspace.so).

All three read the same trainDir and the same coords, so for each layer configuration the only
difference is where (and how) the row normalisation happens.

What to expect, written down before running:
  * B vs C near-exact (~1e-12 relative) -- same formula, different arithmetic path.
  * A vs B/C may differ in the TAIL only: A normalises AFTER robocop.py:33 zeroes stored values
    below 1e-4, B and C before it. On a healthy row the floor removes true values < 1e-4; on a
    blown-up row it corresponds to a far smaller true value and removes almost nothing.

A disagreement beyond the tail means the damage was never a clean scale factor, and A would be
unsafe as a retroactive fix.

    python compare_posterior_fixes.py --prefix robocop_w30k_bv75_49 --chrom chrXIV
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_robocop as S                                              # noqa: E402

LAYERS = [("both", "both layers"), ("fib", "fiber only"), ("seq", "sequence only")]
ARMS = ["A", "B", "C"]


def optable(outdir, chrom, start, end):
    dec = S.load_decode(outdir)
    tab, covered, _ = S.region_optable(dec, chrom, start, end)
    return tab, covered


def compare(x, y, cols):
    """max abs / max rel difference, and which column carries the worst relative one."""
    a, b = x[cols].to_numpy(float), y[cols].to_numpy(float)
    d = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    rel = np.divide(d, denom, out=np.zeros_like(d), where=denom > 1e-12)
    k = int(np.unravel_index(np.argmax(rel), rel.shape)[1]) if rel.size else 0
    return d.max(), rel.max(), cols[k]


def calls(tab, col, thr):
    v = tab[col].to_numpy(float)
    return set(np.nonzero(v >= thr)[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="robocop_w30k_bv75_49")
    ap.add_argument("--chrom", default="chrXIV")
    ap.add_argument("--start", type=int, default=400001)
    ap.add_argument("--end", type=int, default=430000)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--factor", default="Abf1_murphy")
    a = ap.parse_args()

    for lay, label in LAYERS:
        print("\n" + "=" * 78)
        print("LAYER: %s   (%s:%d-%d)" % (label, a.chrom, a.start, a.end))
        print("=" * 78)
        tabs = {}
        for arm in ARMS:
            d = "%s_%s_%s" % (a.prefix, arm, lay)
            if not os.path.isdir(d):
                print("  %s  MISSING (%s)" % (arm, d))
                continue
            try:
                tabs[arm], _ = optable(d, a.chrom, a.start, a.end)
            except Exception as e:                                      # noqa: BLE001
                print("  %s  FAILED: %s" % (arm, e))
        if len(tabs) < 2:
            continue

        cols = [c for c in list(tabs.values())[0].columns
                if all(c in t.columns for t in tabs.values())]

        print("\n  pairwise differences over %d factor columns x %d bp" % (len(cols), a.end - a.start + 1))
        print("  %-10s %14s %14s   %s" % ("pair", "max abs", "max rel", "worst column"))
        for i in range(len(ARMS)):
            for j in range(i + 1, len(ARMS)):
                p, q = ARMS[i], ARMS[j]
                if p in tabs and q in tabs:
                    da, dr, w = compare(tabs[p], tabs[q], cols)
                    print("  %-10s %14.3e %14.3e   %s" % ("%s vs %s" % (p, q), da, dr, w))

        if a.factor in cols:
            print("\n  %s calls at posterior >= %.2f" % (a.factor, a.thr))
            cs = {k: calls(t, a.factor, a.thr) for k, t in tabs.items()}
            for k in ARMS:
                if k in cs:
                    print("    %s : %d positions" % (k, len(cs[k])))
            for i in range(len(ARMS)):
                for j in range(i + 1, len(ARMS)):
                    p, q = ARMS[i], ARMS[j]
                    if p in cs and q in cs:
                        only_p, only_q = cs[p] - cs[q], cs[q] - cs[p]
                        verdict = "identical" if not (only_p or only_q) else \
                                  "DIFFER: %d only in %s, %d only in %s" % (len(only_p), p, len(only_q), q)
                        print("    %s vs %s : %s" % (p, q, verdict))

        print("\n  summed posterior per factor (the `occ` numerator), top 6 by magnitude")
        tot = {k: t[cols].to_numpy(float).sum(axis=0) for k, t in tabs.items()}
        order = np.argsort(-list(tot.values())[0])[:6]
        hdr = "  %-16s" % "factor" + "".join("%14s" % k for k in ARMS if k in tot)
        print(hdr)
        for idx in order:
            row = "  %-16s" % cols[idx][:16]
            for k in ARMS:
                if k in tot:
                    row += "%14.4f" % tot[k][idx]
            print(row)


if __name__ == "__main__":
    main()
