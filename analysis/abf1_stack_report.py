#!/usr/bin/env python
"""Summary tables for the ABF1 call-set comparison (abf1_stack_profile.py output).

References used
  macisaac_slice : inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed, as shipped (CLAUDE.md rule 5)
  macisaac_full  : MacIsaac_p005_c1_V64_SGD.gff3 merged sites, make_conc_targets.load_macisaac_c1_sites
  rossi_pwm      : inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed
                   -- NOT independent of the fiber model: HANDOFF.md:343 says these are the very
                   sites ABF1's fitted fiber footprint was trained on.
  rossi_cx       : /usr/project/xtmp/nd141/projects/data/rossi_strand/Abf1_CX.bed -- Rossi's merged
                   ChExMix summits, no motif anchoring, so independent of both the PWM and the
                   fiber fit. Chromosomes are arabic there and roman here.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROMAN = {"chrI": 1, "chrII": 2, "chrIII": 3, "chrIV": 4, "chrV": 5, "chrVI": 6, "chrVII": 7,
         "chrVIII": 8, "chrIX": 9, "chrX": 10, "chrXI": 11, "chrXII": 12, "chrXIII": 13,
         "chrXIV": 14, "chrXV": 15, "chrXVI": 16}
CX = "/usr/project/xtmp/nd141/projects/data/rossi_strand/Abf1_CX.bed"
TOL = 30
CHROMS = ["chrII", "chrXIV", "chrIV"]


def load_extra():
    out = {"macisaac_full": {}, "rossi_cx": {}}
    import make_conc_targets as T
    for c, tf, center in T.load_macisaac_c1_sites():
        if tf == "ABF1" and c in CHROMS:
            out["macisaac_full"].setdefault(c, []).append(center)
    inv = {"chr%d" % v: k for k, v in ROMAN.items()}
    df = pd.read_csv(CX, sep="\t", header=None,
                     names=["chr", "start", "end", "name", "score", "strand"])
    df["rom"] = df["chr"].map(inv)
    for c, g in df[df["rom"].isin(CHROMS)].groupby("rom"):
        out["rossi_cx"][c] = sorted(((g["start"] + g["end"]) // 2).tolist())
    for k in out:
        for c in out[k]:
            out[k][c] = np.sort(np.asarray(out[k][c]))
        print("%-14s %s" % (k, {c: len(v) for c, v in out[k].items()}))
    return out


def nearest(arr, x):
    if arr is None or len(arr) == 0:
        return 10 ** 9
    i = int(np.searchsorted(arr, x))
    return min(abs(int(arr[j]) - x) for j in (i - 1, i, i + 1) if 0 <= j < len(arr))


def q(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return len(v), np.percentile(v, 25), np.median(v), np.percentile(v, 75), v.mean()


def best_register(df, stack):
    """Best Abf1_murphy 14-mer within +/-WINSEARCH of the decode's register, either strand.

    The decode's own register is the right place to ask "what sequence did this state sit on",
    but it is only informative if the state was placed BY the sequence. For a call placed by
    the fiber layer the register is arbitrary, so this adds the fairer question: is there an
    ABF1 motif ANYWHERE near the call? Columns best_* carry the answer.
    """
    import pickle
    import abf1_stack_profile as AP
    P = np.asarray(pickle.load(open(os.path.join(HERE, "robocop_train_tw_bo09_48", "pwm.p"),
                                    "rb"))["Abf1_murphy"])[:4].T
    LO, LOrc = AP.log_odds(P), AP.log_odds(P[::-1, ::-1])
    nts = {c: np.load(os.path.join(stack, "bo09_%s.npz" % c), allow_pickle=True)["nt"]
           for c in CHROMS}
    lo0 = {c: int(np.load(os.path.join(stack, "bo09_%s.npz" % c),
                          allow_pickle=True)["pos"][0]) for c in CHROMS}
    W = 25
    out = []
    for _, r in df.iterrows():
        nt = nts[r["chrom"]]
        i = int(r["start"]) - lo0[r["chrom"]]
        best = (-1e9, 0, "+", None)
        for off in range(-W, W + 1):
            for st, mat in (("+", LO), ("-", LOrc)):
                s2 = nt[i + off:i + off + 14].astype(int)
                if len(s2) < 14 or (s2 > 3).any():
                    continue
                per = np.array([mat[q, s2[q]] for q in range(14)])
                if per.sum() > best[0]:
                    best = (per.sum(), off, st, per)
        per = best[3]
        out.append(dict(best_score=best[0], best_off=best[1], best_strand=best[2],
                        best_core=per[CORE_C].sum(), best_spacer=per[SPACER_C].sum()))
    return pd.DataFrame(out, index=df.index)


CORE_C = list(range(0, 5)) + list(range(10, 14))
SPACER_C = list(range(5, 10))


def main():
    df = pd.read_csv(sys.argv[1], sep="\t")
    df = pd.concat([df, best_register(df, sys.argv[2])], axis=1)
    extra = load_extra()
    for k in extra:
        df[k] = [nearest(extra[k].get(c), int(x)) for c, x in zip(df["chrom"], df["center"])]

    order = ["both", "only_bo09", "only_bu01"]
    print("\n=== SET SIZES (resolved register, per chromosome) ===")
    print(pd.crosstab(df["set"], df["chrom"]).reindex(order)[CHROMS].to_string())
    print("  totals:", {s: int((df["set"] == s).sum()) for s in order})

    print("\n=== STRAND / REGISTER ===")
    print(pd.crosstab(df["set"], df["strand"]).reindex(order).to_string())

    print("\n=== BEST PWM REGISTER vs THE DECODE'S REGISTER (|offset|, bp, search +/-25) ===")
    print("%-11s %5s %7s %7s %7s %7s" % ("set", "n", "Q1", "median", "Q3", "frac off=0"))
    for s in order:
        d = df[df["set"] == s]
        v = np.abs(d["best_off"].values)
        print("%-11s %5d %7.1f %7.1f %7.1f %7.2f"
              % (s, len(d), np.percentile(v, 25), np.median(v), np.percentile(v, 75),
                 float((d["best_off"] == 0).mean())))
    print("  strand agreement decode vs best-PWM:",
          {s: round(float((df[df["set"] == s]["best_strand"] ==
                           df[df["set"] == s]["strand"]).mean()), 3) for s in order})

    for name, col in [("PWM log-odds TOTAL (14 col, bits)", "score_total"),
                      ("PWM BEST-IN-WINDOW total (bits)", "best_score"),
                      ("PWM BEST-IN-WINDOW core (bits)", "best_core"),
                      ("PWM BEST-IN-WINDOW spacer (bits)", "best_spacer"),
                      ("PWM log-odds CORE (9 col: 0-4,10-13)", "score_core"),
                      ("PWM log-odds SPACER (5 col: 5-9)", "score_spacer"),
                      ("fiber LLR vs background, nats", "llr_bg"),
                      ("fiber LLR vs clc08=0.08, nats", "llr_clc"),
                      ("m6A fraction over the 14 bp", "m6a"),
                      ("peak posterior, bo09", "post_a"),
                      ("peak posterior, bu01", "post_b")]:
        print("\n=== %s ===" % name)
        print("%-11s %5s %9s %9s %9s %9s" % ("set", "n", "Q1", "median", "Q3", "mean"))
        for s in order:
            n, a, m, b, mu = q(df[df["set"] == s][col])
            print("%-11s %5d %9.3f %9.3f %9.3f %9.3f" % (s, n, a, m, b, mu))
        for s1, s2 in [("both", "only_bu01"), ("both", "only_bo09"), ("only_bo09", "only_bu01")]:
            x = df[df["set"] == s1][col].dropna()
            y = df[df["set"] == s2][col].dropna()
            u = stats.mannwhitneyu(x, y, alternative="two-sided")
            print("   %-10s vs %-10s  Mann-Whitney U p = %.3g   (median %.3f vs %.3f)"
                  % (s1, s2, u.pvalue, np.median(x), np.median(y)))

    print("\n=== REFERENCE AGREEMENT (fraction of calls with a site within %d bp) ===" % TOL)
    refs = ["d_macisaac", "macisaac_full", "d_rossi", "rossi_cx"]
    print("%-11s %5s %s" % ("set", "n", "  ".join("%-14s" % r for r in refs)))
    for s in order:
        d = df[df["set"] == s]
        cells = []
        for r in refs:
            k = int((d[r] <= TOL).sum())
            cells.append("%-14s" % ("%d/%d  %4.1f%%" % (k, len(d), 100.0 * k / max(1, len(d)))))
        print("%-11s %5d %s" % (s, len(d), "  ".join(cells)))

    print("\n=== WHAT THE OTHER RUN PUTS THERE (max posterior over the 14 bp) ===")
    for s in ["only_bo09", "only_bu01"]:
        d = df[df["set"] == s]
        print("\n-- %s (n=%d); 'other' = %s" % (s, len(d), "bu01" if s == "only_bo09" else "bo09"))
        print("   top-1 state taken by the other run:")
        print(d["other_top1"].value_counts().head(12).to_string())
        for c in ["other_abf1", "other_unknown", "other_nuc", "other_bg", "other_top1_p"]:
            n, a, m, b, mu = q(d[c])
            print("   %-14s median %6.3f  Q1 %6.3f  Q3 %6.3f  mean %6.3f" % (c, m, a, b, mu))

    print("\n=== PER-CALL TABLE, only_bu01 sorted by PWM total ===")
    cols = ["chrom", "center", "strand", "seq", "score_total", "score_core", "score_spacer",
            "llr_bg", "m6a", "post_b", "other_top1", "other_top1_p", "d_macisaac", "rossi_cx"]
    print(df[df["set"] == "only_bu01"][cols].sort_values("score_total").to_string(index=False))
    print("\n=== PER-CALL TABLE, only_bo09 sorted by PWM total ===")
    cols2 = [c if c != "post_b" else "post_a" for c in cols]
    print(df[df["set"] == "only_bo09"][cols2].sort_values("score_total").to_string(index=False))

    out = sys.argv[1].replace(".tsv", "_withrefs.tsv")
    df.to_csv(out, sep="\t", index=False, float_format="%.5f")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
