#!/usr/bin/env python
"""The stacked view: three ABF1 call sets, aligned on the motif, three measures.

Rows 1: median +/- IQR profiles over the motif frame (offset 0 = motif column 0, the 14 bp
block shaded): PWM 14-mer log-odds of the window starting at that offset; m6A fraction;
ABF1 posterior in each run.
Row 2: per-site distributions (box + strip) of PWM total / core / spacer and the fiber LLR.

The sequence profile is RECOMPUTED here in the motif's own frame rather than read from
abf1_stack_profile.py's npz: for a minus-strand site that script stores the window-score
array mirrored about the array centre, which shifts the window-START register by the motif
width (the m6A and posterior profiles, being per-position, mirror correctly).

Palette: dataviz categorical slots 1-3 (#2a78d6 / #eb6834 / #1baf7a), validated with
scripts/validate_palette.js --mode light --pairs all (all checks PASS; aqua's sub-3:1
contrast WARN is relieved by the direct labels and the legend).
"""
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abf1_stack_profile as AP            # BG, CORE, SPACER, log_odds, L

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SETS = [("both", "#2a78d6", "called by both"),
        ("only_bo09", "#eb6834", "bo09 only"),
        ("only_bu01", "#1baf7a", "bu01 only")]
HALF = 50
L = AP.L


def band(ax, x, mat, color, label, lw=2.0):
    m = np.nanmedian(mat, axis=0)
    q1 = np.nanpercentile(mat, 25, axis=0)
    q3 = np.nanpercentile(mat, 75, axis=0)
    ax.fill_between(x, q1, q3, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, m, color=color, lw=lw, label=label, solid_capstyle="round")
    return m


def style(ax, title, ylab, xlab=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=9.5, loc="left", pad=6)
    ax.set_ylabel(ylab, color=INK2, fontsize=8.5)
    if xlab:
        ax.set_xlabel(xlab, color=INK2, fontsize=8.5)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3, width=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    calls = pd.read_csv(sys.argv[1], sep="\t")
    prof = np.load(sys.argv[2], allow_pickle=True)
    stack = sys.argv[3]
    out = sys.argv[4]

    import pickle
    P = np.asarray(pickle.load(open(os.path.join(HERE, "robocop_train_tw_bo09_48", "pwm.p"),
                                    "rb"))["Abf1_murphy"])[:4].T
    LO, LOrc = AP.log_odds(P), AP.log_odds(P[::-1, ::-1])

    nts = {c: np.load(os.path.join(stack, "bo09_%s.npz" % c), allow_pickle=True)["nt"]
           for c in ("chrII", "chrXIV", "chrIV")}
    los = {c: int(np.load(os.path.join(stack, "bo09_%s.npz" % c),
                          allow_pickle=True)["pos"][0]) for c in nts}

    fib = {c: np.load(os.path.join(stack, "bo09_%s.npz" % c), allow_pickle=True) for c in nts}

    u = np.arange(-HALF, HALF + 1)
    seqprof = np.full((len(calls), len(u)), np.nan)
    for r_i, r in calls.reset_index(drop=True).iterrows():
        nt = nts[r["chrom"]]
        i = int(r["start"]) - los[r["chrom"]]
        mat = LO if r["strand"] == "+" else LOrc
        for k, uu in enumerate(u):
            jj = i + uu if r["strand"] == "+" else i - uu
            if jj < 0 or jj + L > len(nt):
                continue
            s = nt[jj:jj + L].astype(int)
            if (s > 3).any():
                continue
            seqprof[r_i, k] = sum(mat[q, s[q]] for q in range(L))

    pidx = {}
    for n, (s, c, ct) in enumerate(zip(prof["set"], prof["chrom"], prof["center"])):
        pidx[(s, c, int(ct))] = n
    off = np.arange(prof["m6a"].shape[1]) - HALF

    def pooled_m6a(mask):
        """sum(meth)/sum(valid) per motif-frame offset -- a ratio of pooled counts, which is
        stable where a per-site median is not (only A/T positions carry any coverage)."""
        num = np.zeros(len(off)); den = np.zeros(len(off))
        for _, r in calls[mask].iterrows():
            z = fib[r["chrom"]]
            i = int(r["start"]) - los[r["chrom"]]
            sl = slice(i - HALF, i + L + HALF)
            m = (z["meth_w"][sl] + z["meth_c"][sl]).astype(float)
            v = (z["A_w"][sl] + z["A_c"][sl]).astype(float)
            if len(m) != len(off):
                continue
            if r["strand"] == "-":
                m, v = m[::-1], v[::-1]
            num += m; den += v
        r = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        k = 5                                   # 5 bp rolling mean: only A/T positions carry
        pad = np.r_[np.full(k // 2, np.nan), r, np.full(k // 2, np.nan)]   # coverage, so the
        return np.array([np.nanmean(pad[j:j + k]) for j in range(len(r))])  # raw ratio is spiky

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, hspace=0.46, wspace=0.26,
                          left=0.052, right=0.985, top=0.825, bottom=0.085)
    axes1 = [fig.add_subplot(gs[0, j]) for j in range(4)]
    axes2 = [fig.add_subplot(gs[1, j]) for j in range(4)]

    panels1 = [("Sequence: Abf1_murphy 14-mer log-odds",
                "bits (window starting here)", None),
               ("Fiber-seq: m6A fraction (pooled counts)", "methylated A / valid A", "m6a"),
               ("ABF1 posterior in bo09 (9 states live)", "posterior", "post_a"),
               ("ABF1 posterior in bu01 (154 states live)", "posterior", "post_b")]
    for j, (title, ylab, k) in enumerate(panels1):
        ax = axes1[j]
        ax.axvspan(0, L - 1, color="#000000", alpha=0.045, linewidth=0)
        for s, col, lab in SETS:
            if k is None:
                m = calls["set"].values == s
                if m.sum() == 0:
                    continue
                band(ax, u, seqprof[m], col, "%s (n=%d)" % (lab, int(m.sum())))
            elif k == "m6a":
                m = calls["set"].values == s
                if m.sum() == 0:
                    continue
                ax.plot(off, pooled_m6a(m), color=col, lw=(1.3 if s == "both" else 2.0),
                        ls=((0, (5, 2)) if s == "both" else "-"),
                        label="%s (n=%d)" % (lab, int(m.sum())), solid_capstyle="round")
            else:
                rows = [pidx[(s, c, ct)] for s2, c, ct in
                        zip(calls["set"], calls["chrom"], calls["center"])
                        if s2 == s and (s, c, int(ct)) in pidx]
                if not rows:
                    continue
                band(ax, off, prof[k][rows], col, "%s (n=%d)" % (lab, len(rows)))
        style(ax, title, ylab, "offset from motif column 0 (bp, motif frame)")
        if k in ("post_a", "post_b"):
            ax.set_ylim(-0.03, 1.03)
        if k == "m6a":
            ax.set_ylim(0, 0.275)
            for yv, lb in ((0.24894, "background p 0.249"), (0.0843, "ABF1 fitted p 0.084"),
                           (0.08, "clc08 / unknown p 0.080")):
                ax.axhline(yv, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
            bb = dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.92)
            ax.annotate("background p 0.249", (-48, 0.2515), fontsize=7.5, color=MUTED,
                        ha="left", va="bottom", bbox=bb, zorder=6)
            ax.annotate("ABF1 p 0.084  \u2248  `unknown` p 0.080", (-48, 0.0355), fontsize=7.5,
                        color=MUTED, ha="left", va="bottom", bbox=bb, zorder=6)
        if k is None:
            ax.set_ylim(-26, 11)
            ax.annotate("bu01\u2019s calls sit exactly on a motif;\nbo09\u2019s sit on nothing",
                        (1, 5.7), xytext=(14, 8.2), fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.9), va="center")

    panels2 = [("score_total", "PWM total, all 14 columns", "bits"),
               ("score_core", "PWM core (cols 0-4, 10-13)", "bits"),
               ("score_spacer", "PWM spacer (cols 5-9)", "bits"),
               ("llr_bg", "Fiber LLR, ABF1 vs background", "nats")]
    rng = np.random.default_rng(0)
    for j, (col, title, ylab) in enumerate(panels2):
        ax = axes2[j]
        for n, (s, c, lab) in enumerate(SETS):
            v = calls.loc[calls["set"] == s, col].dropna().values
            if len(v) == 0:
                continue
            ax.scatter(n + rng.uniform(-0.17, 0.17, len(v)), v, s=14, color=c,
                       alpha=0.55, linewidths=0.5, edgecolors=SURFACE, zorder=3)
            bp = ax.boxplot([v], positions=[n], widths=0.52, showfliers=False,
                            medianprops=dict(color=INK, lw=2.0),
                            boxprops=dict(color=AXIS, lw=1.0),
                            whiskerprops=dict(color=AXIS, lw=1.0),
                            capprops=dict(color=AXIS, lw=1.0), zorder=2)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["both\nn=%d" % int((calls["set"] == "both").sum()),
                            "bo09 only\nn=%d" % int((calls["set"] == "only_bo09").sum()),
                            "bu01 only\nn=%d" % int((calls["set"] == "only_bu01").sum())],
                           fontsize=8)
        style(ax, title, ylab)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        for n, (s, c, lab) in enumerate(SETS):
            v = calls.loc[calls["set"] == s, col].dropna().values
            if len(v):
                ax.annotate("med %.2f" % np.median(v), (n, np.median(v)),
                            xytext=(0, 9), textcoords="offset points", fontsize=8,
                            color=INK, ha="center", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none"))

    fig.suptitle("ABF1 posterior \u2265 0.5: bo09 and bu01 call almost disjoint site sets \u2014 "
                 "bu01\u2019s are the ones with the motif",
                 color=INK, fontsize=15.5, x=0.052, ha="left", y=0.975)
    fig.text(0.052, 0.947,
             "chrII + chrXIV + chrIV.  Top row: per-set profile on the register the decode itself "
             "used (argmax over the 28 ABF1 states); band = IQR, grey block = the 14 bp motif.  "
             "Bottom row: one dot per call.",
             color=INK2, fontsize=9.5, va="top")
    fig.text(0.052, 0.926,
             "bo09 = pkgvar/seq_maskoff_fit9_bgopen (9 TF states live, `unknown` masked, ABF1 "
             "tf_prob 1.4e-51).   bu01 = pkgvar/seq_maskoff_all153 (154 live, "
             "combined_low_count 0.08, ABF1 tf_prob 3.37e-05).",
             color=INK2, fontsize=9.5, va="top")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=c, lw=2.6,
                               label="%s  (n=%d)" % (lab, int((calls["set"] == s).sum())))
                        for s, c, lab in SETS],
               loc="upper left", bbox_to_anchor=(0.052, 0.905), ncol=3, frameon=False,
               fontsize=10, labelcolor=INK, handlelength=1.6, columnspacing=2.4)
    fig.savefig(out, facecolor=SURFACE)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
