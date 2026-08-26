"""The 5 chrI MacIsaac ABF1 sites vs the user's own FIMO runs (Murphy and JASPAR).

Input: analysis/trash/fimo_{murphy,jaspar}.tsv -- FIMO 5.5.9, p<1e-4 default, run with
`--bgfile --motif--`, i.e. the background declared in each MEME header.

That background was RECOVERED FROM THE REPORTED SCORES (not assumed): least-squares
solve for log2(bg) across all hits, with FIMO's --motif-pseudo 0.1 spread by background.
Both files solve to A .300 / C .200 / G .200 / T .300 (residual rms ~0.006 bits), i.e.
rounded yeast -- the exact genome count is A .3098 C .1909 G .1906 T .3087, a difference
of hundredths of a bit. So these runs ARE comparable to chrI_fimo/ and jaspar/, which
used the exact genome composition via --bfile.

Both files use motif_id 'Abf1_murphy'; the matrices differ.

Coordinates: FIMO reports 1-based inclusive, the MacIsaac bed is 0-based half-open.
Converted with -1 on entry so both live in the bed frame used elsewhere in analysis/.

    python plot_macisaac_vs_user_fimo.py
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SITES = [(1, 45318, 45332, ""), (2, 45498, 45512, ""),
         (3, 61163, 61177, "upstream of ERV46"), (4, 62657, 62671, "downstream of ERV46"),
         (5, 108788, 108802, "")]
THRESH = 1e-4
# Murphy keeps the purple it carried in the earlier 5-site figures; JASPAR takes aqua.
# Every point is direct-labelled with its p-value and rank, so identity never rests on
# colour alone.
MURPHY_C, JASPAR_C = "#6a3fb5", "#1baf7a"
INK, MUTED, GRID, BAND = "#0b0b0b", "#6b6a66", "#e7e6e2", "#cfcec9"
RED = "#a2322b"


def load(path):
    d = pd.read_csv(path, sep="\t", comment="#")
    d = d[d["motif_id"].notna()].copy()
    d["bs"] = d["start"].astype(int) - 1
    d["be"] = d["stop"].astype(int) - 1
    return d.sort_values("p-value").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--murphy", default="trash/fimo_murphy.tsv")
    ap.add_argument("--jaspar", default="trash/fimo_jaspar.tsv")
    ap.add_argument("--out", default="macisaac_vs_user_fimo.png")
    args = ap.parse_args()
    M, J = load(args.murphy), load(args.jaspar)

    runs = [("Murphy", M, MURPHY_C), ("JASPAR", J, JASPAR_C)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.6),
                                  gridspec_kw=dict(width_ratios=[3.05, 1.0], wspace=0.28))

    ax.set_facecolor("white")
    ax.axvline(-np.log10(THRESH), color=RED, ls=":", lw=1.5, zorder=2)
    # label sits INSIDE the axes just right of the line -- above it collides with the subtitle
    ax.text(-np.log10(THRESH) + 0.12, -0.44, "FIMO default p < 1e-4", ha="left", va="center",
            fontsize=8.4, color=RED, fontweight="bold", zorder=6)
    yy = {}
    for row, (n, m0, m1, note) in enumerate(SITES):
        y = len(SITES) - 1 - row
        yy[n] = y
        ax.axhspan(y - 0.42, y + 0.42, color=BAND, alpha=0.20 if row % 2 == 0 else 0.07, zorder=0)
        for k, (lab, d, col) in enumerate(runs):
            off = 0.19 if k == 0 else -0.19
            ov = d[(d.be >= m0) & (d.bs <= m1)]
            if len(ov) == 0:
                ax.plot([0.12], [y + off], marker="o", ms=8, mfc="white", mec=col,
                        mew=1.6, zorder=4)
                ax.text(0.42, y + off, "no hit", va="center", fontsize=8.4,
                        color=col, style="italic", zorder=5)
                continue
            b = ov.iloc[0]
            rank = int(d.index[(d.bs == b.bs) & (d.strand == b.strand)][0]) + 1
            x = -np.log10(b["p-value"])
            ax.plot([0, x], [y + off, y + off], color=col, lw=2.4, alpha=0.55, zorder=3,
                    solid_capstyle="round")
            ax.plot([x], [y + off], marker="o", ms=9, color=col,
                    markeredgecolor="white", markeredgewidth=1.3, zorder=5)
            ax.text(x + 0.16, y + off, "p=%.1e   rank %d/%d" % (b["p-value"], rank, len(d)),
                    va="center", fontsize=8.4, color=col, fontweight="bold", zorder=5)
    ax.set_yticks(list(yy.values()))
    ax.set_yticklabels(["site #%d\n%d–%d%s" % (n, m0, m1, ("\n" + note) if note else "")
                        for n, m0, m1, note in SITES], fontsize=9)
    ax.set_xlabel("$-\\log_{10}$ p-value of the best FIMO hit overlapping the site", fontsize=10)
    ax.set_xlim(0, 10.4)
    ax.set_ylim(-0.6, len(SITES) - 0.25)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#b8b7b2")
    ax.tick_params(length=0, labelsize=9)
    ax.set_title("Do the FIMO hits land on the 5 MacIsaac ABF1 sites?",
                 fontsize=11.5, fontweight="bold", loc="left", pad=8)
    ax.legend(handles=[Line2D([0], [0], color=c, lw=2.4, marker="o", ms=8,
                              markeredgecolor="white", label="%s PWM" % l)
                       for l, _, c in runs] +
                      [Line2D([0], [0], color=MUTED, lw=0, marker="o", ms=8, mfc="white",
                              mec=MUTED, label="no hit at p<1e-4")],
              frameon=False, fontsize=8.8, loc="lower right", ncol=1)

    # ---- right panel: what the whole scan looks like ----
    ax2.set_facecolor("white")
    stats = []
    for lab, d, col in runs:
        rec = sum(1 for n, m0, m1, _ in SITES if len(d[(d.be >= m0) & (d.bs <= m1)]) > 0)
        gc = 100 * np.mean([(s.count("G") + s.count("C")) / 14 for s in d["matched_sequence"]])
        stats.append((lab, col, len(d), rec, int((d["q-value"] < 0.05).sum()), gc))
    rows = [("hits at p<1e-4", lambda s: "%d" % s[2]),
            ("MacIsaac found", lambda s: "%d / 5" % s[3]),
            ("q < 0.05", lambda s: "%d" % s[4]),
            ("mean GC", lambda s: "%.1f%%" % s[5])]
    ax2.text(0.5, 1.055, "whole-chrI scan summary", transform=ax2.transAxes, ha="center",
             va="bottom", fontsize=10.5, fontweight="bold", color=INK)
    for j, (lab, col, *_ ) in enumerate(stats):
        ax2.text(0.635 + 0.275 * j, 0.95, lab, ha="center", va="center", fontsize=9.5,
                 fontweight="bold", color=col, transform=ax2.transAxes)
    for i, (name, fn) in enumerate(rows):
        y = 0.80 - i * 0.175
        ax2.text(0.0, y, name, ha="left", va="center", fontsize=9.0, color=MUTED,
                 transform=ax2.transAxes)
        for j, s in enumerate(stats):
            ax2.text(0.635 + 0.275 * j, y, fn(s), ha="center", va="center", fontsize=10.5,
                     fontweight="bold", color=s[1], transform=ax2.transAxes)
    ax2.text(0.0, 0.055,
             "chrI genome GC = 39.3%.  Murphy's hits are GC-enriched;\n"
             "JASPAR's sit at/below the genome average.",
             ha="left", va="bottom", fontsize=8.4, color=MUTED, transform=ax2.transAxes)
    ax2.axis("off")

    fig.suptitle("MacIsaac ABF1 sites vs FIMO — Murphy PWM against JASPAR MA0265.3, chrI",
                 fontsize=13.5, fontweight="bold", y=1.005)
    fig.text(0.5, 0.958,
             "FIMO 5.5.9, p<1e-4, `--bgfile --motif--`. Background recovered from the reported scores "
             "by least-squares: A .300 C .200 G .200 T .300 for both runs — the AT-rich yeast "
             "composition. Ranks are within that run's own hit list.",
             ha="center", va="top", fontsize=8.6, color=MUTED)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.845, bottom=0.115)
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close("all")
    print("wrote", args.out)
    for lab, d, _ in runs:
        rec = [n for n, m0, m1, _ in SITES if len(d[(d.be >= m0) & (d.bs <= m1)]) > 0]
        print("%-7s %3d hits | sites found %s | best p %.3g | q<0.05: %d"
              % (lab, len(d), rec, d["p-value"].min(), int((d["q-value"] < 0.05).sum())))


if __name__ == "__main__":
    main()
