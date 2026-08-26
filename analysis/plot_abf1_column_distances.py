#!/usr/bin/env python
"""Figure: which columns each metric blames for the Murphy/JASPAR ABF1 gap.

ED and KLD are in different units (probability distance vs nats), so plotting
them raw on one axis would be a dual-scale chart.  Instead each column's
contribution is shown as a SHARE of that metric's own 14-column total -- the
question here is where the weight lands, not how big the number is.

Palette: dataviz categorical slots 1 (blue) and 2 (orange); ALL CHECKS PASS.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "abf1_column_distances.tsv")
PNG = os.path.join(HERE, "abf1_column_distances.png")

ED_C = "#2a78d6"
KL_C = "#eb6834"
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#8a8983"
SURFACE = "#fcfcfb"

MURPHY_CONS = "ATCACATGGCACGA"
JASPAR_CONS = "ATCACTATATACGA"


def main():
    with open(TSV) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    ed = [float(r["ed"]) for r in rows]
    kl = [float(r["kld"]) for r in rows]
    eds = [100 * v / sum(ed) for v in ed]
    kls = [100 * v / sum(kl) for v in kl]
    spacer = [r["region"] == "spacer" for r in rows]

    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = range(14)
    w = 0.38
    ax.axvspan(4.5, 9.5, color="#f0efe9", zorder=0)
    ax.text(7.0, 27.6, "the 5 spacer columns", ha="center", va="center",
            fontsize=10, color=SECOND, style="italic")

    ax.bar([i - w / 2 - 0.012 for i in x], eds, width=w, color=ED_C,
           edgecolor=SURFACE, lw=1.0, zorder=3, label="Euclidean distance")
    ax.bar([i + w / 2 + 0.012 for i in x], kls, width=w, color=KL_C,
           edgecolor=SURFACE, lw=1.0, zorder=3, label="KL divergence")

    for i in x:
        if spacer[i]:
            ax.text(i - w / 2 - 0.012, eds[i] + 0.6, "%.0f" % eds[i],
                    ha="center", va="bottom", fontsize=8.5, color=ED_C)
            ax.text(i + w / 2 + 0.012, kls[i] + 0.6, "%.0f" % kls[i],
                    ha="center", va="bottom", fontsize=8.5, color=KL_C)

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        ["%d\n%s\n%s" % (i, MURPHY_CONS[i], JASPAR_CONS[i]) for i in x],
        fontsize=9, color=SECOND, family="monospace", linespacing=1.6)
    ax.text(-1.15, -1.05, "col\nMurphy\nJASPAR", fontsize=8.5, color=MUTED,
            ha="left", va="top", linespacing=1.6)

    ax.set_ylabel("share of that metric's total 14-column distance (%)",
                  fontsize=10.5, color=SECOND)
    ax.set_ylim(0, 30)
    ax.set_xlim(-1.2, 13.7)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=SECOND, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#e8e7e2", lw=0.8, zorder=1)
    ax.set_axisbelow(True)

    ax.set_title("Both metrics blame the spacer — KL divergence blames it harder",
                 fontsize=14, fontweight="bold", color=INK, loc="left", pad=34)
    ax.text(0, 1.035,
            "ABF1: Murphy vs JASPAR MA0265.3  ·  spacer share of the total: "
            "ED 85.7%,  KL 95.4%  ·  spacer-to-core weight ratio: "
            "ED 10.8×,  KL 37.7×",
            transform=ax.transAxes, fontsize=10, color=SECOND, va="bottom")

    ax.legend(handles=[Patch(facecolor=ED_C, label="Euclidean distance (TOMTOM default)"),
                       Patch(facecolor=KL_C, label="KL divergence (-dist kullback)")],
              loc="upper left", frameon=False, fontsize=9.5, labelcolor=SECOND,
              bbox_to_anchor=(0.005, 0.99))

    fig.subplots_adjust(left=0.085, right=0.985, top=0.845, bottom=0.16)
    fig.savefig(PNG, dpi=180, facecolor=SURFACE)
    print("wrote", PNG)


if __name__ == "__main__":
    main()
