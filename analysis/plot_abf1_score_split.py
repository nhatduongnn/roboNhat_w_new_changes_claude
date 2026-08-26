#!/usr/bin/env python
"""Figure: where the ABF1 score is won and lost, site by site.

Reads analysis/abf1_score_split.tsv (written by explain_abf1_score_split.py) and
draws one grouped bar per MacIsaac chrI site: the 9 shared core columns as a
neutral base, then the 5 spacer columns as a coloured extension (gain) or a
hatched setback (penalty), against each matrix's own p<1e-4 threshold.

Palette: dataviz categorical slots 1 (blue) and 2 (orange) -- validated
ALL CHECKS PASS on the all-pairs list.  The core segment is deliberately a
neutral grey, not a third slot: it encodes 'both matrices agree here', so it
must NOT read as an identity colour.  (The validator flags it as below the
chroma floor, which is the intended reading.)
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
TSV = os.path.join(HERE, "abf1_score_split.tsv")
PNG = os.path.join(HERE, "abf1_score_split.png")

JASPAR_C = "#2a78d6"
MURPHY_C = "#eb6834"
CORE_C = "#c9c8c2"       # neutral: the part both matrices agree on
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#8a8983"
SURFACE = "#fcfcfb"

THR = {"murphy": 9.93, "jaspar": 10.30}   # score for p<1e-4, from the DP null


def main():
    with open(TSV) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    fig, ax = plt.subplots(figsize=(12.0, 6.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    h, gap = 0.34, 0.06
    yticks, ylabels = [], []

    for i, r in enumerate(rows):
        y0 = -i * 1.0
        for k, (src, col) in enumerate((("jaspar", JASPAR_C),
                                        ("murphy", MURPHY_C))):
            y = y0 - k * (h + gap)
            core = float(r["core_%s" % src])
            spac = float(r["spacer_%s" % src])
            total = float(r["score_%s" % src])
            passed = r["pass_%s" % src] == "True"

            ax.barh(y, core, height=h, color=CORE_C, edgecolor=SURFACE, lw=1.2,
                    zorder=3)
            if spac >= 0:
                ax.barh(y, spac, left=core, height=h, color=col,
                        edgecolor=SURFACE, lw=1.2, zorder=3)
            else:
                # a penalty: draw it back from the core end, hatched
                ax.barh(y, -spac, left=core + spac, height=h,
                        color=SURFACE, edgecolor=col, lw=1.4, hatch="////",
                        zorder=4)
            ax.plot([total, total], [y - h / 2, y + h / 2], color=INK, lw=1.6,
                    zorder=5, solid_capstyle="butt")
            ax.text(max(total, core) + 0.45, y,
                    "%.1f  %s" % (total, "recovered" if passed else "MISSED"),
                    va="center", ha="left", fontsize=9,
                    color=INK if passed else MURPHY_C,
                    fontweight="bold" if not passed else "normal", zorder=6)
            yticks.append(y)
            ylabels.append("JASPAR" if src == "jaspar" else "Murphy")

        tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(-0.185, y0 - (h + gap) / 2 + 0.10, "site %s" % r["site"],
                transform=tr, va="center", ha="left", fontsize=10,
                color=INK, fontweight="bold")
        ax.text(-0.185, y0 - (h + gap) / 2 - 0.16, r["seq_murphy"],
                transform=tr, va="center", ha="left", fontsize=8.5,
                color=SECOND, family="monospace")

    for src, col in (("jaspar", JASPAR_C), ("murphy", MURPHY_C)):
        ax.axvline(THR[src], color=col, ls=(0, (4, 3)), lw=1.4, zorder=2,
                   alpha=0.85)
    ax.annotate("FIMO p < 1e-4 threshold", xy=(THR["jaspar"], 0.30),
                xytext=(13.6, 0.52), fontsize=9, color=SECOND,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0,
                                shrinkA=3, shrinkB=0))
    ax.set_ylim(-4.92, 0.78)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9, color=SECOND)
    ax.set_xlabel("FIMO log-odds score (bits)", fontsize=10.5, color=SECOND)
    ax.set_xlim(-0.2, 22.5)
    ax.axvline(0, color=MUTED, lw=1.0, zorder=2)

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=SECOND, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8e7e2", lw=0.8, zorder=1)
    ax.set_axisbelow(True)

    ax.set_title("The two matrices agree on the core and disagree on the spacer",
                 fontsize=14, fontweight="bold", color=INK, loc="left", pad=34)
    ax.text(0, 1.035,
            "both matrices select the identical 14-mer at all 5 MacIsaac ABF1 "
            "sites on chrI  ·  TOMTOM calls them the same motif at q = 2.5e-07",
            transform=ax.transAxes, fontsize=10, color=SECOND, va="bottom")

    ax.legend(handles=[
        Patch(facecolor=CORE_C, label="9 core columns (shared)"),
        Patch(facecolor=JASPAR_C, label="5 spacer columns — JASPAR, gain"),
        Patch(facecolor=MURPHY_C, label="5 spacer columns — Murphy, gain"),
        Patch(facecolor=SURFACE, edgecolor=MURPHY_C, hatch="////",
              label="spacer penalty (score pulled back)"),
    ], loc="upper center", frameon=False, fontsize=9, labelcolor=SECOND,
        ncol=4, columnspacing=1.6, handlelength=1.4,
        bbox_to_anchor=(0.42, -0.115))

    fig.subplots_adjust(left=0.155, right=0.985, top=0.845, bottom=0.20)
    fig.savefig(PNG, dpi=180, facecolor=SURFACE)
    print("wrote", PNG)


if __name__ == "__main__":
    main()
