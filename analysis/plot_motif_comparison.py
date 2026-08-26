"""Figures for the three-way motif comparison (shipped vs JASPAR vs Rossi).

Reads motif_comparison.tsv from compare_motif_sources.py.

  1  agreement_vs_behaviour.png   TOMTOM q (does it match) vs r_track (does it behave the
                                  same). The whole point: these are different questions,
                                  and ABF1 is the worked proof.
  2  agreement_by_source.png      r_track distribution per motif source -- how bad is the
                                  problem, and is it concentrated in one collection.
  3  fitted_tfs.png               the 12 TFs with Fiber-seq parameters, agreement with
                                  JASPAR and with Rossi side by side.
  4  logos_worked_examples.png    aligned logos for the clearest cases.

A NULL IS COMPUTED, NOT ASSUMED. r_track between two unrelated yeast motifs is not 0 --
the genome is AT-rich, so any two AT-rich matrices correlate somewhat. The null band is
sampled from random mismatched pairs so "0.84" can be read against something.

    conda activate pyranges_env3        # needs logomaker
    python plot_motif_comparison.py
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import logomaker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_motif_dbs import parse_meme, compute_background
from compare_motif_sources import (read_chrom, encode, matrix_of, log_odds,
                                   best_track_corr, FITTED)

# Categorical slots 1-3 of the validated default palette. A scatter uses the all-pairs
# pairlist, where only the first three slots clear the floors, so the handful of motifs
# that are neither zhu/badis/murphy fold into a neutral "Other" rather than taking a
# fourth hue. #1baf7a sits below 3:1 on the light surface -> the relief rule applies, met
# by direct labels on every called-out point plus motif_comparison.tsv as the table view.
SRC_COLOR = {"murphy": "#2a78d6", "zhu": "#eb6834", "badis": "#1baf7a", "other": "#8a8983"}
SURFACE = "#fcfcfb"
INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#6b6a66", "#e7e6e2"
NULLBAND = "#d8d7d2"


def srcof(s):
    return s if s in ("murphy", "zhu", "badis") else "other"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b8b7b2")
    ax.tick_params(length=0, labelsize=9, colors=SECOND)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- null

def compute_null(cache, n_pairs=200, seed=0):
    if os.path.isfile(cache):
        return json.load(open(cache))
    print("  sampling null from %d mismatched pairs..." % n_pairs, flush=True)
    bgd = compute_background("inputs/SacCer3.fa")
    bg = np.array([bgd[b] for b in "ACGT"])
    code = encode(read_chrom("inputs/SacCer3.fa", "chrI"))
    sh = parse_meme(open("motifdb/shipped.meme").read())
    ja = parse_meme(open("motifdb/jaspar.meme").read())
    rnd = random.Random(seed)
    vals = []
    for k in range(n_pairs):
        a, b = rnd.choice(sh), rnd.choice(ja)
        if a["name"].rsplit("_", 1)[0].upper() == b["alt"].upper():
            continue
        r, _, _ = best_track_corr(code, log_odds(matrix_of(a), bg),
                                  log_odds(matrix_of(b), bg))
        if np.isfinite(r):
            vals.append(r)
        if (k + 1) % 50 == 0:
            print("    %d/%d" % (k + 1, n_pairs), flush=True)
    out = {"n": len(vals), "median": float(np.median(vals)),
           "p95": float(np.percentile(vals, 95)), "p99": float(np.percentile(vals, 99)),
           "max": float(np.max(vals))}
    json.dump(out, open(cache, "w"), indent=1)
    return out


# ---------------------------------------------------------------- fig 1

def fig_agreement_vs_behaviour(d, null, out):
    s = d[(d.q_jaspar_ed.notna()) & (d.r_track_jaspar.notna())].copy()
    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.axhspan(0, null["p95"], color=NULLBAND, alpha=0.55, zorder=0)
    ax.text(2.2, null["p95"] - 0.015, "unrelated-motif null (95th pct %.2f)" % null["p95"],
            fontsize=8.8, color=SECOND, va="top", ha="right")
    ax.axvline(0.05, color=MUTED, ls=":", lw=1.2, zorder=1)
    ax.text(0.043, 0.985, "q = 0.05", rotation=90, fontsize=8.4,
            color=SECOND, va="top", ha="right")

    for src in ("badis", "zhu", "murphy", "other"):
        sub = s[s.source.map(srcof) == src]
        ax.scatter(sub.q_jaspar_ed.clip(lower=1e-24), sub.r_track_jaspar,
                   s=52, c=SRC_COLOR[src], alpha=0.9, linewidths=1.6,
                   edgecolors=SURFACE, zorder=4, label="%s (%d)" % (src, len(sub)))

    lab = {"Abf1_murphy": (55, -18), "Rap1_zhu": (18, 14), "Gal4_zhu": (20, -20),
           "Fkh1_zhu": (25, 12), "Spt15_zhu": (18, 14), "Mcm1_zhu": (30, -22),
           "Rap1_telomeric": (22, 12), "Reb1_badis": (18, 12), "Hap1_murphy": (22, -20)}
    for _, r in s.iterrows():
        if r.shipped_motif in lab:
            dx, dy = lab[r.shipped_motif]
            bold = r.shipped_motif == "Abf1_murphy"
            ax.annotate(r.shipped_motif, (max(r.q_jaspar_ed, 1e-24), r.r_track_jaspar),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=9.2 if bold else 8.5, color=INK if bold else SECOND,
                        fontweight="bold" if bold else "normal", zorder=6,
                        arrowprops=dict(arrowstyle="-", color="#b8b7b2", lw=0.9))

    ax.set_xscale("log")
    ax.set_xlim(3e-25, 3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("TOMTOM q-value vs JASPAR (Euclidean)   ←  stronger match      "
                  "weaker match  →", fontsize=10.5, color=SECOND)
    ax.set_ylabel("r of chrI score profiles at best alignment\n(does it bind the same DNA)",
                  fontsize=10.5, color=SECOND)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_title("Matching the same motif is not the same as behaving like it",
                 fontsize=14, fontweight="bold", color=INK, loc="left", pad=30)
    ax.text(0, 1.012, "each point is one shipped motif vs its own JASPAR entry  ·  "
                      "%d motifs  ·  colour = source collection" % len(s),
            transform=ax.transAxes, fontsize=9.4, color=SECOND)
    ax.legend(frameon=False, fontsize=9, loc="lower left", title="source",
              title_fontsize=9, labelcolor=SECOND, bbox_to_anchor=(0.0, 0.02))
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("  wrote", out)


# ---------------------------------------------------------------- fig 2

def fig_by_source(d, null, out):
    s = d[d.r_track_jaspar.notna()].copy()
    s["src"] = s.source.map(srcof)
    order = ["murphy", "zhu", "badis", "other"]
    order = [o for o in order if (s.src == o).any()]
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.axhspan(0, null["p95"], color=NULLBAND, alpha=0.55, zorder=0)
    rnd = np.random.RandomState(1)
    for i, o in enumerate(order):
        v = s.loc[s.src == o, "r_track_jaspar"].values
        ax.scatter(i + rnd.uniform(-0.17, 0.17, len(v)), v, s=42, c=SRC_COLOR[o],
                   alpha=0.85, linewidths=1.4, edgecolors=SURFACE, zorder=4)
        med = float(np.median(v))
        ax.plot([i - 0.32, i + 0.32], [med, med], color=INK, lw=2.4, zorder=6)
        ax.text(i + 0.36, med, "median %.2f" % med, fontsize=9, color=INK,
                va="center", fontweight="bold")
        ax.text(i, -0.075, "n = %d" % len(v), ha="center", fontsize=9, color=SECOND)
    ax.text(len(order) - 0.5, null["p95"] - 0.015,
            "unrelated-motif null (95th pct %.2f)" % null["p95"],
            fontsize=8.6, color=SECOND, va="top", ha="right")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10.5, color=INK)
    ax.set_xlim(-0.6, len(order) - 0.25)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("r of chrI score profiles vs JASPAR", fontsize=10.5, color=SECOND)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_title("How bad is it, by source collection", fontsize=14,
                 fontweight="bold", color=INK, loc="left", pad=28)
    ax.text(0, 1.012, "higher = our matrix picks the same sites as JASPAR's; "
                      "grey band = where unrelated motifs land",
            transform=ax.transAxes, fontsize=9.4, color=SECOND)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("  wrote", out)


# ---------------------------------------------------------------- fig 3

def fig_fitted(d, null, out):
    s = d[d.gene.isin(FITTED)].copy()
    s["rj"] = pd.to_numeric(s.r_track_jaspar, errors="coerce")
    s["rr"] = pd.to_numeric(s.r_track_rossi, errors="coerce")
    s = s.sort_values("rj", na_position="first")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(10.8, 7.4))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.axvspan(0, null["p95"], color=NULLBAND, alpha=0.55, zorder=0)
    for i, (_, r) in enumerate(s.iterrows()):
        if np.isfinite(r.rj) and np.isfinite(r.rr):
            ax.plot([min(r.rj, r.rr), max(r.rj, r.rr)], [i, i],
                    color="#c9c8c3", lw=2.0, zorder=2)
    ax.scatter(s.rj, y, s=95, c=SRC_COLOR["murphy"], edgecolors=SURFACE,
               linewidths=1.8, zorder=5, label="vs JASPAR")
    ax.scatter(s.rr, y, s=95, c=SRC_COLOR["badis"], edgecolors=SURFACE,
               linewidths=1.8, zorder=5, label="vs Rossi")
    for i, (_, r) in enumerate(s.iterrows()):
        if not np.isfinite(r.rj):
            ax.text(0.02, i, "no JASPAR entry", fontsize=8.6, color=SECOND, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(s.shipped_motif, fontsize=9.6, color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, len(s) - 0.3)
    ax.set_xlabel("r of chrI score profiles at best alignment", fontsize=10.5, color=SECOND)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_title("The 12 TFs that have fitted Fiber-seq parameters",
                 fontsize=14, fontweight="bold", color=INK, loc="left", pad=28)
    ax.text(0, 1.010, "these are the only states RoboCOP can decode distinctly  ·  "
                      "grey band = unrelated-motif null",
            transform=ax.transAxes, fontsize=9.4, color=SECOND)
    ax.legend(frameon=False, fontsize=9.4, loc="lower right", labelcolor=SECOND)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("  wrote", out)


# ---------------------------------------------------------------- fig 4

def fig_logos(d, out, genes=("ABF1", "RAP1", "GAL4", "FKH1")):
    dbs = {n: {m["name"]: m for m in parse_meme(open("motifdb/%s.meme" % n).read())}
           for n in ("shipped", "jaspar", "rossi")}
    rows = []
    for g in genes:
        sub = d[(d.gene == g) & d.match_jaspar.notna()]
        if not len(sub):
            continue
        r = sub.sort_values("r_track_jaspar").iloc[0]
        rows.append(r)
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.6, 2.35 * len(rows)))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        entries = [("shipped", r.shipped_motif, r.shipped_motif),
                   ("jaspar", r.match_jaspar, "JASPAR"),
                   ("rossi", r.match_rossi if isinstance(r.match_rossi, str) else None, "Rossi")]
        for j, (dbn, key, title) in enumerate(entries):
            ax = axes[i, j]
            style(ax)
            if key and key in dbs[dbn]:
                mat = matrix_of(dbs[dbn][key])
                if j and str(r.get("orient_" + dbn, "+")) == "-":
                    mat = mat[::-1, ::-1]
                df = pd.DataFrame(mat, columns=list("ACGT"))
                logomaker.Logo(logomaker.transform_matrix(
                    df, from_type="probability", to_type="information"),
                    ax=ax, color_scheme="colorblind_safe", show_spines=False)
                lbl = title if j else title
                ax.set_title(lbl, fontsize=10, color=INK, loc="left", pad=4)
            else:
                ax.text(0.5, 0.5, "none", ha="center", va="center",
                        fontsize=10, color=SECOND, transform=ax.transAxes)
                ax.set_title(title, fontsize=10, color=SECOND, loc="left", pad=4)
            ax.set_ylim(0, 2)
            ax.set_yticks([0, 1, 2])
            if j == 0:
                ax.set_ylabel("bits", fontsize=9, color=SECOND)
        rj = r.r_track_jaspar
        axes[i, 2].text(1.01, 0.5, "r vs JASPAR\n%.2f" % float(rj) if pd.notna(rj) else "",
                        transform=axes[i, 2].transAxes, fontsize=9.6, color=INK,
                        va="center", ha="left", fontweight="bold")
    fig.suptitle("Aligned logos, worked examples  (targets shown in the orientation TOMTOM matched)",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.95, 0.97))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("  wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="motif_comparison.tsv")
    ap.add_argument("--outdir", default="motifdb/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d = pd.read_csv(args.tsv, sep="\t")
    for c in ("q_jaspar_ed", "q_jaspar_kullback", "r_track_jaspar", "r_track_rossi",
              "q_rossi_ed"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    print("== null ==")
    null = compute_null("motifdb/.null_track.json")
    print("  n=%d  median %.3f  p95 %.3f  p99 %.3f  max %.3f"
          % (null["n"], null["median"], null["p95"], null["p99"], null["max"]))

    print("\n== figures ==")
    P = lambda f: os.path.join(args.outdir, f)
    fig_agreement_vs_behaviour(d, null, P("agreement_vs_behaviour.png"))
    fig_by_source(d, null, P("agreement_by_source.png"))
    fig_fitted(d, null, P("fitted_tfs.png"))
    fig_logos(d, P("logos_worked_examples.png"))

    print("\n== headline numbers ==")
    s = d[d.r_track_jaspar.notna()]
    print("  motifs comparable to JASPAR : %d" % len(s))
    print("  at/below the null p95 (%.2f): %d" % (null["p95"], int((s.r_track_jaspar <= null["p95"]).sum())))
    print("  r_track < 0.9               : %d" % int((s.r_track_jaspar < 0.9).sum()))
    print("  r_track >= 0.95             : %d" % int((s.r_track_jaspar >= 0.95).sum()))
    print("  median r_track by source:")
    for src in ("murphy", "zhu", "badis", "other"):
        v = s[s.source.map(srcof) == src].r_track_jaspar
        if len(v):
            print("    %-8s n=%-4d median %.3f  below-null %d"
                  % (src, len(v), v.median(), int((v <= null["p95"]).sum())))


if __name__ == "__main__":
    main()
