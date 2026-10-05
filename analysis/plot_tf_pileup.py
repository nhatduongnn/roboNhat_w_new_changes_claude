"""Per-TF raw m6A profile plots: Watson up, Crick down.

Reads the npz written by pileup_tf_windows.py (per-TF, per-strand Nmod/Nvalid_cov summed
over that TF's own Rossi motifs, in the motif frame with minus-strand sites mirrored and
channel-crossed) and draws, for every TF in the bed:

  * one PNG per TF: the +/-200 bp profile with a +/-50 bp zoom beside it
  * one overview sheet with all TFs as small multiples, sorted by footprint depth

Watson is drawn upward and Crick downward -- the convention plot_factor_p_values.py uses --
so a symmetric shape means the two strands agree and an asymmetric one does not. Rates are
trial-weighted (sum Nmod / sum Nvalid_cov per column), never an unweighted column mean:
motif columns that are rarely A carry ~1% of their neighbours' coverage, and an unweighted
mean lets those near-empty cells dominate.

Shading marks the motif span. The dotted reference line is that TF's OWN flank level at
20-50 bp -- the NDR the site sits in. That, not the far flank, is the right comparator:
at 100-200 bp the signal is nucleosomal and protected too, so scoring a motif against it
compares protection to protection and hides the footprint.

    python plot_tf_pileup.py --npz tf_pileup_pm200.npz --outdir figures/tf_pileup_pm200
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W_COLOR = "#2a78d6"      # Watson  -- validated pair, see dataviz validator
C_COLOR = "#eb6834"      # Crick
MOTIF_BAND = "#8a8a8a"
INK, MUTED = "#1b1b1b", "#6b6b6b"

NDR_LO, NDR_HI = 20, 50


def rates(meth, valid):
    """Trial-weighted rate per column, NaN where a column has no coverage."""
    m, v = meth.astype(float), valid.astype(float)
    out = np.full(m.shape, np.nan)
    np.divide(m, v, out=out, where=v > 0)
    return out


def summarise(meth, valid, off, L):
    half = (len(off) - 1) // 2
    c0 = half - L // 2
    M, V = meth.sum(0).astype(float), valid.sum(0).astype(float)
    mo = slice(c0, c0 + L)
    ndr = (np.abs(off) > NDR_LO) & (np.abs(off) <= NDR_HI)
    mr = M[mo].sum() / V[mo].sum() if V[mo].sum() else np.nan
    nr = M[ndr].sum() / V[ndr].sum() if V[ndr].sum() else np.nan
    return mr, nr, (nr / mr if mr else np.nan), c0


def draw_panel(ax, off, rw, rc, c0, L, ndr_level, ymax, compact=False):
    ax.axhline(0, color=INK, lw=0.6, zorder=3)
    ax.axvspan(off[c0], off[c0 + L - 1], color=MOTIF_BAND, alpha=0.18, lw=0, zorder=0)
    ax.fill_between(off, 0, np.nan_to_num(rw), color=W_COLOR, lw=0, alpha=0.85, zorder=2)
    ax.fill_between(off, 0, -np.nan_to_num(rc), color=C_COLOR, lw=0, alpha=0.85, zorder=2)
    if ndr_level == ndr_level:
        for s in (1, -1):
            ax.axhline(s * ndr_level, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=4)
    ax.set_ylim(-ymax, ymax)
    ax.grid(axis="y", color="#e4e4e4", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#cfcfcf")
    # ticks must come from THIS panel's range: a tick outside the limits makes
    # matplotlib autoscale the view back out, which silently un-zooms the panel.
    lo, hi = int(off[0]), int(off[-1])
    ax.set_xticks([lo, 0, hi])
    ax.tick_params(labelsize=6 if compact else 8, length=2 if compact else 3,
                   colors=MUTED)
    ax.set_xlim(lo, hi)          # last, so nothing can widen it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="tf_pileup_pm200.npz")
    ap.add_argument("--outdir", default="figures/tf_pileup_pm200")
    ap.add_argument("--ymax", type=float, default=0.55)
    ap.add_argument("--view", type=int, default=None,
                    help="x half-range for the overview sheet (default: the full window). "
                         "e.g. --view 50 for a +/-50 bp overview.")
    ap.add_argument("--overview-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    z = np.load(a.npz, allow_pickle=True)
    tfs = [str(t) for t in z["tfs"]]
    meth, valid, ml, ns = z["meth"], z["valid"], z["motif_len"], z["n_sites"]
    half = int(z["half"])
    off = np.arange(-half, half + 1)

    recs = []
    for j, t in enumerate(tfs):
        L = int(ml[j])
        mr, nr, ratio, c0 = summarise(meth[j], valid[j], off, L)
        recs.append(dict(j=j, tf=t, n=int(ns[j]), L=L, motif=mr, ndr=nr,
                         ratio=ratio, c0=c0))
    recs.sort(key=lambda r: -(r["ratio"] if r["ratio"] == r["ratio"] else -9))

    # ---- one PNG per TF: +/-200 with a +/-50 zoom ---------------------------------
    for r in ([] if a.overview_only else recs):
        j, L, c0 = r["j"], r["L"], r["c0"]
        rw, rc = rates(meth[j, 0], valid[j, 0]), rates(meth[j, 1], valid[j, 1])
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6),
                                 gridspec_kw=dict(width_ratios=[2.4, 1], wspace=0.18))
        draw_panel(axes[0], off, rw, rc, c0, L, r["ndr"], a.ymax)
        axes[0].set_xlabel("position relative to motif centre (bp)", fontsize=9, color=MUTED)
        axes[0].set_ylabel("m6A / A      Crick  ←   →  Watson", fontsize=9, color=MUTED)

        k = np.abs(off) <= 50
        draw_panel(axes[1], off[k], rw[k], rc[k], int(np.nonzero(k)[0].searchsorted(c0)),
                   L, r["ndr"], a.ymax)
        axes[1].set_xlabel("±50 bp", fontsize=9, color=MUTED)

        ratio_txt = "n/a" if r["ratio"] != r["ratio"] else "%.2f" % r["ratio"]
        verdict = ("protected" if (r["ratio"] == r["ratio"] and r["ratio"] >= 1.3)
                   else "inverted" if (r["ratio"] == r["ratio"] and r["ratio"] <= 0.9)
                   else "flat")
        fig.suptitle("%s      %d sites      motif %.3f  vs  NDR flank %.3f      "
                     "ratio %s  (%s)%s"
                     % (r["tf"], r["n"], r["motif"], r["ndr"], ratio_txt, verdict,
                        "      ⚠ n < 10" if r["n"] < 10 else ""),
                     fontsize=10.5, color=INK, y=0.99)
        fig.legend(handles=[Line2D([], [], color=W_COLOR, lw=6, label="Watson"),
                            Line2D([], [], color=C_COLOR, lw=6, label="Crick"),
                            Line2D([], [], color=MUTED, lw=0.9, ls=(0, (2, 2)),
                                   label="flank 20–50 bp (NDR)"),
                            Line2D([], [], color=MOTIF_BAND, lw=6, alpha=0.4,
                                   label="motif span")],
                   loc="lower center", ncol=4, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
        fig.subplots_adjust(top=0.84, bottom=0.21)
        fig.savefig(os.path.join(a.outdir, "%s.png" % r["tf"]), dpi=150)
        plt.close(fig)

    # ---- overview sheet -----------------------------------------------------------
    n = len(recs)
    ncol = 6 if a.view else 8
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * (2.9 if a.view else 2.05),
                                      nrow * (2.0 if a.view else 1.55)),
                             squeeze=False)
    for i, r in enumerate(recs):
        ax = axes[i // ncol][i % ncol]
        j, L, c0 = r["j"], r["L"], r["c0"]
        rw, rc = rates(meth[j, 0], valid[j, 0]), rates(meth[j, 1], valid[j, 1])
        if a.view:
            k = np.abs(off) <= a.view
            draw_panel(ax, off[k], rw[k], rc[k],
                       int(np.nonzero(k)[0].searchsorted(c0)), L, r["ndr"], a.ymax,
                       compact=True)
        else:
            draw_panel(ax, off, rw, rc, c0, L, r["ndr"], a.ymax, compact=True)
        ratio_txt = "n/a" if r["ratio"] != r["ratio"] else "%.2f" % r["ratio"]
        ax.set_title("%s\nn=%d   ratio %s%s"
                     % (r["tf"].replace("_", "\n", 0), r["n"], ratio_txt,
                        "  ⚠" if r["n"] < 10 else ""),
                     fontsize=6.6, color=INK, pad=2)
    for i in range(n, nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    fig.suptitle("Raw Fiber-seq m6A over each TF's own Rossi motifs  —  Watson up, "
                 "Crick down, trial-weighted, no pseudocount\n"
                 "sorted by footprint depth (NDR flank 20–50 bp ÷ motif); "
                 "shaded band = motif span; dotted = that TF's NDR level; ⚠ = fewer "
                 "than 10 sites",
                 fontsize=11, color=INK)
    fig.legend(handles=[Line2D([], [], color=W_COLOR, lw=6, label="Watson"),
                        Line2D([], [], color=C_COLOR, lw=6, label="Crick")],
               loc="upper right", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.995, 0.985))
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    ov = os.path.join(a.outdir, "_overview_all_TFs%s.png"
                      % ("_pm%d" % a.view if a.view else ""))
    fig.savefig(ov, dpi=150)
    plt.close("all")

    # ---- the table ----------------------------------------------------------------
    tsv = os.path.join(a.outdir, "_footprint_table.tsv")
    with open(tsv, "w") as fh:
        fh.write("tf\tn_sites\tmotif_len\tmotif_rate\tndr_flank_rate\tratio_ndr_over_motif\n")
        for r in recs:
            fh.write("%s\t%d\t%d\t%.6f\t%.6f\t%.6f\n"
                     % (r["tf"], r["n"], r["L"], r["motif"], r["ndr"], r["ratio"]))
    print("wrote %d per-TF PNGs + %s + %s" % (len(recs), ov, tsv))


if __name__ == "__main__":
    main()
