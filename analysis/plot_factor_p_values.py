"""Per-TF Fiber-seq P(m6A) profiles, over the motif +/- 50 bp.

Rewrite of the plotter that produced robocop_all_abf1/figures/factor_p_values/ (that
script lived in a scratch dir that has been cleaned). Same layout -- Watson bars up,
Crick bars down, background and combined_low_count reference lines, motif logo beneath --
but over the 101-column window from make_params_pm50.py instead of the motif span alone.

The flanks are the point: the motif columns show the footprint dipping below background,
while the surrounding 50 bp show the open NDR the site sits in. For ABF1 the motif mean is
0.084 but the 101-column mean is 0.211, which is the whole reason `combined_low_count`
(fit on peak spans) came out above background.

Reads inputs/all_TFs_1000pealVal_params_pseudo_pm50bp.pkl -- a PLOTTING-ONLY pkl; RoboCOP
would mis-index a 101-long p vector.

    conda activate pyranges_env3          # needs logomaker
    python plot_factor_p_values.py
    python plot_factor_p_values.py --tf Abf1_murphy
"""
import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import logomaker

PM50 = "inputs/all_TFs_1000pealVal_params_pseudo_pm50bp.pkl"
MOTIF_MEME = "inputs/motifs_meme.txt"
BED = "inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed"
OUTDIR = "robocop_all_abf1/figures/factor_p_values_pm50"

WATSON, CRICK = "#2a78d6", "#eb6834"
INK, MUTED, GRID, BAND = "#0b0b0b", "#6b6a66", "#e7e6e2", "#cfcec9"
# Logo colours carry meaning here rather than convention: A and T are the ONLY bases the
# fiber layers read (layer 5 at reference A, layer 6 at reference T), so they take the
# Watson/Crick colours and C/G are greyed out as carrying no m6A signal.
LOGO_COLORS = {"A": WATSON, "T": CRICK, "C": "#9a9993", "G": "#9a9993"}


def load_meme(path, motif):
    L = open(path).read().split("\n")
    try:
        mi = next(i for i, l in enumerate(L) if l.startswith("MOTIF " + motif))
    except StopIteration:
        return None
    li = next(i for i in range(mi, mi + 8) if L[i].startswith("letter-probability"))
    w = int(re.search(r"w=\s*(\d+)", L[li]).group(1))
    return pd.DataFrame([[float(x) for x in L[li + 1 + j].split()] for j in range(w)],
                        columns=list("ACGT"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default=None, help="one TF; default = all")
    ap.add_argument("--pm50", default=PM50)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    par = pickle.load(open(args.pm50, "rb"))["p"]
    bg = pickle.load(open("inputs/bg_params.pkl", "rb"))
    bgw = float(np.ravel(bg["p"]["watson_signal"]["A"])[0])
    bgc = float(np.ravel(bg["p"]["crick_signal"]["A"])[0])
    lw = float(np.ravel(par["combined_low_count"]["watson_signal"]["A"]).mean())
    lc = float(np.ravel(par["combined_low_count"]["crick_signal"]["A"]).mean())
    bed = pd.read_csv(BED, sep="\t")

    os.makedirs(args.outdir, exist_ok=True)
    tfs = [args.tf] if args.tf else sorted(k for k in par if k != "combined_low_count")

    # shared y-limit across all TFs so the panels are comparable at a glance
    ymax = max(max(np.ravel(par[t][s]["A"]).max() for s in ("watson_signal", "crick_signal"))
               for t in tfs)
    ymax = max(ymax, lw, lc, bgw, bgc) * 1.12

    for tf in tfs:
        w = np.ravel(np.asarray(par[tf]["watson_signal"]["A"], dtype=float))
        c = np.ravel(np.asarray(par[tf]["crick_signal"]["A"], dtype=float))
        W = len(w)
        half = W // 2
        x = np.arange(W) - half

        sub = bed[bed.TF == tf]
        npl, nmi = int((sub.strand == "+").sum()), int((sub.strand == "-").sum())
        mlen = int((sub.end - sub.start).iloc[0]) if len(sub) else 0
        m_lo, m_hi = -(mlen // 2), mlen - mlen // 2 - 1        # motif span in x coords

        pwm = load_meme(MOTIF_MEME, tf)
        fig = plt.figure(figsize=(13.6, 7.4))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.30)
        ax, lax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        ax.axvspan(m_lo - 0.5, m_hi + 0.5, color=BAND, alpha=0.45, zorder=0)
        ax.bar(x, w, width=0.9, color=WATSON, zorder=3)
        ax.bar(x, -c, width=0.9, color=CRICK, zorder=3)
        # reference-line labels go at the LEFT edge with a white bbox; on the right they
        # collide with the flanking bars, which sit at roughly the combined_low_count level
        for v, ls, lab in ((bgw, "--", "background %.3f" % bgw),
                           (lw, ":", "combined_low_count %.3f" % lw)):
            ax.axhline(v, color=MUTED, ls=ls, lw=1.3, zorder=4)
            ax.text(-half + 0.5, v, lab, ha="left", va="bottom", fontsize=8.4,
                    color=MUTED, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
        ax.axhline(-bgc, color=MUTED, ls="--", lw=1.3, zorder=4)
        ax.axhline(-lc, color=MUTED, ls=":", lw=1.3, zorder=4)
        ax.axhline(0, color=INK, lw=1.1, zorder=5)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlim(-half - 0.5, half + 0.5)
        ax.set_ylabel("Watson up / Crick down\nP(m6A)", fontsize=10)
        ax.set_yticks([t for t in ax.get_yticks() if abs(t) <= ymax])
        ax.set_yticklabels(["%.2f" % abs(t) for t in ax.get_yticks()])
        ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#b8b7b2")
        ax.tick_params(length=0, labelsize=9)
        mo_w = w[half + m_lo:half + m_hi + 1].mean()
        mo_c = c[half + m_lo:half + m_hi + 1].mean()
        ax.set_title("%s  —  %d sites (%d +, %d −)  ·  motif length %d  ·  window ±%d bp"
                     % (tf, npl + nmi, npl, nmi, mlen, half),
                     fontsize=13.5, fontweight="bold", pad=26)
        ax.text(0.5, 1.055,
                "ON MOTIF  mean P(m6A) %.3f W / %.3f C  =  %.2f× / %.2f× background      "
                "FULL ±%d WINDOW  %.3f W / %.3f C  =  %.2f× / %.2f× background"
                % (mo_w, mo_c, mo_w / bgw, mo_c / bgc, half,
                   w.mean(), c.mean(), w.mean() / bgw, c.mean() / bgc),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9.2, color=MUTED)
        ax.legend(handles=[
            Line2D([0], [0], color=MUTED, ls="--", lw=1.3, label="background"),
            Line2D([0], [0], color=MUTED, ls=":", lw=1.3, label="combined_low_count"),
            Patch(fc=BAND, alpha=0.45, label="motif span"),
            Patch(fc=WATSON, label="Watson"), Patch(fc=CRICK, label="Crick")],
            frameon=False, fontsize=8.8, loc="lower right", ncol=5)

        if pwm is not None:
            df = logomaker.transform_matrix(pwm, from_type="probability", to_type="information")
            df.index = range(m_lo, m_lo + len(df))
            logomaker.Logo(df, ax=lax, color_scheme=LOGO_COLORS, show_spines=False)
        lax.set_xlim(-half - 0.5, half + 0.5)
        lax.set_ylabel("bits", fontsize=9.5)
        lax.set_xlabel("position relative to motif centre (bp)", fontsize=10.5)
        lax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
        lax.set_axisbelow(True)
        for s in ("top", "right"): lax.spines[s].set_visible(False)
        lax.tick_params(length=0, labelsize=9)

        fig.text(0.5, 0.028,
                 "p: %s  (PLOTTING ONLY — RoboCOP would mis-index a %d-long vector).  "
                 "background: inputs/bg_params.pkl.  logo: %s.\n"
                 "A blue / T orange = the bases the fiber layers read (layer 5 at A, layer 6 at T); "
                 "C and G grey = no m6A signal.  sites: %s"
                 % (args.pm50, W, MOTIF_MEME, os.path.basename(BED)),
                 ha="center", va="top", fontsize=8.0, color=MUTED)
        fig.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.145)
        out = os.path.join(args.outdir, "p_values_%s.png" % tf)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")
        print("  %-18s motif %.3f/%.3f  window %.3f/%.3f  -> %s"
              % (tf, mo_w, mo_c, w.mean(), c.mean(), out))


if __name__ == "__main__":
    main()
