"""One dot per factor: the mean m6A rate the decoder's Fiber-seq emission actually uses.

WHAT THE DECODER USES (all line numbers in pkg/robocop/robocop.py).  56 of the 59
analysis/pkgvar/*/robocop/robocop.py copies are byte-identical over the `ps` block
(md5 of the `ps = np.zeros` .. `assert (ps > 0)` span); the three that differ change
nothing in the mapping: seq_maskoff_wide and seq_maskoff_widenull add a length-guard
raise, and permol_seq_maskoff (the single-fiber prototype, not this aggregate decoder)
carries a second verbatim copy of the same block.

  628  ps = np.zeros(dshared['silent_states_begin'])       # one p per emitting state
  631  ps[:] = bg_params['p'][strand]['A']                 # DEFAULT = background
  650  for i in range(dshared['n_tfs']):                   # tfs = sorted(tf_prob.keys())
  655      if tf_name in loaded_params['p']:               # exact string match, no fallback
  658/659     forward block <- p[tf][strand]['A'];  reverse block <- p[tf][other][::-1]
  672      else: combined_low_count                        # every unmatched name
  675/676     forward half <- clc[strand];  reverse half <- clc[other]
  680-683  nucleosome: 147-long vector -> 531 states as 9 + 128x4 + 10
  801  binom.pmf(k, n, p_j)                                # p reaches the emission RAW

`dshared['tfs']` is `sorted(tf_prob.keys())` (robocop.py:83-84), and tf_prob is every
dbf_conc key except 'background' and 'nucleosome' (robocop_em.py:93-95).  dbf_conc holds
the 153 MEME motifs plus 'unknown', which parameterize.py:77,83 synthesises.  So there are
154 TF blocks; 12 of them match a key in the fitted pkl and 142 (141 motifs + 'unknown')
fall through to the single scalar `combined_low_count`.  State 0 is background: the TF loop
starts at state 1 (robocop.py:172), so the line-631 default is never overwritten there.

Reads (read-only):
  inputs/all_TFs_1000pealVal_params_pseudo.pkl   fitted p, 12 TFs + combined_low_count
  inputs/bg_params.pkl                           background p (scalar per strand)
  inputs/nucleosome_params.pkl                   147-long p per strand
  inputs/motifs_meme.txt                         the 153 motif names and their widths
  inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed
                                                 the training sites; n per TF, and the
                                                 >= 50 rule that decides fitted vs pooled
                                                 (generator line 54 / line 147)

Draws:
  main panel   every factor on x, sorted by mean rate; Watson and Crick markers joined by
               a tick; filled = individually fitted, hollow = pooled on combined_low_count;
               background drawn as a horizontal reference line as well as a point
  lower strip  the per-position footprint of each fitted TF, plus the 147-bp nucleosome

    python plot_factor_meth_rates.py
    python plot_factor_meth_rates.py --out figures/factor_meth_rates/factor_meth_rates
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TF_PKL = "inputs/all_TFs_1000pealVal_params_pseudo.pkl"
BG_PKL = "inputs/bg_params.pkl"
NUC_PKL = "inputs/nucleosome_params.pkl"
MEME = "inputs/motifs_meme.txt"
BED = "inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed"
OUT = "figures/factor_meth_rates/factor_meth_rates"

# Repo convention (plot_factor_p_values.py:39): Watson blue, Crick orange.
WATSON, CRICK = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e7e6e2"
NUC_C, UNK_C = "#4a3aa7", "#008300"

MIN_SITES = 50          # generator line 54: tf_counts >= 50 gets its own vector
N_NUC_STATES = 531      # robocop.py:154


def meme_motifs(path):
    """MOTIF names in file order, with the number of letter-probability rows each has."""
    out, cur, n = [], None, 0
    for line in open(path):
        s = line.strip()
        if s.startswith("MOTIF"):
            if cur is not None:
                out.append((cur, n))
            cur, n = s.split()[1], 0
            continue
        if cur is None or not s or s.startswith("letter") or s.startswith("URL"):
            continue
        p = s.split()
        if len(p) >= 4:
            try:
                float(p[0])
            except ValueError:
                continue
            n += 1
    if cur is not None:
        out.append((cur, n))
    return out


def nuc_state_weights():
    """Weight of each of the 147 positions among the 531 nucleosome states.

    robocop.py:680-683 lays the 147-long vector out as 9 singles, then positions 9..136
    repeated 4x each (the dinucleotide model's 128 4-state groups), then 10 singles.
    9 + 128*4 + 10 == 531.
    """
    w = np.ones(147)
    w[9:137] = 4.0
    assert w.sum() == N_NUC_STATES, w.sum()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT, help="output basename; .png/.pdf/.tsv appended")
    ap.add_argument("--tf-pkl", default=TF_PKL)
    ap.add_argument("--bg-pkl", default=BG_PKL)
    ap.add_argument("--nuc-pkl", default=NUC_PKL)
    ap.add_argument("--meme", default=MEME)
    ap.add_argument("--bed", default=BED)
    a = ap.parse_args()

    P = pickle.load(open(a.tf_pkl, "rb"))["p"]
    bg = pickle.load(open(a.bg_pkl, "rb"))["p"]
    nuc = pickle.load(open(a.nuc_pkl, "rb"))["p"]

    vec = {s: {} for s in ("watson_signal", "crick_signal")}   # factor -> A vector as fed
    bg_w = np.ravel(bg["watson_signal"]["A"]).astype(float)
    bg_c = np.ravel(bg["crick_signal"]["A"]).astype(float)
    clc_w = np.ravel(P["combined_low_count"]["watson_signal"]["A"]).astype(float)
    clc_c = np.ravel(P["combined_low_count"]["crick_signal"]["A"]).astype(float)

    sites = pd.read_csv(a.bed, sep="\t").groupby("TF")["chr"].count()
    fitted = sorted(k for k in P if k != "combined_low_count")
    assert set(fitted) == set(sites[sites >= MIN_SITES].index), "pkl keys != >=50-site TFs"

    motifs = meme_motifs(a.meme)
    rows = []
    for name, width in motifs:
        if name in P:
            w = np.ravel(P[name]["watson_signal"]["A"]).astype(float)
            c = np.ravel(P[name]["crick_signal"]["A"]).astype(float)
            assert len(w) == width, (name, len(w), width)
            cls = "fitted"
        else:
            w, c = clc_w, clc_c
            cls = "pooled"
        rows.append(dict(factor=name, kind=cls, state="TF", n_sites=int(sites.get(name, 0)),
                         fp_len=width, w=w, c=c))
    # 'unknown': a 10-column flat PWM added in parameterize.py:77; it is a tf_prob key
    # (robocop_em.py:93-95) so it is one of the 154 TF blocks, and it is not in the pkl.
    rows.append(dict(factor="unknown", kind="pooled", state="unknown",
                     n_sites=0, fp_len=10, w=clc_w, c=clc_c))
    rows.append(dict(factor="background", kind="fitted", state="background",
                     n_sites=-1, fp_len=1, w=bg_w, c=bg_c))
    nw = np.ravel(nuc["watson_signal"]["A"]).astype(float)
    nc = np.ravel(nuc["crick_signal"]["A"]).astype(float)
    rows.append(dict(factor="nucleosome", kind="fitted", state="nucleosome",
                     n_sites=-1, fp_len=len(nw), w=nw, c=nc))

    wt = nuc_state_weights()
    for r in rows:
        if r["state"] == "nucleosome":
            r["mean_w"] = float(np.average(r["w"], weights=wt))
            r["mean_c"] = float(np.average(r["c"], weights=wt))
            r["unwt_w"] = float(r["w"].mean())
            r["unwt_c"] = float(r["c"].mean())
        else:
            r["mean_w"], r["mean_c"] = float(r["w"].mean()), float(r["c"].mean())
            r["unwt_w"], r["unwt_c"] = r["mean_w"], r["mean_c"]
        r["min_w"], r["max_w"] = float(r["w"].min()), float(r["w"].max())
        r["min_c"], r["max_c"] = float(r["c"].min()), float(r["c"].max())
        r["mean_both"] = 0.5 * (r["mean_w"] + r["mean_c"])

    rows.sort(key=lambda r: -r["mean_both"])
    x = np.arange(len(rows))

    tsv = a.out + ".tsv"
    os.makedirs(os.path.dirname(tsv) or ".", exist_ok=True)
    pd.DataFrame([{k: r[k] for k in ("factor", "kind", "state", "n_sites", "fp_len",
                                     "mean_w", "mean_c", "mean_both", "min_w", "max_w",
                                     "min_c", "max_c", "unwt_w", "unwt_c")}
                  for r in rows]).to_csv(tsv, sep="\t", index=False, float_format="%.6f")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(24, 11.5))
    gs = fig.add_gridspec(2, 14, height_ratios=[3.0, 1.05], hspace=0.78,
                          wspace=0.34, left=0.045, right=0.985, top=0.945, bottom=0.075)
    ax = fig.add_subplot(gs[0, :])

    for r, xi in zip(rows, x):
        big = r["kind"] == "fitted"
        ax.plot([xi, xi], [r["mean_w"], r["mean_c"]], color=MUTED,
                lw=1.1 if big else 0.5, alpha=0.9 if big else 0.35, zorder=2)
        if big:
            ax.plot(xi, r["mean_w"], "o", ms=9, mfc=WATSON, mec="white", mew=1.2, zorder=4)
            ax.plot(xi, r["mean_c"], "D", ms=7.5, mfc=CRICK, mec="white", mew=1.2, zorder=4)
        else:
            ax.plot(xi, r["mean_w"], "o", ms=5, mfc="none", mec=WATSON, mew=1.0, zorder=3)
            ax.plot(xi, r["mean_c"], "D", ms=4.5, mfc="none", mec=CRICK, mew=1.0, zorder=3)

    ax.axhline(bg_w[0], color=INK, ls="--", lw=1.3, zorder=1)
    ax.text(-0.6, bg_w[0] + 0.004,
            "background  %.4f W / %.4f C  -- the level every contrast is measured against"
            % (bg_w[0], bg_c[0]), ha="left", va="bottom", fontsize=10, color=INK)
    for lvl, lab, col in ((clc_w[0], "combined_low_count Watson %.4f" % clc_w[0], WATSON),
                          (clc_c[0], "combined_low_count Crick %.4f" % clc_c[0], CRICK)):
        ax.axhline(lvl, color=col, ls=":", lw=1.0, alpha=0.55, zorder=1)
        # right end: the last dozen columns are the low-rate fitted factors, so the
        # 0.25 band is clear of markers there
        ax.text(len(rows) + 0.2, lvl, lab, ha="right", va="bottom", fontsize=9.5, color=col)

    unk = next(i for i, r in enumerate(rows) if r["state"] == "unknown")
    ax.plot(unk, rows[unk]["mean_w"], "*", ms=15, mfc=UNK_C, mec="white", mew=1.0, zorder=5)
    ax.plot(unk, rows[unk]["mean_c"], "*", ms=15, mfc=UNK_C, mec="white", mew=1.0, zorder=5)
    ax.annotate("unknown state", (unk, rows[unk]["mean_c"]),
                textcoords="offset points", xytext=(-4, 16), ha="right", fontsize=9.5,
                color=UNK_C, fontweight="bold")

    ax.set_xlim(-1.2, len(rows) + 0.2)
    ax.set_ylim(0.0, 0.335)
    ax.set_xticks(x)
    def tick(r):
        if r["state"] == "nucleosome":
            return "nucleosome  (147 bp -> %d states)" % N_NUC_STATES
        if r["state"] == "background":
            return "background  (1 bp)"
        if r["state"] == "unknown":
            return "unknown  (pooled)"
        if r["kind"] == "fitted":
            return "%s   n=%d" % (r["factor"], r["n_sites"])
        return r["factor"]

    ax.set_xticklabels([tick(r) for r in rows], rotation=90, fontsize=5.6)
    for t, r in zip(ax.get_xticklabels(), rows):
        if r["kind"] == "fitted":
            t.set_color(NUC_C if r["state"] == "nucleosome" else INK)
            t.set_fontsize(8.0)
            t.set_fontweight("bold")
        elif r["state"] == "unknown":
            t.set_color(UNK_C)
            t.set_fontsize(8.0)
            t.set_fontweight("bold")
        else:
            t.set_color(MUTED)
    ax.set_ylabel("mean P(m6A) at A positions, as fed to binom.pmf", fontsize=11)
    ax.set_title("Every emitting factor in the RoboCOP Fiber-seq model and the mean m6A rate its "
                 "state uses\n12 TFs have a fitted footprint; 141 motifs + the unknown state all "
                 "share one scalar, combined_low_count", fontsize=13.5, color=INK, pad=16)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    n_fit = sum(1 for r in rows if r["kind"] == "fitted" and r["state"] == "TF")
    n_pool = sum(1 for r in rows if r["kind"] == "pooled")
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=9, mfc=WATSON, mec="white",
               label="Watson  p['watson_signal']['A']"),
        Line2D([], [], ls="", marker="D", ms=7.5, mfc=CRICK, mec="white",
               label="Crick  p['crick_signal']['A']"),
        Line2D([], [], ls="", marker="o", ms=9, mfc="0.35", mec="white",
               label="individually fitted, filled  (%d TFs, >= %d Rossi sites)" % (n_fit, MIN_SITES)),
        Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="0.35",
               label="pooled on combined_low_count, hollow  (%d states)" % n_pool),
        Line2D([], [], ls="", marker="*", ms=14, mfc=UNK_C, mec="white", label="unknown state"),
        Line2D([], [], color=INK, ls="--", lw=1.3, label="background reference"),
    ], loc="upper right", fontsize=9.5, frameon=False, ncol=2,
        bbox_to_anchor=(0.995, 0.995))

    # ---------------- lower strip: the footprints behind the means ----------------
    fit_rows = [r for r in rows if r["kind"] == "fitted" and r["state"] == "TF"]
    fit_rows.sort(key=lambda r: -r["n_sites"])
    for i, r in enumerate(fit_rows):
        axi = fig.add_subplot(gs[1, i])
        pos = np.arange(len(r["w"])) + 1
        axi.axhline(bg_w[0], color=INK, ls="--", lw=0.8)
        axi.axhline(clc_w[0], color=MUTED, ls=":", lw=0.8)
        axi.plot(pos, r["w"], "-o", color=WATSON, lw=1.6, ms=3, mec="white", mew=0.5)
        axi.plot(pos, r["c"], "-D", color=CRICK, lw=1.6, ms=3, mec="white", mew=0.5)
        axi.set_ylim(0, 0.48)
        axi.set_xlim(0.4, len(r["w"]) + 0.6)
        axi.set_title("%s\n%d bp, n=%d" % (r["factor"].split("_")[0], r["fp_len"], r["n_sites"]),
                      fontsize=8.5, color=INK, pad=3)
        axi.tick_params(labelsize=6.5)
        axi.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        if i:
            axi.set_yticklabels([])
        else:
            axi.set_ylabel("P(m6A)", fontsize=8.5)
        for s in ("top", "right"):
            axi.spines[s].set_visible(False)

    axn = fig.add_subplot(gs[1, 12:])
    axn.axhline(bg_w[0], color=INK, ls="--", lw=0.8)
    axn.axhline(clc_w[0], color=MUTED, ls=":", lw=0.8)
    nr = next(r for r in rows if r["state"] == "nucleosome")
    axn.plot(np.arange(1, 148), nr["w"], color=WATSON, lw=1.6)
    axn.plot(np.arange(1, 148), nr["c"], color=CRICK, lw=1.6)
    axn.set_ylim(0, 0.48)
    axn.set_xlim(0, 148)
    axn.set_title("nucleosome  147 bp -> %d states\n"
                  "mean %.4f W / %.4f C  (unwt. %.4f / %.4f)"
                  % (N_NUC_STATES, nr["mean_w"], nr["mean_c"], nr["unwt_w"], nr["unwt_c"]),
                  fontsize=8.0, color=NUC_C, pad=3)
    axn.tick_params(labelsize=6.5)
    axn.set_yticklabels([])
    axn.set_xlabel("position in footprint (bp)", fontsize=8.5)
    for s in ("top", "right"):
        axn.spines[s].set_visible(False)

    fig.text(0.045, 0.012,
             "Dotted lines: combined_low_count, the single scalar %d of the %d TF states use. "
             "Dashed line: background. Lower strip shows each fitted footprint (Watson blue, "
             "Crick orange) against those two levels.\n"
             "The mean is over the footprint's own positions; the decoder mirrors and "
             "channel-crosses these vectors for the reverse-strand block (robocop.py:658-659, "
             "commit 90b05c3), which leaves the mean unchanged. Values are raw: nothing floors, "
             "caps or scales p between the pickle and binom.pmf (robocop.py:801)."
             % (n_pool, n_fit + n_pool), fontsize=9, color=MUTED, va="bottom")

    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (a.out, ext), dpi=200 if ext == "png" else None,
                    facecolor="white")
    plt.close("all")
    print("wrote %s.png %s.pdf %s" % (a.out, a.out, tsv))
    print("%-22s %8s %8s %6s %6s" % ("factor", "meanW", "meanC", "len", "n"))
    for r in rows:
        if r["kind"] == "fitted" or r["state"] == "unknown":
            print("%-22s %8.4f %8.4f %6d %6d"
                  % (r["factor"], r["mean_w"], r["mean_c"], r["fp_len"], r["n_sites"]))


if __name__ == "__main__":
    main()
