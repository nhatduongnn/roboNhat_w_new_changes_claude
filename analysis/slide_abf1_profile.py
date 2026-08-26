"""Slide the fitted ABF1 Fiber-seq footprint across chrI as a matched filter.

Rebuilds the side experiment of HANDOFF.md section 6, whose code was never saved, and
extends it to the width sweep section 6.6 asked for -- notably +/-10 (21 bp), which was
never run.

WHAT IS COMPUTED (section 6.1)
------------------------------
Template, per strand channel, from the LOCO-refit p-vector:
    w_j = p_j - mean(p)                    # mean-centred => SHAPE only, level discarded
Mean-centring is what makes this an ABF1 detector rather than a generic protection
detector: a window must show the elevated rim, not just the notch.

Genome side, per position, from the modkit pileup (k = methylated calls, n = A trials):
    y_j = (k_j - n_j*p_hat) / sqrt(n_j * p_hat * (1 - p_hat))
a variance-stabilised residual -- observed minus expected in units of its own standard
error. Coverage weighting falls out for free: 0/100 is a loud protection signal, 0/3 is
barely anything. Positions with n = 0 get y = 0 (no evidence, neutral contribution).

Score:
    S = sum_j (w_j * y_j) / sqrt(sum_j w_j^2)
The denominator cancels the template's arbitrary scale, puts S on a unit-variance z scale
under the null, and makes different widths comparable. The data side is deliberately NOT
normalised, so this is a matched filter and not a Pearson r -- amplitude should count.

p_hat is a LOCAL background pooled over +/-500 bp, NOT inputs/bg_params.pkl. Section 6.4:
that pickle holds p = 0.1383 while the genome-wide pooled rate is 0.0790, so it looks like
an accessible-region fit; scoring with it turns the ABF1 template into a nucleosome
detector that ranks true sites at 56.8%, worse than chance.

Both orientations are scored and the better kept. Per the reverse-strand fix (commit
90b05c3), reversing needs BOTH a mirror and a Watson<->Crick channel cross, because
['watson_signal'] means "the motif's own strand", not "the reference plus strand":
    forward: ref-watson <- w_watson       ,  ref-crick <- w_crick
    reverse: ref-watson <- w_crick[::-1]  ,  ref-crick <- w_watson[::-1]

CIRCULARITY (section 6.6)
-------------------------
Rossi IS the training set for these parameters and MacIsaac is 77.5% inside it, so the
template is refit leave-one-chromosome-out: chrI sites are dropped before fitting. Without
that this measures memorisation. Refitting reuses make_params_pm50's own machinery
(Pileup / window_for / fit_group -> combine_motif_counts_binom /
add_pseudocounts_binomial(3,58) / fit_binomial_parameters) rather than re-deriving it; an
ad-hoc extractor previously drifted by max|diff| 0.053 W / 0.170 C.

CALLING AND MATCHING CRITERIA (explicit, and the same as the rest of the repo)
-----------------------------------------------------------------------------
  call   : a local maximum of S, at least `width` bp from any higher call, with
           S >= threshold. Minimum separation = the template width, so calls are
           non-overlapping footprints.
  match  : a reference site is RECOVERED if some unused call centre lies within
           --tol bp of the motif midpoint. tol defaults to 20 bp, the same
           tol_abf1 score_robocop.py uses, and matching is done by
           score_robocop.match_peaks (greedy nearest-neighbour) so the numbers are
           directly comparable to the decode results.

Usage:
    python slide_abf1_profile.py                       # halves 7,10,12,25,50 on chrI
    python slide_abf1_profile.py --halves 7,25 --tol 20
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

import fiber_params_lib
import score_robocop as S
from make_params_pm50 import BED, PILEUP, Pileup, fit_group

BASES = ["A", "C", "G", "T"]
TF = "Abf1_murphy"
BG_HALF = 500          # local background window half-width (section 6.1)
MACISAAC = "inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed"


# ---------------------------------------------------------------------------
def loco_template(ns, pile, bed, chrom, half):
    """Refit the ABF1 p-vector with `chrom` held out. Returns (p_watson, p_crick)."""
    sites = bed[(bed.TF == TF) & (bed.chr != chrom)]
    p, nw, nc = fit_group(ns, pile, sites, half, pseudo=True)
    if p is None:
        raise RuntimeError("no sites for %s excluding %s" % (TF, chrom))
    pw = np.asarray(p["p"]["watson_signal"]["A"], dtype=float)
    pc = np.asarray(p["p"]["crick_signal"]["A"], dtype=float)
    print("   LOCO half=%2d: %d+ %d- sites (chrI held out), mean p W %.4f C %.4f"
          % (half, nw, nc, pw.mean(), pc.mean()), flush=True)
    return pw, pc


def dense_counts(pile, chrom, length):
    """Per-position (k, n) on the A channel, split by pileup strand.

    Buckets on the modkit base column (section 6.3) -- filtering on the reference base
    instead is what made an earlier extractor drift.
    """
    d = pile.by_chrom[chrom]
    keep = d["base"] == "A"
    out = {}
    for name, sym in (("watson", "+"), ("crick", "-")):
        m = keep & (d["strand"] == sym)
        k = np.zeros(length); n = np.zeros(length)
        np.add.at(k, d["pos"][m], d["succ"][m])
        np.add.at(n, d["pos"][m], d["trials"][m])
        out[name] = (k, n)
    return out


def residuals(counts, length):
    """y = (k - n*p_hat)/sqrt(n*p_hat*(1-p_hat)) with p_hat pooled locally over +/-BG_HALF."""
    kw, nw = counts["watson"]; kc, nc = counts["crick"]
    ksum, nsum = kw + kc, nw + nc
    win = 2 * BG_HALF + 1
    kloc = uniform_filter1d(ksum, win, mode="nearest")
    nloc = uniform_filter1d(nsum, win, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        phat = np.where(nloc > 0, kloc / nloc, np.nan)
    phat = np.clip(phat, 1e-4, 1 - 1e-4)
    y = {}
    for name, (k, n) in counts.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (k - n * phat) / np.sqrt(n * phat * (1 - phat))
        y[name] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)  # n=0 -> neutral
    return y, phat


def slide(y, pw, pc):
    """Matched-filter score at every position, both orientations, better kept.

    Returns S aligned so index i is the CENTRE of the window.
    """
    def score(w_watson, w_crick):
        w = np.concatenate([w_watson, w_crick])
        nrm = np.sqrt(np.sum(w * w))
        if nrm == 0:
            return np.zeros_like(y["watson"])
        # np.correlate with the template gives sum_j w_j * y_{i+j}; 'same' centres it.
        s = (np.correlate(y["watson"], w_watson, mode="same")
             + np.correlate(y["crick"], w_crick, mode="same"))
        return s / nrm

    ww, wc = pw - pw.mean(), pc - pc.mean()
    fwd = score(ww, wc)
    rev = score(wc[::-1], ww[::-1])       # mirror AND channel cross (commit 90b05c3)
    return np.maximum(fwd, rev), fwd, rev


def call_peaks(s, width, threshold):
    """Local maxima of s, >= threshold, no two within `width` bp. Greedy by score."""
    idx = np.flatnonzero(s >= threshold)
    if idx.size == 0:
        return np.array([], dtype=int)
    order = idx[np.argsort(-s[idx])]
    taken = []
    occupied = np.zeros(len(s), dtype=bool)
    for i in order:
        if occupied[i]:
            continue
        taken.append(i)
        lo, hi = max(0, i - width), min(len(s), i + width + 1)
        occupied[lo:hi] = True
    return np.array(sorted(taken), dtype=int)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default="chrI")
    ap.add_argument("--halves", default="7,10,12,25,50")
    ap.add_argument("--tol", type=int, default=20,
                    help="match tolerance in bp (default 20 = score_robocop's tol_abf1)")
    ap.add_argument("--out", default="abf1_slide")
    args = ap.parse_args()
    halves = [int(x) for x in args.halves.split(",")]

    sizes = pd.read_csv("inputs/sacCer3.chrom.sizes", sep="\t", header=None,
                        names=["chr", "len"])
    length = int(sizes.loc[sizes.chr == args.chrom, "len"].iloc[0])
    ref = S.load_abf1(MACISAAC)
    ref = sorted(ref.loc[ref.chr == args.chrom, "center"].tolist())
    print("%s: %d bp, %d MacIsaac ABF1 reference sites: %s"
          % (args.chrom, length, len(ref), ref))

    ns = fiber_params_lib.load(verbose=False)
    bed = pd.read_csv(BED, sep="\t")
    pile = Pileup(PILEUP)

    counts = dense_counts(pile, args.chrom, length)
    y, phat = residuals(counts, length)
    ntot = (counts["watson"][1] + counts["crick"][1])
    print("A positions with coverage: %d / %d (%.1f%%); local p_hat median %.4f"
          % ((ntot > 0).sum(), length, 100.0 * (ntot > 0).mean(), np.nanmedian(phat)))

    rows, curves = [], {}
    for half in halves:
        width = 2 * half + 1
        pw, pc = loco_template(ns, pile, bed, args.chrom, half)
        s, fwd, rev = slide(y, pw, pc)
        curves[half] = dict(width=width, s=s)

        # percentile rank of each true site: best S within +/-tol of the midpoint,
        # ranked against every scored position (this is HANDOFF section 6.2's metric)
        valid = np.flatnonzero(ntot > 0)
        svals = s[valid]
        site_best, site_pct = [], []
        for c in ref:
            lo, hi = max(0, c - args.tol), min(length, c + args.tol + 1)
            b = float(np.max(s[lo:hi]))
            site_best.append(b)
            site_pct.append(100.0 * float(np.mean(svals >= b)))
        worst_pct, med_pct = max(site_pct), float(np.median(site_pct))

        # operating point that recovers ALL 5 sites: threshold = the weakest true site
        thr5 = min(site_best)
        calls5 = call_peaks(s, width, thr5)
        m5 = S.match_peaks([int(x) for x in calls5], ref, args.tol)

        row = dict(half=half, width=width,
                   median_site_pct=round(med_pct, 4), worst_site_pct=round(worst_pct, 4),
                   thr_for_5of5=round(thr5, 4), n_calls_at_5of5=int(len(calls5)),
                   recovered_at_5of5=int(m5["tp"]),
                   precision_at_5of5=round(m5["precision"], 6),
                   site_scores={str(c): round(b, 3) for c, b in zip(ref, site_best)},
                   site_pcts={str(c): round(p, 4) for c, p in zip(ref, site_pct)})
        rows.append(row)
        print("   half=%2d (%2d bp): median %.3f%%  worst %.3f%%  |  S>=%.2f -> %d calls, "
              "%d/5 recovered, precision %.4f"
              % (half, width, med_pct, worst_pct, thr5, len(calls5), m5["tp"],
                 m5["precision"]), flush=True)

        # threshold sweep at fixed call criteria
        sweep = []
        for thr in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
            cs = call_peaks(s, width, float(thr))
            mm = S.match_peaks([int(x) for x in cs], ref, args.tol)
            sweep.append(dict(threshold=thr, n_calls=int(len(cs)), recovered=int(mm["tp"]),
                              precision=round(mm["precision"], 6),
                              recall=round(mm["recall"], 4)))
        row["sweep"] = sweep

    with open(args.out + ".json", "w") as f:
        json.dump(dict(chrom=args.chrom, tol=args.tol, n_ref=len(ref), ref=ref,
                       bg_half=BG_HALF, loco=True, rows=rows), f, indent=2)
    np.savez_compressed(args.out + "_tracks.npz",
                        **{("s_half%d" % h): curves[h]["s"] for h in halves})

    with open(args.out + ".tsv", "w") as f:
        cols = ["half", "width", "median_site_pct", "worst_site_pct", "thr_for_5of5",
                "n_calls_at_5of5", "recovered_at_5of5", "precision_at_5of5"]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print("\nwrote %s.json / .tsv / _tracks.npz" % args.out)


if __name__ == "__main__":
    main()
