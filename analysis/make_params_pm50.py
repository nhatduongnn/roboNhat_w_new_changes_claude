"""TASK B: re-fit the per-TF Fiber-seq binomial p over a fixed 101 bp window
(motif centre +/- 50 bp) instead of just the motif span.

For PLOTTING ONLY. RoboCOP expects p vectors of length tf_len and would mis-index a
101-long vector, so this pkl must never be handed to a decode. It writes to a distinct
filename and never touches inputs/all_TFs_1000pealVal_params_pseudo.pkl.

Reuses combine_motif_counts_binom / fit_binomial_parameters / add_pseudocounts_binomial
verbatim from the original generator (via fiber_params_lib, which execs only its function
defs -- importing that file directly would re-run its pipeline and clobber the shipped
pkl). Only the count-extraction step is new, because the original keys its window to the
bed's own start/end.

REGISTER (the part that silently corrupts everything if wrong).
combine_motif_counts_binom pools minus-strand sites with a [::-1] mirror plus a
Watson<->Crick channel cross. For the mirror to map the motif block onto itself, the
motif must sit symmetrically in the window. In a fixed odd window an even-length motif
cannot be exactly centred, so the centre is defined per strand:

    plus :  c = start + L//2
    minus:  c = end   - L//2 - 1

Verified for every motif length present in the bed (5..20): plus-strand motif columns and
mirrored minus-strand columns land on identical window indices. The blocking check in
verify() re-tests this against the shipped 14-column ABF1 vector.

PERFORMANCE. The original scans all 11.4M pileup rows per site (2839 sites). This indexes
the pileup by chromosome once and uses searchsorted, which is why it finishes in minutes.
Identical arithmetic, just not O(sites x rows).

    conda activate robocop-2024
    export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
    python make_params_pm50.py
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

import fiber_params_lib

BED = "inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed"
PILEUP = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/"
          "combine_sequencing_runs/merged_Mar20_barcode01_Jun25_barcode21-24_"
          "May07_barcode03-04_sup_model_sorted_pileup_all_chr")
OUT = "inputs/all_TFs_1000pealVal_params_pseudo_pm50bp.pkl"
BASES = ["A", "C", "G", "T"]
MIN_SITES = 50            # same threshold the original uses to split individual vs pooled
SUCC, TRIALS = 11, 9      # Nmod, Nvalid_cov -- after the col-9 space split


def window_for(start, end, strand, half):
    """Fixed (2*half+1) bp window whose centre keeps plus/minus sites in register."""
    L = end - start
    c = start + L // 2 if strand == "+" else end - L // 2 - 1
    return c - half, c + half + 1


class Pileup:
    """Per-chromosome position index over the modkit pileup."""

    def __init__(self, path):
        t0 = time.time()
        df = pd.read_csv(path, sep="\t", header=None)
        split = df[9].str.split(" ", expand=True)
        split.columns = [i for i in range(9, 9 + split.shape[1])]
        df = pd.concat([df.drop(columns=[9]), split], axis=1)
        for col in split.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        self.by_chrom = {}
        for chrm, sub in df.groupby(0):
            sub = sub.sort_values(1)
            self.by_chrom[chrm] = dict(
                pos=sub[1].to_numpy(np.int64),
                base=sub[3].astype(str).str.upper().to_numpy(),
                strand=sub[5].astype(str).to_numpy(),
                succ=sub[SUCC].to_numpy(np.int64),
                trials=sub[TRIALS].to_numpy(np.int64))
        print("pileup: %d rows, %d chromosomes, indexed in %.1fs"
              % (len(df), len(self.by_chrom), time.time() - t0))

    def slice(self, chrm, lo, hi):
        d = self.by_chrom.get(chrm)
        if d is None:
            return None
        a, b = np.searchsorted(d["pos"], lo), np.searchsorted(d["pos"], hi)
        if a == b:
            return None
        return {k: v[a:b] for k, v in d.items()}


def extract(pile, sites, half):
    """Counts over the widened window. Mirrors compute_individual_Fiber_seq_TF_binom's
    output shape exactly, with tf_len = 2*half+1 so combine_motif_counts_binom reshapes
    correctly."""
    W = 2 * half + 1
    out = {s: {"successes": {b: [] for b in BASES}, "trials": {b: [] for b in BASES}}
           for s in ("watson", "crick")}
    for _, r in sites.iterrows():
        lo, hi = window_for(int(r["start"]), int(r["end"]), r["strand"], half)
        succ = {s: {b: [0] * W for b in BASES} for s in ("watson", "crick")}
        tri = {s: {b: [0] * W for b in BASES} for s in ("watson", "crick")}
        sl = pile.slice(r["chr"], lo, hi)
        if sl is not None:
            rel = sl["pos"] - lo
            for i in range(len(rel)):
                b = sl["base"][i]
                if b not in BASES:
                    continue
                p = int(rel[i])
                if not (0 <= p < W):
                    continue
                sk = "watson" if sl["strand"][i] == "+" else ("crick" if sl["strand"][i] == "-" else None)
                if sk is None:
                    continue
                succ[sk][b][p] += int(sl["succ"][i])
                tri[sk][b][p] += int(sl["trials"][i])
        for s in ("watson", "crick"):
            for b in BASES:
                out[s]["successes"][b].extend(succ[s][b])
                out[s]["trials"][b].extend(tri[s][b])
    return {"watson_signal": out["watson"], "crick_signal": out["crick"],
            "tf_len": W, "num_sites": len(sites)}


def fit_group(ns, pile, sites, half, pseudo):
    w = extract(pile, sites[sites.strand == "+"], half)
    c = extract(pile, sites[sites.strand == "-"], half)
    comb = ns["combine_motif_counts_binom"](w, c)
    if comb["num_sites"] == 0:
        return None, 0, 0
    if pseudo:
        comb = ns["add_pseudocounts_binomial"](comb, 3, 58)
    return ns["fit_binomial_parameters"](comb, comb["tf_len"], comb["num_sites"]), \
        w["num_sites"], c["num_sites"]


def verify(out_path, half):
    """BLOCKING: the central motif columns of the widened ABF1 vector must reproduce the
    shipped 14-column fit. If they do not, the window centring is wrong."""
    new = pickle.load(open(out_path, "rb"))
    old = pickle.load(open("inputs/all_TFs_1000pealVal_params_pseudo.pkl", "rb"))
    L = 14
    lo = half - L // 2
    ok = True
    print("\n%-16s %-9s %8s %8s %8s" % ("REGISTER CHECK", "strand", "old", "new", "|diff|"))
    for s in ("watson_signal", "crick_signal"):
        o = np.asarray(old["p"]["Abf1_murphy"][s]["A"], dtype=float)
        n = np.asarray(new["p"]["Abf1_murphy"][s]["A"], dtype=float)[lo:lo + L]
        d = float(np.abs(o - n).max())
        print("  %-14s %-9s %8.4f %8.4f %8.4f  %s"
              % ("Abf1 centre", s, o.mean(), n.mean(), d, "OK" if d < 0.02 else "MISMATCH"))
        ok &= d < 0.02
        print("     old: %s" % np.round(o, 3))
        print("     new: %s" % np.round(n, 3))
    print("\n  central %d columns reproduce the shipped fit: %s" % (L, "PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, default=50)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ns = fiber_params_lib.load()
    bed = pd.read_csv(BED, sep="\t")
    counts = bed.groupby("TF")["chr"].count()
    ind = sorted(counts[counts >= MIN_SITES].index)
    low = sorted(counts[counts < MIN_SITES].index)
    print("\nTFs: %d individual (>=%d sites), %d pooled into combined_low_count"
          % (len(ind), MIN_SITES, len(low)))
    print("window: motif centre +/- %d  ->  %d columns\n" % (args.half, 2 * args.half + 1))

    pile = Pileup(PILEUP)
    params = {"mu": {}, "phi": {}, "p": {}}
    t0 = time.time()
    for i, tf in enumerate(ind, 1):
        p, nw, nc = fit_group(ns, pile, bed[bed.TF == tf], args.half, pseudo=True)
        if p is None:
            continue
        params["p"][tf] = {"watson_signal": p["p"]["watson_signal"],
                           "crick_signal": p["p"]["crick_signal"]}
        v = np.asarray(p["p"]["watson_signal"]["A"], dtype=float)
        print("  [%2d/%2d] %-18s %3d+ %3d-  len %3d  mean p %.4f  (%.0fs)"
              % (i, len(ind), tf, nw, nc, len(v), v.mean(), time.time() - t0), flush=True)
    p, nw, nc = fit_group(ns, pile, bed[bed.TF.isin(low)], args.half, pseudo=False)
    params["p"]["combined_low_count"] = {"watson_signal": p["p"]["watson_signal"],
                                         "crick_signal": p["p"]["crick_signal"]}
    print("  combined_low_count: %d TFs, %d+ %d- sites, mean p %.4f"
          % (len(low), nw, nc, np.asarray(p["p"]["watson_signal"]["A"], dtype=float).mean()))

    with open(args.out, "wb") as f:
        pickle.dump(params, f)
    print("\nwrote %s  (%d TFs + combined_low_count)" % (args.out, len(params["p"]) - 1))
    if not verify(args.out, args.half):
        sys.exit("REGISTER CHECK FAILED -- do not plot from this pkl")


if __name__ == "__main__":
    main()
