"""Raw m6A pileup over every TF's Rossi sites, in one streaming pass.

What this is
------------
For each TF in a Rossi bed, sum the modkit pileup's Nmod and Nvalid_cov over a fixed
window centred on each of that TF's motifs, keeping Watson and Crick signal separate and
keeping plus/minus motifs in register. No threshold, no pseudocount, no binomial fit, no
pkl format -- just the counts, for every TF in the bed.

How it differs from make_params_pm50.py
---------------------------------------
That script does the same extraction but (a) drops every TF with < MIN_SITES = 50 sites
(:52, :173), (b) adds a synthetic pseudo-site of 3/58 per column, and (c) collapses to a
pooled ratio for the decoder. Here nothing is dropped and nothing is added, and both the
numerator and the denominator are kept so a proper test can be run downstream.

It is also faster: make_params_pm50 loads the whole 786 MB pileup with pd.read_csv
(~87 s, ~10 GB). This streams it once (a full awk pass over the 11.4 M rows is ~4 s), and
keeps only the ~2.5 M positions that fall inside a window.

Registration (copied from the existing code, not invented)
----------------------------------------------------------
- window centre: `start + L//2` for a + motif, `end - L//2 - 1` for a - motif
  (make_params_pm50.py:56-60 window_for).
- a - motif is MIRRORED and its channels CROSSED before being added to the + frame:
  a minus motif's Crick signal reversed aligns with a plus motif's Watson signal
  (abf1_reb1_dms_parameter_Fiber-seq_w_binom.py combine_motif_counts_binom, the vstack of
  `c_rs['crick_signal'][...][:, ::-1]` into `combined['watson_signal']`). This is what
  commit 90b05c3 fixed.

Pileup columns: tab field 1 chrom, 2 start0, 4 base, 6 strand, 10 space-packed; after the
split, index 9 = Nvalid_cov (trials) and index 11 = Nmod (successes)
(make_params_pm50.py:53 `SUCC, TRIALS = 11, 9`).

Usage
-----
    python pileup_tf_windows.py --half 200 --out tf_pileup_pm200.npz
    python pileup_tf_windows.py --bed inputs/..._peakVal.bed --half 200 --out ...

Extract once at the widest half; narrower windows are a slice of the result.
"""
import argparse
import sys
import time

import numpy as np

BED_DEFAULT = "inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed"
PILEUP = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/"
          "combine_sequencing_runs/merged_Mar20_barcode01_Jun25_barcode21-24_"
          "May07_barcode03-04_sup_model_sorted_pileup_all_chr")


def window_for(start, end, strand, half):
    """Identical to make_params_pm50.py:56-60."""
    L = end - start
    c = start + L // 2 if strand == "+" else end - L // 2 - 1
    return c - half, c + half + 1


def read_bed(path):
    sites, header = [], None
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if header is None:
                header = p
                idx = {n: i for i, n in enumerate(p)}
                continue
            sites.append((p[idx["chr"]], int(p[idx["start"]]), int(p[idx["end"]]),
                          p[idx["strand"]], p[idx["TF"]]))
    return sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", default=BED_DEFAULT)
    ap.add_argument("--pileup", default=PILEUP)
    ap.add_argument("--half", type=int, default=200)
    ap.add_argument("--out", default="tf_pileup_pm200.npz")
    a = ap.parse_args()

    W = 2 * a.half + 1
    sites = read_bed(a.bed)
    tfs = sorted({s[4] for s in sites})
    ti = {t: i for i, t in enumerate(tfs)}
    n_sites = np.zeros(len(tfs), dtype=np.int64)
    motif_len = {}
    for _, st, en, _, tf in sites:
        n_sites[ti[tf]] += 1
        motif_len.setdefault(tf, en - st)
    print("bed %s: %d sites, %d TFs, window +/-%d (%d cols)"
          % (a.bed, len(sites), len(tfs), a.half, W))

    # (chrom, pos) -> list of (tf_index, offset, minus_strand_flag)
    want = {}
    for chrom, st, en, strand, tf in sites:
        lo, hi = window_for(st, en, strand, a.half)
        minus = strand == "-"
        j = ti[tf]
        for pos in range(lo, hi):
            off = (hi - 1 - pos) if minus else (pos - lo)
            want.setdefault((chrom, pos), []).append((j, off, minus))
    print("window positions to watch: %d" % len(want))

    # meth[tf, channel, offset]; channel 0 = Watson in the motif frame, 1 = Crick
    meth = np.zeros((len(tfs), 2, W), dtype=np.int64)
    valid = np.zeros((len(tfs), 2, W), dtype=np.int64)

    t0 = time.time()
    n_rows = n_hit = 0
    with open(a.pileup) as fh:
        for line in fh:
            n_rows += 1
            f = line.split("\t", 10)
            key = (f[0], int(f[1]))
            hits = want.get(key)
            if hits is None:
                continue
            if f[3].upper() != "A":
                continue
            g = f[9].split(" ")
            trials = int(g[0])          # Nvalid_cov
            succ = int(g[2])            # Nmod
            if trials == 0:
                continue
            plus = f[5] == "+"
            n_hit += 1
            for j, off, minus in hits:
                # minus motif: channels cross (its Crick -> the motif frame's Watson)
                ch = (0 if plus else 1) if not minus else (1 if plus else 0)
                meth[j, ch, off] += succ
                valid[j, ch, off] += trials
    dt = time.time() - t0
    print("streamed %d rows in %.1f s (%d inside a window)" % (n_rows, dt, n_hit))

    np.savez_compressed(
        a.out, tfs=np.array(tfs), n_sites=n_sites, half=a.half,
        motif_len=np.array([motif_len[t] for t in tfs]), meth=meth, valid=valid,
        bed=a.bed, pileup=a.pileup)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
