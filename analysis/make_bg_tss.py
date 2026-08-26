"""VARIANT 1 (bg_tss): re-estimate the Fiber-seq background from TSS-proximal open
regions instead of nucleosome-flanking linkers.

Why. The shipped background (inputs/bg_params.pkl, p=0.1383) is fit by computeLinkers()
on two 15 bp windows pressed against each Chereji +1/-1 dyad (dyad+/-73..88). Those are
still partly nucleosome-occluded, so the estimate is far below open chromatin. Meanwhile
combined_low_count -- which 142 of 154 TFs fall back to -- is fit at Rossi peak centres in
wide-open NDRs and comes out at 0.2489. Background BELOW the fallback TF p means those TF
states outscore background wherever DNA is most accessible, putting TFs in linkers.

ORIENTATION MATTERS AND THE ORIGINAL CODE HAS NO CONCEPT OF IT. Park_2014_TSS.csv carries
no strand column; yeast ORF names encode it in the trailing W/C (YAL067C -> Crick). Pooled
over chrI, TSS-200..-100 measures 0.113 UNORIENTED and 0.144 ORIENTED. This script orients.

MEASURED EXPECTATION (chrI, before running): the default -200..-100 window gives ~0.144.
That is above the current background (0.138) but still BELOW combined_low_count (0.249),
so the "new bg > low_count" check will FAIL. -150..-50 measures ~0.192, still below. This
is reported explicitly rather than buried -- see the verdict block at the end.

Writes inputs/bg_params_tss.pkl. Does not modify inputs/bg_params.pkl or the generator.

    conda activate robocop-2024
    export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
    python make_bg_tss.py                     # default -200..-100
    python make_bg_tss.py --up 150 --down 50  # the more open window
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

import fiber_params_lib

TSS_FILE = "inputs/Park_2014_TSS.csv"
PILEUP = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/"
          "combine_sequencing_runs/merged_Mar20_barcode01_Jun25_barcode21-24_"
          "May07_barcode03-04_sup_model_sorted_pileup_all_chr")
FASTA = "inputs/SacCer3.fa"
OUT = "inputs/bg_params_tss.pkl"
ORF_STRAND = re.compile(r"^Y[A-P][LR]\d{3}([WC])")


def computeTSSOpenRegions(tssFile, up=200, down=100, verbose=True):
    """Open regions upstream of each TSS, ORIENTED by ORF strand.

    Mirrors computeLinkers()'s return shape -- [{"chrm","start","stop"}, ...] -- so it
    drops straight into compute_Fiber_seq_background().

    `up`/`down` are distances UPSTREAM of the TSS, so the window is [TSS-up, TSS-down)
    for a Watson gene and mirrored to [TSS+down, TSS+up) for a Crick gene. Without that
    flip the window lands in the gene body for half the genes, which is what drags the
    unoriented estimate down to 0.113.
    """
    tss = pd.read_csv(tssFile, sep="\t")
    tss["strand"] = tss["ORF"].astype(str).str.extract(ORF_STRAND)[0]
    n_all = len(tss)
    tss = tss[(tss["internal"] == 0) & (tss["flag"] == 0) & tss["strand"].notna()]
    segments = []
    for _, r in tss.iterrows():
        c, chrm = int(r["coordinate"]), r["chr"]
        if "micron" in str(chrm):
            continue
        lo, hi = (c - up, c - down) if r["strand"] == "W" else (c + down, c + up)
        if lo > 0:
            segments.append({"chrm": chrm, "start": int(lo), "stop": int(hi)})
    if verbose:
        print("TSS open regions: %d segments from %d TSSes (%d rows in file)"
              % (len(segments), len(tss), n_all))
        print("  window: TSS-%d..TSS-%d upstream, strand-flipped for Crick genes" % (up, down))
        print("  strand split: %s" % tss["strand"].value_counts().to_dict())
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=int, default=200, help="upstream far edge")
    ap.add_argument("--down", type=int, default=100, help="upstream near edge")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ns = fiber_params_lib.load()
    segments = computeTSSOpenRegions(TSS_FILE, args.up, args.down)

    print("\nreading pileup (this is the slow part)...", flush=True)
    df = pd.read_csv(PILEUP, sep="\t", header=None)
    split = df[9].str.split(" ", expand=True)
    split.columns = [i for i in range(9, 9 + split.shape[1])]
    df = pd.concat([df.drop(columns=[9]), split], axis=1)
    # The split yields STRING columns. compute_individual_Fiber_seq_TF_binom() casts with
    # int(row[...]) per cell so the TF path survives this, but compute_Fiber_seq_background()
    # calls .to_numpy() straight into arithmetic and dies on str vs int. Cast here rather
    # than editing the generator, which must stay untouched.
    for col in split.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    print("  pileup rows: %d   (split cols %s cast to int)"
          % (len(df), list(split.columns)))

    bg = ns["compute_Fiber_seq_background"](
        df, segments, FASTA, dist="binomial", successes_col=11, trials_col=9)

    w = float(np.ravel(bg["p"]["watson_signal"]["A"])[0])
    c = float(np.ravel(bg["p"]["crick_signal"]["A"])[0])
    with open(args.out, "wb") as f:
        pickle.dump(bg, f)
    print("\nwrote %s" % args.out)

    old = pickle.load(open("inputs/bg_params.pkl", "rb"))
    ow = float(np.ravel(old["p"]["watson_signal"]["A"])[0])
    oc = float(np.ravel(old["p"]["crick_signal"]["A"])[0])
    lp = pickle.load(open("inputs/all_TFs_1000pealVal_params_pseudo.pkl", "rb"))
    lw = float(np.ravel(lp["p"]["combined_low_count"]["watson_signal"]["A"])[0])
    lc = float(np.ravel(lp["p"]["combined_low_count"]["crick_signal"]["A"])[0])

    print("\n%-34s %10s %10s" % ("", "watson", "crick"))
    print("%-34s %10.4f %10.4f" % ("OLD background (linkers)", ow, oc))
    print("%-34s %10.4f %10.4f" % ("NEW background (TSS open regions)", w, c))
    print("%-34s %10.4f %10.4f" % ("combined_low_count (the target)", lw, lc))
    print("\nVERDICT on the requested check (new bg > combined_low_count):")
    for lab, new, low in (("watson", w, lw), ("crick", c, lc)):
        ok = new > low
        print("   %-7s %.4f vs %.4f  ->  %s" % (lab, new, low,
              "PASS" if ok else "FAIL (background still below the fallback TF p)"))
    print("\nsecondary check (new bg > old bg, i.e. did it move the right way):")
    for lab, new, o in (("watson", w, ow), ("crick", c, oc)):
        print("   %-7s %.4f vs %.4f  ->  %s (%.2fx)"
              % (lab, new, o, "higher" if new > o else "LOWER", new / o))


if __name__ == "__main__":
    main()
