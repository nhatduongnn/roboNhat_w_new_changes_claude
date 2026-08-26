"""How much posterior mass do TF states hold in linkers and NDRs, baseline vs variants?

This is the number the three variants exist to move. The artifact: `combined_low_count`
(0.2489) sits ABOVE the background p (0.1383), and 142 of 154 TFs fall back to it, so those
states outscore background exactly where DNA is most accessible -- linkers and NDRs.

All values are DECODE OUTPUT, read from each run's posterior table via
plot_abf1_5sites_decoded.state_composition. Regions come from the same Chereji bed the
background estimator uses, so "linker" here means literally the windows computeLinkers()
trains on.

    conda activate pyranges_env3
    python compare_variants.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
import score_robocop as S
import plot_abf1_5sites_decoded as PD

RUNS = [
    ("baseline (shipped params)", "robocop_chrI_seq_maskoff_revfix"),
    ("V1 bg_tss",                 "robocop_chrI_seq_maskoff_bgtss"),
    ("V2 low_abf1",               "robocop_chrI_seq_maskoff_lowabf1"),
    ("V3 12_tfs",                 "robocop_chrI_seq_maskoff_12tfs"),
]
NUCS = "inputs/Chereji_2018_+1_-1_nucs.bed"


def regions(chrm="chrI", ndr_max=400):
    nu = pd.read_csv(NUCS, sep="\t", header=None)
    d = sorted(((nu[1] + nu[2]) // 2)[nu[0] == chrm].astype(int))
    linkers = [(x - 73 - 15, x - 73) for x in d if x - 88 > 0] + \
              [(x + 73, x + 73 + 15) for x in d]
    ndrs = [(d[i] + 73, d[i + 1] - 73) for i in range(len(d) - 1)
            if 0 < d[i + 1] - 73 - (d[i] + 73) <= ndr_max]
    nucbody = [(x - 73, x + 73) for x in d if x - 73 > 0]
    return {"linker (dyad±73..88)": linkers, "NDR (gap<=%d)" % ndr_max: ndrs,
            "nucleosome body": nucbody}


def mass(dec, chrm, spans, pad=0):
    """Mean posterior mass per group over the union of spans."""
    tot = {k: 0.0 for k in ("background", "abf1_fwd", "abf1_rev", "other_TFs",
                            "unknown", "nucleosome")}
    n = 0
    for lo, hi in spans:
        comp, pos, _ = PD.state_composition(dec, chrm, lo - pad, hi + pad)
        if comp is None:
            continue
        m = (pos >= lo) & (pos <= hi)
        if not m.any():
            continue
        for k in tot:
            tot[k] += float(comp[k][m].sum())
        n += int(m.sum())
    return {k: v / n for k, v in tot.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default="chrI")
    args = ap.parse_args()
    regs = regions(args.chrom)
    avail = [(l, o) for l, o in RUNS if os.path.isdir(os.path.join(o, "tmpDir"))]
    miss = [o for l, o in RUNS if not os.path.isdir(os.path.join(o, "tmpDir"))]
    if miss:
        print("NOT AVAILABLE: %s\n" % ", ".join(miss))

    for rname, spans in regs.items():
        print("=" * 104)
        print("%s   (%d intervals on %s)" % (rname, len(spans), args.chrom))
        print("%-28s %9s %9s %9s %9s %9s | %11s" %
              ("run", "backgrnd", "nucleos", "ABF1", "otherTF", "unknown", "ALL TF mass"))
        print("-" * 104)
        for lab, od in avail:
            dec = S.load_decode(od)
            m, n = mass(dec, args.chrom, spans)
            tfmass = m["abf1_fwd"] + m["abf1_rev"] + m["other_TFs"] + m["unknown"]
            print("%-28s %9.4f %9.4f %9.4f %9.4f %9.4f | %11.4f"
                  % (lab, m["background"], m["nucleosome"],
                     m["abf1_fwd"] + m["abf1_rev"], m["other_TFs"], m["unknown"], tfmass))
        print()

    print("=" * 104)
    print("READ: 'ALL TF mass' in linkers/NDRs is the artifact. Lower is better there --")
    print("those positions should be background. ABF1 at real sites is the signal to keep.")


if __name__ == "__main__":
    main()
