#!/usr/bin/env python
"""Is a MEME E-value comparable across TFs, and when is the top motif 'clearly' best?

MEME's E-value is the expected number of motifs scoring as well or better in a
shuffled dataset of the same size.  It therefore scales with how much data the
factor had -- number of sites, sequence count, motif width, information content.
That makes it a statement about EVIDENCE WITHIN ONE RUN, not a quality score you
can compare between factors.  This script measures that directly, and measures
the gap between the best and second-best motif within a run so the 'clearly
better by multiple folds' rule can be set from data instead of guessed.

Everything is in log10(E) because the values span ~500 orders of magnitude.
"""
import os
import re
import csv
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "motifdb")
BED = os.path.join(HERE, "inputs", "rossi_peak_w_strand_all_TFs.bed")

FLOOR = -500.0          # log10(E) floor; MEME underflows to 0 for the strongest


def load_replicates():
    """sample_id -> (TF, replicate). All Rossi dirs here are _YEP, so there is
    no condition axis to carry -- see the report at the bottom."""
    m = {}
    with open(BED) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            m[r["sample_id"]] = (r["TF"].upper(), r["replicate"])
    return m


def load_motifs():
    """-> list of dicts from motifdb/rossi.meme (already E<=0.05 filtered)."""
    out, cur = [], None
    with open(os.path.join(DB, "rossi.meme")) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                mid = line.split()[1]
                mm = re.match(r"(.+)_rossi_(\d+)_m(\d+)$", mid)
                cur = dict(id=mid, gene=mm.group(1).upper(),
                           sample=mm.group(2), k=int(mm.group(3)))
                out.append(cur)
            elif cur is not None and line.startswith("letter-probability"):
                cur["w"] = int(line.split("w=")[1].split()[0])
                cur["nsites"] = int(float(line.split("nsites=")[1].split()[0]))
                e = float(line.split("E=")[1].split()[0])
                cur["E"] = e
                cur["logE"] = np.log10(e) if e > 0 else FLOOR
    return out


def main():
    reps = load_replicates()
    mots = load_motifs()
    for m in mots:
        tf, rep = reps.get(m["sample"], (m["gene"], "?"))
        m["replicate"] = rep
    print("Rossi motifs surviving the E<=0.05 filter: %d over %d samples, %d genes"
          % (len(mots), len({m["sample"] for m in mots}),
             len({m["gene"] for m in mots})))

    # ---- 1. is a single E-value threshold meaningful across TFs? ------------
    best_per_gene = {}
    for m in mots:
        g = m["gene"]
        if g not in best_per_gene or m["logE"] < best_per_gene[g]["logE"]:
            best_per_gene[g] = m
    bl = np.array([m["logE"] for m in best_per_gene.values()])
    print("\n1. BEST log10(E) per gene, over %d genes" % len(bl))
    print("   min %.0f   p10 %.0f   median %.0f   p90 %.0f   max %.1f"
          % (bl.min(), np.percentile(bl, 10), np.median(bl),
             np.percentile(bl, 90), bl.max()))
    print("   -> the best motif of one factor can be 1e%.0f while the best motif"
          % bl.min())
    print("      of another is 1e%.1f. Any single cutoff either keeps junk for"
          % bl.max())
    print("      the data-rich factors or discards the real motif of the sparse ones.")

    ls = np.array([m["logE"] for m in mots])
    ns = np.array([m["nsites"] for m in mots], float)
    ws = np.array([m["w"] for m in mots], float)
    ok = ls > FLOOR            # drop the floored ones from the correlation
    r_ns = np.corrcoef(np.log10(ns[ok]), ls[ok])[0, 1]
    r_w = np.corrcoef(ws[ok], ls[ok])[0, 1]
    print("\n   correlation of log10(E) with log10(nsites): r = %+.2f  (n=%d)"
          % (r_ns, ok.sum()))
    print("   correlation of log10(E) with motif width:   r = %+.2f" % r_w)
    print("   -> E tracks how much data the run had. It is evidence within a run,")
    print("      not a quality score comparable between factors.")

    # ---- 2. within one sample, how far ahead is the best motif? -------------
    by_sample = defaultdict(list)
    for m in mots:
        by_sample[m["sample"]].append(m)
    gaps, singles = [], 0
    for s, ms in by_sample.items():
        if len(ms) < 2:
            singles += 1
            continue
        ms.sort(key=lambda x: x["logE"])
        gaps.append(ms[1]["logE"] - ms[0]["logE"])
    gaps = np.array(gaps)
    print("\n2. WITHIN a sample: log10 gap between best and second-best motif")
    print("   %d samples have >1 surviving motif, %d have exactly 1"
          % (len(gaps), singles))
    print("   min %.1f  p10 %.1f  p25 %.1f  median %.1f  p75 %.1f  max %.1f"
          % (gaps.min(), np.percentile(gaps, 10), np.percentile(gaps, 25),
             np.median(gaps), np.percentile(gaps, 75), gaps.max()))
    for t in (1, 2, 3, 5, 10, 20):
        print("   gap >= %2d orders of magnitude: %3d of %d samples (%.0f%%)"
              % (t, (gaps >= t).sum(), len(gaps), 100 * (gaps >= t).mean()))
    print("   -> a RELATIVE rule works where an absolute one cannot: the top motif")
    print("      is 'clearly best' when it leads the runner-up by a wide log gap.")

    # ---- 3. how consistent are replicates of the same TF? ------------------
    by_gene_sample = defaultdict(dict)
    for m in mots:
        g, s = m["gene"], m["sample"]
        if s not in by_gene_sample[g] or m["logE"] < by_gene_sample[g][s]["logE"]:
            by_gene_sample[g][s] = m
    multi = {g: d for g, d in by_gene_sample.items() if len(d) > 1}
    spans = np.array([max(x["logE"] for x in d.values())
                      - min(x["logE"] for x in d.values())
                      for d in multi.values()])
    print("\n3. ACROSS replicates of one TF: log10(E) span of each replicate's own")
    print("   best motif (%d genes with >1 sample)" % len(multi))
    print("   median %.1f  p90 %.1f  max %.1f"
          % (np.median(spans), np.percentile(spans, 90), spans.max()))
    print("   -> replicates of the same factor differ by orders of magnitude too,")
    print("      so 'best E-value across all samples' silently prefers the")
    print("      deepest-sequenced replicate rather than the best motif.")

    # ---- 4. condition ------------------------------------------------------
    print("\n4. CONDITION: every Rossi sample directory on disk is <id>_YEP")
    print("   (781 of 781). This download carries one growth condition, so there")
    print("   is no heat-shock / perturbation axis to annotate. What does exist is")
    print("   the replicate label:")
    rc = defaultdict(set)
    for m in mots:
        rc[m["gene"]].add(m["replicate"])
    from collections import Counter
    print("   replicates per gene:",
          dict(sorted(Counter(len(v) for v in rc.values()).items())))
    print("   replicate labels seen:",
          sorted({m["replicate"] for m in mots}))

    print("\nWorked example -- ABF1, every surviving motif:")
    print("   %-24s %-6s %-5s %5s %7s %12s"
          % ("motif", "rep", "m#", "w", "nsites", "E"))
    for m in sorted([x for x in mots if x["gene"] == "ABF1"],
                    key=lambda x: (x["sample"], x["k"])):
        print("   %-24s %-6s %-5d %5d %7d %12.1e"
              % (m["id"], m["replicate"], m["k"], m["w"], m["nsites"], m["E"]))


if __name__ == "__main__":
    main()
