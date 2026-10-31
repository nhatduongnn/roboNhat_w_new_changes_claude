#!/usr/bin/env python
"""Genome-wide count of the single-fiber consensus FLAG rule (consensus_seq.py), flag only.

Same counting (consensus_seq.pileup) and same rule (consensus_seq.classify) as the per-read
driver, applied to whole chromosomes in CHUNK-bp pieces. Read filters are read_calls'
(flags, no sequence; MAPQ minimum 0 = no MAPQ filter; reads without an MM A+a tag dropped,
require_mm=True, as read_calls does; the per-read driver's flagging keeps them, see
consensus_seq.pileup). n_reads_without_mm is recorded so the difference is visible.

  scan     --chrom C : writes <out>/per_chrom/<C>.flags.tsv (every flagged position),
                       <C>.arrays.npz (per-position counts) and <C>.json (run info)
  summary            : reads every npz -> <out>/summary.tsv, alt_frac_hist.tsv,
                       strand_split.tsv, reconcile.tsv, summary.json

Usage (from analysis/):
  python singlefiber/consensus_scan_genome.py scan --chrom chrI --out /usr/project/xtmp/nd141/consensus_scan
  python singlefiber/consensus_scan_genome.py summary --out /usr/project/xtmp/nd141/consensus_scan
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pysam
from Bio import SeqIO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import consensus_seq as CS                                        # noqa: E402

FASTA = os.path.join(os.path.dirname(HERE), "inputs", "SacCer3.fa")
BAM = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/combine_sequencing_runs/"
       "merged_Mar20_barcode01_Jun25_barcode21-24_May07_barcode03-04.bam")
CHUNK = 100_000
# named sub-regions reported separately (1-based inclusive)
REGIONS = {"chrXII_rDNA": ("chrXII", 450_001, 470_000)}
LUT = np.full(256, 4, np.uint8)
for _b, _c in CS.CODE.items():
    LUT[ord(_b)] = _c
    LUT[ord(_b.lower())] = _c


def ref_codes(chrom):
    for rec in SeqIO.parse(open(FASTA), "fasta"):
        if rec.name == chrom:
            return LUT[np.frombuffer(str(rec.seq).encode("ascii"), np.uint8)]
    raise KeyError(chrom)


def scan(a):
    t0 = time.time()
    ref = ref_codes(a.chrom)
    bam = pysam.AlignmentFile(a.bam)
    L = dict(zip(bam.references, bam.lengths))[a.chrom]
    assert L == len(ref), (a.chrom, L, len(ref))
    keys = ("cov", "cov_fwd", "top_cnt", "alt_fwd", "altsum", "altsum_fwd", "dele", "dele_fwd",
            "ins_top")
    arr = {k: np.zeros(L, np.int32) for k in keys}
    arr["top_alt"] = np.zeros(L, np.uint8)
    pdir = os.path.join(a.out, "per_chrom")
    os.makedirs(pdir, exist_ok=True)
    tsv = os.path.join(pdir, a.chrom + ".flags.tsv")
    n_reads_chunks = n_nomm_chunks = 0
    with open(tsv + ".part", "w") as f:
        f.write("\t".join(["chrom"] + CS.SITE_COLS) + "\n")
        for s0 in range(0, L, CHUNK):
            e0 = min(L, s0 + CHUNK)
            pu = CS.pileup(a.bam, a.chrom, s0, e0, require_mm=True, bam=bam)
            n_reads_chunks += pu["n_reads"]
            n_nomm_chunks += pu["n_reads_nomm"]
            r = ref[s0:e0]
            c = CS.classify(pu, r)
            for row in CS.site_rows(pu, r, c, s0 + 1, chrom=a.chrom):
                f.write("\t".join(str(row[k]) for k in ["chrom"] + CS.SITE_COLS) + "\n")
            n = e0 - s0
            altc = pu["base"][:, :4].copy()
            altf = pu["base_fwd"][:, :4].copy()
            kn = r < 4
            altc[np.arange(n)[kn], r[kn]] = 0
            altf[np.arange(n)[kn], r[kn]] = 0
            sl = slice(s0, e0)
            arr["cov"][sl] = c["cov"]
            arr["cov_fwd"][sl] = c["cov_fwd"]
            arr["top_cnt"][sl] = c["top_cnt"]
            arr["alt_fwd"][sl] = c["alt_fwd"]
            arr["top_alt"][sl] = c["top_alt"]
            arr["altsum"][sl] = altc.sum(axis=1)
            arr["altsum_fwd"][sl] = altf.sum(axis=1)
            arr["dele"][sl] = pu["dele"]
            arr["dele_fwd"][sl] = pu["dele_fwd"]
            arr["ins_top"][sl] = pu["ins_top"]
            print("%s %d-%d reads %d nomm %d (%.0fs)" % (a.chrom, s0, e0, pu["n_reads"],
                                                         pu["n_reads_nomm"], time.time() - t0))
            sys.stdout.flush()
    bam.close()
    np.savez_compressed(os.path.join(pdir, a.chrom + ".arrays.npz"), ref=ref, **arr)
    os.replace(tsv + ".part", tsv)
    json.dump(dict(chrom=a.chrom, length=L, bam=a.bam, fasta=FASTA, chunk=CHUNK,
                   min_cov=CS.MIN_COV, frac=CS.FRAC, require_mm=True,
                   read_chunk_visits=n_reads_chunks, nomm_chunk_visits=n_nomm_chunks,
                   secs=time.time() - t0, slurm_job_id=os.environ.get("SLURM_JOB_ID")),
              open(os.path.join(pdir, a.chrom + ".json"), "w"), indent=1)


def stats(z, lo=0, hi=None):
    """All counts over [lo, hi) of one chromosome's arrays."""
    g = {k: z[k][lo:hi].astype(np.int64) for k in z.files}
    cov, covf = g["cov"], g["cov_fwd"]
    covr = cov - covf
    top, topf = g["top_cnt"], g["alt_fwd"]
    topr = top - topf
    ok = cov >= CS.MIN_COV
    alt = ok & (top > CS.FRAC * cov)
    indel = ok & ((g["dele"] > CS.FRAC * cov) | (g["ins_top"] > CS.FRAC * cov))
    frac = top / np.maximum(cov, 1)
    fwd_maj = topf > CS.FRAC * covf
    rev_maj = topr > CS.FRAC * covr
    s = dict(length=int(len(cov)), cov_ge10=int(ok.sum()), median_cov=float(np.median(cov)),
             flagged_alt=int(alt.sum()), flagged_low_cov=int((~ok).sum()),
             flagged_indel=int(indel.sum()))
    s["flagged_alt_per_Mb_cov_ge10"] = s["flagged_alt"] / max(s["cov_ge10"], 1) * 1e6
    # alt-base fraction distribution of flagged_alt
    s["alt_frac_50_60"] = int((alt & (frac <= 0.6)).sum())
    s["alt_frac_60_90"] = int((alt & (frac > 0.6) & (frac <= 0.9)).sum())
    s["alt_frac_gt90"] = int((alt & (frac > 0.9)).sum())
    # strand split of flagged_alt (same top alt base, > 50% of that strand's coverage)
    both10 = (covf >= 10) & (covr >= 10)
    s["alt_both_strands_cov10"] = int((alt & both10).sum())
    s["alt_majority_on_both"] = int((alt & both10 & fwd_maj & rev_maj).sum())
    s["alt_majority_fwd_only"] = int((alt & both10 & fwd_maj & ~rev_maj).sum())
    s["alt_majority_rev_only"] = int((alt & both10 & ~fwd_maj & rev_maj).sum())
    s["alt_one_strand_cov_lt10"] = int((alt & ~both10).sum())
    s["alt_majority_on_both_gt90"] = int((alt & both10 & fwd_maj & rev_maj & (frac > 0.9)).sum())
    # reconcile with the earlier "4.5 / kb" estimate: >= 50% of reads disagree on BOTH strands,
    # >= 10 reads per strand, under three meanings of "disagree"
    altsum, altsumf = g["altsum"], g["altsum_fwd"]
    dele, delf = g["dele"], g["dele_fwd"]
    d_top = (topf >= 0.5 * covf) & (topr >= 0.5 * covr)                      # same alt base
    d_anyb = (altsumf >= 0.5 * covf) & ((altsum - altsumf) >= 0.5 * covr)    # any alt base
    nr_f = altsumf + delf
    nr_r = (altsum - altsumf) + (dele - delf)
    d_anyd = (nr_f >= 0.5 * covf) & (nr_r >= 0.5 * covr)                      # alt base or del
    s["recon_bases_both10"] = int(both10.sum())
    s["recon_top_alt_ge50_both"] = int((both10 & d_top).sum())
    s["recon_any_alt_ge50_both"] = int((both10 & d_anyb).sum())
    s["recon_alt_or_del_ge50_both"] = int((both10 & d_anyd).sum())
    s["recon_del_ge50_both"] = int((both10 & (delf >= 0.5 * covf) & ((dele - delf) >= 0.5 * covr)).sum())
    return s, dict(alt=alt, frac=frac)


def summary(a):
    pdir = os.path.join(a.out, "per_chrom")
    order = [r.name for r in SeqIO.parse(open(FASTA), "fasta")]
    rows, hist_all = [], np.zeros(10, np.int64)
    edges = np.linspace(0.5, 1.0, 11)
    for ch in order:
        p = os.path.join(pdir, ch + ".arrays.npz")
        if not os.path.exists(p):
            print("MISSING", ch)
            continue
        z = np.load(p)
        s, x = stats(z)
        rows.append(dict(region=ch, **s))
        h, _ = np.histogram(x["frac"][x["alt"]], bins=edges)
        if ch != "chrM":
            hist_all += h
        for nm, (c2, s1, e1) in REGIONS.items():
            if c2 == ch:
                rows.append(dict(region=nm, **stats(z, s1 - 1, e1)[0]))
                lo, hi = stats(z, 0, s1 - 1)[0], stats(z, e1, None)[0]
                rows.append(dict(region=ch + "_minus_rDNA",
                                 **{k: (lo[k] + hi[k]) for k in lo if k not in
                                    ("median_cov", "flagged_alt_per_Mb_cov_ge10")},
                                 median_cov=float("nan"), flagged_alt_per_Mb_cov_ge10=0.0))
                rows[-1]["flagged_alt_per_Mb_cov_ge10"] = rows[-1]["flagged_alt"] / \
                    max(rows[-1]["cov_ge10"], 1) * 1e6
    nuc = [r for r in rows if r["region"] in order and r["region"] != "chrM"]
    tot = {k: sum(r[k] for r in nuc) for k in nuc[0] if k not in
           ("region", "median_cov", "flagged_alt_per_Mb_cov_ge10")}
    tot["flagged_alt_per_Mb_cov_ge10"] = tot["flagged_alt"] / max(tot["cov_ge10"], 1) * 1e6
    rows.append(dict(region="nuclear_total(excl_chrM)", median_cov=float("nan"), **tot))
    nr = [r for r in rows if r["region"] in order and r["region"] not in ("chrM", "chrXII")] + \
        [r for r in rows if r["region"] == "chrXII_minus_rDNA"]
    t2 = {k: sum(r[k] for r in nr) for k in tot if k != "flagged_alt_per_Mb_cov_ge10"}
    t2["flagged_alt_per_Mb_cov_ge10"] = t2["flagged_alt"] / max(t2["cov_ge10"], 1) * 1e6
    rows.append(dict(region="nuclear_excl_chrM_rDNA", median_cov=float("nan"), **t2))
    cols = ["region"] + [k for k in rows[0] if k != "region"]
    with open(os.path.join(a.out, "summary.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(("%.2f" % r[c]) if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    with open(os.path.join(a.out, "alt_frac_hist_nuclear.tsv"), "w") as f:
        f.write("frac_lo\tfrac_hi\tn_flagged_alt\n")
        for i in range(10):
            f.write("%.2f\t%.2f\t%d\n" % (edges[i], edges[i + 1], hist_all[i]))
    json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(open(os.path.join(a.out, "summary.tsv")).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "summary"])
    ap.add_argument("--chrom")
    ap.add_argument("--bam", default=BAM)
    ap.add_argument("--out", default="/usr/project/xtmp/nd141/consensus_scan")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else summary(a)


if __name__ == "__main__":
    main()
