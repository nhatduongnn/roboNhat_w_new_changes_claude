#!/usr/bin/env python
"""Flag positions where the reads disagree with the reference. The decode ALWAYS uses the reference.

Rule (user decision, 2026-10-28; replaces the 2026-10-26 substitution rule). The per-read
decoder sticks with the reference genome at all times: nothing here changes the sequence it
decodes. At each reference position of the window, with coverage = reads with a base OR a
deletion there (same read filters as read_calls.read_molecules, except that reads without an MM
tag are kept; see pileup()):

    flagged_alt       coverage >= MIN_COV (10) and ONE non-reference base (A/C/G/T) is carried
                      by more than FRAC (0.5) x coverage reads. Written to the TSV only (later it
                      may be drawn on the decode plot). The reference base is still decoded.
    flagged_low_cov   coverage < MIN_COV.
    flagged_indel     coverage >= MIN_COV and a deletion, or one identical inserted sequence
                      (insertion placed after this reference position), is carried by more
                      than FRAC x coverage reads.
    (not written)     everything else.

A position can carry more than one action (joined with ';'). Each TSV row also carries the
per-strand split of coverage and of the top alt base (fwd = forward-aligned reads = Watson,
rev = reverse-aligned = Crick): a real variant should show on both strands, a basecall error is
often strand-specific.
"""
import json
from collections import Counter

import numpy as np
import pysam

MIN_COV = 10
FRAC = 0.5
BASES = "ACGT"
CODE = {"A": 0, "C": 1, "G": 2, "T": 3}

# pysam cigar ops
M_OPS = (0, 7, 8)      # M, =, X : aligned base
OP_I, OP_D, OP_N, OP_S = 1, 2, 3, 4


def _has_mm(read):
    """True when read_calls._ml_by_query_pos would find an A+a / A+21891 key (it skips reads
    without one: 'skipped_nomm')."""
    mb = read.modified_bases
    return bool(mb) and any(k[0] == "A" and k[2] in ("a", 21891) for k in mb)


def pileup(bam_path, chrom, start0, end0, require_mm=False, bam=None):
    """Per-position base / deletion / insertion counts over [start0, end0).

    Returns dict(base=(n,5) int [A,C,G,T,other], dele=(n,) int, ins_top=(n,) int,
    ins_seq=list of top inserted sequence or '', n_reads=int, n_reads_nomm=int, and the
    forward-strand part base_fwd=(n,5), dele_fwd=(n,); reverse = total - fwd).
    Read filters as read_calls.read_molecules (unmapped/secondary/supplementary/qcfail/duplicate
    dropped; no query sequence dropped; its MAPQ minimum is 0, i.e. no MAPQ filter). Reads
    WITHOUT an MM A+a tag are kept unless require_mm=True (read_calls drops them).
    """
    n = end0 - start0
    base = np.zeros((n, 5), np.int64)
    dele = np.zeros(n, np.int64)
    base_f = np.zeros((n, 5), np.int64)
    dele_f = np.zeros(n, np.int64)
    ins = {}
    own = bam is None
    if own:
        bam = pysam.AlignmentFile(bam_path)
    n_reads = n_nomm = 0
    for read in bam.fetch(chrom, start0, end0):
        if read.is_unmapped or read.is_secondary or read.is_supplementary \
                or read.is_qcfail or read.is_duplicate:
            continue
        seq = read.query_sequence
        if seq is None:
            continue
        if not _has_mm(read):
            n_nomm += 1
            if require_mm:
                continue
        n_reads += 1
        fwd = not read.is_reverse
        qarr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        r, q = read.reference_start, 0
        for op, L in read.cigartuples:
            if op in M_OPS:
                lo, hi = max(r, start0), min(r + L, end0)
                if lo < hi:
                    qb = qarr[q + (lo - r): q + (hi - r)]
                    idx = np.arange(lo - start0, hi - start0)
                    col = np.full(hi - lo, 4, np.int64)
                    for b, c in CODE.items():
                        col[qb == ord(b)] = c
                    np.add.at(base, (idx, col), 1)
                    if fwd:
                        np.add.at(base_f, (idx, col), 1)
                r += L
                q += L
            elif op == OP_D:
                lo, hi = max(r, start0), min(r + L, end0)
                if lo < hi:
                    dele[lo - start0:hi - start0] += 1
                    if fwd:
                        dele_f[lo - start0:hi - start0] += 1
                r += L
            elif op == OP_N:
                r += L                          # skipped region: not covered
            elif op == OP_I:
                p = r - 1                       # insertion sits after reference position r-1
                if start0 <= p < end0 and read.reference_start <= p:
                    ins.setdefault(p - start0, Counter())[seq[q:q + L]] += 1
                q += L
            elif op == OP_S:
                q += L
            # H (5), P (6): consume nothing
    if own:
        bam.close()
    ins_top = np.zeros(n, np.int64)
    ins_seq = [""] * n
    for i, c in ins.items():
        s, k = c.most_common(1)[0]
        ins_top[i], ins_seq[i] = k, s
    return dict(base=base, dele=dele, ins_top=ins_top, ins_seq=ins_seq, n_reads=n_reads,
                n_reads_nomm=n_nomm, base_fwd=base_f, dele_fwd=dele_f)


def classify(pu, ref, min_cov=MIN_COV, frac=FRAC):
    """The rule on a pileup `pu` (from pileup()) against reference codes `ref` (A=0..T=3, N=4).

    Returns a dict of per-position arrays: cov, top_alt, top_cnt, alt (flagged_alt), dmaj, imaj,
    indel, low, plus the strand split cov_fwd/cov_rev and top_alt count alt_fwd/alt_rev.
    """
    ref = np.asarray(ref)
    n = len(ref)
    cov = pu["base"].sum(axis=1) + pu["dele"]
    cov_f = pu["base_fwd"].sum(axis=1) + pu["dele_fwd"]
    altc = pu["base"][:, :4].copy()
    known = ref < 4
    altc[np.arange(n)[known], ref[known]] = 0             # only NON-reference bases compete
    top_alt = altc.argmax(axis=1)
    top_cnt = altc[np.arange(n), top_alt]
    alt_f = pu["base_fwd"][np.arange(n), top_alt]
    ok = cov >= min_cov
    alt = ok & (top_cnt > frac * cov)
    dmaj = ok & (pu["dele"] > frac * cov)
    imaj = ok & (pu["ins_top"] > frac * cov)
    return dict(cov=cov, top_alt=top_alt, top_cnt=top_cnt, ok=ok, alt=alt, dmaj=dmaj, imaj=imaj,
                indel=dmaj | imaj, low=~ok, cov_fwd=cov_f, cov_rev=cov - cov_f,
                alt_fwd=alt_f, alt_rev=top_cnt - alt_f)


def site_rows(pu, ref, c, pos1_of_index0, chrom=None):
    """TSV rows (list of dicts) for every flagged position of classify() output `c`."""
    rows = []
    for i in np.where(c["alt"] | c["indel"] | c["low"])[0]:
        acts = []
        if c["low"][i]:
            acts.append("flagged_low_cov")
        if c["alt"][i]:
            acts.append("flagged_alt")
        if c["indel"][i]:
            acts.append("flagged_indel")
        row = dict(pos=int(pos1_of_index0 + i), coverage=int(c["cov"][i]),
                   ref="ACGTN"[int(ref[i])], top_alt=BASES[int(c["top_alt"][i])],
                   alt_count=int(c["top_cnt"][i]), del_count=int(pu["dele"][i]),
                   ins_top_count=int(pu["ins_top"][i]), ins_top_seq=pu["ins_seq"][i],
                   cov_fwd=int(c["cov_fwd"][i]), cov_rev=int(c["cov_rev"][i]),
                   alt_fwd=int(c["alt_fwd"][i]), alt_rev=int(c["alt_rev"][i]),
                   action=";".join(acts))
        if chrom is not None:
            row["chrom"] = chrom
        rows.append(row)
    return rows


def consensus(bam_path, chrom, win_start1, win_end1, codes_ext, ext_start0,
              min_cov=MIN_COV, frac=FRAC):
    """Flag the window part of codes_ext ([ext_start0, ext_start0+len)). Changes nothing.

    Returns (sites: list of dicts for the TSV, summary dict). The caller keeps decoding the
    reference codes it already has; codes_ext is only read.
    """
    codes_ext = np.asarray(codes_ext)
    w0 = win_start1 - 1
    w1 = win_end1
    pu = pileup(bam_path, chrom, w0, w1)
    n = w1 - w0
    off = w0 - ext_start0
    ref = codes_ext[off:off + n]
    c = classify(pu, ref, min_cov, frac)
    sites = site_rows(pu, ref, c, win_start1)
    cov, ok = c["cov"], c["ok"]
    maf = c["top_cnt"] / np.maximum(cov, 1)
    summary = dict(rule="reference always decoded; flag only: flagged_alt = cov>=%d and one alt "
                        "base > %.2f x cov; flagged_indel; flagged_low_cov = cov<%d"
                        % (min_cov, frac, min_cov),
                   sequence_changed=False,
                   min_cov=min_cov, frac=frac, n_reads=int(pu["n_reads"]),
                   n_reads_without_mm=int(pu["n_reads_nomm"]),
                   n_positions=int(n), median_coverage=float(np.median(cov)),
                   min_coverage=int(cov.min()),
                   n_flagged_alt=int(c["alt"].sum()),
                   n_flagged_alt_both_strands=int((c["alt"] & (c["alt_fwd"] > frac * c["cov_fwd"])
                                                   & (c["alt_rev"] > frac * c["cov_rev"])).sum()),
                   n_flagged_low_cov=int(c["low"].sum()),
                   n_flagged_deletion=int(c["dmaj"].sum()), n_flagged_insertion=int(c["imaj"].sum()),
                   n_flagged_indel=int(c["indel"].sum()),
                   max_top_alt_frac_cov_ge_min=float(maf[ok].max()) if ok.any() else None,
                   max_del_frac_cov_ge_min=float((pu["dele"] / np.maximum(cov, 1))[ok].max())
                   if ok.any() else None,
                   max_ins_frac_cov_ge_min=float((pu["ins_top"] / np.maximum(cov, 1))[ok].max())
                   if ok.any() else None)
    return sites, summary


SITE_COLS = ["pos", "coverage", "ref", "top_alt", "alt_count", "del_count",
             "ins_top_count", "ins_top_seq", "cov_fwd", "cov_rev", "alt_fwd", "alt_rev", "action"]


def write_sites(path, sites):
    with open(path, "w") as f:
        f.write("\t".join(SITE_COLS) + "\n")
        for s in sites:
            f.write("\t".join(str(s[c]) for c in SITE_COLS) + "\n")


if __name__ == "__main__":
    import argparse
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    an = os.path.dirname(here)
    sys.path.insert(0, os.path.join(an, "pkgvar/permol_seq_maskoff_logfull/"))
    from robocop.utils.getNucleotides import getNucleotideSequence
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="chrI:60001-65000")
    ap.add_argument("--bam", required=True)
    ap.add_argument("--out", required=True, help="consensus_sites.tsv path")
    a = ap.parse_args()
    chrom, rng = a.window.split(":")
    s1, e1 = (int(x) for x in rng.split("-"))
    codes = np.asarray(getNucleotideSequence(os.path.join(an, "inputs", "SacCer3.fa"), chrom, s1, e1))
    sites, summ = consensus(a.bam, chrom, s1, e1, codes, s1 - 1)
    write_sites(a.out, sites)
    print(json.dumps(summ, indent=1))
