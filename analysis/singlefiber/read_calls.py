#!/usr/bin/env python
"""MM/ML -> per-molecule binary m6A calls, matched to `modkit pileup` semantics.

Everything else in this repo decodes the AGGREGATE Fiber-seq signal: `modkit pileup`
pools all molecules into (k methylated, n valid trials) per reference position and the
emission is a binomial. This module goes back to the BAM and recovers, for one molecule
at a time, the same call that modkit would have contributed to that pileup.

Call rule (modkit `pileup` with a confidence threshold, here 0.8105469 as recorded in
`pileup.log_whole_genome`): for a two-state model the called state is the more likely of
{m6A, canonical} and the call is kept only when its probability reaches the threshold.
With p_mod = ML/255 that is

    methylated  ML/255 >= 0.8105469
    canonical   ML/255 <= 1 - 0.8105469 = 0.1894531, OR the position is absent from MM
                (the tag here is mode `A+a.`, i.e. unlisted adenines are implicitly
                 canonical)
    dropped     anything in between -- an ambiguous call, which modkit also excludes
                from Nvalid_cov, so it is "no observation" for the HMM as well.

Strand: `dx:i:0` on every sampled read (no duplex), so a forward-aligned read carries
m6A on the reference forward strand (Watson channel, reference base A) and a
reverse-aligned read on the reverse strand (Crick channel, reference base T). That is
exactly the aggregate contract: `robocop.py` uses `nucleotide_ref = 0` (A) for the
Watson layer and `3` (T) for the Crick layer.

`pysam.AlignedSegment.modified_bases` reports positions in STORED-query coordinates --
verified: for a reverse read the key is ('A', 1, 'a') and `query_sequence[pos]` is 'T'
-- so they line up with `get_aligned_pairs()` query positions without any flipping.
"""
import numpy as np
import pysam

# modkit's confidence threshold for this dataset (pileup.log_whole_genome)
THRESH = 0.8105469
ML_METH = THRESH * 255.0            # >= this ML value -> methylated
ML_CANON = (1.0 - THRESH) * 255.0   # <= this ML value -> canonical

NO_OBS, CANON, METH = -1, 0, 1

# nucleotide codes as getNucleotides.mapNucToInt writes them: A=0 C=1 G=2 T=3 N=4
CODE_A, CODE_T = 0, 3


class Molecule(object):
    """One molecule's m6A calls on one reference interval.

    calls      int8, length (ext_end0 - ext_start0), values NO_OBS/CANON/METH, indexed
               by reference 0-based coordinate minus ext_start0
    strand     'watson' (forward read) or 'crick' (reverse read)
    """
    __slots__ = ("name", "strand", "is_reverse", "ref_start0", "ref_end0", "calls",
                 "n_informative", "n_meth", "n_ambig", "flank_n", "flank_meth",
                 "span_n", "span_meth", "mapq")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def rate_flank(self):
        return self.flank_meth / self.flank_n if self.flank_n else np.nan

    @property
    def rate_span(self):
        return self.span_meth / self.span_n if self.span_n else np.nan


def mm_modes(read):
    """The per-group mode flag of the MM tag, e.g. {'A+a': '.'}."""
    tag = None
    for t in ("MM", "Mm"):
        if read.has_tag(t):
            tag = read.get_tag(t)
            break
    if tag is None:
        return {}
    out = {}
    for group in str(tag).split(";"):
        if not group:
            continue
        head = group.split(",")[0]
        if head and head[-1] in ".?":
            out[head[:-1]] = head[-1]
        else:
            out[head] = ""
    return out


def _ml_by_query_pos(read, qlen):
    """Dense ML lookup over stored-query positions; -1 where the position is not in MM."""
    ml = np.full(qlen, -1, dtype=np.int16)
    mb = read.modified_bases
    if not mb:
        return ml, None
    keys = [k for k in mb if k[0] == "A" and k[2] in ("a", 21891)]
    if not keys:
        return ml, None
    for k in keys:
        arr = np.asarray(mb[k], dtype=np.int64)
        if arr.size:
            ml[arr[:, 0]] = arr[:, 1]
    return ml, keys[0]


def read_molecules(bam_path, chrom, win_start1, win_end1, ref_codes_ext, flank=400,
                   min_mapq=0, require_implicit_canonical=True):
    """Per-molecule calls for every read overlapping chrom:win_start1-win_end1.

    win_start1/win_end1 are 1-based inclusive, as coords.tsv writes them.
    ref_codes_ext must be the nucleotide codes for [ext_start0, ext_end0) where
    ext_start0 = max(0, win_start1 - 1 - flank) and ext_end0 = win_end1 + flank.

    Returns (molecules, ext_start0, ext_end0, stats).
    """
    win_start0 = win_start1 - 1
    win_end0 = win_end1                      # exclusive
    ext_start0 = max(0, win_start0 - flank)
    ext_end0 = win_end0 + flank
    n_ext = ext_end0 - ext_start0
    assert len(ref_codes_ext) == n_ext, (len(ref_codes_ext), n_ext)
    ref_codes_ext = np.asarray(ref_codes_ext)

    bam = pysam.AlignmentFile(bam_path)
    mols = []
    stats = dict(fetched=0, skipped_flag=0, skipped_mapq=0, skipped_nomm=0,
                 mode_not_dot=0, duplex=0, informative=0, meth=0, ambiguous=0,
                 unlisted_canonical=0)

    for read in bam.fetch(chrom, win_start0, win_end0):
        stats["fetched"] += 1
        if read.is_unmapped or read.is_secondary or read.is_supplementary \
                or read.is_qcfail or read.is_duplicate:
            stats["skipped_flag"] += 1
            continue
        if read.mapping_quality < min_mapq:
            stats["skipped_mapq"] += 1
            continue
        if read.has_tag("dx") and int(read.get_tag("dx")) != 0:
            stats["duplex"] += 1
        seq = read.query_sequence
        if seq is None:
            stats["skipped_nomm"] += 1
            continue
        qlen = len(seq)
        ml, key = _ml_by_query_pos(read, qlen)
        if key is None:
            stats["skipped_nomm"] += 1
            continue
        modes = mm_modes(read)
        mode = modes.get("A+a", modes.get("A+a?", ""))
        if mode != ".":
            stats["mode_not_dot"] += 1
            if require_implicit_canonical:
                # mode '?' means unlisted adenines carry no information; only listed
                # positions may be called.
                pass

        target_code = CODE_A if not read.is_reverse else CODE_T
        target_base = b"A" if not read.is_reverse else b"T"
        strand = "watson" if not read.is_reverse else "crick"

        pairs = read.get_aligned_pairs(matches_only=True)
        if not pairs:
            continue
        pa = np.asarray(pairs, dtype=np.int64)
        qp, rp = pa[:, 0], pa[:, 1]
        sel = (rp >= ext_start0) & (rp < ext_end0)
        if not sel.any():
            continue
        qp, rp = qp[sel], rp[sel]
        ri = rp - ext_start0
        qbase = np.frombuffer(seq.encode("ascii"), dtype="S1")[qp]
        on_target = (ref_codes_ext[ri] == target_code) & (qbase == target_base)
        if not on_target.any():
            continue
        ri_t, qp_t = ri[on_target], qp[on_target]
        mlv = ml[qp_t].astype(np.float64)

        calls = np.full(n_ext, NO_OBS, dtype=np.int8)
        unlisted = mlv < 0
        is_meth = (~unlisted) & (mlv >= ML_METH)
        is_canon = unlisted | ((~unlisted) & (mlv <= ML_CANON))
        if mode != "." and require_implicit_canonical:
            # no implicit canonical for mode '?'
            is_canon = (~unlisted) & (mlv <= ML_CANON)
        ambig = ~(is_meth | is_canon)
        calls[ri_t[is_meth]] = METH
        calls[ri_t[is_canon]] = CANON

        stats["informative"] += int(is_meth.sum() + is_canon.sum())
        stats["meth"] += int(is_meth.sum())
        stats["ambiguous"] += int(ambig.sum())
        stats["unlisted_canonical"] += int(unlisted.sum())

        # --- per-read efficiency statistics ---
        # flank = the +/-`flank` bp outside the window (the "candidate" window excluded)
        in_win = np.zeros(n_ext, dtype=bool)
        in_win[win_start0 - ext_start0: win_end0 - ext_start0] = True
        inf = calls != NO_OBS
        fl = inf & ~in_win
        mols.append(Molecule(
            name=read.query_name, strand=strand, is_reverse=bool(read.is_reverse),
            mapq=int(read.mapping_quality),
            ref_start0=int(max(read.reference_start, ext_start0)),
            ref_end0=int(min(read.reference_end, ext_end0)),
            calls=calls,
            n_informative=int(inf.sum()), n_meth=int((calls == METH).sum()),
            n_ambig=int(ambig.sum()),
            flank_n=int(fl.sum()), flank_meth=int((calls[fl] == METH).sum()),
            span_n=int(inf.sum()), span_meth=int((calls == METH).sum())))
    bam.close()
    return mols, ext_start0, ext_end0, stats


def pileup_from_molecules(mols, ext_start0, ext_end0):
    """Sum per-molecule calls back into (k, n) per reference position, per channel.

    This is the reader's correctness test: these four arrays must reproduce the
    `Fiber_count_*` arrays that `modkit pileup` produced, up to calls modkit and this
    reader classify differently.
    """
    n = ext_end0 - ext_start0
    out = dict(k_watson=np.zeros(n, np.int64), n_watson=np.zeros(n, np.int64),
               k_crick=np.zeros(n, np.int64), n_crick=np.zeros(n, np.int64))
    for m in mols:
        ks, ns = ("k_watson", "n_watson") if m.strand == "watson" else ("k_crick", "n_crick")
        out[ns] += (m.calls != NO_OBS)
        out[ks] += (m.calls == METH)
    return out
