#!/usr/bin/env python
"""Prove the per-molecule reader before anything is decoded with it.

Sum `read_calls.read_molecules` output over molecules and compare, position by position,
against the `Fiber_count_*` arrays an existing aggregate decode stored -- those came from
`modkit pileup`, so agreement means the per-molecule reader reproduces the aggregate
(k, n) the binomial emission has been using all along.

    python singlefiber/validate_reader.py \
        --h5 robocop_erv46_maskoff/tmpDir/info.h5 --segment 0 \
        --window chrI:60001-65000

Run from the analysis/ directory.
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import read_calls  # noqa: E402

BAM = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/"
       "combine_sequencing_runs/merged_Mar20_barcode01_Jun25_barcode21-24_"
       "May07_barcode03-04.bam")
FASTA = os.path.join(ANALYSIS, "inputs", "SacCer3.fa")


def sparse_todense(f, k):
    g = f[k]
    v = np.array(sparse.csr_matrix(
        (g["data"][:], g["indices"][:], g["indptr"][:]), g.attrs["shape"]).todense())
    return v[0] if v.shape[0] == 1 else v


def ref_codes(chrom, start1, stop1):
    """Nucleotide codes for chrom:start1-stop1 (1-based inclusive), via SeqIO.

    SeqIO, not faidx: the shipped SacCer3.fa.fai misindexes 12/17 chromosomes (see
    CLAUDE.md / memory), and getNucleotides.getNucleotideSequence uses SeqIO too.
    """
    sys.path.insert(0, os.path.join(ANALYSIS, "pkgvar", "permol_seq_maskoff"))
    from robocop.utils.getNucleotides import getNucleotideSequence
    return getNucleotideSequence(FASTA, chrom, start1, stop1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="robocop_erv46_maskoff/tmpDir/info.h5")
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--window", default="chrI:60001-65000")
    ap.add_argument("--tech2", default="Fiber")
    ap.add_argument("--flank", type=int, default=400)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    chrom, rng = a.window.split(":")
    start1, end1 = (int(x) for x in rng.split("-"))

    ext_start0 = max(0, start1 - 1 - a.flank)
    ext_end0 = end1 + a.flank
    codes = ref_codes(chrom, ext_start0 + 1, ext_end0)
    mols, es, ee, stats = read_calls.read_molecules(
        BAM, chrom, start1, end1, codes, flank=a.flank)
    assert (es, ee) == (ext_start0, ext_end0)
    print("molecules: %d   watson %d  crick %d" % (
        len(mols), sum(m.strand == "watson" for m in mols),
        sum(m.strand == "crick" for m in mols)))
    print("reader stats:", json.dumps(stats, sort_keys=True))
    tot = stats["informative"] + stats["ambiguous"]
    print("ambiguity loss: %d / %d = %.4f  (calls with %.4f < ML/255 < %.4f)"
          % (stats["ambiguous"], tot, stats["ambiguous"] / max(tot, 1),
             1 - read_calls.THRESH, read_calls.THRESH))

    pu = read_calls.pileup_from_molecules(mols, es, ee)
    # window slice of the per-molecule pileup, in the aggregate arrays' frame:
    # the stored Fiber_count_* arrays run from (start1 - 1) for end1 - start1 + 2 entries.
    o = (start1 - 1) - es
    n_store = end1 - start1 + 2
    mine = {k: v[o:o + n_store] for k, v in pu.items()}

    f = h5py.File(a.h5, "r")
    g = "segment_%d/%s" % (a.segment, a.tech2)
    agg = dict(k_watson=sparse_todense(f, g + "_count_meth_watson"),
               n_watson=sparse_todense(f, g + "_count_A_watson"),
               k_crick=sparse_todense(f, g + "_count_meth_crick"),
               n_crick=sparse_todense(f, g + "_count_A_crick"))
    f.close()

    res = {"molecules": len(mols), "stats": stats,
           "ambiguity_loss": stats["ambiguous"] / max(tot, 1)}

    # --- primary criterion: the positions the emission actually reads ---------------
    # robocop.py:791-792 consumes the Watson layer only where the REFERENCE base is A
    # (nucleotide_ref = 0) and the Crick layer only where it is T (=3); everything else
    # is zeroed across states. modkit, in contrast, also emits rows at reference
    # positions where a READ carries an A/T through a mismatch, and those rows are
    # invisible to the model. So the test that matters is per-position equality on the
    # on-reference positions.
    codes_win = codes[(start1 - 1) - es:(start1 - 1) - es + n_store]
    print("\n-- on-reference positions (the ones the emission reads) --")
    print("%-10s %8s %10s %10s %10s %8s" % ("array", "n_pos", "agg", "mine", "equal_at", "max|d|"))
    exact = True
    for key, mask in (("n_watson", codes_win == read_calls.CODE_A),
                      ("k_watson", codes_win == read_calls.CODE_A),
                      ("n_crick", codes_win == read_calls.CODE_T),
                      ("k_crick", codes_win == read_calls.CODE_T)):
        A = agg[key][mask].astype(np.int64)
        B = mine[key][mask].astype(np.int64)
        eq = int((A == B).sum())
        md = int(np.abs(A - B).max()) if mask.sum() else 0
        print("%-10s %8d %10d %10d %10d %8d" % (key, mask.sum(), A.sum(), B.sum(), eq, md))
        res["onref_" + key] = dict(n_pos=int(mask.sum()), agg=int(A.sum()), mine=int(B.sum()),
                                   equal_at=eq, max_abs_diff=md)
        if md != 0:
            exact = False
    res["onref_exact"] = bool(exact)
    print("on-reference agreement:", "EXACT" if exact else "NOT EXACT")

    print("\n-- all positions modkit reports, including read-only A/T at mismatches --")
    print("%-10s %10s %10s %8s %8s %8s %8s" % (
        "array", "agg_total", "mine_total", "ratio", "r", "n_defined", "max|d|"))
    ok = exact
    for key in ("n_watson", "k_watson", "n_crick", "k_crick"):
        A, B = agg[key].astype(float), mine[key].astype(float)
        assert A.shape == B.shape, (key, A.shape, B.shape)
        d = B - A
        both = (A > 0) | (B > 0)
        r = np.corrcoef(A[both], B[both])[0, 1] if both.sum() > 2 else np.nan
        ratio = B.sum() / A.sum() if A.sum() else np.nan
        print("%-10s %10d %10d %8.4f %8.4f %8d %8d" % (
            key, A.sum(), B.sum(), ratio, r, both.sum(), np.abs(d).max()))
        res[key] = dict(agg=float(A.sum()), mine=float(B.sum()), ratio=float(ratio),
                        r=float(r), max_abs_diff=float(np.abs(d).max()),
                        n_positions_either=int(both.sum()),
                        n_positions_equal=int((A == B).sum()))
        if not (0.85 <= ratio <= 1.0 + 1e-9 and r > 0.99):
            ok = False
    # positions where the aggregate has coverage the reader does not see at all
    for key in ("n_watson", "n_crick"):
        A, B = agg[key], mine[key]
        res[key]["agg_nonzero_mine_zero"] = int(((A > 0) & (B == 0)).sum())
        res[key]["mine_nonzero_agg_zero"] = int(((B > 0) & (A == 0)).sum())
        print("%s: agg>0 & mine==0 at %d positions; mine>0 & agg==0 at %d"
              % (key, res[key]["agg_nonzero_mine_zero"], res[key]["mine_nonzero_agg_zero"]))
    res["pass"] = bool(ok)
    print("\nreader validation:", "PASS" if ok else "CHECK")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print("wrote", a.json)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
