#!/usr/bin/env python
"""Extract the collapsed per-factor posterior table for one decode + chromosome.

Why a new reader instead of score_robocop.region_optable
-------------------------------------------------------
Two reasons, both checkable:

1. Every decode's `config.ini:17` says `trainDir = ./robocop_train_fiberonly/`, which is NOT
   the trainDir the decode was actually run with (`RUN_INFO.json:"traindir"` ->
   robocop_train_tw_bo09_48 / robocop_train_tw_bu01_11, passed as argv[2] by the driver).
   `score_robocop._resolve_train_dir` trusts config.ini, so it loads the wrong HMMconfig.
   The STATE LAYOUT is identical across the three (verified: same tfs/tf_starts/tf_lens
   hashes, n_states 3485, nuc_start 2799, padding 0), so the collapse is unaffected -- but
   tf_prob is not (1.4001e-51 vs 3.37063e-05 vs 1.79737e-07), so this reader takes the
   trainDir from RUN_INFO.json and says which it used.
2. region_optable allocates (n, 3485) per chunk; this streams segment by segment and keeps
   only the collapsed (n, 160) table, so a whole chromosome fits.

Everything load-bearing is copied, not reinvented:
  - overlapping segments are SUMMED then divided by the coverage count
    (score_robocop.py:220-221);
  - the collapse is singlefiber/collapse.collapse_matrix (asserted equal to
    robocop.sum_for_dbf_probs by its own assert_matches);
  - validity is the [0,1] bound on the float32 cast of EVERY column
    (count_calls.py:131-141), not finiteness;
  - positions are 1-based genomic, index i of a segment array <-> coord seg_start + i
    (getNucleotides.getNucleotideSequence slices [start-1:stop]; getReads
    getValuesFiber_seqOneFileNucleotide:211 writes index pos0-(start-1) = coord-start).

Writes <out>/<tag>_<chrom>.npz   : pos, abf1, ok, nt, meth_w, meth_c, A_w, A_c, cols
       <out>/<tag>_<chrom>_full.npy : (n_pos, n_cols) float32 collapsed table (all states)
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "singlefiber"))
import collapse as C                      # singlefiber/collapse.py
import pickle
import pandas as pd

VALID_TOL = 1e-4                           # count_calls.py:63


def sparse_todense(f, k):
    g = f[k]
    v = np.array(sparse.csr_matrix(
        (g['data'][:], g['indices'][:], g['indptr'][:]), g.attrs['shape']).todense())
    return v[0] if v.shape[0] == 1 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outDir")
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    info = json.load(open(os.path.join(a.outDir, "RUN_INFO.json")))
    traindir = os.path.join(HERE, info["traindir"])
    dshared = pickle.load(open(os.path.join(traindir, "HMMconfig.pkl"), "rb"))
    print("%s %s: trainDir from RUN_INFO = %s (config.ini says %s)"
          % (a.tag, a.chrom, info["traindir"], "robocop_train_fiberonly"))

    coords = pd.read_csv(os.path.join(a.outDir, "coords.tsv"), sep="\t")
    sel = coords[coords["chr"] == a.chrom]
    lo, hi = int(sel["start"].min()), int(sel["end"].max())
    n = hi - lo + 1
    cols = C.column_names(dshared)
    M = C.collapse_matrix(dshared)
    print("  %d segments, span %d-%d (%d bp), %d collapsed columns"
          % (len(sel), lo, hi, n, len(cols)))

    acc = np.zeros((n, len(cols)), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int32)
    nt = np.full(n, 4, dtype=np.int8)
    fib = {k: np.zeros(n, dtype=np.int32) for k in
           ("meth_watson", "meth_crick", "A_watson", "A_crick")}
    fib_conflict = 0
    seen = set()

    for infofile in sorted(glob.glob(os.path.join(a.outDir, "tmpDir", "info*.h5"))):
        with h5py.File(infofile, "r") as f:
            for k in f.keys():
                idx = int(k.split("_")[1])
                if idx not in sel.index:
                    continue
                assert idx not in seen, "segment %d seen twice" % idx
                seen.add(idx)
                s0 = int(coords.loc[idx, "start"]) - lo      # array offset of segment start
                dp = sparse_todense(f, k + "/posterior")
                L = dp.shape[0]
                acc[s0:s0 + L] += C.collapse(dp, M)
                cnt[s0:s0 + L] += 1
                nt[s0:s0 + L] = sparse_todense(f, k + "/nucleotides")[:L].astype(np.int8)
                for w in fib:
                    v = sparse_todense(f, "%s/Fiber_count_%s" % (k, w))[:L].astype(np.int32)
                    old = fib[w][s0:s0 + L]
                    ov = cnt[s0:s0 + L] > 1
                    fib_conflict += int((old[ov] != v[ov]).sum())
                    fib[w][s0:s0 + L] = v
                del dp
    missing = set(sel.index) - seen
    assert not missing, "segments absent from tmpDir: %s" % sorted(missing)[:10]
    print("  read %d segments; fiber-count disagreements in overlaps: %d" % (len(seen), fib_conflict))

    assert (cnt > 0).all(), "uncovered positions: %d" % int((cnt == 0).sum())
    print("  coverage counts: %s" % dict(zip(*[x.tolist() for x in np.unique(cnt, return_counts=True)])))
    acc /= cnt[:, None]
    v32 = acc.astype(np.float32)
    ok = (np.isfinite(v32).all(axis=1) & (v32 >= -VALID_TOL).all(axis=1)
          & (v32 <= 1.0 + VALID_TOL).all(axis=1))
    nbad = int((~ok).sum())
    print("  INVALID positions dropped: %d of %d (%.4f%%)" % (nbad, n, 100.0 * nbad / n))

    pos = np.arange(lo, hi + 1, dtype=np.int64)
    ai = cols.index("Abf1_murphy")
    np.savez_compressed(
        os.path.join(a.out, "%s_%s.npz" % (a.tag, a.chrom)),
        pos=pos, abf1=v32[:, ai], ok=ok, nt=nt, cols=np.array(cols),
        meth_w=fib["meth_watson"], meth_c=fib["meth_crick"],
        A_w=fib["A_watson"], A_c=fib["A_crick"], n_invalid=nbad,
        traindir=info["traindir"], pkg_tree=info["pkg_tree"])
    np.save(os.path.join(a.out, "%s_%s_full.npy" % (a.tag, a.chrom)), v32)
    print("  ABF1 column: max %.4f  sum %.1f" % (v32[ok, ai].max(), v32[ok, ai].sum()))


if __name__ == "__main__":
    main()
