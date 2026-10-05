#!/usr/bin/env python
"""Write a decode's per-factor posterior table ("optable") for one array task's info file.

For each segment in <outdir>/tmpDir/info_<idx>_<total>.h5, collapse the sparse state posterior to
per-factor columns with get_posterior_binding_probability_df from the DRIVER'S OWN package tree
(widened trees sum different state slices) and the REAL trainDir's HMMconfig.pkl (config.ini's
trainDir is not reliable). Result: <outdir>/factor_tables/part_<idx>_<total>.npz

    q          uint16 (rows, ncols)   round(clip(p,0,1) * 65535), segments concatenated
    cols       str (ncols,)           every optable column (all motifs, background, unknown, nuc_*)
    seg_idx    int  (nseg,)           coords.tsv row index of each segment
    seg_chr/seg_start/seg_end         coords of each segment (1-based inclusive, as coords.tsv)
    row_off    int  (nseg+1,)         q[row_off[k]:row_off[k+1]] is segment k; row r <-> seg_start+r
    n_invalid  int  (nseg,)           positions with any column outside [0, 1+1e-6] before clipping
                                      (chrXII rDNA emits 1e171 posteriors; see CLAUDE.md)
    fiber_md5  str  (nseg,)           md5 of the 4 Fiber_count_* arrays over the posterior rows, for the
                                      viewer's Fiber-seq sameness check; "" when absent
    meta       str                    JSON: tree, traindir, driver, scale, written

Stitching segments and averaging overlaps equally reproduces score_robocop.region_optable, since the
collapse (sum_for_dbf_probs) is a sum of state-column slices and so linear in the posterior.

    python write_factor_table.py --outdir D --driver X --traindir T --idx i --total N \
        [--segments all|windows|3,17,40] [--windows viewer_site/windows.tsv] [--force]

Segments already in an existing part file are skipped (the file is extended with any new ones),
so a window-only backfill and a later --segments all coexist in one file. --force recomputes.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
import pickle
import sys
import time

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_robocop as S  # noqa: E402
from write_run_info import driver_tree  # noqa: E402

SCALE = 65535
FIBER_KEYS = ("count_A_watson", "count_A_crick", "count_meth_watson", "count_meth_crick")


def window_segments(coords, windows_tsv):
    w = pd.read_csv(windows_tsv, sep="\t")
    keep = set()
    for _, r in w.iterrows():
        c = coords[(coords["chr"] == r["chrom"]) & (coords["start"] <= r["end"]) & (coords["end"] >= r["start"])]
        keep.update(int(i) for i in c.index)
    return keep


def fiber_md5(g, tech2, n):
    """md5 over the first n (= posterior rows) values of the 4 Fiber_count_* arrays, as float64.
    The stored arrays carry one trailing element beyond the posterior; row r <-> seg_start + r, as
    score_robocop.region_optable slices them. Hashing exactly the posterior span lets the viewer
    recompute the same hash from any other decode's counts over the same coordinates."""
    if not tech2:
        return ""
    h = hashlib.md5()
    for k in FIBER_KEYS:
        key = "%s_%s" % (tech2, k)
        if key not in g:
            return ""
        h.update(np.asarray(S._get_sparse_todense(g, key), dtype=np.float64)[:n].tobytes())
    return h.hexdigest()


def load_part(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--traindir", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--variant", default=None, help="argv[6] of a variant driver, if one was passed")
    ap.add_argument("--tree", default=None, help="override the tree parsed from the driver")
    ap.add_argument("--segments", default="all")
    ap.add_argument("--windows", default=os.path.join(HERE, "viewer_site", "windows.tsv"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()

    od = (a.outdir if os.path.isabs(a.outdir) else os.path.join(HERE, a.outdir)).rstrip("/")
    info = os.path.join(od, "tmpDir", "info_%d_%d.h5" % (a.idx, a.total))
    if not os.path.isfile(info):
        print("write_factor_table: missing %s" % info, file=sys.stderr)
        return 2
    tree = a.tree
    if tree is None:
        tree, how = driver_tree(a.driver, a.variant)
        if tree is None:
            print("write_factor_table: cannot resolve tree from %s (%s)" % (a.driver, how), file=sys.stderr)
            return 2
    td = a.traindir if os.path.isabs(a.traindir) else os.path.join(HERE, a.traindir)
    S.use_pkg(os.path.join(HERE, tree) if not os.path.isabs(tree) else tree)
    dshared = pickle.load(open(os.path.join(td, "HMMconfig.pkl"), "rb"))
    dshared["info_file"] = None
    coords = pd.read_csv(os.path.join(od, "coords.tsv"), sep="\t")
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(od, "config.ini"))
    tech2 = cfg.get("main", "tech2", fallback=None)

    try:
        f = h5py.File(info, "r")
        present = sorted(int(k.split("_")[1]) for k in f.keys() if k.startswith("segment_"))
    except Exception as e:
        print("write_factor_table: unreadable %s: %r" % (info, e), file=sys.stderr)
        return 3
    if a.segments == "all":
        want = set(present)
    elif a.segments == "windows":
        want = window_segments(coords, a.windows) & set(present)
    else:
        want = {int(x) for x in a.segments.split(",")} & set(present)

    outd = os.path.join(od, "factor_tables")
    out = os.path.join(outd, "part_%d_%d.npz" % (a.idx, a.total))
    old = None
    if os.path.exists(out) and not a.force:
        old = load_part(out)
        have = set(int(x) for x in old["seg_idx"])
        want -= have
    if not want:
        print("write_factor_table: nothing to do for %s (all requested segments present)" % out)
        return 0

    blocks, recs, cols = [], [], None
    tc = 0.0
    for idx in sorted(want):
        g = f["segment_%d" % idx]
        dp = S._get_sparse_todense(g, "posterior")
        if dp.ndim == 1:
            dp = dp[np.newaxis, :]
        t1 = time.time()
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            o = S.get_posterior_binding_probability_df(dshared, dp)
        tc += time.time() - t1
        c = list(o.columns)
        if cols is None:
            cols = c
        elif c != cols:
            raise RuntimeError("column set changed between segments")
        v = o.values
        bad = ~np.all((v >= 0) & (v <= 1 + 1e-6), axis=1)
        blocks.append(np.round(np.clip(np.nan_to_num(v, nan=0.0), 0, 1) * SCALE).astype(np.uint16))
        r = coords.loc[idx]
        recs.append((idx, r["chr"], int(r["start"]), int(r["end"]), int(bad.sum()), fiber_md5(g, tech2, v.shape[0])))
    f.close()

    seg_idx = [x[0] for x in recs]
    seg_chr = [x[1] for x in recs]
    seg_start = [x[2] for x in recs]
    seg_end = [x[3] for x in recs]
    n_invalid = [x[4] for x in recs]
    fmd5 = [x[5] for x in recs]
    if old is not None:
        if list(old["cols"]) != cols:
            raise RuntimeError("existing %s has different columns; use --force" % out)
        ro = old["row_off"]
        oblocks = [old["q"][ro[k]:ro[k + 1]] for k in range(len(old["seg_idx"]))]
        blocks = oblocks + blocks
        seg_idx = list(old["seg_idx"]) + seg_idx
        seg_chr = list(old["seg_chr"]) + seg_chr
        seg_start = list(old["seg_start"]) + seg_start
        seg_end = list(old["seg_end"]) + seg_end
        n_invalid = list(old["n_invalid"]) + n_invalid
        fmd5 = list(old["fiber_md5"]) + fmd5
    order = np.argsort(seg_idx, kind="stable")
    blocks = [blocks[k] for k in order]
    lens = [b.shape[0] for b in blocks]
    row_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    meta = dict(tree=tree, traindir=os.path.relpath(td, HERE), driver=os.path.basename(a.driver),
                scale=SCALE, written=datetime.datetime.now().isoformat(timespec="seconds"),
                info_file=os.path.basename(info))
    os.makedirs(outd, exist_ok=True)
    tmp = out + ".tmp%d.npz" % os.getpid()
    np.savez_compressed(
        tmp, q=np.concatenate(blocks, axis=0), cols=np.array(cols),
        seg_idx=np.array(seg_idx, dtype=np.int64)[order], seg_chr=np.array(seg_chr)[order],
        seg_start=np.array(seg_start, dtype=np.int64)[order], seg_end=np.array(seg_end, dtype=np.int64)[order],
        row_off=row_off, n_invalid=np.array(n_invalid, dtype=np.int64)[order],
        fiber_md5=np.array(fmd5)[order], meta=np.array(json.dumps(meta)))
    os.replace(tmp, out)
    print("write_factor_table: wrote %s: %d new segments (%d total), %d cols; collapse %.1fs, total %.1fs"
          % (out, len(want), len(seg_idx), len(cols), tc, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
