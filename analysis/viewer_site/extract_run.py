#!/usr/bin/env python
"""Per-run, per-chromosome viewer data from the decode's factor tables (write_factor_table.py).

    python viewer_site/extract_run.py <run_id> <chrom> [--out DIR] [--force]
    python viewer_site/extract_run.py --all [--out DIR]            # every run x chrom in runs.json

Reads <decode dir>/factor_tables/part_*.npz (no HMM collapse here), stitches the segments that
cover each windows.tsv window on <chrom>, averaging overlaps equally (as
score_robocop.region_optable does), and writes <out>/runs/<run_id>/<chrom>.bin.gz:

    b"RVB1" | uint32 LE header length | header JSON | uint8 payload
    header: run_id, chrom, dir, q (=100), cols, blocks [{start, len, off}], fiber {"s-e": md5},
            n_invalid, written
    payload: for each block, for each column, `len` bytes of round(p * q), clipped to 0..q.

Columns: nucleosome, background, nuc_center, unknown (if live), every LIVE motif of the run
(its KEEP set; for an unmasked run every motif whose max in the extracted blocks >= 1e-3), and
`other TFs` = the sum of the remaining motif columns (clipped at 1). Coordinates are 1-based
whole-chromosome positions, so blocks for more windows / whole chromosomes slot in later.

Incremental: skipped when the output is newer than every part file it reads (--force rebuilds).
"""
import argparse
import glob
import gzip
import json
import os
import struct
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, A)
OUT_DEFAULT = "/usr/project/xtmp/nd141/viewer_site"
Q = 100
ALWAYS = ["nucleosome", "background", "nuc_center"]
NON_MOTIF = {"background", "unknown", "nuc_padding", "nucleosome", "nuc_center", "nuc_start", "nuc_end"}


def load_parts(d, chrom):
    """-> list of (seg_start, seg_end, float32 array rows x cols, cols, fiber_md5, n_invalid), newest mtime."""
    segs, cols, newest = {}, None, 0.0
    for p in sorted(glob.glob(os.path.join(A, d, "factor_tables", "part_*.npz"))):
        newest = max(newest, os.path.getmtime(p))
        z = np.load(p, allow_pickle=False)
        c = [str(x) for x in z["cols"]]
        if cols is None:
            cols = c
        elif c != cols:
            raise RuntimeError("%s: column set differs between part files" % d)
        scale = json.loads(str(z["meta"])).get("scale", 65535)
        ro = z["row_off"]
        q = None
        for k in range(len(z["seg_idx"])):
            if str(z["seg_chr"][k]) != chrom:
                continue
            if q is None:
                q = z["q"]
            idx = int(z["seg_idx"][k])
            segs[idx] = (int(z["seg_start"][k]), int(z["seg_end"][k]),
                         q[ro[k]:ro[k + 1]].astype(np.float32) / scale,
                         str(z["fiber_md5"][k]), int(z["n_invalid"][k]))
    return [segs[k] for k in sorted(segs)], cols, newest


def stitch(segs, ncol, ws, we):
    """Average overlapping segments over [ws, we] -> (values n x ncol, covered bool n)."""
    n = we - ws + 1
    acc = np.zeros((n, ncol), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int32)
    for s, e, v, _, _ in segs:
        if e < ws or s > we:
            continue
        a, b = max(s, ws), min(e, we, s + v.shape[0] - 1)
        acc[a - ws:b - ws + 1] += v[a - s:b - s + 1]
        cnt[a - ws:b - ws + 1] += 1
    cov = cnt > 0
    acc[cov] /= cnt[cov, None]
    return acc, cov


def spans(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > 1)
    st = np.concatenate(([idx[0]], idx[br + 1]))
    en = np.concatenate((idx[br], [idx[-1]]))
    return list(zip(st, en))


def live_keep(run):
    """KEEP set of the run's tree (None = unmasked) via build_run_matrix.tree_attrs."""
    import build_run_matrix as B
    return B.tree_attrs(run.get("pkgvar") if run.get("pkgvar") != "?" else None).get("keep")


def extract(run, chrom, out, force=False, windows=None):
    d = run["dirs"].get(chrom)
    dst = os.path.join(out, "runs", run["run_id"], chrom + ".bin.gz")
    if not d:
        return "no decode on %s" % chrom
    segs, cols, newest = load_parts(d, chrom)
    if not segs:
        return "no factor-table segments on %s (backfill pending?)" % chrom
    if os.path.exists(dst) and not force and os.path.getmtime(dst) > newest:
        return "up to date"
    windows = windows if windows is not None else pd.read_csv(os.path.join(HERE, "windows.tsv"), sep="\t")
    w = windows[windows["chrom"] == chrom]
    blocks_raw = []
    for _, r in w.iterrows():
        vals, cov = stitch(segs, len(cols), int(r["start"]), int(r["end"]))
        for a, b in spans(cov):
            blocks_raw.append((int(r["start"]) + a, vals[a:b + 1]))
    if not blocks_raw:
        return "no segments overlap the %s window" % chrom
    keep = live_keep(run)
    motifs = [c for c in cols if c not in NON_MOTIF]
    if keep is not None:
        live = [m for m in motifs if m in keep]
        unk = "unknown" in keep
    else:
        mx = np.max(np.vstack([v for _, v in blocks_raw]), axis=0)
        live = [m for m in motifs if mx[cols.index(m)] >= 1e-3]
        unk = "unknown" in cols
    keep_cols = ALWAYS + (["unknown"] if unk and "unknown" in cols else []) + live
    rest = [cols.index(m) for m in motifs if m not in live]
    ci = [cols.index(c) for c in keep_cols]
    out_cols = keep_cols + (["other TFs"] if rest else [])
    payload, blocks = [], []
    off = 0
    for start, v in blocks_raw:
        m = v[:, ci]
        if rest:
            m = np.hstack([m, np.minimum(1.0, v[:, rest].sum(axis=1, keepdims=True))])
        qv = np.clip(np.round(m * Q), 0, Q).astype(np.uint8)
        payload.append(np.ascontiguousarray(qv.T).tobytes())
        blocks.append(dict(start=int(start), len=int(v.shape[0]), off=off))
        off += qv.size
    fiber = {"%d-%d" % (s, e): md5 for s, e, _, md5, _ in segs
             if any(not (e < b["start"] or s > b["start"] + b["len"] - 1) for b in blocks)}
    hdr = dict(run_id=run["run_id"], chrom=chrom, dir=d, q=Q, cols=out_cols, blocks=blocks, fiber=fiber,
               n_invalid=int(sum(x[4] for x in segs)), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp%d" % os.getpid()
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(b"RVB1" + struct.pack("<I", len(hb)) + hb + b"".join(payload))
    os.replace(tmp, dst)
    return "wrote %d blocks, %d cols" % (len(blocks), len(out_cols))


def read_bin(path):
    """Inverse of extract(): -> (header, {col: [(start, float array)]})."""
    raw = gzip.open(path, "rb").read()
    assert raw[:4] == b"RVB1"
    n = struct.unpack("<I", raw[4:8])[0]
    hdr = json.loads(raw[8:8 + n])
    body = np.frombuffer(raw[8 + n:], dtype=np.uint8)
    out = {c: [] for c in hdr["cols"]}
    for b in hdr["blocks"]:
        m = body[b["off"]:b["off"] + b["len"] * len(hdr["cols"])].reshape(len(hdr["cols"]), b["len"])
        for k, c in enumerate(hdr["cols"]):
            out[c].append((b["start"], m[k].astype(np.float32) / hdr["q"]))
    return hdr, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("chrom", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    reg = json.load(open(os.path.join(a.out, "runs.json")))
    runs = {r["run_id"]: r for r in reg["runs"]}
    wins = pd.read_csv(os.path.join(HERE, "windows.tsv"), sep="\t")
    todo = []
    if a.all:
        for r in reg["runs"]:
            for c in wins["chrom"]:
                if c in r["dirs"]:
                    todo.append((r["run_id"], c))
    else:
        todo = [(a.run_id, a.chrom)]
    stat = {}
    for rid, c in todo:
        try:
            msg = extract(runs[rid], c, a.out, a.force, wins)
        except Exception as e:
            msg = "ERROR %r" % e
        stat[msg.split(" ")[0]] = stat.get(msg.split(" ")[0], 0) + 1
        if not a.all or msg.startswith("ERROR"):
            print("%s %s: %s" % (rid, c, msg))
    print("extract_run: %d run x chrom: %s" % (len(todo), stat))


if __name__ == "__main__":
    main()
