#!/usr/bin/env python
"""Fiber-seq sameness check -> <out>/fiber_check.json (shown as a banner + a run-matrix column).

    python viewer_site/fiber_check.py [--out DIR] [--corrupt RUN_ID:CHROM]

For every runs/<run>/<chrom>.bin.gz, each segment hash in its header (md5 of that decode's four
Fiber_count_* arrays over the segment, written by write_factor_table.py) is recomputed from the
shared reference decode's counts (scratch_viewer/fiber_ref/<chrom>.npz, extract_shared.py) over the
same coordinates, and compared. Segments the reference does not fully cover are "not compared".
--corrupt flips one hash in memory to prove the banner fires (the files are not touched).
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_run import OUT_DEFAULT, read_bin  # noqa: E402

SCRATCH = "/usr/project/xtmp/nd141/scratch_viewer"


def load_ref(chrom):
    p = os.path.join(SCRATCH, "fiber_ref", chrom + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return [(int(z["spans"][i][0]), z["cnt_%d" % i], z["cov_%d" % i]) for i in range(len(z["spans"]))], str(z["ref_dir"])


def ref_md5(ref, s, e):
    for lo, cnt, cov in ref:
        hi = lo + cnt.shape[1] - 1
        if s >= lo and e <= hi and cov[s - lo:e - lo + 1].all():
            h = hashlib.md5()
            for j in (2, 3, 0, 1):             # write_factor_table order: A_w, A_c, meth_w, meth_c
                h.update(cnt[j, s - lo:e - lo + 1].astype(np.float64).tobytes())
            return h.hexdigest()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--corrupt", default=None)
    a = ap.parse_args()
    refs, res = {}, {}
    tot = dict(match=0, mismatch=0, not_compared=0)
    for p in sorted(glob.glob(os.path.join(a.out, "runs", "*", "*.bin.gz"))):
        run, chrom = os.path.basename(os.path.dirname(p)), os.path.basename(p)[:-len(".bin.gz")]
        if chrom not in refs:
            refs[chrom] = load_ref(chrom)
        hdr, _ = read_bin(p)
        fib = dict(hdr["fiber"])
        if a.corrupt == "%s:%s" % (run, chrom) and fib:
            k0 = sorted(fib)[0]
            fib[k0] = "0" * 32
        r = dict(match=0, mismatch=0, not_compared=0, mismatches=[])
        for span, md5 in sorted(fib.items()):
            s, e = map(int, span.split("-"))
            want = ref_md5(refs[chrom][0], s, e) if refs[chrom] and md5 else None
            if want is None:
                r["not_compared"] += 1
            elif want == md5:
                r["match"] += 1
            else:
                r["mismatch"] += 1
                r["mismatches"].append(span)
        for k in tot:
            tot[k] += r[k]
        res.setdefault(run, {})[chrom] = r
    bad = sorted("%s:%s" % (run, c) for run, cs in res.items() for c, r in cs.items() if r["mismatch"])
    out = dict(built=time.strftime("%Y-%m-%d %H:%M:%S"),
               reference={c: (v[1] if v else None) for c, v in refs.items()},
               totals=tot, differing=bad, runs=res, corrupted_for_test=a.corrupt)
    with open(os.path.join(a.out, "fiber_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("fiber check: %d run x chrom files; segments %s; differing: %s"
          % (sum(len(v) for v in res.values()), tot, bad or "none"))


if __name__ == "__main__":
    main()
