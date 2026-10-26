"""Scan every stored segment of a set of decodes for posterior rows that do not sum to 1.

    cd analysis && python posterior_fix/fibercop_audit/extent/scan_rowsums.py <out.tsv> <decode dir> [...]

Emitting row sum = sum of posterior[:, :silent_states_begin] (POSTERIOR_SCALING.md section 1); ssb is
read from the run's REAL trainDir (RUN_INFO.json), not config.ini. A row is "bad" when
|rowsum - 1| > 0.01; separately, rows with any SILENT cell > 1 are counted (defect 4) (the save-time floor alone keeps sums within ~[0.995, 1]). Read-only.
One output row per segment.
"""
import sys, os, json, glob, pickle
import numpy as np, h5py
from multiprocessing import Pool

def ssb_of(d):
    ri = json.load(open(os.path.join(d, "RUN_INFO.json")))
    return int(pickle.load(open(os.path.join(ri["traindir"], "HMMconfig.pkl"), "rb"))["silent_states_begin"]), ri

def scan(d):
    ssb, ri = ssb_of(d)
    rows = []
    for fn in sorted(glob.glob(os.path.join(d, "tmpDir", "info*.h5"))):
        with h5py.File(fn, "r") as f:
            for k in f.keys():
                if not k.startswith("segment_"):
                    continue
                g = f[k]; a = g.attrs
                pg = g["posterior"]; data = pg["data"][:]; ind = pg["indices"][:]; ptr = pg["indptr"][:]
                n = len(ptr) - 1
                emit = np.where(ind < ssb, data, 0.0)
                rs = np.add.reduceat(np.concatenate([emit, [0.0]]), ptr[:-1]) if len(data) else np.zeros(n)
                rs[ptr[:-1] == ptr[1:]] = 0.0          # empty rows
                sil = np.where(ind >= ssb, data, 0.0)
                silmax = float(sil.max()) if len(sil) else 0.0
                rowof = np.repeat(np.arange(n), np.diff(ptr))
                silbad = np.zeros(n, bool); silbad[rowof[sil > 1.0 + 1e-6]] = True
                bad = np.abs(rs - 1) > 0.01
                nb = int(bad.sum())
                lr = np.log10(np.maximum(rs, 1e-300))
                steps = np.flatnonzero(np.abs(np.diff(lr)) > 0.3)
                idx = np.flatnonzero(bad)
                prefix = bool(nb and idx[0] == 0 and idx[-1] == nb - 1)
                rows.append(dict(dir=d, campaign=ri.get("campaign"), round=ri.get("round"), role=ri.get("role"),
                                 segment=int(a["segment"]), chrom=str(a["chr"]), start=int(a["start"]), end=int(a["end"]),
                                 n_rows=n, n_bad=nb, n_over=int((rs > 1.01).sum()), n_under=int((rs < 0.99).sum()),
                                 n_zero=int((rs == 0).sum()), rs_max=float(rs.max()), rs_min=float(rs.min()),
                                 first_bad=int(idx[0]) if nb else -1, last_bad=int(idx[-1]) if nb else -1,
                                 prefix=prefix, n_steps=len(steps), steps=";".join(str(int(s)) for s in steps[:20]),
                                 silent_max=silmax, n_silent_gt1=int(silbad.sum()),
                                 n_silent_gt1_rs_ok=int((silbad & ~bad).sum()),
                                 silent_rows=";".join(str(int(i)) for i in np.flatnonzero(silbad)[:30])))
    return rows

if __name__ == "__main__":
    out, dirs = sys.argv[1], [d.rstrip("/") for d in sys.argv[2:]]
    with Pool(min(16, len(dirs))) as p:
        res = p.map(scan, dirs)
    import pandas as pd
    df = pd.DataFrame([r for rr in res for r in rr]).sort_values(["dir", "segment"])
    df.to_csv(out, sep="\t", index=False)
    print("wrote", out, len(df), "segments from", len(dirs), "decodes")
