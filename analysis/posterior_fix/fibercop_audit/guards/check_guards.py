"""Do the posterior '> 1' guards still fire? Measure each guard's trigger on a set of decode dirs.

    cd analysis && python posterior_fix/fibercop_audit/guards/check_guards.py <out.tsv> <decode dir> [...]

Per decode dir (read-only):
  rows               posterior rows stored
  rowsum_max         largest emitting row sum (sum over [:silent_states_begin])
  rows_gt1           rows with emitting sum > 1 + 1e-4 (count_calls' VALID_TOL)
  rows_lt099         rows with emitting sum < 0.99 (beyond the save-time floor's ~0.995)
  silent_max         largest silent-state cell
  guard_possible     rows where ANY collapsed factor column could exceed 1 + 1e-4. Every factor column is a sum over
                     a subset of states (emitting, plus nuc_end = the first silent column), so a column can exceed the
                     bound only if rowsum or a silent cell does. 0 here proves that write_factor_table's clip/n_invalid,
                     count_calls' exclusion, abf1_stack_extract's mask and build_tuned_viewer's mask cannot fire.
  normalise_maxshift largest |rowsum - 1| on rows with mass: how far score_robocop.region_optable's normalisation moves a row
  ft_n_invalid       n_invalid recorded in factor_tables/part_*.npz by write_factor_table.py (positions clipped)
"""
import sys, os, glob, json, pickle
import numpy as np, pandas as pd, h5py

TOL = 1e-4


def ssb_of(d):
    ri = json.load(open(os.path.join(d, "RUN_INFO.json")))
    return int(pickle.load(open(os.path.join(ri["traindir"], "HMMconfig.pkl"), "rb"))["silent_states_begin"]), ri


rows = []
for d in [x.rstrip("/") for x in sys.argv[2:]]:
    ssb, ri = ssb_of(d)
    n = gt1 = lt099 = poss = 0
    rmax, smax, shift = 0.0, 0.0, 0.0
    for fn in glob.glob(os.path.join(d, "tmpDir", "info*.h5")):
        with h5py.File(fn, "r") as f:
            for k in f.keys():
                g = f[k + "/posterior"]; data = g["data"][:]; ind = g["indices"][:]; ptr = g["indptr"][:]
                m = len(ptr) - 1; n += m
                rs = np.add.reduceat(np.concatenate([np.where(ind < ssb, data, 0.0), [0.0]]), ptr[:-1])
                rs[ptr[:-1] == ptr[1:]] = 0.0
                sil = np.zeros(m)
                rowof = np.repeat(np.arange(m), np.diff(ptr))
                sel = ind >= ssb
                if sel.any():
                    np.maximum.at(sil, rowof[sel], data[sel])
                gt1 += int((rs > 1 + TOL).sum()); lt099 += int(((rs < 0.99) & (rs > 0)).sum())
                poss += int(((rs > 1 + TOL) | (sil > 1 + TOL)).sum())
                rmax = max(rmax, float(rs.max())); smax = max(smax, float(sil.max()))
                if (rs > 0).any():
                    shift = max(shift, float(np.abs(rs[rs > 0] - 1).max()))
    ftinv = 0
    for p in glob.glob(os.path.join(d, "factor_tables", "part_*.npz")):
        ftinv += int(np.load(p, allow_pickle=False)["n_invalid"].sum())
    rows.append(dict(dir=d, so_md5=str(ri.get("so_md5")), log_emission=ri.get("log_emission"), rows=n, rowsum_max=rmax,
                     rows_gt1=gt1, rows_lt099=lt099, silent_max=smax, guard_possible=poss,
                     normalise_maxshift=shift, ft_n_invalid=ftinv))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], sep="\t", index=False)
print("wrote", sys.argv[1])
