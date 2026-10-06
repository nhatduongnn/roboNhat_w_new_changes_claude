"""Count linear-recursion underflow positions over every segment of a coords file (Stage 2 pred. 5).

    cd analysis && python posterior_fix/underflow_scan.py <tree> <coords> <outDir> <out.json> [--traindir T]

Runs the REAL pipeline of a stock (linear) pkgvar tree (as harness_capture.py does), and for each
segment, in place of the tree's recursion, runs linear fward/bward + posterior_decoding_v2
(librobocop.so) and fward_log/bward_log/posterior_decoding_log (librobocop_logfull.so) on the
log of the SAME floored emission (= arm D0's input). Per segment it records the positions where the
linear scaling factor (pre-normalisation emitting sum, incl. ssb*DBL_MIN) is tiny, the rows where
forward / posterior differ, and the B silent-column > 1 rows. Writes JSON only; the info h5 in
<outDir> is throwaway. Read-only on all inputs and both .so files.
"""
import sys, os, json, argparse
import numpy as np
from ctypes import CDLL
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import harness_gates as HG

ap = argparse.ArgumentParser()
ap.add_argument('tree'); ap.add_argument('coords'); ap.add_argument('outdir'); ap.add_argument('json')
ap.add_argument('--traindir', default='robocop_train_tw_bv75_49')
a = ap.parse_args()
sys.path.insert(0, a.tree)
import robocop
import run_robocop
assert os.path.abspath(os.path.dirname(robocop.__file__)).startswith(os.path.abspath(a.tree))

DBL_MIN = np.finfo(np.double).tiny
prod = HG.bind(CDLL(HG.SO_PROD))
logl = HG.bind_log(HG.bind(CDLL(HG.SO_LOG)))
md5_0 = HG.md5(HG.SO_PROD)
OUT = dict(tree=a.tree, coords=a.coords, segments={})


def lse_rows(x):
    m = x.max(1, keepdims=True)
    mm = np.where(np.isfinite(m), m, 0.0)
    return (mm + np.log(np.exp(x - mm).sum(1, keepdims=True))).ravel()


def runs(idx):
    """contiguous runs of an index array -> [[start, end], ...]"""
    idx = np.asarray(idx)
    if idx.size == 0: return []
    br = np.flatnonzero(np.diff(idx) > 1)
    st = np.r_[idx[0], idx[br + 1]]; en = np.r_[idx[br], idx[-1]]
    return [[int(s), int(e)] for s, e in zip(st, en)]


def scan(d, segment, dshared):
    k = 'segment_' + str(segment)
    n = int(dshared['info_file'][k].attrs['n_obs'])
    c = dict(n_states=int(dshared['n_states']), ssb=int(dshared['silent_states_begin']), n_obs=n,
             n_vars=int(dshared['n_vars']), n_tfs=int(dshared['n_tfs']), nuc_present=int(dshared['nuc_present']),
             nuc_start=int(dshared['nuc_start']), nuc_len=int(dshared['nuc_len']),
             init=np.ascontiguousarray(dshared['initial_probs'], dtype=np.double),
             end=np.ascontiguousarray(dshared['end_probs'], dtype=np.double),
             T=np.ascontiguousarray(dshared['transition_matrix'], dtype=np.double),
             ms=np.ascontiguousarray(dshared['tf_starts'], dtype=np.longlong),
             ml=np.ascontiguousarray(dshared['tf_lens'], dtype=np.longlong),
             E=np.ascontiguousarray(d['emission'], dtype=np.double))
    ns, ssb = c['n_states'], c['ssb']
    B = HG.run_linear(prod, c)
    with np.errstate(divide='ignore'):
        LE = np.ascontiguousarray(np.log(c['E']).sum(axis=0))
    del c['E']; d['emission'] = None
    par, chi, npar, nchi = B['graph']
    lf = np.full((n, ns), -np.inf); lb = np.full((n, ns), -np.inf)
    lfs = np.zeros(n); lbs = np.zeros(n); plog = np.zeros((n, ns))
    assert logl.fward_log(c['init'], c['T'], LE, c['end'], ns, ssb, n, par, npar, lf, lfs, HG.ABI) == HG.ABI
    assert logl.bward_log(c['T'], LE, c['end'], ns, ssb, n, chi, nchi, lb, lbs, HG.ABI) == HG.ABI
    assert logl.posterior_decoding_log(lf, lb, ns, n, ssb, plog, HG.ABI) == HG.ABI
    fl = B['f'][:, :ssb]; fl = fl / fl.sum(1, keepdims=True)
    fg = np.exp(lf[:, :ssb] - lse_rows(lf[:, :ssb])[:, None])
    dF = np.abs(fl - fg).max(1)
    dP = np.abs(B['p'][:, :ssb] - plog[:, :ssb]).max(1)
    # linear pre-normalisation emitting sum WITHOUT the DBL_MIN terms
    pre_f = B['fs'] - ssb * DBL_MIN
    pre_b = B['bs'] - ssb * DBL_MIN
    sil = B['p'][:, ssb:].max(1)
    R = dict(n_obs=n, chr=str(dshared['info_file'][k].attrs['chr']), start=int(dshared['info_file'][k].attrs['start']),
             loglik_linear=float(B['ll']), loglik_log=float(lfs.sum()), log_minus_linear=float(lfs.sum() - B['ll']),
             # underflow flags: exp(offset) below the smallest normal double => the linear row cannot hold it
             n_lfs_lt_log_DBL_MIN=int((lfs < np.log(DBL_MIN)).sum()),
             n_lbs_lt_log_DBL_MIN=int((lbs < np.log(DBL_MIN)).sum()),
             n_lfs_lt_m690=int((lfs < -690).sum()), n_lbs_lt_m690=int((lbs < -690).sum()),
             min_lfs=float(lfs.min()), min_lbs=float(lbs.min()),
             n_fs_le_2x_floor=int((B['fs'] <= 2 * ssb * DBL_MIN).sum()),
             n_bs_le_2x_floor=int((B['bs'] <= 2 * ssb * DBL_MIN).sum()),
             fwd_under_rows=runs(np.flatnonzero(lfs < np.log(DBL_MIN))),
             bwd_under_rows=runs(np.flatnonzero(lbs < np.log(DBL_MIN))),
             n_rows_dF_gt_1em6=int((dF > 1e-6).sum()), first_dF_gt_1em6=int(np.argmax(dF > 1e-6)) if (dF > 1e-6).any() else None,
             n_rows_dP_gt_1em6=int((dP > 1e-6).sum()), n_rows_dP_gt_0p01=int((dP > 0.01).sum()), max_dP=float(dP.max()),
             dP_gt_0p01_runs=runs(np.flatnonzero(dP > 0.01)),
             B_silent_gt1_rows=[int(i) for i in np.flatnonzero(sil > 1)],
             B_silent_gt1_max=float(sil.max()),
             B_silent_gt1_detail=[dict(row=int(i), sil=float(sil[i]), lfs=float(lfs[i]), lbs=float(lbs[i]),
                                       lfs_prev=float(lfs[i - 1]) if i else None,
                                       lbs_next=float(lbs[i + 1]) if i + 1 < n else None)
                                  for i in np.flatnonzero(sil > 1)])
    OUT['segments'][str(segment)] = R
    print("SCAN", segment, json.dumps({kk: v for kk, v in R.items() if not kk.endswith('detail')}, default=float))
    sys.stdout.flush()
    dshared['info_file'][k].attrs['log_likelihood'] = np.nan
    json.dump(OUT, open(a.json, 'w'), indent=1, default=float)


robocop.posterior_forward_backward = scan
run_robocop.run_robocop_without_em(a.coords, a.traindir, a.outdir, idx=0, total=1)
assert HG.md5(HG.SO_PROD) == md5_0
json.dump(OUT, open(a.json, 'w'), indent=1, default=float)
print("done")
