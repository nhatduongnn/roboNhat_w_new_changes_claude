"""Phase A harness, step 2: gates A1 and A2 (POSTERIOR_FIX_PLAN.md "Phase A verification"), plus an
in-process preview of A3 and a check that the logfull tree's FLOORED emission is the log of the stock
tree's linear emission.

    cd analysis && python posterior_fix/harness_gates.py <cap_stock.npz> [<cap_logD0.npz> <logD0_info.h5>] [--json out.json]

A1  fward / bward / calc_sr / posterior_decoding_v2 from librobocop_logfull.so vs librobocop.so on the SAME
    captured linear inputs -> must be bit-identical (proves -O2 -ffp-contract=off + the extra TU changed
    nothing in the old symbols).
A2  in-process, same emission matrix: linear path (fward/bward + posterior_decoding_normlin) vs log path
    (log of the same matrix, collapsed over channels -> fward_log/bward_log/posterior_decoding_log).
    Gate: max abs < 1e-9 over all cells, max rel < 1e-9 over cells >= 1e-4 (the save floor).
D0  (optional) the logfull tree in ROBOCOP_LOG_EMISSION=floored: its captured 3-D log emission vs
    log(stock linear emission), per channel; and its stored segment_0 log_likelihood / posterior vs the
    linear path computed here (A3 preview -- A3 proper is read from the D0 decodes).
Read-only on everything; loads both .so files by absolute path and never writes them.
"""
import sys, os, json, argparse, hashlib
from ctypes import CDLL, c_int, c_double
import numpy as np
from numpy.ctypeslib import ndpointer

PKG = '/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/pkg/robocop/'
SO_PROD = PKG + 'librobocop.so'
SO_LOG = PKG + 'librobocop_logfull.so'
ABI = 20261005
D = ndpointer(np.double, flags="C_CONTIGUOUS")
L = ndpointer(np.longlong, flags="C_CONTIGUOUS")
LD = ndpointer(np.longdouble, flags="C_CONTIGUOUS")


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def bind(lib):
    lib.find_parents_and_children.argtypes = [L, L, L, L, c_int, c_int, D]
    lib.fward.argtypes = [D, D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.bward.argtypes = [D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.calc_sr.argtypes = [D, D, c_int, LD]
    lib.calc_sr.restype = c_double
    lib.posterior_decoding_v2.argtypes = [D, D, D, LD, c_int, c_int, c_int, D]
    return lib


def bind_log(lib):
    lib.fward_log.argtypes = [D, D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]
    lib.fward_log.restype = c_int
    lib.bward_log.argtypes = [D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]
    lib.bward_log.restype = c_int
    lib.posterior_decoding_log.argtypes = [D, D, c_int, c_int, c_int, D, c_int]
    lib.posterior_decoding_log.restype = c_int
    lib.posterior_decoding_normlin.argtypes = [D, D, c_int, c_int, c_int, D, c_int]
    lib.posterior_decoding_normlin.restype = c_int
    lib.log_add.argtypes = [c_double, c_double]
    lib.log_add.restype = c_double
    return lib


def graph(lib, c):
    ns = c['n_states']
    par = np.zeros((ns, ns), dtype=np.longlong); chi = np.zeros((ns, ns), dtype=np.longlong)
    npar = np.zeros(ns, dtype=np.longlong); nchi = np.zeros(ns, dtype=np.longlong)
    lib.find_parents_and_children(par, chi, npar, nchi, ns, c['ssb'], c['T'])
    return par, chi, npar, nchi


def run_linear(lib, c):
    """Exactly the call sequence of robocop.py:887-980 (B tree)."""
    ns, n, ssb = c['n_states'], c['n_obs'], c['ssb']
    par, chi, npar, nchi = graph(lib, c)
    f = np.zeros((n, ns)); b = np.zeros((n, ns)); p = np.zeros((n, ns))
    fs = np.zeros(n); bs = np.zeros(n); sr = np.zeros(n).astype(np.longdouble)
    lib.fward(c['init'], c['T'], c['E'], c['end'], ns, ssb, n, c['n_vars'], par, npar,
              c['ms'], c['ml'], c['n_tfs'], c['nuc_present'], c['nuc_start'], c['nuc_len'], f, fs)
    lib.bward(c['T'], c['E'], c['end'], ns, ssb, n, c['n_vars'], chi, nchi,
              c['ms'], c['ml'], c['n_tfs'], c['nuc_present'], c['nuc_start'], c['nuc_len'], b, bs)
    ll = lib.calc_sr(fs, bs, n, sr)
    lib.posterior_decoding_v2(f, b, bs, sr, ns, n, ssb, p)
    return dict(f=f, b=b, fs=fs, bs=bs, sr=sr, ll=ll, p=p, graph=(par, chi, npar, nchi))


def bits_equal(x, y):
    x = np.ascontiguousarray(x); y = np.ascontiguousarray(y)
    if x.dtype == np.longdouble:          # 80-bit value in 16 bytes; compare values, NaN==NaN
        return bool(np.array_equal(x, y, equal_nan=True))
    return x.shape == y.shape and bool(np.array_equal(x.view(np.uint64), y.view(np.uint64)))


def nfirst_diff(x, y):
    d = np.flatnonzero(np.ascontiguousarray(x).ravel() != np.ascontiguousarray(y).ravel())
    return int(d.size)


def errs(a, ref, floor):
    ad = np.abs(a - ref)
    m = ref >= floor
    rel = (ad[m] / ref[m]).max() if m.any() else 0.0
    return float(ad.max()), float(rel), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cap_stock'); ap.add_argument('cap_logD0', nargs='?'); ap.add_argument('logD0_h5', nargs='?')
    ap.add_argument('--json')
    a = ap.parse_args()
    R = {}
    md5_before = md5(SO_PROD)
    R['md5'] = dict(librobocop_so=md5_before, librobocop_logfull_so=md5(SO_LOG))

    z = np.load(a.cap_stock)
    c = dict(n_states=int(z['n_states']), ssb=int(z['silent_states_begin']), n_obs=int(z['n_obs']),
             n_vars=int(z['n_vars']), n_tfs=int(z['n_tfs']), nuc_present=int(z['nuc_present']),
             nuc_start=int(z['nuc_start']), nuc_len=int(z['nuc_len']),
             init=np.ascontiguousarray(z['initial_probs'], dtype=np.double),
             end=np.ascontiguousarray(z['end_probs'], dtype=np.double),
             T=np.ascontiguousarray(z['transition_matrix'], dtype=np.double),
             ms=np.ascontiguousarray(z['tf_starts'], dtype=np.longlong),
             ml=np.ascontiguousarray(z['tf_lens'], dtype=np.longlong),
             E=np.ascontiguousarray(z['emission'], dtype=np.double))
    ns, n, ssb = c['n_states'], c['n_obs'], c['ssb']
    R['inputs'] = dict(npz=a.cap_stock, robocopC_in_pipeline=str(z['robocopC']), tree=str(z['tree']),
                       n_obs=n, n_states=ns, ssb=ssb, n_vars=c['n_vars'],
                       emission_min=float(c['E'].min()), emission_max=float(c['E'].max()),
                       n_emission_zero=int((c['E'] == 0).sum()),
                       n_emission_eq_1e30=int((c['E'] == 1e-30).sum()),
                       min_nonzero_transition=float(c['T'][c['T'] != 0].min()))
    T = c['T']
    R['inputs']['silent_self_loops'] = int(sum(T[j, j] != 0 for j in range(ssb, ns)))
    R['inputs']['silent_to_silent_edges'] = int((T[ssb:, ssb:] != 0).sum())
    R['inputs']['silent_to_higher_silent_edges'] = int(np.triu(T[ssb:, ssb:] != 0, 1).sum())
    print(json.dumps(R, indent=1)); sys.stdout.flush()

    # ------------------------------------------------------------------ A1
    prod = bind(CDLL(SO_PROD))
    logl = bind_log(bind(CDLL(SO_LOG)))
    out_p = run_linear(prod, c)
    out_l = run_linear(logl, c)
    a1 = {}
    for k in ['f', 'b', 'fs', 'bs', 'sr', 'p']:
        a1[k] = dict(bit_identical=bits_equal(out_p[k], out_l[k]), n_diff=nfirst_diff(out_p[k], out_l[k]))
    for i, nm in enumerate(['parents', 'children', 'n_parents', 'n_children']):
        a1[nm] = dict(bit_identical=bool(np.array_equal(out_p['graph'][i], out_l['graph'][i])))
    a1['loglik'] = dict(prod=out_p['ll'], logfull=out_l['ll'], bit_identical=bool(
        np.float64(out_p['ll']).view(np.uint64) == np.float64(out_l['ll']).view(np.uint64)))
    a1['PASS'] = all(v['bit_identical'] for v in a1.values() if isinstance(v, dict))
    R['A1'] = a1
    print("A1", json.dumps(a1, indent=1)); sys.stdout.flush()

    # ------------------------------------------------------------------ A2
    B = out_l   # linear tables from librobocop_logfull.so (bit-identical to prod if A1 passed)
    p_normlin = np.zeros((n, ns))
    rc = logl.posterior_decoding_normlin(B['f'], B['b'], ns, n, ssb, p_normlin, ABI)
    assert rc == ABI, rc
    with np.errstate(divide='ignore'):
        LE = np.ascontiguousarray(np.log(c['E']).sum(axis=0))
    par, chi, npar, nchi = B['graph']
    lf = np.full((n, ns), -np.inf); lb = np.full((n, ns), -np.inf)
    lfs = np.zeros(n); lbs = np.zeros(n); plog = np.zeros((n, ns))
    rc = logl.fward_log(c['init'], T, LE, c['end'], ns, ssb, n, par, npar, lf, lfs, ABI); assert rc == ABI, rc
    rc = logl.bward_log(T, LE, c['end'], ns, ssb, n, chi, nchi, lb, lbs, ABI); assert rc == ABI, rc
    rc = logl.posterior_decoding_log(lf, lb, ns, n, ssb, plog, ABI); assert rc == ABI, rc
    ll_log = 0.0
    for i in range(n - 1, 0, -1):
        ll_log += lfs[i]
    ll_log += lfs[0]
    a2 = {}
    mabs, mrel, ncell = errs(plog, p_normlin, 1e-4)
    a2['posterior_log_vs_normlin'] = {'max_abs': mabs, 'max_rel_ge_1e-4': mrel, 'n_cells_ge_1e-4': ncell}
    _, mrel12, ncell12 = errs(plog, p_normlin, 1e-12)
    a2['posterior_log_vs_normlin']['max_rel_ge_1e-12'] = mrel12
    a2['posterior_log_vs_normlin']['n_cells_ge_1e-12'] = ncell12
    # normlin vs fix B's posterior_decoding, emitting columns
    a2['normlin_vs_B_emitting_bit_identical'] = bits_equal(p_normlin[:, :ssb], B['p'][:, :ssb])
    a2['normlin_vs_B_emitting_max_abs'] = float(np.abs(p_normlin[:, :ssb] - B['p'][:, :ssb]).max())
    a2['B_silent_nnz_raw'] = int((B['p'][:, ssb:] != 0).sum())
    a2['B_silent_nnz_ge_1e-4'] = int((B['p'][:, ssb:] >= 1e-4).sum())
    a2['log_silent_nnz'] = int((plog[:, ssb:] != 0).sum())
    mabsB, mrelB, _ = errs(plog[:, :ssb], B['p'][:, :ssb], 1e-4)
    a2['posterior_log_vs_B_emitting'] = {'max_abs': mabsB, 'max_rel_ge_1e-4': mrelB}
    a2['loglik'] = dict(linear_calc_sr=B['ll'], log=ll_log, abs_diff=abs(ll_log - B['ll']))
    a2['log_scaling_vs_log_linear_scaling_max_abs'] = float(np.abs(lfs - np.log(B['fs'])).max())
    a2['emitting_rowsum_log_minmax'] = [float(plog[:, :ssb].sum(1).min()), float(plog[:, :ssb].sum(1).max())]
    # independent log_add sanity
    a2['log_add_check'] = [logl.log_add(np.log(0.25), np.log(0.5)) - np.log(0.75),
                           logl.log_add(-np.inf, -3.0) + 3.0, logl.log_add(-np.inf, -np.inf)]
    a2['PASS'] = bool(mabs < 1e-9 and mrel < 1e-9 and a2['loglik']['abs_diff'] < 1e-6)
    R['A2'] = a2
    print("A2", json.dumps(a2, indent=1)); sys.stdout.flush()

    # ------------------------------------------------------------------ D0 python + A3 preview
    if a.cap_logD0:
        y = np.load(a.cap_logD0)
        assert str(y['log_emission_env']) == 'floored', y['log_emission_env']
        E_log = y['emission']
        d0 = dict(npz=a.cap_logD0, tree=str(y['tree']), robocopC=str(y['robocopC']))
        with np.errstate(divide='ignore'):
            ref = np.log(c['E'])
        per = []
        for ch in range(7):
            x, r = E_log[ch], ref[ch]
            same_inf = bool(np.array_equal(np.isneginf(x), np.isneginf(r)))
            fin = np.isfinite(r)
            per.append(dict(channel=ch, neginf_pattern_identical=same_inf,
                            n_neginf=int(np.isneginf(x).sum()),
                            max_abs_finite=float(np.abs(x[fin] - r[fin]).max()) if fin.any() else 0.0,
                            bit_identical=bool(np.array_equal(x.view(np.uint64), r.view(np.uint64)))))
        d0['per_channel_vs_log_of_stock'] = per
        d0['nanmax'] = float(np.nanmax(E_log))
        d0['n_eq_log1e-30'] = int((E_log == np.log(1e-30)).sum())
        if a.logD0_h5:
            import h5py
            from scipy import sparse
            h = h5py.File(a.logD0_h5, 'r')
            g = h['segment_0']
            ll_d0 = float(g.attrs['log_likelihood'])
            gp = g['posterior']
            P = np.asarray(sparse.csr_matrix((gp['data'][:], gp['indices'][:], gp['indptr'][:]),
                                             shape=tuple(gp.attrs['shape'])).todense())
            Bfl = B['p'].copy(); Bfl[Bfl < 1e-4] = 0       # the save-time floor, robocop.py:33
            d0['stored_loglik'] = ll_d0
            d0['stored_loglik_minus_linear_calc_sr_here'] = ll_d0 - B['ll']
            d0['stored_loglik_minus_B_both_stored'] = ll_d0 - (-489955.03749446274)
            d0['stored_posterior_vs_B_emitting_max_abs'] = float(np.abs(P[:, :ssb] - Bfl[:, :ssb]).max())
            d0['stored_posterior_silent_nnz'] = int((P[:, ssb:] != 0).sum())
            d0['stored_posterior_emitting_cells_crossing_1e-4_floor'] = int(((P[:, :ssb] > 0) != (Bfl[:, :ssb] > 0)).sum())
        R['D0'] = d0
        print("D0", json.dumps(d0, indent=1)); sys.stdout.flush()

    md5_after = md5(SO_PROD)
    R['md5']['librobocop_so_after'] = md5_after
    assert md5_after == md5_before
    if a.json:
        json.dump(R, open(a.json, 'w'), indent=1, default=float)
    print("A1 PASS" if R['A1']['PASS'] else "A1 FAIL", "|", "A2 PASS" if R['A2']['PASS'] else "A2 FAIL")


if __name__ == '__main__':
    main()
