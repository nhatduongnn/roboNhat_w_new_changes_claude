"""Stage 2 prediction 5 follow-up: why D0 (log recursion, floored emission) != B (linear, fix B) on
chrXII rDNA. Runs linear fward/bward (librobocop.so) and fward_log/bward_log (librobocop_logfull.so)
in-process on ONE captured segment (harness_capture.py npz, stock tree => linear floored emission),
finds the first forward row where the two diverge and dumps the per-row diagnostics.

    cd analysis && python posterior_fix/divergence_chrXII.py <cap.npz> <info.h5> <segment> [--B h5] [--D0 h5] [--json out]

<info.h5>/<segment> supplies the fiber (n,k) per strand (Fiber_count_*). --B / --D0 are the stored
decodes' info h5 (to check the in-process run reproduces the stored divergence). Read-only.
"""
import sys, json, argparse
import numpy as np
import h5py
from scipy import sparse
from ctypes import CDLL
sys.path.insert(0, 'posterior_fix')
import harness_gates as HG

DBL_MIN = np.finfo(np.double).tiny


def dense(f, key):
    g = f[key]
    return np.asarray(sparse.csr_matrix((g['data'][:], g['indices'][:], g['indptr'][:]),
                                        shape=tuple(g.attrs['shape'])).todense())


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    mm = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide='ignore'):
        return (mm + np.log(np.exp(x - mm).sum(axis=axis, keepdims=True))).squeeze(axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz'); ap.add_argument('info_h5'); ap.add_argument('segment', type=int)
    ap.add_argument('--B'); ap.add_argument('--D0'); ap.add_argument('--json')
    ap.add_argument('--tol', type=float, default=1e-6)
    a = ap.parse_args()
    z = np.load(a.npz)
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
    R = dict(npz=a.npz, tree=str(z['tree']), n_obs=n, n_states=ns, ssb=ssb)

    prod = HG.bind(CDLL(HG.SO_PROD))
    logl = HG.bind_log(HG.bind(CDLL(HG.SO_LOG)))
    md5_0 = HG.md5(HG.SO_PROD)
    B = HG.run_linear(prod, c)                       # f, b, fs, bs, ll, p (fix B posterior)
    with np.errstate(divide='ignore'):
        LEc = np.log(c['E'])                         # per channel
    LE = np.ascontiguousarray(LEc.sum(axis=0))
    par, chi, npar, nchi = B['graph']
    lf = np.full((n, ns), -np.inf); lb = np.full((n, ns), -np.inf)
    lfs = np.zeros(n); lbs = np.zeros(n); plog = np.zeros((n, ns))
    assert logl.fward_log(c['init'], c['T'], LE, c['end'], ns, ssb, n, par, npar, lf, lfs, HG.ABI) == HG.ABI
    assert logl.bward_log(c['T'], LE, c['end'], ns, ssb, n, chi, nchi, lb, lbs, HG.ABI) == HG.ABI
    assert logl.posterior_decoding_log(lf, lb, ns, n, ssb, plog, HG.ABI) == HG.ABI
    R['loglik'] = dict(linear=B['ll'], log=float(lfs.sum()), log_minus_linear=float(lfs.sum() - B['ll']))

    # ---- forward / backward row comparison (emitting columns), position order
    fl = B['f'][:, :ssb]
    fl_n = fl / fl.sum(1, keepdims=True)
    lfe = lf[:, :ssb]
    fg_n = np.exp(lfe - lse(lfe)[:, None])
    dF = np.abs(fl_n - fg_n).max(1)
    bl = B['b'][::-1, :ssb]                           # reversed storage -> position order
    bl_n = bl / bl.sum(1, keepdims=True)
    lbe = lb[::-1, :ssb]
    bg_n = np.exp(lbe - lse(lbe)[:, None])
    dBk = np.abs(bl_n - bg_n).max(1)
    dP = np.abs(B['p'][:, :ssb] - plog[:, :ssb]).max(1)
    first_f = int(np.argmax(dF > a.tol)) if (dF > a.tol).any() else None
    first_b = int(n - 1 - np.argmax(dBk[::-1] > a.tol)) if (dBk > a.tol).any() else None
    R['forward'] = dict(n_rows_diff=int((dF > a.tol).sum()), first_row=first_f, max=float(dF.max()))
    R['backward'] = dict(n_rows_diff=int((dBk > a.tol).sum()), last_row=first_b, max=float(dBk.max()))
    R['posterior'] = dict(n_rows_diff=int((dP > a.tol).sum()), max=float(dP.max()),
                          rows_gt_0p01=np.flatnonzero(dP > 0.01).tolist()[:20])

    # ---- scaling: log(linear fs) vs log-space offset. fs includes ssb*DBL_MIN.
    with np.errstate(divide='ignore'):
        R['scaling_floor_ssb_DBL_MIN'] = ssb * DBL_MIN
    tiny_f = fs_tiny = B['fs'] < 1e-290
    tiny_b = B['bs'] < 1e-290
    R['n_pos_fs_lt_1e-290'] = int(tiny_f.sum()); R['n_pos_bs_lt_1e-290'] = int(tiny_b.sum())
    R['n_pos_lfs_lt_-668'] = int((lfs < -668).sum()); R['n_pos_lbs_lt_-668'] = int((lbs < -668).sum())
    R['fs_tiny_rows'] = np.flatnonzero(tiny_f).tolist()[:50]
    R['bs_tiny_rows'] = np.flatnonzero(tiny_b).tolist()[:50]

    # ---- fiber (n,k)
    h = h5py.File(a.info_h5, 'r')
    sg = 'segment_%d/' % a.segment
    kw = dense(h, sg + 'Fiber_count_meth_watson').ravel(); nw = dense(h, sg + 'Fiber_count_A_watson').ravel()
    kc = dense(h, sg + 'Fiber_count_meth_crick').ravel(); nc = dense(h, sg + 'Fiber_count_A_crick').ravel()
    nuc = dense(h, sg + 'nucleotides').ravel()

    # ---- per-row diagnostics around the first divergence(s)
    emis_max = np.where(np.isfinite(LE[:, :ssb]), LE[:, :ssb], -np.inf).max(1)   # log, emitting
    # Python one-step reconstruction of the linear forward pre-normalisation row
    import scipy.sparse as sp
    Tsp = sp.csr_matrix(c['T'][:, :ssb])            # parent -> emitting child
    def step(i):
        s = (sp.csr_matrix(B['f'][i - 1][None, :]) @ Tsp).toarray().ravel()   # sum_p f[i-1,p] T[p,j]
        with np.errstate(divide='ignore'):
            ls = np.log(s)
        pre_log = ls + LE[i, :ssb]                  # exact log of the pre-norm value
        lin = s.copy()
        for l in range(c['n_vars']):                # C order: sequential in-place products
            lin = lin * c['E'][l, i, :ssb]
        return pre_log, lin
    rows = []
    want = set()
    for fr in [first_f] + [int(x) for x in np.flatnonzero(tiny_f)[:3]]:
        if fr is None: continue
        for r in range(max(1, fr - 4), min(n - 1, fr + 2)):
            want.add(r)
    for i in sorted(want):
        pre_log, lin = step(i)
        rows.append(dict(
            row=i, dF=float(dF[i]), dB=float(dBk[i]), dP=float(dP[i]),
            fs_linear=float(B['fs'][i]), log_fs_linear=float(np.log(B['fs'][i])), lfs_log=float(lfs[i]),
            pre_norm_max_log10_exact=float(pre_log.max() / np.log(10)),
            pre_norm_max_linear_reconstructed=float(lin.max()),
            n_emitting_nonzero_linear=int((lin > 0).sum()),
            n_emitting_subnormal_linear=int(((lin > 0) & (lin < DBL_MIN)).sum()),
            emis_max_log10=float(emis_max[i] / np.log(10)),
            emis_ch5_max_log10=float(LEc[5, i, :ssb].max() / np.log(10)),
            emis_ch6_max_log10=float(LEc[6, i, :ssb].max() / np.log(10)),
            n_ch5_floored=int((c['E'][5, i, :ssb] == 1e-30).sum()),
            n_ch6_floored=int((c['E'][6, i, :ssb] == 1e-30).sum()),
            nucleotide=int(nuc[i]), watson_nk=(int(nw[i]), int(kw[i])), crick_nk=(int(nc[i]), int(kc[i])),
            argmax_log=int(np.argmax(fg_n[i])), argmax_lin=int(np.argmax(fl_n[i])),
            masked_mass_linear=float(fl_n[i][~np.isfinite(LE[i, :ssb])].sum()),
        ))
    R['rows'] = rows

    # ---- B silent columns > 1
    sil = B['p'][:, ssb:]
    bad = np.flatnonzero(sil.max(1) > 1)
    R['B_silent_gt1_rows'] = [dict(row=int(i), max=float(sil[i].max()), col=int(ssb + sil[i].argmax()),
                                   fs=float(B['fs'][i]), bs=float(B['bs'][i]),
                                   lfs=float(lfs[i]), lbs=float(lbs[i]),
                                   f_sil=float(B['f'][i, ssb + sil[i].argmax()]),
                                   b_sil=float(B['b'][n - 1 - i, ssb + sil[i].argmax()]),
                                   emit_rowsum_unnorm=float((B['f'][i, :ssb] * B['b'][n - 1 - i, :ssb]).sum()))
                              for i in bad]

    # ---- stored decodes
    for nm, path in [('B_stored', a.B), ('D0_stored', a.D0)]:
        if not path: continue
        g = h5py.File(path, 'r')
        P = dense(g, sg + 'posterior')
        ll = float(g['segment_%d' % a.segment].attrs['log_likelihood'])
        Bfl = B['p'].copy(); Bfl[Bfl < 1e-4] = 0
        Lfl = plog.copy(); Lfl[Lfl < 1e-4] = 0
        R[nm] = dict(ll=ll, ll_minus_inproc_linear=ll - B['ll'], ll_minus_inproc_log=ll - float(lfs.sum()),
                     maxabs_vs_inproc_linear_emitting=float(np.abs(P[:, :ssb] - Bfl[:, :ssb]).max()),
                     maxabs_vs_inproc_log_emitting=float(np.abs(P[:, :ssb] - Lfl[:, :ssb]).max()))
    assert HG.md5(HG.SO_PROD) == md5_0
    out = json.dumps(R, indent=1, default=float)
    print(out)
    if a.json:
        open(a.json, 'w').write(out)
    if a.json:
      np.savez_compressed(a.json.replace('.json', '_rows.npz'),
                        dF=dF, dB=dBk, dP=dP, fs=B['fs'], bs=B['bs'], lfs=lfs, lbs=lbs, emis_max=emis_max,
                        kw=kw, nw=nw, kc=kc, nc=nc)


if __name__ == '__main__':
    main()
