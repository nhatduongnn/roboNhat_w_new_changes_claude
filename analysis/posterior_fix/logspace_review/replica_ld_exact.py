"""(d) Extended-precision reference for an EXACT-mode log capture.
    python replica_ld_exact.py <cap_log.npz> <out.json> [--save post.npz]
<cap_log.npz>: harness_capture.py of a *_logfull tree (layers hold LOGS). The 7 layers are summed
(float64, as robocop.py does), then exponentiated in np.longdouble (x87 80-bit, min normal ~3.4e-4932)
and run through an independent linear forward/backward written from algo.c's recursion order
(same as posterior_fix/replica_longdouble.py, eps = 0: no DBL_MIN). The C log path
(librobocop_logfull.so, bound as robocop.py binds it) runs on the same float64 log emission.
Compared: emitting posterior (row-normalised) and log-likelihood (sum of log forward offsets)."""
import sys, json, argparse, ctypes
import numpy as np
from ctypes import c_int
from numpy.ctypeslib import ndpointer

ap = argparse.ArgumentParser(); ap.add_argument('cap'); ap.add_argument('json'); ap.add_argument('--save')
a = ap.parse_args()
LD = np.longdouble
z = np.load(a.cap)
ns, ssb, n = int(z['n_states']), int(z['silent_states_begin']), int(z['n_obs'])
LE = np.ascontiguousarray(z['emission'].sum(0), dtype=np.double)
T = np.ascontiguousarray(z['transition_matrix'], dtype=np.double)
init = np.ascontiguousarray(z['initial_probs'], dtype=np.double); endp = np.ascontiguousarray(z['end_probs'], dtype=np.double)
R = dict(cap=a.cap, tree=str(z['tree']), env=str(z['log_emission_env']), n_obs=n,
         min_log_emis_finite=float(LE[np.isfinite(LE)].min()),
         min_rowmax_log_emis=float(LE[:, :ssb].max(1).min()))

# ---------------- C log path
LIB = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/pkg/robocop/librobocop_logfull.so"
ABI = 20261005
D = ndpointer(np.double, flags="C_CONTIGUOUS"); L = ndpointer(np.longlong, flags="C_CONTIGUOUS")
lib = ctypes.CDLL(LIB)
lib.find_parents_and_children.argtypes = [L, L, L, L, c_int, c_int, D]
lib.fward_log.argtypes = [D, D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.fward_log.restype = c_int
lib.bward_log.argtypes = [D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.bward_log.restype = c_int
lib.posterior_decoding_log.argtypes = [D, D, c_int, c_int, c_int, D, c_int]; lib.posterior_decoding_log.restype = c_int
par = np.zeros((ns, ns), np.longlong); chi = np.zeros((ns, ns), np.longlong)
npar = np.zeros(ns, np.longlong); nchi = np.zeros(ns, np.longlong)
lib.find_parents_and_children(par, chi, npar, nchi, ns, ssb, T)
fl = np.full((n, ns), -np.inf); bl = np.full((n, ns), -np.inf); pl = np.zeros((n, ns)); lfs = np.zeros(n); lbs = np.zeros(n)
with np.errstate(divide='ignore'):
    rc = (lib.fward_log(init, T, LE, endp, ns, ssb, n, par, npar, fl, lfs, ABI),
          lib.bward_log(T, LE, endp, ns, ssb, n, chi, nchi, bl, lbs, ABI))
rc = rc + (lib.posterior_decoding_log(fl, bl, ns, n, ssb, pl, ABI),)
R['rc'] = list(rc); R['loglik_C'] = float(lfs.sum())
del fl, bl

# ---------------- 80-bit linear replica (algo.c order, no DBL_MIN)
Ep = np.exp(LE.astype(LD))
R['Ep_zero_cells_not_neginf'] = int(((Ep == 0) & np.isfinite(LE)).sum())   # longdouble underflow check
Tl = T
par_e, ch_e = np.nonzero(Tl[:, :ssb]); o = np.argsort(ch_e, kind='stable'); par_e, ch_e = par_e[o], ch_e[o]
w_f = Tl[par_e, ch_e].astype(LD); uc, st_f = np.unique(ch_e, return_index=True)
pb, cb = np.nonzero(Tl[:ssb, :]); o = np.argsort(pb, kind='stable'); pb, cb = pb[o], cb[o]
w_b = Tl[pb, cb].astype(LD); up, st_b = np.unique(pb, return_index=True)
sil_par = [(np.flatnonzero(Tl[:, j]), Tl[np.flatnonzero(Tl[:, j]), j].astype(LD)) for j in range(ssb, ns)]
sil_ch = {j: (np.flatnonzero(Tl[j, :]), Tl[j, np.flatnonzero(Tl[j, :])].astype(LD)) for j in range(ssb, ns)}
initL, endL = init.astype(LD), endp.astype(LD)

def norm(r):
    s = r[:ssb].sum(); return r / s, s

f = np.zeros((n, ns), dtype=LD); fs = np.zeros(n, dtype=LD)
r = np.zeros(ns, dtype=LD); r[:ssb] = initL[:ssb] * Ep[0, :ssb]
for k, j in enumerate(range(ssb, ns)):
    p, w = sil_par[k]; r[j] += (r[p] * w).sum()
f[0], fs[0] = norm(r)
for i in range(1, n):
    r = np.zeros(ns, dtype=LD)
    r[uc] = np.add.reduceat(f[i - 1][par_e] * w_f, st_f)
    if i == n - 1: r[:ssb] *= endL[:ssb]
    r[:ssb] *= Ep[i, :ssb]
    for k, j in enumerate(range(ssb, ns)):
        p, w = sil_par[k]; r[j] += (r[p] * w).sum()
    f[i], fs[i] = norm(r)
b = np.zeros((n, ns), dtype=LD); bs = np.zeros(n, dtype=LD)
r = np.zeros(ns, dtype=LD); r[:ssb] = endL[:ssb]
for j in range(ns - 1, ssb - 1, -1):
    c, w = sil_ch[j]; r[j] += (r[c] * w * Ep[n - 1, c]).sum()
b[n - 1], bs[n - 1] = norm(r)
for pos in range(n - 2, -1, -1):
    r = np.zeros(ns, dtype=LD)
    x = b[pos + 1] * Ep[pos + 1]
    r[up] = np.add.reduceat(x[cb] * w_b, st_b)
    for j in range(ns - 1, ssb - 1, -1):
        c, w = sil_ch[j]; r[j] += (r[c] * w * Ep[pos, c]).sum()
    b[pos], bs[pos] = norm(r)
pe = f[:, :ssb] * b[:, :ssb]; rs = pe.sum(1, keepdims=True)
pref = (pe / rs).astype(float)
d = np.abs(pref - pl[:, :ssb]).max(1)
R.update(loglik_LD=float(np.log(fs).sum()), loglik_diff=float(np.log(fs).sum()) - R['loglik_C'],
         post_maxabs=float(d.max()), post_rows_gt_1e6=int((d > 1e-6).sum()), post_rows_gt_1e9=int((d > 1e-9).sum()),
         ld_zero_rowsum=int((rs == 0).sum()), min_log10_fs=float(np.log10(fs.min())),
         C_rowsum_range=[float(pl[:, :ssb].sum(1).min()), float(pl[:, :ssb].sum(1).max())],
         C_silent_max=float(pl[:, ssb:].max()), C_nan=int(np.isnan(pl).sum()))
print(json.dumps(R, indent=1))
json.dump(R, open(a.json, 'w'), indent=1)
if a.save:
    np.savez(a.save, pl=pl, pref=pref)
