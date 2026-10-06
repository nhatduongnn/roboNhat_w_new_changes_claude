"""Linear fward/bward replica of algo.c in np.longdouble (x87 80-bit: min normal ~3.4e-4932) with
eps = 0 (plain divide-by-sum, no DBL_MIN injection). If this matches the log-space reference, the
log code is right and B's divergence is a float64 range artefact.

    python posterior_fix/replica_longdouble.py <cap.npz> <tables.npz> [--eps-dblmin] [--json out]
"""
import sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('cap'); ap.add_argument('tables'); ap.add_argument('--json')
ap.add_argument('--eps-dblmin', action='store_true', help='inject float64 DBL_MIN per cell like algo.c')
a = ap.parse_args()
LD = np.longdouble
z = np.load(a.cap); t = np.load(a.tables)
E = z['emission']; T = z['transition_matrix'].astype(float); init = z['initial_probs'].astype(LD); endp = z['end_probs'].astype(LD)
ns, ssb, n = int(z['n_states']), int(z['silent_states_begin']), int(z['n_obs'])
eps = LD(np.finfo(float).tiny) if a.eps_dblmin else LD(0)
print("longdouble finfo tiny", np.finfo(LD).tiny, "eps", eps)
Ep = np.ones((n, ns), dtype=LD)
for l in range(E.shape[0]):
    Ep = Ep * E[l].astype(LD)
del E
# edge lists, grouped by child (forward emitting) and by parent (backward emitting)
par_e, ch_e = np.nonzero(T[:, :ssb]); o = np.argsort(ch_e, kind='stable'); par_e, ch_e = par_e[o], ch_e[o]
w_f = T[par_e, ch_e].astype(LD); uc, st_f = np.unique(ch_e, return_index=True)
pb, cb = np.nonzero(T[:ssb, :]); o = np.argsort(pb, kind='stable'); pb, cb = pb[o], cb[o]
w_b = T[pb, cb].astype(LD); up, st_b = np.unique(pb, return_index=True)
sil_par = [(np.flatnonzero(T[:, j]), T[np.flatnonzero(T[:, j]), j].astype(LD)) for j in range(ssb, ns)]
sil_ch = {j: (np.flatnonzero(T[j, :]), T[j, np.flatnonzero(T[j, :])].astype(LD)) for j in range(ssb, ns)}

def norm(r):
    s = (r[:ssb] + eps).sum(); return (r + eps) / s, s

def fward():
    f = np.zeros((n, ns), dtype=LD); fs = np.zeros(n, dtype=LD)
    r = np.zeros(ns, dtype=LD); r[:ssb] = init[:ssb] * Ep[0, :ssb]
    for k, j in enumerate(range(ssb, ns)):
        p, w = sil_par[k]; r[j] += (r[p] * w).sum()
    f[0], fs[0] = norm(r)
    for i in range(1, n):
        r = np.zeros(ns, dtype=LD)
        r[uc] = np.add.reduceat(f[i - 1][par_e] * w_f, st_f)
        if i == n - 1: r[:ssb] *= endp[:ssb]
        r[:ssb] *= Ep[i, :ssb]
        for k, j in enumerate(range(ssb, ns)):
            p, w = sil_par[k]; r[j] += (r[p] * w).sum()
        f[i], fs[i] = norm(r)
    return f, fs

def bward():
    b = np.zeros((n, ns), dtype=LD); bs = np.zeros(n, dtype=LD)
    r = np.zeros(ns, dtype=LD); r[:ssb] = endp[:ssb]
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
    return b, bs

f, fs = fward(); b, bs = bward()
pfull = f * b; rs = pfull[:, :ssb].sum(1, keepdims=True); p = (pfull[:, :ssb] / rs).astype(float)
psil = (pfull[:, ssb:] / rs).astype(float)          # silent columns, normalised like fix B (algo.c:300-303)
sC = t['p'][:, ssb:]
pC = t['p'][:, :ssb]; pL = t['pl'][:, :ssb]
dL = np.abs(p - pL).max(1); dC = np.abs(p - pC).max(1)
ll = float(np.log(fs).sum())
R = dict(eps=float(eps), loglik=ll, loglik_minus_log=ll - float(t['lfs'].sum()), loglik_minus_C=ll - float(np.log(t['fs']).sum()),
         post_maxabs_vs_log=float(dL.max()), post_rows_gt_1em6_vs_log=int((dL > 1e-6).sum()),
         post_maxabs_vs_C=float(dC.max()), post_rows_gt_0p01_vs_C=int((dC > 0.01).sum()),
         n_zero_rowsum=int((rs == 0).sum()), min_fs=float(np.log10(fs.min())), min_bs=float(np.log10(bs.min())),
         silent_max=float(psil.max()), silent_rows_gt1=np.flatnonzero(psil.max(1) > 1).tolist(),
         silent_max_C=float(sC.max()), silent_rows_gt1_C=np.flatnonzero(sC.max(1) > 1).tolist(),
         silent_maxabs_vs_C=float(np.abs(psil - sC).max()),
         log10_fs_vs_lfs_maxabs=float(np.abs(np.log(fs).astype(float) - t['lfs']).max()))
print(json.dumps(R))
if a.json: json.dump(R, open(a.json, 'w'), indent=1)
