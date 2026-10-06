"""Python replica of algo.c fward/bward (linear) with the normalize() injection `eps` as a parameter.

    python posterior_fix/replica_eps.py <cap.npz> <tables.npz> [--json out]

eps = DBL_MIN reproduces algo.c:8-28 (normalize adds DBL_MIN to every cell); eps = 0 is the same
linear recursion with a plain divide-by-sum. Compares each against librobocop.so's tables (from
<tables.npz>, written by the scratch dump script) and against the log-space reference (fward_log /
bward_log / posterior_decoding_log, same file). Isolates whether the B-vs-D0 divergence is the
DBL_MIN injection or double-precision underflow. Read-only.
"""
import sys, json, argparse
import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser(); ap.add_argument('cap'); ap.add_argument('tables'); ap.add_argument('--json')
a = ap.parse_args()
z = np.load(a.cap); t = np.load(a.tables)
E = z['emission']; T = z['transition_matrix'].astype(float); init = z['initial_probs']; endp = z['end_probs']
ns, ssb, n = int(z['n_states']), int(z['silent_states_begin']), int(z['n_obs'])
DM = np.finfo(float).tiny
# per-position emission product, multiplied channel by channel in C order (algo.c:86-87, :176-177)
Ep = np.ones((n, ns))
for l in range(E.shape[0]):
    Ep = Ep * E[l]
del E
Tem = sp.csc_matrix(T[:, :ssb])                      # any parent -> emitting child
sil_par = [(np.flatnonzero(T[:, j]), T[np.flatnonzero(T[:, j]), j]) for j in range(ssb, ns)]
Tch_em = sp.csr_matrix(T[:ssb, :])                   # emitting parent -> any child
sil_ch = [(np.flatnonzero(T[j, :]), T[j, np.flatnonzero(T[j, :])]) for j in range(ns)]


def norm(row, eps):
    s = (row[:ssb] + eps).sum()
    return (row + eps) / s, s


def fward(eps):
    f = np.zeros((n, ns)); fs = np.zeros(n)
    r = np.zeros(ns); r[:ssb] = init[:ssb] * Ep[0, :ssb]
    for k, j in enumerate(range(ssb, ns)):
        p, w = sil_par[k]; r[j] += (r[p] * w).sum()
    f[0], fs[0] = norm(r, eps)
    for i in range(1, n):
        r = np.zeros(ns)
        r[:ssb] = Tem.T.dot(f[i - 1])
        if i == n - 1: r[:ssb] *= endp[:ssb]
        r[:ssb] *= Ep[i, :ssb]
        for k, j in enumerate(range(ssb, ns)):
            p, w = sil_par[k]; r[j] += (r[p] * w).sum()
        f[i], fs[i] = norm(r, eps)
    return f, fs


def bward(eps):   # position order (not reversed)
    b = np.zeros((n, ns)); bs = np.zeros(n)
    r = np.zeros(ns); r[:ssb] = endp[:ssb]
    for j in range(ns - 1, ssb - 1, -1):
        c, w = sil_ch[j]; r[j] += (r[c] * w * Ep[n - 1, c]).sum()
    b[n - 1], bs[n - 1] = norm(r, eps)
    for pos in range(n - 2, -1, -1):
        r = np.zeros(ns)
        r[:ssb] = Tch_em.dot(b[pos + 1] * Ep[pos + 1])
        for j in range(ns - 1, ssb - 1, -1):
            c, w = sil_ch[j]; r[j] += (r[c] * w * Ep[pos, c]).sum()
        b[pos], bs[pos] = norm(r, eps)
    return b, bs


def post(f, b):
    p = f[:, :ssb] * b[:, :ssb]
    return p / p.sum(1, keepdims=True)


def rown(x):
    return x[:, :ssb] / x[:, :ssb].sum(1, keepdims=True)


fC = t['f']; bC = t['b'][::-1]; pC = t['p'][:, :ssb]; pL = t['pl'][:, :ssb]
R = {}
for name, eps in [('eps_DBL_MIN', DM), ('eps_0', 0.0)]:
    f, fs = fward(eps); b, bs = bward(eps); p = post(f, b)
    ll = float(np.log(fs).sum())
    dpC = np.abs(p - pC).max(1); dpL = np.abs(p - pL).max(1)
    R[name] = dict(eps=eps, loglik=ll,
                   fwd_maxabs_vs_C=float(np.abs(rown(f) - rown(fC)).max()),
                   bwd_maxabs_vs_C=float(np.abs(rown(b) - rown(bC)).max()),
                   post_maxabs_vs_C=float(dpC.max()), post_rows_gt_1em6_vs_C=int((dpC > 1e-6).sum()),
                   post_maxabs_vs_log=float(dpL.max()), post_rows_gt_1em6_vs_log=int((dpL > 1e-6).sum()),
                   post_rows_gt_0p01_vs_log=int((dpL > 0.01).sum()),
                   n_rows_all_emitting_zero_before_norm=int((fs - ssb * eps <= 0).sum() + (bs - ssb * eps <= 0).sum()),
                   min_fs=float(fs.min()), min_bs=float(bs.min()))
    R[name]['loglik_minus_C'] = ll - float(np.log(t['fs']).sum())
    R[name]['loglik_minus_log'] = ll - float(t['lfs'].sum())
    print(name, json.dumps(R[name])); sys.stdout.flush()
if a.json: json.dump(R, open(a.json, 'w'), indent=1)
