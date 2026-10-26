"""Defect 3 on bo09 chrIV:1,200,001-1,205,000 (seg 300): where does the forward pass with +DBL_MIN
leave the true (log-space) forward, and which states carry the fake mass?

    python seg300_forward_replay.py <cap.npz> <out.json>

Runs, on the SAME captured inputs, in one process:
  A  fward/bward from pkg/robocop/librobocop.so            (production fix B: +DBL_MIN)
  N  fward/bward from build/lib_nodblmin.so                (same algo.c with '+ DBL_MIN' -> '+ 0.0')
  L  fward_log/bward_log from pkg/robocop/librobocop_logfull.so (exact log space, no DBL_MIN)
All forward rows are normalised over emitting states, so they are directly comparable
(A, N linear; L exponentiated).
"""
import sys, os, json, ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import c_int

R = '/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude'
HERE = os.path.dirname(os.path.abspath(__file__))
ABI = 20261005
D = ndpointer(np.double, flags='C_CONTIGUOUS'); L_ = ndpointer(np.longlong, flags='C_CONTIGUOUS')
LDp = ndpointer(np.longdouble, flags='C_CONTIGUOUS')
DBL_MIN = np.finfo(float).tiny

z = np.load(sys.argv[1]); out_json = sys.argv[2]
E = np.ascontiguousarray(z['emission'], dtype=np.double)
T = np.ascontiguousarray(z['transition_matrix'], dtype=np.double)
init = np.ascontiguousarray(z['initial_probs'], dtype=np.double); endp = np.ascontiguousarray(z['end_probs'], dtype=np.double)
ns, ssb, n, nv = int(z['n_states']), int(z['silent_states_begin']), int(z['n_obs']), int(z['n_vars'])
nuc_start, nuc_len = int(z['nuc_start']), int(z['nuc_len'])
tf_starts, tf_lens = np.array(z['tf_starts']), np.array(z['tf_lens'])
print('n_obs', n, 'n_states', ns, 'ssb', ssb, 'n_vars', nv, 'nuc_start', nuc_start, 'nuc_len', nuc_len, flush=True)

def state_name(j):
    if j >= ssb: return 'silent_%d' % (j - ssb)
    if nuc_start <= j < nuc_start + nuc_len: return 'nuc_pos%d' % (j - nuc_start)
    for k, (s, l) in enumerate(zip(tf_starts, tf_lens)):
        if s <= j < s + l: return 'tf%d_col%d' % (k, j - s)
    return 'state%d' % j

def graph(lib):
    P = np.zeros((ns, ns), np.longlong); C = np.zeros((ns, ns), np.longlong)
    nP = np.zeros(ns, np.longlong); nC = np.zeros(ns, np.longlong)
    lib.find_parents_and_children.argtypes = [L_, L_, L_, L_, c_int, c_int, D]
    lib.find_parents_and_children(P, C, nP, nC, ns, ssb, T)
    return P, C, nP, nC

def linear(path):
    lib = ctypes.CDLL(path); P, C, nP, nC = graph(lib); zz = np.zeros(1, np.longlong)
    f = np.zeros((n, ns)); b = np.zeros((n, ns)); sf = np.zeros(n); sb = np.zeros(n)
    lib.fward.argtypes = [D, D, D, D, c_int, c_int, c_int, c_int, L_, L_, L_, L_, c_int, c_int, c_int, c_int, D, D]
    lib.bward.argtypes = [D, D, D, c_int, c_int, c_int, c_int, L_, L_, L_, L_, c_int, c_int, c_int, c_int, D, D]
    lib.fward(init, T, E, endp, ns, ssb, n, nv, P, nP, zz, zz, 0, 0, 0, 0, f, sf)
    lib.bward(T, E, endp, ns, ssb, n, nv, C, nC, zz, zz, 0, 0, 0, 0, b, sb)
    p = np.zeros((n, ns)); sr = np.ones(n, np.longdouble)
    lib.posterior_decoding_v2.argtypes = [D, D, D, LDp, c_int, c_int, c_int, D]
    lib.posterior_decoding_v2(f, b, sb, sr, ns, n, ssb, p)
    return f, sf, b, p

def logspace():
    lib = ctypes.CDLL(R + '/pkg/robocop/librobocop_logfull.so'); P, C, nP, nC = graph(lib)
    with np.errstate(divide='ignore'):
        le = np.ascontiguousarray(np.log(E).sum(0))   # floored emission, as captured => arm "D0"
    f = np.full((n, ns), -np.inf); b = np.full((n, ns), -np.inf); sf = np.zeros(n); sb = np.zeros(n)
    p = np.zeros((n, ns))
    lib.fward_log.argtypes = [D, D, D, D, c_int, c_int, c_int, L_, L_, D, D, c_int]; lib.fward_log.restype = c_int
    lib.bward_log.argtypes = [D, D, D, c_int, c_int, c_int, L_, L_, D, D, c_int]; lib.bward_log.restype = c_int
    lib.posterior_decoding_log.argtypes = [D, D, c_int, c_int, c_int, D, c_int]; lib.posterior_decoding_log.restype = c_int
    with np.errstate(divide='ignore'):
        r1 = lib.fward_log(init, T, le, endp, ns, ssb, n, P, nP, f, sf, ABI)
        r2 = lib.bward_log(T, le, endp, ns, ssb, n, C, nC, b, sb, ABI)
        r3 = lib.posterior_decoding_log(f, b, ns, n, ssb, p, ABI)
    assert r1 == r2 == r3 == ABI, (r1, r2, r3)
    return f, sf, b, p

with np.errstate(all='ignore'):
    fA, sfA, bA, pA = linear(R + '/pkg/robocop/librobocop.so')
    fN, sfN, bN, pN = linear(HERE + '/build/lib_nodblmin.so')
lf, lsf, lb, pL = logspace()
print('tables done', flush=True)

res = {}
res['loglik'] = dict(A=float(np.log(sfA).sum()), N=float(np.log(sfN).sum()) if np.all(sfN > 0) else None, L=float(lsf.sum()))
res['loglik_A_minus_L'] = res['loglik']['A'] - res['loglik']['L']
dpA = np.abs(pA[:, :ssb] - pL[:, :ssb]).max(1); dpN = np.abs(pN[:, :ssb] - pL[:, :ssb]).max(1)
res['posterior_rows_gt_0p01'] = dict(A_vs_L=int((dpA > .01).sum()), N_vs_L=int((dpN > .01).sum()))
badA = np.flatnonzero(dpA > .01)
res['posterior_bad_rows_A'] = [int(badA.min()), int(badA.max())] if badA.size else None
res['posterior_maxabs'] = dict(A_vs_L=float(dpA.max()), N_vs_L=float(np.nanmax(dpN)))
res['N_nonfinite_forward_rows'] = int((~np.isfinite(fN).all(1)).sum())

# forward: true value of each cell (exp of log), row-normalised identically
fL = np.exp(lf)                     # rows already normalised over emitting states by log_normalize
dfA = np.abs(fA[:, :ssb] - fL[:, :ssb]).max(1)
dfN = np.abs(fN[:, :ssb] - fL[:, :ssb]).max(1)
res['forward_rows_gt_0p01'] = dict(A_vs_L=int((dfA > .01).sum()), N_vs_L=int((dfN > .01).sum()))
firstA = int(np.argmax(dfA > .01)) if (dfA > .01).any() else None
res['forward_first_row_gt_0p01_A'] = firstA
res['forward_first_row_gt_1em6_A'] = int(np.argmax(dfA > 1e-6)) if (dfA > 1e-6).any() else None

# per-row scale: log10 of the emitting sum S that normalize() divides by
log10S = np.log10(sfA)
res['log10_S'] = dict(min=float(log10S.min()), median=float(np.median(log10S)), max=float(log10S.max()))
res['log10_fake_per_cell'] = dict(min=float(np.log10(DBL_MIN) - log10S.max()), max=float(np.log10(DBL_MIN) - log10S.min()))
# the floor every cell of A carries after normalize: >= DBL_MIN/S (cells start >= 0)
floor_ok = np.all(fA[:, :] >= (DBL_MIN / sfA)[:, None] * (1 - 1e-12))
res['every_cell_of_A_ge_DBLMIN_over_S'] = bool(floor_ok)
# how many emitting cells are truly impossible (log -inf) in L but carry mass in A
impossible = np.isneginf(lf[:, :ssb])
res['emitting_cells_impossible_in_L'] = int(impossible.sum())
res['of_those_nonzero_in_A'] = int((fA[:, :ssb][impossible] > 0).sum())

if firstA is not None:
    i0 = max(firstA - 3, 1)
    rows = []
    for i in range(i0, firstA + 2):
        diff = fA[i, :ssb] - fL[i, :ssb]
        top = np.argsort(-diff)[:5]
        rows.append(dict(row=i, log10_S=float(log10S[i]), maxabs=float(dfA[i]),
                         top_states_A_gt_L=[dict(state=int(j), name=state_name(j), fA=float(fA[i, j]), fL=float(fL[i, j]),
                                                 log_fL=float(lf[i, j])) for j in top]))
    res['around_first_forward_divergence'] = rows
    # in the row just BEFORE divergence: the states that become the winners, A (fake) vs L (true)
    win = rows[-1]['top_states_A_gt_L'][0]['state']
    res['winner_state'] = state_name(win)

# where do the 147 bad posterior rows sit: dominant state in A vs L
if badA.size:
    mA = pA[badA, :ssb].argmax(1); mL = pL[badA, :ssb].argmax(1)
    res['bad_rows_argmax_A'] = sorted(set(state_name(int(j)).split('_pos')[0] for j in mA))
    res['bad_rows_argmax_L'] = sorted(set(state_name(int(j)).split('_pos')[0] for j in mL))
    res['bad_rows_first5'] = [dict(row=int(i), A=state_name(int(pA[i, :ssb].argmax())), pA=float(pA[i, :ssb].max()),
                                   L=state_name(int(pL[i, :ssb].argmax())), pL=float(pL[i, :ssb].max())) for i in badA[:5]]
    nucA = pA[badA, nuc_start:nuc_start + nuc_len].sum(1); nucL = pL[badA, nuc_start:nuc_start + nuc_len].sum(1)
    res['bad_rows_nuc_occupancy_mean'] = dict(A=float(nucA.mean()), L=float(nucL.mean()))

np.savez_compressed(os.path.splitext(out_json)[0] + '_rows.npz', dfA=dfA, dfN=dfN, dpA=dpA, dpN=dpN, log10S=log10S)
json.dump(res, open(out_json, 'w'), indent=1)
print(json.dumps(res, indent=1))
