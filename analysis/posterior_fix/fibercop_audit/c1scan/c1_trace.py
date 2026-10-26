"""Trace one real C1 event: run original (prenorm) and C1-fixed bward from the C libraries on a captured
segment, then replay rows [lo, hi] in numpy with both orderings to see which cells carry the difference.

    cd analysis && python posterior_fix/fibercop_audit/c1scan/c1_trace.py <cap.npz> <lo_idx> <hi_idx>

idx = position index within the segment (row i of btable holds position n_obs-1-i). Read-only.
"""
import sys, ctypes, numpy as np
from ctypes import c_int
from numpy.ctypeslib import ndpointer
FORK = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude"
WT = "/usr/project/xtmp/nd141/programs/robocop_c_code_test/robocop_c_audit/walkthrough"
D = ndpointer(np.double, flags="C_CONTIGUOUS"); L = ndpointer(np.int64, flags="C_CONTIGUOUS")
z = np.load(sys.argv[1]); lo, hi = int(sys.argv[2]), int(sys.argv[3])
ns, ssb, n, nv = int(z['n_states']), int(z['silent_states_begin']), int(z['n_obs']), int(z['n_vars'])
E = np.ascontiguousarray(z['emission'], dtype=np.double); T = np.ascontiguousarray(z['transition_matrix'], dtype=np.double)
endp = np.ascontiguousarray(z['end_probs'], dtype=np.double)
ms = np.ascontiguousarray(z['tf_starts'], dtype=np.int64); ml = np.ascontiguousarray(z['tf_lens'], dtype=np.int64)
a = (int(z['n_tfs']), int(z['nuc_present']), int(z['nuc_start']), int(z['nuc_len']))

def bw(path):
    lib = ctypes.CDLL(path)
    lib.find_parents_and_children.argtypes = [L] * 4 + [c_int, c_int, D]
    lib.bward.argtypes = [D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    par = np.zeros((ns, ns), np.int64); chi = np.zeros((ns, ns), np.int64); npar = np.zeros(ns, np.int64); nchi = np.zeros(ns, np.int64)
    lib.find_parents_and_children(par, chi, npar, nchi, ns, ssb, T)
    b = np.zeros((n, ns)); s = np.zeros(n)
    lib.bward(T, E, endp, ns, ssb, n, nv, chi, nchi, ms, ml, *a, b, s)
    return b, s, chi, nchi

bo, so, chi, nchi = bw(FORK + "/pkg/robocop/librobocop_prenorm.so")
bf, sf, _, _ = bw(WT + "/variants/librobocop_fixed.so")
d = np.log(so) - np.log(sf)
print("positions where log scaling factor differs (>1e-6):", np.flatnonzero(np.abs(d) > 1e-6).tolist(), "values (nats):", d[np.abs(d) > 1e-6].round(3).tolist())
pe = E.prod(0)
np.savez(sys.argv[1].replace('.npz', '_bw.npz'), bo=bo, bf=bf, so=so, sf=sf, chi=chi, nchi=nchi)

# where do the two backward tables first differ (relative), scanning from the segment end?
for p in range(hi, lo - 1, -1):
    r = n - 1 - p
    em = np.abs(bo[r, :ssb] - bf[r, :ssb]).max()
    with np.errstate(divide='ignore', invalid='ignore'):
        q = bo[r, ssb:] / bf[r, ssb:]
    q = q[np.isfinite(q) & (bf[r, ssb:] > 0)]
    print("idx %d  scale_o %.3e scale_f %.3e  emit max|diff| %.1e  silent o/f ratio range %s  silent sum o %.2e f %.2e" % (
        p, so[p], sf[p], em, ("%.3g..%.3g" % (q.min(), q.max())) if q.size else "-", bo[r, ssb:].sum(), bf[r, ssb:].sum()))
