"""Trace the fake mass of defect 3 back along nucleosome diagonals (nuc_pos k at row i <- nuc_pos k-1 at row i-1).
    python seg300_diagonal_trace.py <cap.npz> unused
For a few (row, state) cells where A (librobocop.so, +DBL_MIN) >> L (log space), walk back along the
diagonal and print fA, fL and the padding floor DBL_MIN/S_row that normalize() put in that cell."""
import sys, os, ctypes, numpy as np
sys.argv_ = sys.argv
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seg300_forward_replay.py')).read().split("with np.errstate(all='ignore'):")[0])
lib = ctypes.CDLL(R + '/pkg/robocop/librobocop.so'); P, C, nP, nC = graph(lib); zz = np.zeros(1, np.longlong)
fA = np.zeros((n, ns)); sfA = np.zeros(n)
lib.fward.argtypes = [D, D, D, D, c_int, c_int, c_int, c_int, L_, L_, L_, L_, c_int, c_int, c_int, c_int, D, D]
lib.fward(init, T, E, endp, ns, ssb, n, nv, P, nP, zz, zz, 0, 0, 0, 0, fA, sfA)
lib2 = ctypes.CDLL(R + '/pkg/robocop/librobocop_logfull.so'); P2, C2, nP2, nC2 = graph(lib2)
with np.errstate(divide='ignore'): le = np.ascontiguousarray(np.log(E).sum(0))
lf = np.full((n, ns), -np.inf); lsf = np.zeros(n)
lib2.fward_log.argtypes = [D, D, D, D, c_int, c_int, c_int, L_, L_, D, D, c_int]; lib2.fward_log.restype = c_int
with np.errstate(divide='ignore'): assert lib2.fward_log(init, T, le, endp, ns, ssb, n, P2, nP2, lf, lsf, ABI) == ABI
floor = DBL_MIN / sfA
print('parents of nuc_pos5:', [state_name(int(p)) for p in P[nuc_start + 5, :nP[nuc_start + 5]]])
print('parents of nuc_pos0:', [state_name(int(p)) for p in P[nuc_start, :nP[nuc_start]]][:10], '... n =', int(nP[nuc_start]))
for (i_end, k_end) in [(1789, 9), (1789, 5), (1790, 16)]:
    print('\n== diagonal ending at row %d nuc_pos%d   (row, state, log10 fA, log10 fL, log10 floor=DBL_MIN/S, log10 emission[row,state], log10 emission max over emitting)' % (i_end, k_end))
    for d in range(k_end, -1, -1):
        i, j = i_end - d, nuc_start + k_end - d
        em = np.log10(np.prod(E[:, i, j])); emmax = np.log10(np.max(np.prod(E[:, i, :ssb], axis=0)))
        print('  %d %-10s fA %8.2f  fL %8.2f  floor %8.2f  emis %8.2f  emis_max %8.2f' % (
            i, state_name(j), np.log10(fA[i, j]), lf[i, j] / np.log(10), np.log10(floor[i]), em, emmax))
    # the cell one step before the diagonal starts (nuc_pos0's predecessor at row i_end-k_end-1 is not on the chain)
