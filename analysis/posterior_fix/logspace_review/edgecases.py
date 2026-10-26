"""Edge cases of librobocop_logfull.so: n_obs = 0 out-of-bounds writes, rc = -2 on an impossible
sequence (and what the recursion leaves behind), and the ABI guard. Uses sentinel buffers so an
out-of-bounds write is visible. Run from analysis/."""
import ctypes, numpy as np
from ctypes import c_int
from numpy.ctypeslib import ndpointer
LIB = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/pkg/robocop/librobocop_logfull.so"
ABI = 20261005
D = ndpointer(np.double, flags="C_CONTIGUOUS"); L = ndpointer(np.longlong, flags="C_CONTIGUOUS")
lib = ctypes.CDLL(LIB)
lib.find_parents_and_children.argtypes = [L, L, L, L, c_int, c_int, D]
lib.fward_log.argtypes = [D, D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.fward_log.restype = c_int
lib.bward_log.argtypes = [D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.bward_log.restype = c_int
lib.posterior_decoding_log.argtypes = [D, D, c_int, c_int, c_int, D, c_int]; lib.posterior_decoding_log.restype = c_int

# 3 emitting + 1 silent: 0->0, 0->S, S->1, 1->2, 2->0
ns, ssb = 4, 3
T = np.zeros((ns, ns)); T[0, 0] = .5; T[0, 3] = .5; T[3, 1] = 1; T[1, 2] = 1; T[2, 0] = 1
par = np.zeros((ns, ns), np.longlong); chi = np.zeros((ns, ns), np.longlong)
npar = np.zeros(ns, np.longlong); nchi = np.zeros(ns, np.longlong)
lib.find_parents_and_children(par, chi, npar, nchi, ns, ssb, T)
init = np.array([1., 0, 0, 0]); endp = np.ones(ns)

# ---- n_obs = 0: pass 0-row views into sentinel buffers
SENT = 777.0
for name in ("fward_log", "bward_log"):
    fbuf = np.full(64, SENT); sbuf = np.full(64, SENT); ebuf = np.zeros(64)
    f = fbuf[:0]; sf = sbuf[:0]
    with np.errstate(divide='ignore'):
        if name == "fward_log":
            rc = lib.fward_log(init, T, ebuf[:0], endp, ns, ssb, 0, par, npar, f, sf, ABI)
        else:
            rc = lib.bward_log(T, ebuf[:0], endp, ns, ssb, 0, chi, nchi, f, sf, ABI)
    # the scaling buffer: bward writes log_scaling_factors[n_obs-1] = [-1] -> one element BEFORE sbuf[0]
    print("n_obs=0 %s rc=%d  table cells overwritten past end: %d (first 6: %s)  scaling[0..2] %s"
          % (name, rc, int((fbuf != SENT).sum()), fbuf[:6], sbuf[:3]))
# For bward, the write to log_scaling_factors[-1] lands before the buffer; show it with an offset view
sbuf = np.full(64, SENT); fbuf = np.full(64, SENT)
with np.errstate(divide='ignore'):
    rc = lib.bward_log(T, np.zeros(1), endp, ns, ssb, 0, chi, nchi, fbuf[8:8], sbuf[8:8], ABI)
print("n_obs=0 bward_log with offset views: sbuf[7] (index -1 of the passed array) =", sbuf[7], " rc", rc)

# ---- impossible sequence: position 1 has every emitting state at -inf -> rc -2
n = 3
le = np.zeros((n, ns)); le[1, :ssb] = -np.inf
f = np.full((n, ns), -np.inf); b = np.full((n, ns), -np.inf); p = np.zeros((n, ns))
fs = np.zeros(n); bs = np.zeros(n)
with np.errstate(divide='ignore'):
    rf = lib.fward_log(init, T, le, endp, ns, ssb, n, par, npar, f, fs, ABI)
    rb = lib.bward_log(T, le, endp, ns, ssb, n, chi, nchi, b, bs, ABI)
rp = lib.posterior_decoding_log(f, b, ns, n, ssb, p, ABI)
print("impossible: rc fward %d bward %d posterior %d | fwd offsets %s | posterior rows sums %s | any NaN f/b/p %s"
      % (rf, rb, rp, fs, p.sum(1), bool(np.isnan(f).any() or np.isnan(b).any() or np.isnan(p).any())))

# ---- unreachable only (emissions finite, but transitions make a later row impossible)
le = np.zeros((n, ns)); le[1, 0] = -np.inf; le[1, 1] = -np.inf   # only state 2 at t=1; 0->2 impossible
f = np.full((n, ns), -np.inf); b = np.full((n, ns), -np.inf); p = np.zeros((n, ns))
with np.errstate(divide='ignore'):
    rf = lib.fward_log(init, T, le, endp, ns, ssb, n, par, npar, f, fs, ABI)
    rb = lib.bward_log(T, le, endp, ns, ssb, n, chi, nchi, b, bs, ABI)
rp = lib.posterior_decoding_log(f, b, ns, n, ssb, p, ABI)
print("unreachable: rc fward %d bward %d posterior %d | posterior row sums %s" % (rf, rb, rp, p.sum(1)))

# ---- ABI guard
print("wrong abi rc:", lib.posterior_decoding_log(f, b, ns, n, ssb, p, 123))
