"""Toy 4-state HMM run through the real C decoders, to show what normalize()'s +DBL_MIN does.

States: 0 bg, 1 motif col 1, 2 motif col 2 (emitting, ssb = 3), 3 silent hub.
Edges: bg->bg 0.8, bg->hub 0.2, m1->m2 1, m2->hub 1, hub->bg 0.5, hub->m1 0.5.
So m2 is ONLY reachable from m1 at the previous position.

Libraries:
  WITH    = pkg/robocop/librobocop.so            (production fix B, has +DBL_MIN)
  WITHOUT = dblmin_history/build/lib_nodblmin.so (same algo.c, '+ DBL_MIN' -> '+ 0.0')
  LOG     = pkg/robocop/librobocop_logfull.so    (log space, algo_log.c)

Cases:
  A  position 3 has every emitting emission = 0 (an 'N' base / forbidden everywhere)
  B  every position has some allowed state, but position 3 allows only m2 and position 2
     forbids m1 -> no valid path exists, although no single row is all zero
"""
import sys, os, ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import c_int

R = '/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude'
HERE = os.path.dirname(os.path.abspath(__file__))
ABI = 20261005
D = ndpointer(np.double, flags='C_CONTIGUOUS'); L = ndpointer(np.longlong, flags='C_CONTIGUOUS')
LD = ndpointer(np.longdouble, flags='C_CONTIGUOUS')
np.set_printoptions(precision=4, linewidth=140)

ns, ssb, n = 4, 3, 6
T = np.zeros((ns, ns)); T[0, 0] = .8; T[0, 3] = .2; T[1, 2] = 1; T[2, 3] = 1; T[3, 0] = .5; T[3, 1] = .5
init = np.array([.5, .5, 0., 0.]); endp = np.array([1., 0., 1., 0.])

def emis(case):
    E = np.ones((1, n, ns))
    E[0, :, 0] = 0.6; E[0, :, 1] = 0.3; E[0, :, 2] = 0.3      # silent column stays 1.0 (bc.c)
    if case == 'A':
        E[0, 3, :ssb] = 0.0
    if case == 'B':
        E[0, 2, 1] = 0.0                     # m1 forbidden at pos 2
        E[0, 3, 0] = 0.0; E[0, 3, 1] = 0.0   # only m2 allowed at pos 3 -> needs m1 at pos 2
    return np.ascontiguousarray(E)

def graph(lib):
    P = np.zeros((ns, ns), np.longlong); C = np.zeros((ns, ns), np.longlong)
    nP = np.zeros(ns, np.longlong); nC = np.zeros(ns, np.longlong)
    lib.find_parents_and_children.argtypes = [L, L, L, L, c_int, c_int, D]
    lib.find_parents_and_children(P, C, nP, nC, ns, ssb, T)
    return P, C, nP, nC

def linear(libpath, E):
    lib = ctypes.CDLL(libpath); P, C, nP, nC = graph(lib)
    z = np.zeros(1, np.longlong)
    f = np.zeros((n, ns)); b = np.zeros((n, ns)); sf = np.zeros(n); sb = np.zeros(n)
    lib.fward.argtypes = [D, D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.fward(init, T, E, endp, ns, ssb, n, 1, P, nP, z, z, 0, 0, 0, 0, f, sf)
    lib.bward.argtypes = [D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.bward(T, E, endp, ns, ssb, n, 1, C, nC, z, z, 0, 0, 0, 0, b, sb)
    p = np.zeros((n, ns)); sr = np.ones(n, np.longdouble)
    lib.posterior_decoding_v2.argtypes = [D, D, D, LD, c_int, c_int, c_int, D]
    lib.posterior_decoding_v2(f, b, sb, sr, ns, n, ssb, p)
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = np.log(sf).sum()
    return f, sf, p, ll

def logspace(E):
    lib = ctypes.CDLL(R + '/pkg/robocop/librobocop_logfull.so'); P, C, nP, nC = graph(lib)
    with np.errstate(divide='ignore'):
        le = np.ascontiguousarray(np.log(E).sum(0))
    out = {}
    # the Python pre-check, copied from pkgvar/seq_maskoff_tune75_logfull/robocop/robocop.py:976-979
    _all_neg_inf = np.all(np.isneginf(le[:, :ssb]), axis=1)
    try:
        assert not _all_neg_inf.any(), \
            "logfull: %d positions have every emitting state at -inf (first at %d)" % (
                _all_neg_inf.sum(), int(np.argmax(_all_neg_inf)))
        out['python_precheck'] = 'passed'
    except AssertionError as e:
        out['python_precheck'] = 'AssertionError: ' + str(e)
    f = np.full((n, ns), -np.inf); b = np.full((n, ns), -np.inf); sf = np.zeros(n); sb = np.zeros(n)
    p = np.zeros((n, ns))
    lib.fward_log.argtypes = [D, D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.fward_log.restype = c_int
    lib.bward_log.argtypes = [D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.bward_log.restype = c_int
    lib.posterior_decoding_log.argtypes = [D, D, c_int, c_int, c_int, D, c_int]; lib.posterior_decoding_log.restype = c_int
    with np.errstate(divide='ignore'):
        out['rc_fward_log'] = lib.fward_log(init, T, le, endp, ns, ssb, n, P, nP, f, sf, ABI)
        out['rc_bward_log'] = lib.bward_log(T, le, endp, ns, ssb, n, C, nC, b, sb, ABI)
        out['rc_posterior_log'] = lib.posterior_decoding_log(f, b, ns, n, ssb, p, ABI)
    out['loglik'] = sf.sum(); out['p'] = p
    return out

sys.stdout.flush()
for case in ['A', 'B']:
    E = emis(case)
    print('=' * 100); print('CASE', case, __doc__.split('Cases:')[1].split('\n')[1 if case == 'A' else 2].strip())
    for name, path in [('WITH DBL_MIN (librobocop.so)', R + '/pkg/robocop/librobocop.so'),
                       ('WITHOUT DBL_MIN (lib_nodblmin.so)', HERE + '/build/lib_nodblmin.so')]:
        with np.errstate(all='ignore'):
            f, sf, p, ll = linear(path, E)
        print('--', name)
        print('   forward row 3 (normalised):', f[3], '  scaling factor sf[3] =', sf[3])
        print('   forward row 4 (normalised):', f[4])
        print('   posterior rows 2..4 (emitting cols bg,m1,m2):'); print(p[2:5, :ssb])
        print('   log-likelihood =', ll, '  any NaN in posterior:', bool(np.isnan(p).any()))
    sys.stdout.flush()
    o = logspace(E)
    print('-- LOG SPACE (librobocop_logfull.so)')
    print('   python pre-check:', o['python_precheck'])
    sys.stdout.flush()
    print('   rc fward_log=%d bward_log=%d posterior_decoding_log=%d  (success = %d)' % (
        o['rc_fward_log'], o['rc_bward_log'], o['rc_posterior_log'], ABI))
    print('   robocop.py would then fail: assert _rc == ROBOCOP_LOG_ABI ->', o['rc_fward_log'] == ABI)
