"""Run every linear / log decoder variant in-process on ONE captured real segment and compare.

    cd analysis && python posterior_fix/fibercop_audit/realseg/compare_variants.py <cap.npz> <out.json> [--save posts.npz]

<cap.npz> is a harness_capture.py capture of a stock (linear, floored-emission) tree. Variants:
  ORIG     pkg/robocop/librobocop_prenorm.so      upstream algo.c, 7-arg posterior_decoding
  ORIG+B   ORIG posterior, each row / its emitting-state sum (in Python)
  FIXB     pkg/robocop/librobocop.so              production fix B, posterior_decoding_v2 (8 args)
  C1FIX    audit variants/librobocop_fixed.so     bward rescales emitting cells before the silent loop
  C1FIX+B  C1FIX posterior, each row / its emitting-state sum (in Python)
  LOG-D0   pkg/robocop/librobocop_logfull.so      fward_log/bward_log/posterior_decoding_log on
                                                  log(E) summed over layers, silent columns 0
The reference is LOG-D0. Bindings mirror fiber_model.py load_lin/run_lin/run_logfull and
harness_gates.py run_linear (= robocop.py call sequence). Read-only on every input.
"""
import sys, json, argparse, hashlib, time, ctypes
import numpy as np
from ctypes import c_int, c_double
from numpy.ctypeslib import ndpointer

FORK = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude"
WT = "/usr/project/xtmp/nd141/programs/robocop_c_code_test/robocop_c_audit/walkthrough"
LIBS = dict(ORIG=FORK + "/pkg/robocop/librobocop_prenorm.so", FIXB=FORK + "/pkg/robocop/librobocop.so",
            C1FIX=WT + "/variants/librobocop_fixed.so", LOG=FORK + "/pkg/robocop/librobocop_logfull.so")
ABI = 20261005
D = ndpointer(np.double, flags="C_CONTIGUOUS"); L = ndpointer(np.int64, flags="C_CONTIGUOUS")
LDp = ndpointer(np.longdouble, flags="C_CONTIGUOUS")

ap = argparse.ArgumentParser(); ap.add_argument('cap'); ap.add_argument('json'); ap.add_argument('--save')
a = ap.parse_args()
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
md5_before = {k: md5(v) for k, v in LIBS.items()}

z = np.load(a.cap)
ns, ssb, n, nv = int(z['n_states']), int(z['silent_states_begin']), int(z['n_obs']), int(z['n_vars'])
E = np.ascontiguousarray(z['emission'], dtype=np.double)
T = np.ascontiguousarray(z['transition_matrix'], dtype=np.double)
init = np.ascontiguousarray(z['initial_probs'], dtype=np.double); endp = np.ascontiguousarray(z['end_probs'], dtype=np.double)
ms = np.ascontiguousarray(z['tf_starts'], dtype=np.int64); ml = np.ascontiguousarray(z['tf_lens'], dtype=np.int64)
n_tfs, nuc_present, nuc_start, nuc_len = int(z['n_tfs']), int(z['nuc_present']), int(z['nuc_start']), int(z['nuc_len'])
print("capture", a.cap, "E", E.shape, "ns", ns, "ssb", ssb, "n", n, "tree", str(z['tree']), flush=True)
R = dict(capture=a.cap, tree=str(z['tree']), n_obs=n, n_states=ns, ssb=ssb, n_vars=nv, md5=md5_before,
         emission_zero_cells=int((E == 0).sum()), emission_floor_cells=int((E == 1e-30).sum()))


def bind_lin(path, v2):
    lib = ctypes.CDLL(path)
    lib.find_parents_and_children.argtypes = [L] * 4 + [c_int, c_int, D]
    lib.fward.argtypes = [D, D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.bward.argtypes = [D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.calc_sr.argtypes = [D, D, c_int, LDp]; lib.calc_sr.restype = c_double
    if v2:
        lib.posterior_decoding_v2.argtypes = [D, D, D, LDp, c_int, c_int, c_int, D]
    else:
        lib.posterior_decoding.argtypes = [D, D, D, LDp, c_int, c_int, D]
    return lib


def graph(lib):
    par = np.zeros((ns, ns), np.int64); chi = np.zeros((ns, ns), np.int64)
    npar = np.zeros(ns, np.int64); nchi = np.zeros(ns, np.int64)
    lib.find_parents_and_children(par, chi, npar, nchi, ns, ssb, T)
    return par, chi, npar, nchi


def run_lin(path, v2):
    lib = bind_lin(path, v2); par, chi, npar, nchi = graph(lib)
    f = np.zeros((n, ns)); b = np.zeros((n, ns)); p = np.zeros((n, ns))
    fs = np.zeros(n); bs = np.zeros(n); sr = np.zeros(n, dtype=np.longdouble)
    lib.fward(init, T, E, endp, ns, ssb, n, nv, par, npar, ms, ml, n_tfs, nuc_present, nuc_start, nuc_len, f, fs)
    lib.bward(T, E, endp, ns, ssb, n, nv, chi, nchi, ms, ml, n_tfs, nuc_present, nuc_start, nuc_len, b, bs)
    ll = lib.calc_sr(fs, bs, n, sr)
    if v2:
        lib.posterior_decoding_v2(f, b, bs, sr, ns, n, ssb, p)
    else:
        lib.posterior_decoding(f, b, bs, sr, ns, n, p)
    with np.errstate(divide='ignore'):
        llb = float(np.log(bs).sum())
    return p, dict(loglik_calc_sr=float(ll), loglik_sum_log_bs=llb)


def run_log():
    lib = ctypes.CDLL(LIBS['LOG'])
    lib.find_parents_and_children.argtypes = [L] * 4 + [c_int, c_int, D]
    lib.fward_log.argtypes = [D, D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.fward_log.restype = c_int
    lib.bward_log.argtypes = [D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.bward_log.restype = c_int
    lib.posterior_decoding_log.argtypes = [D, D, c_int, c_int, c_int, D, c_int]; lib.posterior_decoding_log.restype = c_int
    with np.errstate(divide='ignore'):
        le = np.ascontiguousarray(np.log(E).sum(0))
    le[:, ssb:] = 0.0
    par, chi, npar, nchi = graph(lib)
    lf = np.full((n, ns), -np.inf); lb = np.full((n, ns), -np.inf); sf = np.zeros(n); sb = np.zeros(n); p = np.zeros((n, ns))
    rc = (lib.fward_log(init, T, le, endp, ns, ssb, n, par, npar, lf, sf, ABI),
          lib.bward_log(T, le, endp, ns, ssb, n, chi, nchi, lb, sb, ABI),
          lib.posterior_decoding_log(lf, lb, ns, n, ssb, p, ABI))
    assert rc == (ABI,) * 3, rc
    return p, dict(loglik_sum_log_offsets_fwd=float(sf.sum()), loglik_sum_log_offsets_bwd=float(sb.sum()),
                   rc=list(rc))


def rownorm(p):
    s = p[:, :ssb].sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return p / s


P, LL, SEC = {}, {}, {}
for name, fn in (('LOG-D0', run_log), ('ORIG', lambda: run_lin(LIBS['ORIG'], False)),
                 ('FIXB', lambda: run_lin(LIBS['FIXB'], True)), ('C1FIX', lambda: run_lin(LIBS['C1FIX'], False))):
    t0 = time.time(); P[name], LL[name] = fn(); SEC[name] = round(time.time() - t0, 1)
    print(name, "done", SEC[name], "s", LL[name], flush=True)
P['ORIG+B'] = rownorm(P['ORIG']); LL['ORIG+B'] = LL['ORIG']
P['C1FIX+B'] = rownorm(P['C1FIX']); LL['C1FIX+B'] = LL['C1FIX']
del E

ref = P['LOG-D0'][:, :ssb]
ll_ref = LL['LOG-D0']['loglik_sum_log_offsets_fwd']
V = {}
for name in ('ORIG', 'ORIG+B', 'FIXB', 'C1FIX', 'C1FIX+B', 'LOG-D0'):
    pe = P[name][:, :ssb]; ps = P[name][:, ssb:]
    fin = np.isfinite(pe).all(1)
    d = np.abs(pe - ref).max(1)                    # NaN rows propagate -> counted separately
    rs = pe.sum(1)
    ll = LL[name].get('loglik_calc_sr', ll_ref)
    V[name] = dict(seconds=SEC.get(name), loglik=LL[name], loglik_minus_logD0=float(ll - ll_ref),
                   max_abs_vs_logD0=float(np.nanmax(d)), rows_gt_0p01=int((d > 0.01).sum()),
                   rows_gt_1em6=int((d > 1e-6).sum()), rows_nonfinite=int((~fin).sum()),
                   rowsum_min=float(np.nanmin(rs)), rowsum_max=float(np.nanmax(rs)),
                   silent_max=float(np.nanmax(ps)), silent_rows_gt1=int((np.nanmax(ps, 1) > 1).sum()),
                   rows_gt_0p01_idx_first10=np.flatnonzero(d > 0.01)[:10].tolist())
    print(name, json.dumps({k: v for k, v in V[name].items() if k != 'loglik'}), flush=True)
R['variants'] = V
# pairwise emitting max-abs among the normalised variants
NORM = ['ORIG+B', 'FIXB', 'C1FIX+B', 'LOG-D0']
R['pairwise_emitting_max_abs'] = {f"{x} vs {y}": float(np.nanmax(np.abs(P[x][:, :ssb] - P[y][:, :ssb])))
                                  for i, x in enumerate(NORM) for y in NORM[i + 1:]}
R['pairwise_rows_gt_0p01'] = {f"{x} vs {y}": int((np.abs(P[x][:, :ssb] - P[y][:, :ssb]).max(1) > 0.01).sum())
                              for i, x in enumerate(NORM) for y in NORM[i + 1:]}
R['md5_after'] = {k: md5(v) for k, v in LIBS.items()}
assert R['md5_after'] == md5_before
print(json.dumps(R['pairwise_emitting_max_abs']), json.dumps(R['pairwise_rows_gt_0p01']))
json.dump(R, open(a.json, 'w'), indent=1)
if a.save:
    np.savez(a.save, **{k.replace('+', '_').replace('-', '_'): v for k, v in P.items() if k not in ('ORIG',)})
print("wrote", a.json)
