"""Exact C1 check for FiberCOP, one array task: rebuild each segment's emissions with the model's own
pipeline, then run the ORIGINAL C code and the audit's C1-FIXED C code on the same arrays and compare.

    cd analysis && python posterior_fix/fibercop_audit/c1scan/scan_genome.py \
        <tree> <traindir> <coords.tsv> <workdir> <outdir> <idx> <total>

Why not the MNase-style bound (max emission at t x max emission at t+1 < DBL_MIN): in FiberCOP C1 fires
far above that line (bo09 chrXIV:413,386, smallest pair product in the segment 1e-120). The cause there is
normalize() padding every silent cell with DBL_MIN before dividing by a tiny emitting sum; see README.md.
So this runs the recursion itself.

Per segment (seg_<idx>_<total>.tsv):
  n_events         positions where the backward scaling factor differs (|dlog| > 1e-6 nats) = C1 fired
  max_dnats        largest such difference
  bad_orig         original posterior rows with |emitting row sum - 1| > 1e-4 (count_calls' tolerance)
  bad_fixed        same for the C1-fixed library (still has fward's DBL_MIN padding, so non-C1 causes remain)
  bad_c1           rows bad in the original but not in the fixed library = rows C1 alone broke
  min_scale        smallest backward emitting sum S in the segment
Per event (events_<idx>_<total>.tsv): chrom, pos (1-based), dnats, S_orig, S_fixed.
Nothing is decoded; the pipeline's own recursion is never called. md5 of both .so checked at start and end.
"""
import sys, os, ctypes, hashlib, time
import numpy as np, pandas as pd
from ctypes import c_int, c_double
from numpy.ctypeslib import ndpointer

tree, traindir, coords_f, workdir, outdir, idx, total = sys.argv[1:8]
idx, total = int(idx), int(total)
FORK = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude"
LIB_ORIG = FORK + "/pkg/robocop/librobocop_prenorm.so"      # pre-cutover production binary (md5 6a0724bf)
LIB_FIX = ("/usr/project/xtmp/nd141/programs/robocop_c_code_test/robocop_c_audit/walkthrough/variants/"
           "librobocop_fixed.so")                             # audit C1 fix (md5 6dbd5e6d)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
MD5 = {p: md5(p) for p in (LIB_ORIG, LIB_FIX)}
D = ndpointer(np.double, flags="C_CONTIGUOUS"); L = ndpointer(np.int64, flags="C_CONTIGUOUS")
LDp = ndpointer(np.longdouble, flags="C_CONTIGUOUS")


def bind(path):
    lib = ctypes.CDLL(path)
    lib.find_parents_and_children.argtypes = [L] * 4 + [c_int, c_int, D]
    lib.fward.argtypes = [D, D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.bward.argtypes = [D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.calc_sr.argtypes = [D, D, c_int, LDp]; lib.calc_sr.restype = c_double
    lib.posterior_decoding.argtypes = [D, D, D, LDp, c_int, c_int, D]   # both libraries: 7-arg, un-normalised
    return lib


LO, LF = bind(LIB_ORIG), bind(LIB_FIX)
coords = pd.read_csv(coords_f, sep="\t")
os.makedirs(outdir, exist_ok=True)
SEG, EV = [], []

sys.path.insert(0, tree)
import robocop
import run_robocop
assert os.path.abspath(os.path.dirname(robocop.__file__)).startswith(os.path.abspath(tree)), robocop.__file__


def check(d, segment, dshared):
    t0 = time.time()
    ns, ssb, nv = int(dshared["n_states"]), int(dshared["silent_states_begin"]), int(dshared["n_vars"])
    n = int(dshared["info_file"]["segment_%d/" % segment].attrs["n_obs"])
    E = np.ascontiguousarray(d["emission"], dtype=np.double)
    T = np.ascontiguousarray(dshared["transition_matrix"], dtype=np.double)
    init = np.ascontiguousarray(dshared["initial_probs"], dtype=np.double)
    endp = np.ascontiguousarray(dshared["end_probs"], dtype=np.double)
    ms = np.ascontiguousarray(dshared["tf_starts"], dtype=np.int64); ml = np.ascontiguousarray(dshared["tf_lens"], dtype=np.int64)
    a = (int(dshared["n_tfs"]), int(dshared["nuc_present"]), int(dshared["nuc_start"]), int(dshared["nuc_len"]))
    par = np.zeros((ns, ns), np.int64); chi = np.zeros((ns, ns), np.int64)
    npar = np.zeros(ns, np.int64); nchi = np.zeros(ns, np.int64)
    LO.find_parents_and_children(par, chi, npar, nchi, ns, ssb, T)
    f = np.zeros((n, ns)); fs = np.zeros(n)
    LO.fward(init, T, E, endp, ns, ssb, n, nv, par, npar, ms, ml, *a, f, fs)   # fward is identical in both libraries
    out = {}
    for name, lib in (("orig", LO), ("fix", LF)):
        b = np.zeros((n, ns)); bs = np.zeros(n); sr = np.zeros(n, dtype=np.longdouble); p = np.zeros((n, ns))
        lib.bward(T, E, endp, ns, ssb, n, nv, chi, nchi, ms, ml, *a, b, bs)
        lib.calc_sr(fs, bs, n, sr)
        lib.posterior_decoding(f, b, bs, sr, ns, n, p)
        rs = p[:, :ssb].sum(1)
        out[name] = (bs, ~(np.abs(rs - 1) <= 1e-4))          # NaN/inf row sums count as bad
        del b, p
    (so, bad_o), (sf, bad_f) = out["orig"], out["fix"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dl = np.log(so) - np.log(sf)
    ev = np.flatnonzero(~(np.abs(dl) <= 1e-6))
    chrom, start = coords.loc[segment, "chr"], int(coords.loc[segment, "start"])
    for i in ev:
        EV.append(dict(chrom=chrom, pos=start + int(i), segment=segment, dnats=float(dl[i]), S_orig=float(so[i]), S_fixed=float(sf[i])))
    SEG.append(dict(segment=segment, chrom=chrom, start=start, n_obs=n, n_events=len(ev),
                    max_dnats=float(np.nanmax(np.abs(dl))) if len(ev) else 0.0,
                    bad_orig=int(bad_o.sum()), bad_fixed=int(bad_f.sum()), bad_c1=int((bad_o & ~bad_f).sum()),
                    min_scale=float(so.min()), seconds=round(time.time() - t0, 1)))
    print("segment %d %s:%d  events %d  bad orig/fixed/c1-only %d/%d/%d  (%.0fs)" % (
        segment, chrom, start, len(ev), bad_o.sum(), bad_f.sum(), (bad_o & ~bad_f).sum(), time.time() - t0), flush=True)
    dshared["info_file"]["segment_" + str(segment)].attrs["log_likelihood"] = np.nan   # readers of the throwaway h5 expect it


robocop.posterior_forward_backward = check
run_robocop.run_robocop_without_em(coords_f, traindir, workdir, idx=idx, total=total)
assert {p: md5(p) for p in MD5} == MD5, "a library changed during the run"
tag = "%d_%d" % (idx, total)
pd.DataFrame(SEG).to_csv(os.path.join(outdir, "seg_%s.tsv" % tag), sep="\t", index=False)
pd.DataFrame(EV, columns=["chrom", "pos", "segment", "dnats", "S_orig", "S_fixed"]).to_csv(
    os.path.join(outdir, "events_%s.tsv" % tag), sep="\t", index=False)
print("done", len(SEG), "segments,", len(EV), "events")
