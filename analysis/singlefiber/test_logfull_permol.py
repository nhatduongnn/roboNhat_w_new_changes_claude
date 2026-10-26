#!/usr/bin/env python
"""Unit tests for the log-space per-read decoder: "no information" vs "forbidden" vs "impossible".

In log space there is no DBL_MIN and no 1e-30 floor:
  * no information  -> 0.0 (log 1): uncovered / ambiguous positions, non-target bases (exact mode),
                       silent states. Must contribute NOTHING.
  * forbidden       -> -inf in individual cells (PWM zeros, hard masks). The state gets posterior 0.
  * impossible      -> a position where EVERY emitting state is -inf. Must stop loudly: the tree's
                       Python assert fires before C; C itself returns -2 (algo_log.c:138, :161-164).

Tests (each records pass/fail and numbers into --out JSON; any failure exits 1):
  T1  real read with a 1,500 bp stretch blanked to "no call": fiber layers in the stretch are
      exactly 0.0 on every cell (exact) / 0.0 on target bases and log(1e-30) elsewhere (floored);
      the stretch's total log emission equals the sequence layer alone; decode valid.
  T2  a read with NO calls at all decodes bit-identically to a decode whose layers 1-6 are set
      to 0.0 by hand -- the per-read code adds nothing for "no information".
  T3  forbidden cells: every TF state at one position set to -inf -> decode fine, those states get
      posterior exactly 0 there, row still sums to 1.
  T4  impossible row via the per-read driver: AssertionError ("every emitting state at -inf")
      propagates out of decode_one (nothing catches it).
  T5  impossible row straight into C (fward_log / bward_log, bypassing Python): rc == -2.
  T6  Bernoulli guard: a ps of exactly 0 or 1 is refused (AssertionError), calls outside -1/0/1 too.
  T7  fitted ps range for arms A, B(0.3/0.5/0.7), C, both strands: report, and whether any is 0 or 1.
  Every successful decode: no NaN; emitting rows sum to 1 within 1e-9 (dense posterior; the
  1e-4 save floor is not applied here); silent columns exactly 0.

    ROBOCOP_LOG_EMISSION=exact python singlefiber/test_logfull_permol.py --out .../unit_tests_exact.json
"""
import argparse
import ctypes
import json
import os
import pickle
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
os.chdir(ANALYSIS)
sys.path.insert(0, 'pkgvar/permol_seq_maskoff_logfull/')
sys.path.insert(0, HERE)

import robocop                                                    # noqa: E402
from robocop.utils.getNucleotides import getNucleotideSequence     # noqa: E402
from numpy.ctypeslib import ndpointer                              # noqa: E402

import collapse as C                                              # noqa: E402
import permol_params as PP                                        # noqa: E402
import read_calls as RC                                           # noqa: E402
import run_permol_window_logfull as RPW                           # noqa: E402

WINDOW = ("chrI", 60001, 65000)
FLOOR = float(np.log(1e-30))


def setup():
    dshared = pickle.load(open("robocop_train_fiberonly/HMMconfig.pkl", "rb"))
    dshared["robocopC"] = RPW.LOGFULL_SO
    dshared["tmpDir"] = "/tmp/"
    dshared["nucleotides"] = 1
    dshared["cache_parents_children"] = True
    dshared["info_file"] = None
    return dshared


def valid(p, ssb):
    v = RPW.posterior_validity(p, ssb)
    ok = v["n_nan"] == 0 and v["max_row_sum_dev"] < RPW.ROW_SUM_TOL and v["max_silent"] == 0.0
    return ok, v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bam", default=RPW.BAM)
    a = ap.parse_args(argv)
    mode = robocop.log_emission_mode()
    assert RPW.so_md5(RPW.LOGFULL_SO) == RPW.LOGFULL_SO_MD5
    RPW.install_posterior_grab()
    dshared = setup()
    ssb = dshared["silent_states_begin"]
    M = C.collapse_matrix(dshared)
    cols = C.column_names(dshared)
    keep_idx = np.arange(len(cols))
    ps_w, _s, _i = PP.arm_ps(dshared, "watson_signal", "A")
    ps_c, _s, _i = PP.arm_ps(dshared, "crick_signal", "A")

    chrom, ws, we = WINDOW
    flank = 400
    es = ws - 1 - flank
    codes_ext = getNucleotideSequence(RPW.FASTA, chrom, es + 1, we + flank)
    mols, es2, ee2, _st = RC.read_molecules(a.bam, chrom, ws, we, codes_ext, flank=flank)
    # first molecule spanning the whole window
    m = next(x for x in mols if x.ref_start0 <= ws - 1 and x.ref_end0 >= we)
    lo, hi = (ws - 1) - es, we - es
    codes = np.asarray(codes_ext[lo:hi])
    calls = m.calls[lo:hi].copy()
    no = np.full(hi - lo, RC.NO_OBS, dtype=np.int8)
    on = codes == (RC.CODE_A if m.strand == "watson" else RC.CODE_T)
    res = {"log_emission": mode, "so_md5": RPW.LOGFULL_SO_MD5, "molecule": m.name,
           "strand": m.strand, "n_informative_in_window": int((calls != RC.NO_OBS).sum()),
           "tests": {}}
    T = res["tests"]

    def run(cw, cc, hook=None):
        return RPW.decode_one(dshared, chrom, ws, we, codes, ps_w, ps_c, cw, cc, M, keep_idx,
                              emission_hook=hook)

    def lanes(c):
        return (c, no) if m.strand == "watson" else (no, c)

    # ---- T1 -----------------------------------------------------------------------
    s0, s1 = 1500, 3000
    c1 = calls.copy()
    blanked = int((c1[s0:s1] != RC.NO_OBS).sum())
    c1[s0:s1] = RC.NO_OBS
    grab = {}

    def h1(d):
        e = d["emission"]
        grab["l5"] = e[5, s0:s1, :ssb].copy()
        grab["l6"] = e[6, s0:s1, :ssb].copy()
        grab["l0"] = e[0, s0:s1, :].copy()
        grab["tot"] = e[:, s0:s1, :].sum(axis=0)
        grab["fiber_silent_max"] = float(np.abs(e[1:, :, ssb:]).max())
    p1, _op, ll1 = run(*lanes(c1), hook=h1)
    tgt_w = codes[s0:s1] == RC.CODE_A
    tgt_c = codes[s0:s1] == RC.CODE_T
    nontgt_val = 0.0 if mode == "exact" else FLOOR
    l5_ok = np.all(grab["l5"][tgt_w] == 0.0) and np.all(grab["l5"][~tgt_w] == nontgt_val)
    l6_ok = np.all(grab["l6"][tgt_c] == 0.0) and np.all(grab["l6"][~tgt_c] == nontgt_val)
    if mode == "exact":
        seq_only = bool(np.array_equal(grab["tot"], grab["l0"]))
    else:   # floored: sequence + one per-position constant log(1e-30) per non-target layer
        const = (~tgt_w).astype(float) * FLOOR + (~tgt_c).astype(float) * FLOOR
        exp_tot = grab["l0"].copy()
        exp_tot[:, :ssb] += const[:, None]
        seq_only = bool(np.allclose(grab["tot"], exp_tot, rtol=0, atol=1e-9))
    ok_v, v1 = valid(p1, ssb)
    T["T1_no_coverage_stretch"] = dict(
        stretch=[ws + s0, ws + s1 - 1], calls_blanked=blanked,
        layer5_values_ok=bool(l5_ok), layer6_values_ok=bool(l6_ok),
        neg_inf_cells_in_fiber_layers=int(np.isneginf(grab["l5"]).sum() + np.isneginf(grab["l6"]).sum()),
        total_emission_in_stretch_is_sequence_only=seq_only,
        fiber_layers_silent_cols_max_abs=grab["fiber_silent_max"],
        loglik=ll1, validity=v1,
        passed=bool(l5_ok and l6_ok and seq_only and ok_v and grab["fiber_silent_max"] == 0.0))

    # ---- T2 -----------------------------------------------------------------------
    p2a, _o, ll2a = run(no, no)

    def h2(d):
        d["emission"][1:7] = 0.0
    p2b, _o, ll2b = run(no, no, hook=h2) if mode == "exact" else (None, None, None)
    if mode == "exact":
        d2 = float(np.abs(p2a - p2b).max())
        ok_v2, v2 = valid(p2a, ssb)
        T["T2_uncovered_read_equals_sequence_only"] = dict(
            max_abs_diff=d2, loglik_no_calls=ll2a, loglik_layers_zeroed=ll2b, validity=v2,
            passed=bool(d2 == 0.0 and ll2a == ll2b and ok_v2))
    else:
        # floored: compare after removing the per-position constant -> posteriors equal
        def h2f(d):
            e = d["emission"]
            e[1:7] = 0.0
            e[5, codes != RC.CODE_A, :ssb] = FLOOR
            e[6, codes != RC.CODE_T, :ssb] = FLOOR
        p2b, _o, ll2b = run(no, no, hook=h2f)
        d2 = float(np.abs(p2a - p2b).max())
        ok_v2, v2 = valid(p2a, ssb)
        T["T2_uncovered_read_equals_sequence_only"] = dict(
            max_abs_diff=d2, loglik_no_calls=ll2a, loglik_layers_zeroed=ll2b, validity=v2,
            passed=bool(d2 == 0.0 and ok_v2))

    # ---- T3 -----------------------------------------------------------------------
    pos3 = 2000
    tf_lo, tf_hi = 1, int(dshared["nuc_start"])

    def h3(d):
        d["emission"][5, pos3, tf_lo:tf_hi] = -np.inf
    p3, _o, ll3 = run(*lanes(calls), hook=h3)
    ok_v3, v3 = valid(p3, ssb)
    tf_mass = float(p3[pos3, tf_lo:tf_hi].sum())
    T["T3_forbidden_cells"] = dict(position=ws + pos3, forbidden_states=[tf_lo, tf_hi - 1],
                                   posterior_mass_on_forbidden=tf_mass,
                                   row_sum=float(p3[pos3, :ssb].sum()), validity=v3,
                                   passed=bool(tf_mass == 0.0 and ok_v3))

    # ---- T4 -----------------------------------------------------------------------
    pos4 = 2500

    def h4(d):
        d["emission"][5, pos4, :ssb] = -np.inf
    try:
        run(*lanes(calls), hook=h4)
        T["T4_impossible_row_python"] = dict(raised=None, passed=False)
    except AssertionError as e:
        msg = str(e)
        T["T4_impossible_row_python"] = dict(position=ws + pos4, raised="AssertionError",
                                             message=msg,
                                             passed="every emitting state at -inf" in msg)
    except Exception as e:   # anything else is a different failure; record it
        T["T4_impossible_row_python"] = dict(raised=repr(e), passed=False,
                                             tb=traceback.format_exc())

    # ---- T5 -----------------------------------------------------------------------
    lib = ctypes.CDLL(RPW.LOGFULL_SO)
    n_states = dshared["n_states"]
    n_obs = 60
    tm = dshared["transition_matrix"].astype(np.double)
    pm = np.zeros((n_states, n_states), dtype=np.longlong)
    cm = np.zeros((n_states, n_states), dtype=np.longlong)
    npar = np.zeros(n_states, dtype=np.longlong)
    nch = np.zeros(n_states, dtype=np.longlong)
    lib.find_parents_and_children.argtypes = [ndpointer(np.longlong, flags="C_CONTIGUOUS")] * 4 + \
        [ctypes.c_int, ctypes.c_int, ndpointer(np.double, flags="C_CONTIGUOUS")]
    lib.find_parents_and_children(pm, cm, npar, nch, n_states, ssb, tm)
    le = np.zeros((n_obs, n_states))
    le[30, :ssb] = -np.inf
    ft = np.full((n_obs, n_states), -np.inf)
    bt = np.full((n_obs, n_states), -np.inf)
    fs = np.zeros(n_obs)
    bs = np.zeros(n_obs)
    D = ndpointer(np.double, flags="C_CONTIGUOUS")
    L = ndpointer(np.longlong, flags="C_CONTIGUOUS")
    lib.fward_log.argtypes = [D, D, D, D, ctypes.c_int, ctypes.c_int, ctypes.c_int, L, L, D, D, ctypes.c_int]
    lib.fward_log.restype = ctypes.c_int
    lib.bward_log.argtypes = [D, D, D, ctypes.c_int, ctypes.c_int, ctypes.c_int, L, L, D, D, ctypes.c_int]
    lib.bward_log.restype = ctypes.c_int
    with np.errstate(divide="ignore"):
        rcf = lib.fward_log(dshared["initial_probs"].astype(np.double), tm, le,
                            dshared["end_probs"].astype(np.double), n_states, ssb, n_obs,
                            pm, npar, ft, fs, robocop.ROBOCOP_LOG_ABI)
        rcb = lib.bward_log(tm, le, dshared["end_probs"].astype(np.double), n_states, ssb, n_obs,
                            cm, nch, bt, bs, robocop.ROBOCOP_LOG_ABI)
        le[30, :ssb] = 0.0
        ft[:] = -np.inf
        rc_ok = lib.fward_log(dshared["initial_probs"].astype(np.double), tm, le,
                              dshared["end_probs"].astype(np.double), n_states, ssb, n_obs,
                              pm, npar, ft, fs, robocop.ROBOCOP_LOG_ABI)
    T["T5_impossible_row_C"] = dict(fward_log_rc=int(rcf), bward_log_rc=int(rcb),
                                    fward_log_rc_control=int(rc_ok), abi=robocop.ROBOCOP_LOG_ABI,
                                    passed=bool(rcf == -2 and rcb == -2 and rc_ok == robocop.ROBOCOP_LOG_ABI))

    # ---- T6 -----------------------------------------------------------------------
    out6 = {}
    for nm, bad in (("ps_zero", 0.0), ("ps_one", 1.0)):
        psb = ps_w.copy()
        psb[0] = bad
        try:
            RPW.decode_one(dshared, chrom, ws, we, codes, psb, ps_c, *lanes(calls), M, keep_idx)
            out6[nm] = "NOT RAISED"
        except AssertionError as e:
            out6[nm] = "AssertionError: " + str(e)
    cb = calls.copy()
    cb[10] = 2
    try:
        run(*lanes(cb))
        out6["call_value_2"] = "NOT RAISED"
    except AssertionError as e:
        out6["call_value_2"] = "AssertionError: " + str(e)
    T["T6_bernoulli_guards"] = dict(out6, passed=all(v.startswith("AssertionError") for v in out6.values()))

    # ---- T7 -----------------------------------------------------------------------
    rng = {}
    for arm, pi in (("A", None), ("B", 0.3), ("B", 0.5), ("B", 0.7), ("C", None)):
        for strand in ("watson_signal", "crick_signal"):
            ps, _s, _i = PP.arm_ps(dshared, strand, arm, pi)
            key = "%s%s_%s" % (arm, "" if pi is None else str(pi), strand.split("_")[0])
            rng[key] = dict(min=float(ps.min()), max=float(ps.max()),
                            any_0_or_1=bool(np.any(ps <= 0) or np.any(ps >= 1)))
    T["T7_fitted_ps_range"] = dict(rng, passed=not any(v["any_0_or_1"] for v in rng.values()))

    res["all_passed"] = all(t["passed"] for t in T.values())
    for k, t in T.items():
        print("%-45s %s" % (k, "PASS" if t["passed"] else "FAIL"))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", a.out, "| all passed:", res["all_passed"])
    return 0 if res["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
