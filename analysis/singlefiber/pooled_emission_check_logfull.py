#!/usr/bin/env python
"""Pooled-emission check, LOG-SPACE port (2026-10-25). Twin of pooled_emission_check.py.

Identity (per position, per strand, C(n, k) state-independent):

    sum_over_molecules [ m log ps + (1-m) log(1-ps) ]  =  k log ps + (n-k) log(1-ps)
                                                       =  binom.logpmf(k, n, ps) - log C(n, k)

So adding ALL molecules' log-Bernoulli terms into ONE chain (the shipped log
`update_data_emission_matrix_using_bernoulli_fiber_seq` of pkgvar/permol_seq_maskoff_logfull,
called once per molecule; it ADDS in place) and decoding once must reproduce the log-space
AGGREGATE decode (pkgvar/seq_maskoff_logfull, same trainDir, same ROBOCOP_LOG_EMISSION):

  * posterior: identical up to float rounding (the aggregate's h5 is truncated at 1e-4, so the
    pooled posterior is also compared after the SAME truncation + per-row renormalisation);
  * log-likelihood: loglik_agg - loglik_pooled == sum over on-target positions of log C(n, k),
    computed independently from the aggregate h5's Fiber_count arrays (exact mode only; in
    floored mode the aggregate's 1e-30 floor can bite where pmf underflows).

No 1e-30 floor is applied anywhere (the old twin floored once at the end); "no observation" is 0.0.

    ROBOCOP_LOG_EMISSION=exact python singlefiber/pooled_emission_check_logfull.py \\
        --agg /usr/project/xtmp/nd141/permol_proto_logfull/agg_logfull_exact \\
        --out /usr/project/xtmp/nd141/permol_proto_logfull/pooled_check
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
os.chdir(ANALYSIS)
sys.path.insert(0, 'pkgvar/permol_seq_maskoff_logfull/')
sys.path.insert(0, HERE)
sys.path.insert(0, ANALYSIS)

import h5py                                                   # noqa: E402
import robocop                                                # noqa: E402
from robocop.utils.getNucleotides import getNucleotideSequence  # noqa: E402
from scipy.special import gammaln                              # noqa: E402

import collapse as C                                          # noqa: E402
import permol_params as PP                                    # noqa: E402
import read_calls as RC                                       # noqa: E402
import run_permol_window_logfull as RPW                       # noqa: E402
import score_robocop as S                                     # noqa: E402

TREE = os.path.join(ANALYSIS, "pkgvar", "permol_seq_maskoff_logfull")


def log_choose(n, k):
    n = np.asarray(n, float)
    k = np.asarray(k, float)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="chrI:60001-65000")
    ap.add_argument("--traindir", default="robocop_train_fiberonly")
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bam", default=RPW.BAM)
    a = ap.parse_args(argv)
    t0 = time.time()
    mode = robocop.log_emission_mode()
    assert RPW.so_md5(RPW.LOGFULL_SO) == RPW.LOGFULL_SO_MD5
    RPW.install_posterior_grab()
    chrom, rng = a.window.split(":")
    s1, e1 = (int(x) for x in rng.split("-"))
    n_obs = e1 - s1 + 1
    os.makedirs(a.out, exist_ok=True)

    dshared = pickle.load(open(os.path.join(a.traindir, "HMMconfig.pkl"), "rb"))
    dshared["robocopC"] = RPW.LOGFULL_SO
    dshared["tmpDir"] = a.out + "/"
    dshared["nucleotides"] = 1
    dshared["cache_parents_children"] = True
    ssb = dshared["silent_states_begin"]
    ps_w, _sw, _iw = PP.arm_ps(dshared, "watson_signal", "A")
    ps_c, _sc, _ic = PP.arm_ps(dshared, "crick_signal", "A")

    flank = 400
    es = max(0, s1 - 1 - flank)
    ee = e1 + flank
    codes_ext = getNucleotideSequence(RPW.FASTA, chrom, es + 1, ee)
    mols, es2, ee2, rstats = RC.read_molecules(a.bam, chrom, s1, e1, codes_ext, flank=flank)
    assert (es2, ee2) == (es, ee)
    lo = (s1 - 1) - es
    codes = np.asarray(codes_ext[lo:lo + n_obs])

    mem = h5py.File("pooled_mem_logfull_%d" % os.getpid(), mode="w", driver="core", backing_store=False)
    g = mem.create_group("segment_0")
    g.attrs["n_obs"] = n_obs
    g.attrs["chr"] = chrom
    g.attrs["start"] = s1
    g.attrs["end"] = e1
    g.attrs["segment"] = 0
    RPW.save_sparse_dense(mem, "segment_0/nucleotides", codes)
    dshared["info_file"] = mem
    d = robocop.createDictionary(0, dshared, chrom, s1, e1)

    # ADD every molecule's log-Bernoulli terms into ONE emission array.
    no = np.full(n_obs, RC.NO_OBS, dtype=np.int8)
    kw = np.zeros(n_obs, np.int64); nw = np.zeros(n_obs, np.int64)
    kc = np.zeros(n_obs, np.int64); nc = np.zeros(n_obs, np.int64)
    for m in mols:
        calls = m.calls[lo:lo + n_obs]
        if m.strand == "watson":
            cw, cc = calls, no
            kw += (calls == RC.METH); nw += (calls != RC.NO_OBS)
        else:
            cw, cc = no, calls
            kc += (calls == RC.METH); nc += (calls != RC.NO_OBS)
        robocop.update_data_emission_matrix_using_bernoulli_fiber_seq(d, 0, dshared, ps_w, cw, 5, 0, 'watson')
        robocop.update_data_emission_matrix_using_bernoulli_fiber_seq(d, 0, dshared, ps_c, cc, 6, 0, 'crick')
    robocop.posterior_forward_backward(d, 0, dshared)      # sums the 7 log layers, asserts, log C path
    p = RPW._GRAB.pop("p")
    ll = float(mem["segment_0"].attrs["log_likelihood"])
    mem.close()
    dshared["info_file"] = None
    val = RPW.posterior_validity(p, ssb)
    print("pooled log-Bernoulli decode: loglik %.6f in %.1fs; validity %s" % (ll, time.time() - t0, val))

    # --- aggregate reference ---------------------------------------------------------
    agg_h5 = os.path.join(a.agg, "tmpDir", "info_0_1.h5")
    with h5py.File(agg_h5, "r") as f:
        ga = f["segment_0"]
        ll_agg = float(ga.attrs["log_likelihood"])
        assert int(ga.attrs["start"]) == s1 and int(ga.attrs["end"]) == e1
        cnt = {}
        for nm in ("meth_watson", "A_watson", "meth_crick", "A_crick"):
            cnt[nm] = S._get_sparse_todense(f, "segment_0/Fiber_count_%s" % nm)[:n_obs].astype(np.int64)
        p_agg = S._get_sparse_todense(f, "segment_0/posterior")
    on_w = codes == RC.CODE_A
    on_t = codes == RC.CODE_T
    # the reader must reproduce the aggregate's (k, n) at every on-target position
    cnt_diff = dict(k_watson=int(np.abs(cnt["meth_watson"][on_w] - kw[on_w]).max()),
                    n_watson=int(np.abs(cnt["A_watson"][on_w] - nw[on_w]).max()),
                    k_crick=int(np.abs(cnt["meth_crick"][on_t] - kc[on_t]).max()),
                    n_crick=int(np.abs(cnt["A_crick"][on_t] - nc[on_t]).max()))
    sum_logC = float(log_choose(cnt["A_watson"][on_w], cnt["meth_watson"][on_w]).sum()
                     + log_choose(cnt["A_crick"][on_t], cnt["meth_crick"][on_t]).sum())

    # state level, with the SAME truncation the aggregate's save applied
    pt = p.copy()
    pt[pt < 1e-4] = 0
    rs = pt[:, :ssb].sum(axis=1, keepdims=True)
    pt = pt / rs
    rs_a = p_agg[:, :ssb].sum(axis=1, keepdims=True)
    pa = p_agg / rs_a
    state_maxdiff_trunc = float(np.abs(pt - pa).max())

    M = C.collapse_matrix(dshared)
    cols = C.column_names(dshared)
    op = C.collapse(p, M)
    op_t = C.collapse(pt, M)

    S.use_pkg(TREE)
    dec = S.load_decode(a.agg.rstrip("/") + "/")
    agg_op, _cov, _fr = S.region_optable(dec, chrom, s1, e1)

    res = {"window": a.window, "log_emission": mode, "so_md5": RPW.LOGFULL_SO_MD5,
           "n_molecules": len(mols), "agg": a.agg,
           "loglik_pooled_bernoulli": ll, "loglik_agg": ll_agg,
           "sum_logC_on_target": sum_logC,
           "loglik_identity_residual": ll_agg - ll - sum_logC,
           "reader_vs_agg_counts_maxdiff": cnt_diff,
           "pooled_posterior_validity": val,
           "state_max_abs_diff_after_same_truncation": state_maxdiff_trunc,
           "columns": {}}
    print("\n%-14s %12s %12s %12s %12s" % ("column", "r", "max|d|", "max|d| trunc", "mean_agg"))
    for c in ("nucleosome", "nuc_center", "background", "Abf1_murphy",
              "nuc_padding", "unknown", "Reb1_badis"):
        if c not in cols or c not in agg_op.columns:
            continue
        x = op[:, cols.index(c)]
        xt = op_t[:, cols.index(c)]
        y = agg_op[c].values
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")
        res["columns"][c] = dict(r=r, max_abs_diff=float(np.abs(x - y).max()),
                                 max_abs_diff_same_truncation=float(np.abs(xt - y).max()),
                                 mean_pooled=float(x.mean()), mean_agg=float(y.mean()))
        print("%-14s %12.9f %12.3g %12.3g %12.5f" % (c, r, np.abs(x - y).max(), np.abs(xt - y).max(), y.mean()))
    print("loglik pooled %.6f  agg %.6f  sum logC %.6f  residual %.3g"
          % (ll, ll_agg, sum_logC, res["loglik_identity_residual"]))
    print("reader vs aggregate counts max|d|:", cnt_diff)
    np.savez_compressed(os.path.join(a.out, "pooled_optable.npz"),
                        op=op.astype(np.float32), cols=np.array(cols),
                        agg=agg_op.values.astype(np.float32),
                        agg_cols=np.array(list(agg_op.columns)))
    with open(os.path.join(a.out, "pooled_check.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\nwrote", os.path.join(a.out, "pooled_check.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
