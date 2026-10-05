#!/usr/bin/env python
"""Is the per-molecule Bernoulli emission mis-specified, or is check 1 measuring something else?

Check 1 compares the AVERAGE of 439 per-molecule posteriors against the posterior of the
POOLED data. Those are not the same quantity even when every line of code is right, so a
low correlation does not by itself say the emission is wrong. This script separates the
two possibilities with an identity:

    prod_over_molecules  ps^{m}(1-ps)^{1-m}  =  ps^k (1-ps)^{n-k}
                                            =  binom.pmf(k, n, ps) / C(n, k)

and C(n, k) does not depend on the state, so it cancels in the posterior. So if the reader
and the Bernoulli emission are correct, multiplying ALL 439 molecules' Bernoulli factors
into ONE chain and decoding once must reproduce the aggregate binomial decode almost
exactly. Any real mis-specification -- wrong strand, wrong frame, an off-by-one, the wrong
"no observation" value -- breaks that.

This uses the SHIPPED `robocop.update_data_emission_matrix_using_bernoulli_fiber_seq`
unchanged, called once per molecule on one emission array (it multiplies in place), so it
tests exactly the code the per-molecule runs use.

    python singlefiber/pooled_emission_check.py \\
        --agg /usr/project/xtmp/nd141/permol_proto/agg_baseline \\
        --out /usr/project/xtmp/nd141/permol_proto/pooled_check
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
sys.path.insert(0, 'pkgvar/permol_seq_maskoff/')
sys.path.insert(0, HERE)
sys.path.insert(0, ANALYSIS)

import h5py                                                   # noqa: E402
import numpy as np                                            # noqa: E402,F811
import robocop                                                # noqa: E402
from robocop.utils.getNucleotides import getNucleotideSequence  # noqa: E402

import collapse as C                                          # noqa: E402
import permol_params as PP                                    # noqa: E402
import read_calls as RC                                       # noqa: E402
import run_permol_window as RPW                               # noqa: E402
import score_robocop as S                                     # noqa: E402

TREE = os.path.join(ANALYSIS, "pkgvar", "permol_seq_maskoff")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="chrI:60001-65000")
    ap.add_argument("--traindir", default="robocop_train_fiberonly")
    ap.add_argument("--agg", default="/usr/project/xtmp/nd141/permol_proto/agg_baseline")
    ap.add_argument("--out", default="/usr/project/xtmp/nd141/permol_proto/pooled_check")
    ap.add_argument("--bam", default=RPW.BAM)
    a = ap.parse_args(argv)
    t0 = time.time()
    RPW.install_posterior_grab()
    chrom, rng = a.window.split(":")
    s1, e1 = (int(x) for x in rng.split("-"))
    n_obs = e1 - s1 + 1
    os.makedirs(a.out, exist_ok=True)

    dshared = pickle.load(open(os.path.join(a.traindir, "HMMconfig.pkl"), "rb"))
    dshared["robocopC"] = os.path.join(TREE, "robocop", "librobocop.so")
    dshared["tmpDir"] = a.out + "/"
    dshared["nucleotides"] = 1
    dshared["cache_parents_children"] = True
    ps_w, _sw, _iw = PP.arm_ps(dshared, "watson_signal", "A")
    ps_c, _sc, _ic = PP.arm_ps(dshared, "crick_signal", "A")

    flank = 400
    es = max(0, s1 - 1 - flank)
    ee = e1 + flank
    codes_ext = getNucleotideSequence(RPW.FASTA, chrom, es + 1, ee)
    mols, es2, ee2, rstats = RC.read_molecules(a.bam, chrom, s1, e1, codes_ext, flank=flank)
    assert (es2, ee2) == (es, ee)
    lo = (s1 - 1) - es
    codes = codes_ext[lo:lo + n_obs]

    mem = h5py.File("pooled_mem", mode="w", driver="core", backing_store=False)
    g = mem.create_group("segment_0")
    g.attrs["n_obs"] = n_obs
    g.attrs["chr"] = chrom
    g.attrs["start"] = s1
    g.attrs["end"] = e1
    g.attrs["segment"] = 0
    RPW.save_sparse_dense(mem, "segment_0/nucleotides", np.asarray(codes))
    dshared["info_file"] = mem
    d = robocop.createDictionary(0, dshared, chrom, s1, e1)

    # Multiply every molecule's Bernoulli factors into ONE emission array.
    no = np.full(n_obs, RC.NO_OBS, dtype=np.int8)
    for m in mols:
        calls = m.calls[lo:lo + n_obs]
        if m.strand == "watson":
            robocop.update_data_emission_matrix_using_bernoulli_fiber_seq(
                d, 0, dshared, ps_w, calls, 5, 0, 'watson')
            robocop.update_data_emission_matrix_using_bernoulli_fiber_seq(
                d, 0, dshared, ps_c, no, 6, 0, 'crick')
        else:
            robocop.update_data_emission_matrix_using_bernoulli_fiber_seq(
                d, 0, dshared, ps_w, no, 5, 0, 'watson')
            robocop.update_data_emission_matrix_using_bernoulli_fiber_seq(
                d, 0, dshared, ps_c, calls, 6, 0, 'crick')
    # the SAME 1e-30 floor as the aggregate path, applied once
    emat = d["emission"]
    emat[5][emat[5] == 0] = 1e-30
    emat[6][emat[6] == 0] = 1e-30
    acc = np.array(emat[0], dtype=np.double, order="C")
    for layer in range(1, emat.shape[0]):
        acc *= emat[layer]
    d["emission"] = acc[np.newaxis, :, :]
    del emat, acc
    dshared["n_vars"] = 1
    robocop.posterior_forward_backward(d, 0, dshared)
    p = RPW._GRAB.pop("p")
    ll = float(mem["segment_0"].attrs["log_likelihood"])
    mem.close()
    dshared["info_file"] = None
    print("pooled Bernoulli decode: loglik %.3f in %.1fs" % (ll, time.time() - t0))

    M = C.collapse_matrix(dshared)
    cols = C.column_names(dshared)
    op = C.collapse(p, M)

    S.use_pkg(TREE)
    dec = S.load_decode(a.agg + "/")
    agg_op, _cov, _fr = S.region_optable(dec, chrom, s1, e1)
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(a.agg, "config.ini"))

    res = {"window": a.window, "loglik_pooled_bernoulli": ll,
           "n_molecules": len(mols), "agg": a.agg}
    print("\n%-14s %8s %10s %10s" % ("column", "r", "mean_pool", "mean_agg"))
    for c in ("nucleosome", "nuc_center", "background", "Abf1_murphy",
              "nuc_padding", "unknown", "Reb1_badis"):
        if c not in cols or c not in agg_op.columns:
            continue
        x = op[:, cols.index(c)]
        y = agg_op[c].values
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")
        print("%-14s %8.5f %10.5f %10.5f" % (c, r, x.mean(), y.mean()))
        res[c] = dict(r=r, mean_pooled=float(x.mean()), mean_agg=float(y.mean()),
                      max_abs_diff=float(np.abs(x - y).max()))
    np.savez_compressed(os.path.join(a.out, "pooled_optable.npz"),
                        op=op.astype(np.float32), cols=np.array(cols),
                        agg=agg_op.values.astype(np.float32),
                        agg_cols=np.array(list(agg_op.columns)))
    with open(os.path.join(a.out, "pooled_check.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\nwrote", os.path.join(a.out, "pooled_check.json"))
    rn = res.get("nucleosome", {}).get("r", float("nan"))
    print("VERDICT: emission %s (pooled-vs-aggregate r on nucleosome = %.5f)"
          % ("CORRECT" if rn >= 0.99 else "SUSPECT", rn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
