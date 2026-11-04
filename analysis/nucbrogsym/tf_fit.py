#!/usr/bin/env python
"""TF-footprint m6A params from a modkit pileup, the method that made
inputs/all_TFs_1000pealVal_params_pseudo.pkl (md5 ba0d7e2e...).

What it reproduces: the top-level call at abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:1123-1132,
    compute_Fiber_seq_TFPhisMus("Fiber_seq", <bam, unused>, <pileup>, <Rossi bed>, <tmpDir, unused>,
                                (0, 80), None, 0, dist="binomial")
(successes_col=11 = N_mod, trials_col=9 = N_valid_cov, the defaults). That script pickles to the shipped
path at import (:1138-1139), so it is never run or imported. Instead the functions are pulled out with
`ast` and exec'd unchanged (as nucbrogsym/bg_fit.py does):
    compute_Fiber_seq_TFPhisMus (:28), compute_individual_Fiber_seq_TF_binom (:389),
    compute_combined_Fiber_seq_TF_binom (:462), combine_motif_counts_binom (:534),
    fit_binomial_parameters (:622), create_default_params_binomial (:654),
    add_pseudocounts_binomial (:1035).
Three names are replaced by stubs; none of them changes a number on the binomial path:
    importr               -> returns None (fitdist is only used on the DMS / nbinom paths);
    plot_binomial_boxplots, plot_p_heatmaps -> no-ops (pure plotting; avoids ~50 open figures).
SeqIO is the real one: the function still reads the hard-coded SacCer3.fa (unused on this path).

Method, per TF with >= 50 Rossi sites (12 TFs in the peakVal_1000 bed): for each site and motif strand,
sum N_mod and N_valid_cov per motif position and per pileup strand (+ = watson, - = crick); crick-motif
sites are reversed and their strands swapped; one pseudo-site (3 successes / 58 trials per position) is
added; p = sum(successes) / sum(trials). TFs with < 50 sites are pooled into one position
("combined_low_count", tf_len 1, NO pseudo-count). Bases C/G/T get p = 0 (no trials).

Inputs:  --pileup (modkit bedMethyl, col 9 = space-separated counts), --rossi-bed.
Output:  pickle {'mu': {}, 'phi': {}, 'p': {tf: {'watson_signal'|'crick_signal': {A,C,G,T: ndarray}}}},
         same layout as the shipped pkl. --json adds per-TF site counts, coverage and mean p(A).

  python nucbrogsym/tf_fit.py --pileup <merged pileup> --check inputs/all_TFs_1000pealVal_params_pseudo.pkl
  python nucbrogsym/tf_fit.py --pileup <filtered pileup> --out inputs/all_TFs_..._pseudo_filt.pkl
"""
import argparse
import ast
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
SRC = os.path.join(ANALYSIS, "abf1_reb1_dms_parameter_Fiber-seq_w_binom.py")
FUNCS = ("compute_Fiber_seq_TFPhisMus", "compute_individual_Fiber_seq_TF_binom",
         "compute_combined_Fiber_seq_TF_binom", "combine_motif_counts_binom", "fit_binomial_parameters",
         "create_default_params_binomial", "add_pseudocounts_binomial")
ROSSI = os.path.join(ANALYSIS, "inputs", "rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed")
# arguments of the original call that the Fiber_seq/binomial path never reads (kept for fidelity)
UNUSED_BAM = "/home/rapiduser/projects/DMS-seq/DM1664/DM1664_trim_3prime_18bp_remaining_name_change_sorted.bam"
UNUSED_TMP = "/usr/xtmp/nd141/programs/roboNhat_w_new_changes/analysis/robocop_train/tmpDir"
PROTECTED = {"bg_params.pkl", "nucleosome_params.pkl", "all_TFs_1000pealVal_params_pseudo.pkl",
             "all_TFs_1000pealVal_params.pkl", "motifs_meme.txt"}


def load_functions(stats):
    tree = ast.parse(open(SRC).read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    assert sorted(n.name for n in keep) == sorted(FUNCS), [n.name for n in keep]
    from Bio import SeqIO
    import matplotlib.pyplot as plt
    ns = dict(np=np, pd=pd, SeqIO=SeqIO, plt=plt,
              importr=lambda name: None,
              plot_binomial_boxplots=lambda *a, **k: None,
              plot_p_heatmaps=lambda *a, **k: None)
    exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)
    # record per-call site/coverage stats without changing the result (wrapper returns the same object)
    orig = ns["combine_motif_counts_binom"]

    def combine_logged(w, c):
        out = orig(w, c)
        s = dict(tf_len=int(out["tf_len"]), num_sites=int(out["num_sites"]),
                 sites_watson_motif=int(w["num_sites"]), sites_crick_motif=int(c["num_sites"]))
        for st in ("watson_signal", "crick_signal"):
            tri = out[st]["trials"]["A"]
            s[st] = dict(sum_succ_A=int(out[st]["successes"]["A"].sum()), sum_trials_A=int(tri.sum()),
                         rows_zero_trials_A=int((tri.sum(axis=1) == 0).sum()) if tri.size else 0)
        stats.append(s)
        return out
    ns["combine_motif_counts_binom"] = combine_logged
    return ns, {n.name: (n.lineno, n.end_lineno) for n in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pileup", required=True)
    ap.add_argument("--rossi-bed", default=ROSSI)
    ap.add_argument("--out", default=None, help="NEW pkl (refuses the shipped names and existing files)")
    ap.add_argument("--check", default=None, help="pkl to compare against (exact); writes nothing")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.out:
        assert not a.check, "--check writes nothing; drop --out"
        assert os.path.basename(a.out) not in PROTECTED, "never overwrite %s" % a.out
        if os.path.exists(a.out):
            sys.exit("refusing to overwrite %s" % a.out)
    if a.json and os.path.exists(a.json):
        sys.exit("refusing to overwrite %s" % a.json)
    t0 = time.time()
    stats = []
    ns, lines = load_functions(stats)
    tfs = pd.read_csv(a.rossi_bed, sep="\t").rename(columns={"TF": "tf_name"})
    counts = tfs.groupby("tf_name")["chr"].count()
    fitted = sorted(counts.loc[counts >= 50].index)
    low = sorted(counts.loc[counts < 50].index)
    print("Rossi bed %s: %d sites, %d TFs fitted (>=50 sites), %d pooled" % (a.rossi_bed, len(tfs), len(fitted), len(low)),
          flush=True)
    res = ns["compute_Fiber_seq_TFPhisMus"]("Fiber_seq", UNUSED_BAM, a.pileup, a.rossi_bed, UNUSED_TMP,
                                            (0, 80), None, 0, dist="binomial")
    secs = time.time() - t0
    # stats order = ind_tfs loop order (groupby index order), then combined_low_count
    order = list(counts.loc[counts >= 50].index) + ["combined_low_count"]
    assert len(stats) == len(order), (len(stats), len(order))
    per_tf = {}
    for tf, s in zip(order, stats):
        p = res["p"][tf]
        s["mean_pA"] = {st: float(np.mean(p[st]["A"])) for st in ("watson_signal", "crick_signal")}
        s["bed_sites"] = int(counts[tf]) if tf in counts.index else int(counts[low].sum())
        per_tf[tf] = s
        print("%-20s sites=%-4d len=%-2d mean pA W=%.4f C=%.4f" % (tf, s["bed_sites"], s["tf_len"],
              s["mean_pA"]["watson_signal"], s["mean_pA"]["crick_signal"]), flush=True)
    out = dict(pileup=a.pileup, rossi_bed=a.rossi_bed, source=SRC, source_lines=lines, runtime_s=secs,
               fitted_tfs=fitted, pooled_tfs=low, per_tf=per_tf)
    if a.check:
        ref = pickle.load(open(a.check, "rb"))
        diff, allsame = {}, sorted(ref) == sorted(res) and sorted(ref["p"]) == sorted(res["p"])
        allsame = allsame and all(len(ref[k]) == 0 and len(res[k]) == 0 for k in ("mu", "phi"))
        for tf in sorted(set(ref["p"]) | set(res["p"])):
            if tf not in ref["p"] or tf not in res["p"]:
                diff[tf] = "missing in %s" % ("ref" if tf not in ref["p"] else "new")
                allsame = False
                continue
            d = {}
            for st in ("watson_signal", "crick_signal"):
                for b in "ACGT":
                    x, y = np.asarray(ref["p"][tf][st][b]), np.asarray(res["p"][tf][st][b])
                    same = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
                    allsame &= same
                    d["%s_%s" % (st, b)] = dict(max_abs=float(np.abs(x - y).max()) if x.shape == y.shape else None,
                                                bit_identical=bool(same))
            diff[tf] = d
        out["check"] = dict(file=a.check, all_bit_identical=bool(allsame), diff=diff)
        print("vs %s: all_bit_identical=%s" % (a.check, allsame))
        for tf, d in diff.items():
            if isinstance(d, str):
                print("  %s: %s" % (tf, d))
            else:
                print("  %-20s max_abs=%.3g bit_identical=%s" % (tf, max(v["max_abs"] or 0 for v in d.values()),
                      all(v["bit_identical"] for v in d.values())))
    if a.out:
        with open(a.out, "wb") as f:
            pickle.dump(res, f)
        out["written"] = a.out
        print("wrote", a.out)
    print("runtime %.0f s" % secs)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
