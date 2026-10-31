#!/usr/bin/env python
"""Background m6A params from a modkit pileup, the method that made inputs/bg_params.pkl (2026-10-30).

The functions are NOT re-typed: computeLinkers, compute_Fiber_seq_background,
fit_binomial_parameters and create_default_params_binomial are pulled out of
abf1_reb1_dms_parameter_Fiber-seq_w_binom.py with `ast` and exec'd (that script runs a TF fit at
import, so it cannot be imported). The calls are the commented-out ones in that script
(:1071-1083 pileup load, :1103-1109 background fit): linker segments from the Chereji +1/-1 dyads,
dist="binomial", successes_col=11 (N_mod), trials_col=9 (N_valid_cov).

  python nucbrogsym/bg_fit.py --pileup <barcode01 pileup> --check inputs/bg_params.pkl   # reproduce
  python nucbrogsym/bg_fit.py --pileup <merged pileup> --out inputs/bg_params_all7.pkl     # new file
"""
import argparse
import ast
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
SRC = os.path.join(ANALYSIS, "abf1_reb1_dms_parameter_Fiber-seq_w_binom.py")
FUNCS = ("computeLinkers", "compute_Fiber_seq_background", "fit_binomial_parameters",
         "create_default_params_binomial")
CHEREJI = os.path.join(ANALYSIS, "inputs", "Chereji_2018_+1_-1_nucs.bed")
PROTECTED = {"bg_params.pkl", "nucleosome_params.pkl", "all_TFs_1000pealVal_params_pseudo.pkl"}


def load_functions():
    tree = ast.parse(open(SRC).read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    assert sorted(n.name for n in keep) == sorted(FUNCS), [n.name for n in keep]
    import pyranges as pr
    ns = dict(np=np, pd=pd, pr=pr)
    exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)
    return ns, {n.name: (n.lineno, n.end_lineno) for n in keep}


def load_pileup(path):
    """As the script's commented block: read tab-separated, split column 9 on spaces."""
    df = pd.read_csv(path, sep="\t", header=None)
    split = df[9].str.split(" ", expand=True)
    split.columns = [i for i in range(9, 9 + split.shape[1])]
    df = pd.concat([df.drop(columns=[9]), split], axis=1)
    df[11] = df[11].astype(int)
    df[9] = df[9].astype(int)
    return df


def flat(p):
    return {st: float(np.asarray(p["p"][st]["A"]).ravel()[0]) for st in ("watson_signal", "crick_signal")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pileup", required=True)
    ap.add_argument("--out", default=None, help="NEW pkl under inputs/ (refuses the shipped names)")
    ap.add_argument("--check", default=None, help="pkl to compare against (exact)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ns, lines = load_functions()
    segments = ns["computeLinkers"](CHEREJI)
    df = load_pileup(a.pileup)
    bases = sorted(df[3].astype(str).unique().tolist())
    print("pileup %s: %d rows, base codes %s; %d linker segments" % (a.pileup, len(df), bases, len(segments)))
    bg = ns["compute_Fiber_seq_background"](df, segments, None, dist="binomial",
                                            successes_col=11, trials_col=9)
    res = dict(pileup=a.pileup, n_rows=len(df), base_codes=bases, n_segments=len(segments),
               source=SRC, source_lines=lines, chereji=CHEREJI, p_A=flat(bg))
    for st in ("watson_signal", "crick_signal"):
        for b in ("C", "G", "T"):
            assert np.all(np.asarray(bg["p"][st][b]) == 0.0), (st, b)
    print("p(A):", json.dumps(res["p_A"]))
    if a.check:
        ref = pickle.load(open(a.check, "rb"))
        diff = {}
        for st in ("watson_signal", "crick_signal"):
            for b in ("A", "C", "G", "T"):
                x, y = np.asarray(ref["p"][st][b]), np.asarray(bg["p"][st][b])
                diff["%s_%s" % (st, b)] = dict(same_shape=x.shape == y.shape,
                                               max_abs=float(np.abs(x - y).max()),
                                               bit_identical=bool(x.shape == y.shape and np.array_equal(x, y)))
        res["check"] = dict(file=a.check, keys_match=sorted(ref) == sorted(bg), diff=diff)
        print("vs %s:" % a.check, json.dumps(res["check"]))
    if a.out:
        assert os.path.basename(a.out) not in PROTECTED, "never overwrite %s" % a.out
        if os.path.exists(a.out):
            sys.exit("refusing to overwrite %s" % a.out)
        with open(a.out, "wb") as f:
            pickle.dump(bg, f)
        res["written"] = a.out
        print("wrote", a.out)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
