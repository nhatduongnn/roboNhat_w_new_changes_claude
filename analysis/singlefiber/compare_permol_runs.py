#!/usr/bin/env python
"""Compare the per-molecule posteriors of two per-read runs, molecule by molecule (2026-10-25).

Reads each run's permol.npz (per-molecule collapsed factor columns, float32) and molecules.tsv,
matches molecules by read name, and reports per column: max |difference|, the number of
(molecule, position) cells with |d| > 0.01, and the per-molecule log-likelihood difference.

    python singlefiber/compare_permol_runs.py \\
        --a /usr/project/xtmp/nd141/permol_proto/armA_effoff \\
        --b /usr/project/xtmp/nd141/permol_proto_logfull/armA_effoff_floored_first20 \\
        --out /usr/project/xtmp/nd141/permol_proto_logfull/compare_floored_vs_linear.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def load(d):
    z = np.load(os.path.join(d, "permol.npz"), allow_pickle=False)
    mt = pd.read_csv(os.path.join(d, "molecules.tsv"), sep="\t")
    return dict(per_mol=z["per_mol"], cov=z["per_mol_cov"], cols=[str(c) for c in z["cols"]],
                names=[str(n) for n in z["names"]], mt=mt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference run")
    ap.add_argument("--b", required=True, help="run under test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.01)
    a = ap.parse_args(argv)
    A, B = load(a.a), load(a.b)
    assert A["cols"] == B["cols"], (A["cols"], B["cols"])
    ia = {n: i for i, n in enumerate(A["names"])}
    pairs = [(ia[n], j) for j, n in enumerate(B["names"]) if n in ia]
    res = dict(a=a.a, b=a.b, n_molecules_b=len(B["names"]), n_matched=len(pairs),
               thresh=a.thresh, columns={})
    for c, col in enumerate(A["cols"]):
        mx, nbig, ncell = 0.0, 0, 0
        worst = None
        for i, j in pairs:
            m = A["cov"][i] & B["cov"][j]
            assert np.array_equal(A["cov"][i], B["cov"][j]), "span differs for molecule %s" % B["names"][j]
            d = np.abs(A["per_mol"][i, m, c].astype(float) - B["per_mol"][j, m, c].astype(float))
            ncell += int(m.sum())
            nbig += int((d > a.thresh).sum())
            if d.size and d.max() > mx:
                mx = float(d.max())
                worst = dict(molecule=B["names"][j], offset=int(np.flatnonzero(m)[d.argmax()]))
        res["columns"][col] = dict(max_abs_diff=mx, n_cells_over_thresh=nbig, n_cells=ncell, worst=worst)
    ma = A["mt"].set_index("name")
    mb = B["mt"].set_index("name")
    common = [n for n in mb.index if n in ma.index and "loglik" in ma.columns]
    if common:
        dll = (mb.loc[common, "loglik"] - ma.loc[common, "loglik"]).abs()
        res["loglik_abs_diff_max"] = float(dll.max())
        res["loglik_abs_diff_median"] = float(dll.median())
    if "secs" in ma.columns and "secs" in mb.columns and common:
        res["secs_a_sum"] = float(ma.loc[common, "secs"].sum())
        res["secs_b_sum"] = float(mb.loc[common, "secs"].sum())
    print(json.dumps(res, indent=1, default=float))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
