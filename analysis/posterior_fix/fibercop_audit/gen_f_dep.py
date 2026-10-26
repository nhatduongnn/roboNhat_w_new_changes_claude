#!/usr/bin/env python3
"""
build/f_dep.json -- error vs read depth for fiberCOP (the audit's variants/depth.json "sweep" part).

Coverage n (both strands) at positions 1-4 of AATTAA, k = round(frac * n) methylated, ordinary data
(25 reads, 6 methylated) elsewhere: fiber_model.high(n, frac). For each n, the max |posterior - truth|
over emitting states for current / fixB / fixed / logD0 / logD1, against the floored-model truth
(brute on LE_fl) and the real-model truth (brute on LE_ex).

  sweep  = the frac = 1.0 series (primary; same role as the audit's depth.json "sweep")
  series = {"1.0": [...], "0.0": [...], "0.6": [...]}
  reads  = null (filled later from build/realdata.json by another script)

Run:  python gen_f_dep.py      -> build/f_dep.json
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fiber_model as FM
from gen_f_emp import run_methods, maxerr, floored_count, libs

BUILD = os.path.join(HERE, "build"); os.makedirs(BUILD, exist_ok=True)
POS = (1, 2, 3, 4)
EXTRA = [210, 220, 230, 250, 300, 1000, 4379]
GRID = sorted(set(np.round(np.geomspace(2, 10000, 24)).astype(int).tolist()) | set(EXTRA))
FRACS = [1.0, 0.0, 0.6]
METH = ["current", "fixB", "fixed", "logD0", "logD1"]

def ll_ok(ll, ref):
    return bool(ll is not None and np.isfinite(ll) and abs(ll - ref) < 1e-6 * max(1, abs(ref)))

def row(n, frac):
    m = FM.make_model(libs()["current"], **FM.high(n, frac, POS)); S = m.S
    res, _, _ = run_methods(m)
    pb, lpb = res["brute"]; pe, lpe = res["brute_exact"]
    with np.errstate(divide="ignore"):
        r = dict(K=int(n), n=int(n), frac=frac, k=int(round(frac * n)),
                 maxemit_log10=float(max(m.LE_ex.sum(0)[t, :S].max() for t in POS) / math.log(10)),
                 maxemit_log10_floored=float(max(m.LE_fl.sum(0)[t, :S].max() for t in POS) / math.log(10)),
                 floored_cells=floored_count(m))
    for k in METH:
        post, ll = res[k]
        r[k] = maxerr(post, pb, S)
        r[k + "_exact"] = maxerr(post, pe, S)
        r[k + "_ll_ok"] = ll_ok(ll, lpb)
        r[k + "_ll_ok_exact"] = ll_ok(ll, lpe)
    return r

if __name__ == "__main__":
    series = {}
    for frac in FRACS:
        rows = []
        for n in GRID:
            r = row(n, frac); rows.append(r)
            f = lambda e: "NaN" if e is None else "%.1e" % e
            print("frac %.1f n %5d fl %2d  " % (frac, n, r["floored_cells"]) + "  ".join(f"{k}={f(r[k])}/{f(r[k + '_exact'])}" for k in METH))
        series[str(frac)] = rows
    out = dict(sweep=series["1.0"], series=series, fracs=FRACS, grid=GRID, positions=list(POS), methods=METH,
               reads=None)
    json.dump(out, open(os.path.join(BUILD, "f_dep.json"), "w"), indent=1, allow_nan=False)
    print("json", os.path.getsize(os.path.join(BUILD, "f_dep.json")), "bytes; grid", len(GRID), GRID)
