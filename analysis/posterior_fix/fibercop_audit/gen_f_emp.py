#!/usr/bin/env python3
"""
build/f_emp.json -- the "Empirical results" tab for fiberCOP: fiber_model.SCENARIOS x seven methods.

Same schema as robocop_c_audit/walkthrough/variants/empirical.json, except (see build/SCHEMA.md):
  * methods: current (LIB_PRENORM, the fork's original algo.c), fixB (LIB_FIXB, production),
    fixed (LIB_C1FIX, audit's C1 fix), logD0 (log space on the floored emissions),
    logD1 (log space on exact logpmf), brute (floored-model truth), brute_exact (real-model truth).
  * per method: max_err is vs "brute" (floored truth) and max_err_exact is vs "brute_exact".
  * per scenario: short/long replaced by n_w,k_w,n_c,k_c; new floored_cells, scored, maxemit_log10_floored.
Downstream values (dbf, em) use empirical.py's downstream(), i.e. the audit's upstream
robocop.sum_for_dbf_probs, on the posterior after the 1e-4 sparsification.

Run:  python gen_f_emp.py      -> build/f_emp.json
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fiber_model as FM

BUILD = os.path.join(HERE, "build"); os.makedirs(BUILD, exist_ok=True)
downstream = FM._g["downstream"]          # empirical.py:downstream (read-only exec in fiber_model)
METHODS = ["current", "fixB", "fixed", "logD0", "logD1", "brute", "brute_exact"]
_LIBS = {}
def libs():
    if not _LIBS:
        _LIBS.update(current=FM.load_lin(FM.LIB_PRENORM), fixB=FM.load_lin(FM.LIB_FIXB, v2=True), fixed=FM.load_lin(FM.LIB_C1FIX))
    return _LIBS

def run_methods(m):
    """{method: (post (n,N), ll)} for all seven methods, plus npaths of the brute force."""
    L = libs(); res = {}
    with np.errstate(all="ignore"):
        for k in ("current", "fixB", "fixed"):
            r = FM.run_lin(L[k], m); res[k] = (np.array(r["post"], float), float(r["ll"]))
        r = FM.run_logfull(m, m.LE_fl); res["logD0"] = (r["post"], r["ll"])
        r = FM.run_logfull(m, m.LE_ex); res["logD1"] = (r["post"], r["ll"])
        pb, lpb, npaths = FM.brute(m, m.LE_fl); res["brute"] = (pb, float(lpb))
        pe, lpe, npe = FM.brute(m, m.LE_ex); res["brute_exact"] = (pe, float(lpe))
    return res, npaths, npe

def maxerr(post, ref, S):
    e = np.abs(np.asarray(post)[:, :S] - ref[:, :S])
    return None if np.isnan(e).any() else float(e.max())

fin = lambda x: None if not np.isfinite(x) else float(x)

def scored_of(seq):
    return ["W" if c == "A" else ("C" if c == "T" else None) for c in seq]

def floored_count(m):
    sc = scored_of(m.seq); li = {"W": 0, "C": 1}
    return int(sum(int(np.sum(m.raw_pmf[li[s], t] == 0.0)) for t, s in enumerate(sc) if s is not None))

def scenario(key, title, desc, data):
    lib = libs()["current"]
    m = FM.make_model(lib, **data); S, n = m.S, m.T_obs
    res, npaths, npe = run_methods(m)
    pb, pe = res["brute"][0], res["brute_exact"][0]
    with np.errstate(divide="ignore"):
        mx_ex = [float(np.max(m.LE_ex.sum(0)[t, :S]) / math.log(10)) for t in range(n)]
        mx_fl = [float(np.max(m.LE_fl.sum(0)[t, :S]) / math.log(10)) for t in range(n)]
    sc = dict(key=key, title=title, desc=desc, seq=m.seq,
              n_w=list(map(int, m.n_w)), k_w=list(map(int, m.k_w)), n_c=list(map(int, m.n_c)), k_c=list(map(int, m.k_c)),
              scored=scored_of(m.seq), floored_cells=floored_count(m),
              maxemit_log10=mx_ex, maxemit_log10_floored=mx_fl,
              zero_emission_rows=[int(t) for t in range(n) if np.all(m.E.prod(0)[t, :S] == 0)],
              npaths=npaths, npaths_exact=npe, methods={})
    for meth in METHODS:
        post, ll = res[meth]
        post = np.array(post, dtype=float)
        dbf, em, emn = downstream(m, post)
        sc["methods"][meth] = dict(
            post=[[fin(x) for x in r] for r in post], ll=fin(ll),
            max_err=maxerr(post, pb, S), max_err_exact=maxerr(post, pe, S),
            rowsum=[fin(x) for x in post[:, :S].sum(1)],
            dbf=[[fin(x) for x in r] for r in dbf],
            em=[fin(x) for x in emn])
    return sc, m

if __name__ == "__main__":
    out = dict(scenarios=[], names=FM.W.NAMES, S=None, n=None, methods=METHODS)
    for key, title, desc, data in FM.SCENARIOS:
        sc, m = scenario(key, title, desc, data)
        out["S"], out["n"] = m.S, m.T_obs
        out["scenarios"].append(sc)
        print(f"{key:9s} floored={sc['floored_cells']:2d} npaths={sc['npaths']}/{sc['npaths_exact']}  " +
              "  ".join(f"{k}={sc['methods'][k]['max_err']:.2g}/{sc['methods'][k]['max_err_exact']:.2g}" if sc['methods'][k]['max_err'] is not None
                        else f"{k}=NaN" for k in METHODS[:5]))
        print("          ll: " + "  ".join(f"{k}={sc['methods'][k]['ll']:.6g}" for k in METHODS))
    json.dump(out, open(os.path.join(BUILD, "f_emp.json"), "w"), allow_nan=False)
    print("json", os.path.getsize(os.path.join(BUILD, "f_emp.json")), "bytes")
