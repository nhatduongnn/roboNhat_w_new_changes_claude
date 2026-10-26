#!/usr/bin/env python3
"""
build/f_d.json -- the model / data object ("D" in the audit page) for the fiberCOP "c1" scenario.

Keys kept from the audit (c1_data_bf.json / bf_tab.json): short, names, seq, S, n, emit, init, trans, dbl_min.
New key "fiber": the Fiber-seq counts and the per-layer emission factors (see build/SCHEMA.md).

Run:  python gen_f_d.py      -> build/f_d.json
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fiber_model as FM

BUILD = os.path.join(HERE, "build"); os.makedirs(BUILD, exist_ok=True)
SHORT = ["BG", "F1", "F2", "R1", "R2", "N1", "N2", "N3a", "N3b", "N4", "CENTRAL", "TF-entry"]   # gen_bf.py SHORT
nan2none = lambda x: None if (x is None or not np.isfinite(x)) else float(x)

def build(key="c1"):
    scen = {k: d for k, _, _, d in FM.SCENARIOS}[key]
    lib = FM.load_lin(FM.LIB_PRENORM)
    m = FM.make_model(lib, **scen)
    n, S, N = m.T_obs, m.S, m.N
    E = m.E.prod(0)                                          # (n, N) 7-layer product, silent columns = 1
    scored = ["W" if c == "A" else ("C" if c == "T" else None) for c in m.seq]
    li_of = {"W": 0, "C": 1}
    floored = [[bool(scored[t] is not None and m.raw_pmf[li_of[scored[t]], t, j] == 0.0) for j in range(S)] for t in range(n)]
    # sanity: a floored cell's exact pmf is below the smallest subnormal (log10 < -323.3); an unfloored one is not
    for t in range(n):
        for j in range(S):
            if scored[t] is not None:
                lg = m.raw_log10[li_of[scored[t]], t, j]
                assert (lg < -323.0) if floored[t][j] else (lg > -324.0), (t, j, lg)
    raw_log10 = [[[nan2none(m.raw_log10[s, t, j]) for j in range(S)] for t in range(n)] for s in range(2)]
    with np.errstate(divide="ignore"):
        emit_log10_floored = [[nan2none(v / np.log(10)) for v in m.LE_fl.sum(0)[t, :S]] for t in range(n)]
        emit_log10_exact = [[nan2none(v / np.log(10)) for v in m.LE_ex.sum(0)[t, :S]] for t in range(n)]
    fiber = dict(
        seq=m.seq,
        n_w=list(map(int, m.n_w)), k_w=list(map(int, m.k_w)), n_c=list(map(int, m.n_c)), k_c=list(map(int, m.k_c)),
        scored=scored,
        p_watson=list(map(float, FM.P["watson_signal"])), p_crick=list(map(float, FM.P["crick_signal"])),
        raw_log10=raw_log10,
        floored=floored,
        n_floored=int(sum(map(sum, floored))),
        layers=[[[float(m.E[0, t, j]), float(m.E[5, t, j]), float(m.E[6, t, j])] for j in range(S)] for t in range(n)],
        layer_names=["sequence PWM (layer 0)", "Fiber-seq Watson (layer 5)", "Fiber-seq Crick (layer 6)"],
        floor=FM.FLOOR,
        emit_log10_floored=emit_log10_floored, emit_log10_exact=emit_log10_exact,
    )
    # the three listed factors are the whole product: MNase/ATAC layers 1-4 are exactly 1
    assert np.all(m.E[1:5] == 1.0)
    assert np.allclose(np.array(fiber["layers"]).prod(-1), E[:, :S], rtol=1e-14, atol=0)
    trans = [[int(i), int(j), float(m.T[i, j])] for i in range(N) for j in range(N) if m.T[i, j] != 0.0]
    return dict(scenario=key, short=SHORT, names=FM.W.NAMES, seq=m.seq, S=S, n=n, N=N, n_vars=int(m.E.shape[0]),
                emit=[[float(E[t, j]) for j in range(S)] for t in range(n)],
                maxEmit=[float(E[t, :S].max()) for t in range(n)],
                init=list(map(float, m.init)), end=list(map(float, m.end)), trans=trans,
                dbl_min=float(np.finfo(float).tiny), fiber=fiber)

if __name__ == "__main__":
    d = build("c1")
    json.dump(d, open(os.path.join(BUILD, "f_d.json"), "w"), allow_nan=False)
    f = d["fiber"]
    print("seq", d["seq"], "scored", f["scored"], "floored cells", f["n_floored"])
    for t in range(d["n"]):
        print("t=%d %s  emit log10 floored: %s" % (t, d["seq"][t], " ".join("%.0f" % (v if v is not None else -np.inf) for v in f["emit_log10_floored"][t])))
    print("json", os.path.getsize(os.path.join(BUILD, "f_d.json")), "bytes")
