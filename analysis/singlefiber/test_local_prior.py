#!/usr/bin/env python
"""Tests for the local prior (singlefiber/local_prior.py), 2026-11-03.

1. TOY (numpy, exact): in a small HMM with a silent junction like RoboCOP's (bg -> junction J -> TF silent
   parent S -> TF first state; first state's only parent is S), weighting the first state's emission by w at
   ONE row t gives the same posterior as multiplying the entry transition J->S by w on the step into row t
   only (time-inhomogeneous, unnormalised). Max |posterior difference| must be < 1e-10. Also: the
   '-log(w) on every other emitting state' form gives the same posterior.
2. REAL decoder (the driver's default model, log space, prior-only read over chrI:60001-65000):
   a. w = 1 gives a posterior bit-identical to no hook;
   b. the odds of "ABF1 starts at the motif row" (sum of the two first states' posteriors) change by
      exactly w (relative error < 1e-8), both for w > 1 (other-state form) and w < 1 (first-state form);
   c. every TF first state has exactly one parent (check_single_parent).

Run from analysis/:  ROBOCOP_LOG_EMISSION=exact python singlefiber/test_local_prior.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ---------------------------------------------------------------------------------------------
# 1. toy HMM with silent states, explicit forward-backward in log space
# ---------------------------------------------------------------------------------------------
def toy_model(L=4, p_tf=1e-3):
    # emitting: 0 bg, 1..L TF; silent: L+1 junction J, L+2 TF parent S
    n_e = 1 + L
    J, S = n_e, n_e + 1
    n = n_e + 2
    T = np.zeros((n, n))
    T[0, 0] = 0.9; T[0, J] = 0.1                 # bg -> bg or junction
    T[J, 0] = 1 - p_tf; T[J, S] = p_tf           # junction -> bg or TF
    T[S, 1] = 1.0                                # TF parent -> first state (its ONLY parent)
    for k in range(1, L):
        T[k, k + 1] = 1.0
    T[L, J] = 1.0                                # TF end -> junction
    return T, n_e, J, S


def emit_step(T, n_e):
    """Emitting -> emitting one-step matrix with silent states collapsed (T_EE + T_ES (I-T_SS)^-1 T_SE)."""
    TEE, TES, TSE, TSS = T[:n_e, :n_e], T[:n_e, n_e:], T[n_e:, :n_e], T[n_e:, n_e:]
    return TEE + TES @ np.linalg.solve(np.eye(T.shape[0] - n_e) - TSS, TSE)


def fb(logE, steps, init):
    """logE (n_obs, n_e); steps[t] = emitting->emitting matrix into row t (t >= 1). Posterior (n_obs, n_e)."""
    n_obs, n_e = logE.shape
    with np.errstate(divide="ignore"):
        lf = np.full((n_obs, n_e), -np.inf)
        lb = np.zeros((n_obs, n_e))
        lf[0] = np.log(init) + logE[0]
        for t in range(1, n_obs):
            lf[t] = logsumexp(lf[t - 1][:, None] + np.log(steps[t]), axis=0) + logE[t]
        for t in range(n_obs - 2, -1, -1):
            lb[t] = logsumexp(np.log(steps[t + 1]) + (logE[t + 1] + lb[t + 1])[None, :], axis=1)
    lp = lf + lb
    return np.exp(lp - logsumexp(lp, axis=1, keepdims=True))


def test_toy(seed=0):
    rng = np.random.default_rng(seed)
    L, n_obs, t0, w = 4, 40, 17, 250.0
    T, n_e, J, S = toy_model(L)
    P = emit_step(T, n_e)
    init = np.ones(n_e) / n_e
    logE = np.log(rng.uniform(0.2, 1.0, (n_obs, n_e)))
    steps = [P] * n_obs
    # (a) emission weight on the first state at row t0
    E1 = logE.copy(); E1[t0, 1] += np.log(w)
    pa = fb(E1, steps, init)
    # (b) entry transition J->S times w on the step into row t0 only (unnormalised)
    T2 = T.copy(); T2[J, S] *= w
    steps_b = list(steps); steps_b[t0] = emit_step(T2, n_e)
    pb = fb(logE, steps_b, init)
    # (c) -log(w) on every other emitting state at row t0
    E3 = logE.copy(); E3[t0, :] -= np.log(w); E3[t0, 1] += np.log(w)
    pc = fb(E3, steps, init)
    p0 = fb(logE, steps, init)
    d_ab, d_ac = float(np.abs(pa - pb).max()), float(np.abs(pa - pc).max())
    odds = lambda p: p[t0, 1] / (1 - p[t0, 1])
    print("toy: max|emission weight - entry transition| = %.3g, max|first-state form - other-state form| = %.3g, "
          "odds ratio %.6f (w %g), P(first state at t0) %.4g -> %.4g" % (d_ab, d_ac, odds(pa) / odds(p0), w, p0[t0, 1], pa[t0, 1]))
    assert d_ab < 1e-10 and d_ac < 1e-10
    assert abs(odds(pa) / odds(p0) / w - 1) < 1e-10
    return d_ab, d_ac


# ---------------------------------------------------------------------------------------------
# 2. real decoder
# ---------------------------------------------------------------------------------------------
def test_real():
    import calibrate_local_prior as CL
    import local_prior as LP
    dec = CL.PriorDecoder("chrI:60001-65000", tmp=os.path.join("/tmp", "test_local_prior_%d" % os.getpid()))
    LP.check_single_parent(dec.ds)
    s1, e1 = 62659, 62672
    i = list(dec.ds["tfs"]).index(CL.TF)
    s, L = int(dec.ds["tf_starts"][i]), int(dec.ds["tf_lens"][i])
    row = s1 - dec.s1

    def tab(w):
        return pd.DataFrame([dict(name="site2", chrom="chrI", tf=CL.TF, motif_start1=s1, motif_end1=e1, w=w,
                                  heldout_run="pooled")])
    p0, _ = dec.decode(None)
    p1, _ = dec.decode(tab(1.0))
    assert np.array_equal(p0, p1), "w=1 changed the posterior"
    print("real: w=1 bit-identical to no hook: True")
    q0 = p0[row, s] + p0[row, s + L]
    out = []
    for w in (1e3, 1e5, 1e-2):
        pw, _ = dec.decode(tab(w))
        q = pw[row, s] + pw[row, s + L]
        ratio = (q / (1 - q)) / (q0 / (1 - q0))
        rel = abs(ratio / w - 1)
        print("real: w=%g  P(ABF1 starts at %d) %.4g -> %.4g  odds ratio / w - 1 = %.2e" % (w, s1, q0, q, rel))
        assert rel < 1e-8, rel
        out.append(rel)
    # flipped frame: same placement, weight via the flipped-frame rows
    pf0, _ = dec.decode(None, flipped=True)
    pf, _ = dec.decode(tab(1e5), flipped=True)
    # in the ref frame the flipped first states map to the LAST states of the ref blocks at e1
    qf0 = pf0[e1 - dec.s1, s + 2 * L - 1] + pf0[e1 - dec.s1, s + L - 1]
    qf = pf[e1 - dec.s1, s + 2 * L - 1] + pf[e1 - dec.s1, s + L - 1]
    rel_f = abs((qf / (1 - qf)) / (qf0 / (1 - qf0)) / 1e5 - 1)
    print("real flipped: P(ABF1 on [%d,%d]) %.4g -> %.4g  odds ratio / w - 1 = %.2e" % (s1, e1, qf0, qf, rel_f))
    assert rel_f < 1e-8
    return out, rel_f


if __name__ == "__main__":
    test_toy()
    test_real()
    print("ALL PASS")
