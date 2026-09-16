"""Build a trainDir from decoder WEIGHTS w (tuner v2), with a w-reproduction gate.

What the decoder actually sees
------------------------------
`set_transition` writes one row of the transition matrix: p_bg, p_nuc and p_k for every DBF
(make_conc_trainDir.py explains why nothing else depends on concentration). Along any path the
product of those transitions is  p_bg^N * prod_k (p_k / p_bg^{L_k}),  so the posterior depends
only on the weights

    w_k = p_k / p_bg^{L_k}            (background w = 1 by construction)

which are exactly the "concentrations" getDBFconc feeds convert_to_prob: w_TF = Kd*lambda,
w_unknown = 0.1, w_background = 1, w_nucleosome = 35. This writer takes w as the primary object:

    w_TF         = calculateKD(pwm, TF) * lambda_TF
    w_unknown    = UNKNOWN_BASE * lambda_unknown       UNKNOWN_BASE = 0.001 (REBASED, see below)
    w_nucleosome = 35 * lambda_nucleosome              (tuner v2 never moves it: 35)
    w_background = 1

and solves the unbound root a from  sum_k w_k a^{L_k} = 1  by brentq on x = ln a in log space
(logsumexp; no np.roots), then p_k = w_k a^{L_k} / sum.

unknown is REBASED
------------------
parameterize.getDBFconc hardcodes w_unknown = 0.1. Every campaign since u001 ran unknown at
lambda 0.01, i.e. w = 1e-3. Here the base is 0.001, so lambda_unknown = 1 reproduces today's
value (w_unknown = 1e-3, identical to the old lambda 0.01 x 0.1) and the pristine trainDir's 0.1
is lambda_unknown = 100. pkg/robocop/utils/parameterize.py is NOT edited; only this writer knows.

Gates (every build; any failure aborts before anything is written)
------------------------------------------------------------------
 (a) SOLVER SELF-CHECK: the src's own weights, read back from src/HMMconfig.pkl, pushed through
     this solver reproduce the src row (tf_prob, background_prob, nucleosome_prob) to rel <= 1e-9.
     (np.roots vs brentq, so not bit-exact.)
 (c) BLAST RADIUS: only row silent_states_begin of transition_matrix changes (initial_probs are
     re-derived from it; end_probs must not change).
 (d) W-REPRODUCTION: re-read the WRITTEN HMMconfig.pkl and require
     |ln(tf_prob / bg^L) - ln w_requested| <= 1e-9 for every DBF and the nucleosome, every prob
     finite and > 0, and w^(1/L) <= RHO_MAX for every TF motif (unknown exempt).
Gate (b), the legacy cross-check against robocop_train_ct_fm002_00, is `--compare <trainDir>`.

Usage
-----
    python make_w_trainDir.py --src robocop_train_fiberonly --out D --lam lam.json
        lam.json: {"motif": lambda, ..., "unknown": lambda_unknown}; absent names -> lambda 1
    python make_w_trainDir.py --out D --lam '{}' --compare robocop_train_ct_fm002_00
"""
import argparse
import json
import math
import os
import pickle
import shutil
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '../pkg/'))
import make_conc_trainDir as MC                                     # noqa: E402  (import only)
from robocop.utils.parameterize import calculateKD                  # noqa: E402

UNKNOWN_BASE = 0.001      # rebased from parameterize.py's 0.1 (lambda_unknown 1 -> w 1e-3)
NUC_BASE = 35.0
NUC_LEN = 147
RHO_MAX = 0.70            # per-bp weight cap on TF motifs: w^(1/L) <= 0.70
TOL = 1e-9


def motif_len(pwm, name):
    return int(pwm[name].shape[1])


def base_weights(pwm, tfs):
    """lambda = 1 weights, as ln w, for every DBF in cfg order, plus background/nucleosome."""
    th = {}
    for t in tfs:
        if t == 'unknown':
            th[t] = math.log(UNKNOWN_BASE)
        else:
            kd = float(calculateKD(pwm, t))
            if not (kd > 0 and math.isfinite(kd)):
                raise ValueError("Kd of %s is %r" % (t, kd))
            th[t] = math.log(kd)
    th['background'] = 0.0
    th['nucleosome'] = math.log(NUC_BASE)
    return th


def solve_probs(theta, lens):
    """theta: {name: ln w}; lens: {name: L}. Returns ({name: p}, ln a). All in log space."""
    names = list(theta)
    th = np.array([theta[n] for n in names], dtype=float)
    L = np.array([lens[n] for n in names], dtype=float)
    if not np.all(np.isfinite(th)):
        raise ValueError("non-finite ln w for %s" % [n for n, v in zip(names, th) if not math.isfinite(v)])

    def g(x):
        return logsumexp(th + L * x)
    lo = -1.0
    while g(lo) > 0:
        lo *= 2.0
        if lo < -1e6:
            raise ValueError("cannot bracket the unbound root")
    hi = 0.0
    if not g(hi) > 0:
        raise ValueError("sum of weights <= 1; no root in (0,1)")
    x = brentq(g, lo, hi, xtol=1e-17, rtol=4 * np.finfo(float).eps, maxiter=500)
    lp = th + L * x
    lp = lp - logsumexp(lp)                  # renormalise, as getDBFconc does
    p = np.exp(lp)
    if not (np.all(np.isfinite(p)) and np.all(p > 0)):
        raise ValueError("a transition probability underflowed to 0 or is non-finite")
    return dict(zip(names, p.tolist())), float(x)


def cfg_weights(cfg, pwm):
    """ln w read back from a (written or source) HMMconfig: ln(p_k) - L_k ln(p_bg)."""
    bg = float(cfg['background_prob'])
    tfs = list(cfg['tfs'])
    tfp = np.asarray(cfg['tf_prob'], dtype=float)
    lens = np.asarray(cfg['tf_lens'])
    out = {t: math.log(tfp[i]) - int(lens[i]) * math.log(bg) for i, t in enumerate(tfs)}
    out['nucleosome'] = math.log(float(cfg['nucleosome_prob'])) - NUC_LEN * math.log(bg)
    out['background'] = 0.0
    return out


def row_of(cfg):
    return (np.asarray(cfg['tf_prob'], dtype=float).copy(), float(cfg['background_prob']),
            float(cfg['nucleosome_prob']))


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a / b - 1.0)))


def lens_of(cfg, pwm):
    tfs = list(cfg['tfs'])
    lens = {t: int(cfg['tf_lens'][i]) for i, t in enumerate(tfs)}
    for t in tfs:
        if motif_len(pwm, t) != lens[t]:
            raise ValueError("%s: pwm length %d != cfg tf_len %d (widened trainDir? not supported)"
                             % (t, motif_len(pwm, t), lens[t]))
    lens['background'] = 1
    lens['nucleosome'] = NUC_LEN
    return lens


def build(src, lam, out=None, rho_check=True, compare=None, quiet=False):
    """lam: {name: lambda} over cfg['tfs'] (incl unknown) and optionally 'nucleosome'.
    Returns the patch dict. out=None runs every gate but writes nothing."""
    say = (lambda *a: None) if quiet else print
    pwm = pickle.load(open(os.path.join(src, "pwm.p"), "rb"))
    cfg = pickle.load(open(os.path.join(src, "HMMconfig.pkl"), "rb"), encoding="latin1")
    cfg['robocopC'] = MC.CSHARED
    tfs = list(cfg['tfs'])
    ssb = int(cfg['silent_states_begin'])
    lens = lens_of(cfg, pwm)
    bad = [k for k in lam if k not in set(tfs) | {'nucleosome'}]
    if bad:
        raise ValueError("not in this model: %s" % bad)
    for k, v in lam.items():
        if not (isinstance(v, (int, float)) and v > 0 and math.isfinite(v)):
            raise ValueError("lambda %s = %r is not a finite positive number" % (k, v))

    src_tf, src_bg, src_nuc = row_of(cfg)
    src_tmat = cfg['transition_matrix'].copy()
    src_ip = np.asarray(cfg['initial_probs']).copy()
    src_ep = np.asarray(cfg['end_probs']).copy()

    # ---- gate (a): the solver reproduces the src row from the src's own weights ----
    th_src = cfg_weights(cfg, pwm)
    p_a, _ = solve_probs(th_src, lens)
    da = max(rel([p_a[t] for t in tfs], src_tf), rel(p_a['background'], src_bg),
             rel(p_a['nucleosome'], src_nuc))
    say("[gate a] solver on src weights reproduces src row: max rel %.3e" % da)
    if not da <= TOL:
        raise RuntimeError("GATE (a) FAILED: rel %.3e > %g" % (da, TOL))

    # ---- requested weights ----
    th = base_weights(pwm, tfs)
    for k, v in lam.items():
        th[k] += math.log(v)
    if rho_check:
        over = {t: math.exp(th[t] / lens[t]) for t in tfs
                if t != 'unknown' and th[t] / lens[t] > math.log(RHO_MAX) + 1e-12}
        if over:
            raise RuntimeError("RHO CAP: w^(1/L) > %.2f for %s" % (
                RHO_MAX, ", ".join("%s %.3f" % kv for kv in sorted(over.items()))))
    prob, lna = solve_probs(th, lens)
    MC.apply_probs(cfg, prob)

    # ---- gate (c): blast radius ----
    diff = np.abs(cfg['transition_matrix'] - src_tmat)
    rows = sorted(set(np.argwhere(diff > 0)[:, 0].tolist()))
    n_ep = int((np.asarray(cfg['end_probs']) != src_ep).sum())
    n_ip = int((np.asarray(cfg['initial_probs']) != src_ip).sum())
    say("[gate c] transition rows changed %s (ssb %d); initial_probs changed %d; end_probs changed %d"
        % (rows, ssb, n_ip, n_ep))
    if any(r != ssb for r in rows) or n_ep:
        raise RuntimeError("GATE (c) FAILED: rows %s, end_probs %d" % (rows, n_ep))

    def check_w(c, label):
        back = cfg_weights(c, pwm)
        dw = max(abs(back[k] - th[k]) for k in list(tfs) + ['nucleosome'])
        probs = np.concatenate([np.asarray(c['tf_prob'], float), [c['background_prob'], c['nucleosome_prob']]])
        finite = bool(np.all(np.isfinite(probs)) and np.all(probs > 0))
        rho = max(math.exp(back[t] / lens[t]) for t in tfs if t != 'unknown')
        say("[gate d] %s: max |ln w_back - ln w_req| %.3e; probs finite>0 %s; max TF rho %.4f"
            % (label, dw, finite, rho))
        if not (dw <= TOL and finite and (rho <= RHO_MAX + 1e-9 or not rho_check)):
            raise RuntimeError("GATE (d) FAILED (%s): dw %.3e finite %s rho %.4f" % (label, dw, finite, rho))
        return dw, rho

    dw, rho = check_w(cfg, "in memory")

    cmp = None
    if compare:
        ref = pickle.load(open(os.path.join(compare, "HMMconfig.pkl"), "rb"), encoding="latin1")
        r_tf, r_bg, r_nuc = row_of(ref)
        n_tf = np.asarray(cfg['tf_prob'], float)
        d_row = max(rel(n_tf, r_tf), rel(cfg['background_prob'], r_bg), rel(cfg['nucleosome_prob'], r_nuc))
        rw = cfg_weights(ref, pwm)
        d_w = max(abs(rw[k] - th[k]) for k in list(tfs) + ['nucleosome'])
        d_t = float(np.max(np.abs(cfg['transition_matrix'] - ref['transition_matrix'])))
        d_ip = float(np.max(np.abs(np.asarray(cfg['initial_probs']) - np.asarray(ref['initial_probs']))))
        cmp = dict(ref=compare, max_rel_row=d_row, max_abs_lnw=d_w, max_abs_tmat=d_t, max_abs_initial=d_ip)
        say("[compare] vs %s: row max rel %.3e | ln w max abs %.3e | tmat max abs %.3e | initial max abs %.3e"
            % (compare, d_row, d_w, d_t, d_ip))
        if not (d_row <= TOL and d_w <= TOL):
            raise RuntimeError("COMPARE FAILED vs %s" % compare)

    patch = dict(
        src=os.path.abspath(src), out=os.path.abspath(out) if out else None,
        unknown_base=UNKNOWN_BASE, nucleosome_base=NUC_BASE, rho_max=RHO_MAX,
        lam={k: float(v) for k, v in lam.items()},
        w={k: math.exp(v) for k, v in th.items()},
        ln_w={k: v for k, v in th.items()},
        ln_unbound_root=lna,
        background_prob_before=src_bg, background_prob_after=float(prob['background']),
        nucleosome_prob_before=src_nuc, nucleosome_prob_after=float(prob['nucleosome']),
        transition_rows_changed=rows, silent_states_begin=ssb,
        gate_a_rel=da, gate_d_max_abs_lnw=dw, max_tf_rho=rho, compare=cmp,
        note="w_unknown = UNKNOWN_BASE * lam_unknown; lam_unknown 1 -> 1e-3, identical to the "
             "legacy lambda 0.01 x 0.1")
    if out is None:
        return patch

    if os.path.abspath(out) == os.path.abspath(src):
        raise RuntimeError("--out must differ from --src")
    if os.path.exists(out):
        raise RuntimeError("%s exists; refusing to overwrite" % out)
    os.makedirs(out)
    for fn in MC.SIDECARS:
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(out, fn))
    with open(os.path.join(out, "HMMconfig.pkl"), "wb") as f:
        pickle.dump(cfg, f)
    # gate (d) on the WRITTEN file
    written = pickle.load(open(os.path.join(out, "HMMconfig.pkl"), "rb"), encoding="latin1")
    try:
        check_w(written, "written pkl")
    except RuntimeError:
        os.rename(out, out + ".FAILED_GATE_D")
        raise
    with open(os.path.join(out, "w_patch.json"), "w") as f:
        json.dump(patch, f, indent=1, sort_keys=True)
    say("wrote %s (HMMconfig.pkl + w_patch.json + sidecars)" % out)
    return patch


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="robocop_train_fiberonly")
    ap.add_argument("--out", default=None, help="omit to run the gates only")
    ap.add_argument("--lam", required=True, help="JSON file or inline JSON {name: lambda}")
    ap.add_argument("--compare", default=None, help="trainDir whose row/weights must match (gate b)")
    ap.add_argument("--no-rho-check", action="store_true", help="testing only")
    a = ap.parse_args()
    lam = json.load(open(a.lam)) if os.path.isfile(a.lam) else json.loads(a.lam)
    try:
        build(a.src, lam, a.out, rho_check=not a.no_rho_check, compare=a.compare)
    except (RuntimeError, ValueError) as e:
        sys.exit("ABORT: %s" % e)


if __name__ == "__main__":
    main()
