#!/usr/bin/env python
"""Calibrate the local-prior weight w per site (2026-11-03, plan let-s-go-with-mad-snazzy-lollipop, Part 2).

For each site and each f (pooled, and f with each sequencing run held out), find w such that the
PRIOR-ONLY single-fiber decode gives P(ABF1 at the site) = f within --tol (default 1e-3).

  prior-only  = the driver's default model (traindir, stationary start, arm A ps), sequence layer on,
                m6A layers carry no information (every call NO_OBS, so layers 5/6 are log 1 = 0.0);
                one synthetic read spanning the whole --window, decoded in the reference frame.
  P(ABF1 at the site) = max over the motif interval of the collapsed ABF1 column, the per-read metric
                of abf1sweep/analyze_sweep.py and of the comparison report.

The sites of one table are calibrated jointly (coordinate passes: each site bisected in log w with the
others' current w applied) until no site moves; the final check decodes once with all weights and reports
|P - f| per site in the reference AND the flipped (Crick 5'->3') frame.

Usage (from analysis/):
  ROBOCOP_LOG_EMISSION=exact python singlefiber/calibrate_local_prior.py \
      --sites-f /usr/project/xtmp/nd141/tf_compare/mixture_filtnordna_pb0p02/sites_f.tsv \
      --names mac_chrI_61165,mac_chrI_62659 --window chrI:60001-65000 --out <dir>/sites_w.tsv
Writes sites_w.tsv (+ sites_w.json with the bisection trace and the final check).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, sys.argv[:1]          # the driver module parses --pkg-tree at import
import run_permol_window_logfull_5to3 as R5        # noqa: E402  (os.chdir(analysis), tree on sys.path)
sys.argv = _argv
import local_prior as LP                           # noqa: E402
from robocop.utils.getNucleotides import getNucleotideSequence  # noqa: E402

D, PP, C, RC = R5.D, R5.PP, R5.C, R5.RC
TF = "Abf1_murphy"


class PriorDecoder:
    """The driver's default model, set up once; decode a call-free read over a window."""

    def __init__(self, window, traindir=R5.TRAINDIR_DEFAULT, tmp="/tmp"):
        import pickle
        R5.robocop.log_emission_mode()
        assert D.so_md5(R5.LOGFULL_SO) == R5.LOGFULL_SO_MD5
        D.install_posterior_grab()
        ds = pickle.load(open(os.path.join(traindir, "HMMconfig.pkl"), "rb"))
        ds["robocopC"] = R5.LOGFULL_SO
        os.makedirs(os.path.join(tmp, "tmpDir"), exist_ok=True)
        ds["tmpDir"] = os.path.join(tmp, "tmpDir") + "/"
        ds["nucleotides"] = 1
        ds["cache_parents_children"] = True
        ds["info_file"] = None
        init, self.start_info = R5.stationary_initial(ds)
        ds["initial_probs"] = init
        pf = dict(nuc_file=R5.NUC_PARAMS_DEFAULT, bg_file=R5.BG_PARAMS_DEFAULT, tf_file=R5.TF_PARAMS_DEFAULT)
        PP.assert_verbatim(ds, "watson_signal", **pf)
        PP.assert_verbatim(ds, "crick_signal", **pf)
        self.ps_w = PP.arm_ps(ds, "watson_signal", "A", None, **pf)[0]
        self.ps_c = PP.arm_ps(ds, "crick_signal", "A", None, **pf)[0]
        self.sigma, self.special = R5.state_mirror(ds)
        self.ps_flip = R5.mirror_ps(self.ps_c, self.sigma, self.special, ds)
        self.ds = ds
        self.ssb = ds["silent_states_begin"]
        self.M = C.collapse_matrix(ds)
        cols = C.column_names(ds)
        self.cols = cols
        self.keep_idx = np.arange(len(cols))
        self.ia = cols.index(TF)
        LP.check_single_parent(ds)
        chrom, rng = window.split(":")
        self.chrom = chrom
        self.s1, self.e1 = (int(x) for x in rng.split("-"))
        self.codes = np.asarray(getNucleotideSequence(R5.FASTA, chrom, self.s1, self.e1))
        self.traindir = traindir

    def decode(self, rows, flipped=False, hook_extra=None):
        """rows: local-prior table rows (DataFrame) or None. Returns (posterior ref frame, optable)."""
        n = self.e1 - self.s1 + 1
        no = np.full(n, RC.NO_OBS, dtype=np.int8)
        pl = LP.placements(rows, self.ds, self.s1, self.e1, self.chrom, flipped) if rows is not None else []
        hook = LP.make_hook(pl, self.ssb, inner=hook_extra) if (pl or hook_extra) else None
        if not flipped:
            p, op, _ = D.decode_one(self.ds, self.chrom, self.s1, self.e1, self.codes, self.ps_w, self.ps_c,
                                    no, no, self.M, self.keep_idx, emission_hook=hook)
        else:
            cf = R5.revcomp_codes(self.codes)[::-1].copy()
            pf, _, _ = D.decode_one(self.ds, self.chrom, self.s1, self.e1, cf, self.ps_flip, self.ps_c,
                                    no, no, self.M, self.keep_idx, emission_hook=hook)
            p = R5.map_posterior(pf, self.sigma, self.special, self.codes, self.ds)
            op = C.collapse(p, self.M)
        return p, op

    def site_p(self, op, s1, e1):
        return float(op[s1 - self.s1:e1 - self.s1 + 1, self.ia].max())


def calibrate(dec, tab, tol=1e-3, max_pass=6, log_lo=-6.0, log_hi=14.0, trace=None):
    """tab: rows of ONE table (same heldout_run). Adds column w. Coordinate bisection in log10 w."""
    tab = tab.copy()
    tab["w"] = 1.0
    for ps in range(max_pass):
        moved = False
        for k in tab.index:
            r = tab.loc[k]
            f = float(r.f)

            def P(lw):
                t2 = tab.copy(); t2.loc[k, "w"] = 10.0 ** lw
                return dec.site_p(dec.decode(t2)[1], int(r.motif_start1), int(r.motif_end1))
            cur = P(np.log10(tab.loc[k, "w"]))
            if abs(cur - f) < tol / 4:
                continue
            lo, hi = log_lo, log_hi
            plo, phi = P(lo), P(hi)
            assert plo <= f <= phi, "site %s: f=%g outside [P(w=1e%g)=%g, P(w=1e%g)=%g]" % (r["name"], f, lo, plo, hi, phi)
            it = 0
            while True:
                mid = 0.5 * (lo + hi)
                pm = P(mid)
                it += 1
                if trace is not None:
                    trace.append(dict(pass_=ps, name=r["name"], heldout_run=r.heldout_run, log10w=mid, P=pm, f=f))
                if abs(pm - f) < tol / 4 or it > 60:
                    break
                if pm < f:
                    lo = mid
                else:
                    hi = mid
            if abs(np.log10(tab.loc[k, "w"]) - mid) > 1e-9:
                moved = True
            tab.loc[k, "w"] = 10.0 ** mid
        if not moved:
            break
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-f", required=True)
    ap.add_argument("--names", required=True, help="comma-separated site names from sites_f.tsv")
    ap.add_argument("--window", default="chrI:60001-65000")
    ap.add_argument("--traindir", default=R5.TRAINDIR_DEFAULT)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    sf = pd.read_csv(a.sites_f, sep="\t")
    names = a.names.split(",")
    sf = sf[sf.name.isin(names)]
    assert len(sf) == len(names), (names, sf.name.tolist())
    runs = sorted(c[len("f_heldout_"):] for c in sf.columns if c.startswith("f_heldout_"))
    rows = []
    for _, r in sf.iterrows():
        base = dict(name=r["name"], chrom=r.chrom, tf=TF, motif_start1=int(r.motif_start1),
                    motif_end1=int(r.motif_end1), site_strand=r.strand)
        rows.append(dict(base, heldout_run="pooled", f=float(r.f)))
        for rn in runs:
            assert np.isfinite(r["f_heldout_" + rn]), (r["name"], rn)
            rows.append(dict(base, heldout_run=rn, f=float(r["f_heldout_" + rn])))
    tab = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    dec = PriorDecoder(a.window, a.traindir, tmp=out_dir)
    # w = 1 baseline (no weight)
    _, op0 = dec.decode(None)
    base = {r["name"]: dec.site_p(op0, r.motif_start1, r.motif_end1) for _, r in sf.iterrows()}
    print("prior-only P(ABF1) with no local prior:", base, flush=True)
    trace, outs, check = [], [], []
    for rn, g in tab.groupby("heldout_run", sort=False):
        c = calibrate(dec, g, tol=a.tol, trace=trace)
        for flipped in (False, True):
            _, op = dec.decode(c, flipped=flipped)
            for k in c.index:
                pv = dec.site_p(op, int(c.loc[k, "motif_start1"]), int(c.loc[k, "motif_end1"]))
                c.loc[k, "P_flipped" if flipped else "P_ref"] = pv
                check.append(dict(heldout_run=rn, name=c.loc[k, "name"], frame="flipped" if flipped else "ref",
                                  f=float(c.loc[k, "f"]), w=float(c.loc[k, "w"]), P=pv,
                                  abs_err=abs(pv - float(c.loc[k, "f"]))))
        print(c[["name", "heldout_run", "f", "w", "P_ref", "P_flipped"]].to_string(), flush=True)
        outs.append(c)
    res = pd.concat(outs)
    res["P_no_prior"] = res.name.map(base)
    res.to_csv(a.out, sep="\t", index=False)
    ck = pd.DataFrame(check)
    summ = dict(sites_f=a.sites_f, window=a.window, traindir=a.traindir, tol=a.tol, tf=TF,
                start_info=dec.start_info, P_no_prior=base,
                max_abs_err_ref=float(ck[ck.frame == "ref"].abs_err.max()),
                max_abs_err_flipped=float(ck[ck.frame == "flipped"].abs_err.max()),
                check=check, n_decodes=len(trace), secs=time.time() - t0, trace=trace)
    json.dump(summ, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
    print("max |P - f|: ref %.2e  flipped %.2e  (%d bisection decodes, %.0fs)" % (
        summ["max_abs_err_ref"], summ["max_abs_err_flipped"], len(trace), summ["secs"]))
    assert summ["max_abs_err_ref"] < a.tol, "calibration failed"
    return 0


if __name__ == "__main__":
    sys.exit(main())
