"""Calibrate the sliding-ABF1 matched filter: what is comparable, and where to cut.

Two different questions get two different statistics, and conflating them is what makes
S confusing:

  S = <w, y> / ||w||            "how much EVIDENCE that ABF1 is here"
  r = <w, y-ybar> / (||w|| ||y-ybar||)   "how much this window LOOKS LIKE ABF1"   in [-1, 1]

and they are related exactly by

  S = r * ||y - ybar||                  evidence = shape agreement x signal magnitude

so S mixes a bounded shape score with an unbounded, depth-growing magnitude. This script
measures both genome-wide, tests empirically whether S's NULL depends on read depth (if
it does not, S is comparable across depths as a detection statistic even though it is not
comparable as a similarity), and converts S into a calibrated FDR using a permuted-
template null.

Null model: randomly permute the template's column order (per channel), keeping the exact
same weights. That destroys ABF1's shape while preserving its marginal weight
distribution, the data, the coverage structure and the local background -- so anything a
permuted template still scores is what the filter would find by chance.

    conda activate robocop-2024
    export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
    python abf1_slide_calibrate.py --halves 7,12 --nperm 40
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

import fiber_params_lib
import score_robocop as S_
from make_params_pm50 import BED, PILEUP, Pileup, fit_group
from slide_abf1_profile import BG_HALF, MACISAAC, TF, dense_counts, residuals, call_peaks

SEED = 0


def stepup(fdr):
    """Monotone (step-up) FDR, nan-safe: nan means 'no calls at this threshold'."""
    out = np.array(fdr, float)
    run = np.inf
    for i in range(len(out) - 1, -1, -1):
        if np.isnan(out[i]):
            continue
        run = min(run, out[i])
        out[i] = run
    return out


def phase_randomize(w, rng):
    """Same power spectrum -> same autocorrelation/smoothness, different shape.

    A plain column permutation destroys the template's spatial smoothness as well as its
    shape, so a smooth template beats a permuted one at ANY position where y varies
    smoothly -- which is everywhere. Phase randomisation is the null that isolates shape.
    """
    F = np.fft.rfft(w)
    ph = rng.uniform(0, 2 * np.pi, F.size)
    ph[0] = 0.0
    if w.size % 2 == 0:
        ph[-1] = 0.0
    out = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=w.size)
    return out - out.mean()


def windowed(y, width):
    """Per-window mean and centred L2 norm of y, aligned to the window CENTRE."""
    k = np.ones(width)
    s1 = np.convolve(y, k, mode="same")
    s2 = np.convolve(y * y, k, mode="same")
    mean = s1 / width
    var = np.maximum(s2 / width - mean ** 2, 0.0)
    return mean, np.sqrt(var * width)


def score_tracks(y, ww, wc):
    """S and r for both orientations; returns the elementwise-better of the two."""
    W = len(ww)
    nrm = np.sqrt(np.sum(ww ** 2) + np.sum(wc ** 2))
    outS, outR = None, None
    for a, b in ((ww, wc), (wc[::-1], ww[::-1])):
        s = (np.correlate(y["watson"], a, mode="same")
             + np.correlate(y["crick"], b, mode="same")) / nrm
        # r needs the window-centred data norm, pooled over both channels
        mw, nw = windowed(y["watson"], W)
        mc, nc = windowed(y["crick"], W)
        # <w, y - ybar> = <w, y> because w is already mean-centred per channel
        dnorm = np.sqrt(nw ** 2 + nc ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dnorm > 0, s * nrm / (nrm * dnorm), 0.0)
        if outS is None:
            outS, outR, outD = s, r, dnorm
        else:
            take = s > outS
            outS = np.where(take, s, outS)
            outR = np.where(take, r, outR)
            outD = np.where(take, dnorm, outD)
    return outS, outR, outD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default="chrI")
    ap.add_argument("--halves", default="7,12")
    ap.add_argument("--nperm", type=int, default=40)
    ap.add_argument("--tol", type=int, default=20)
    ap.add_argument("--out", default="abf1_slide_calib")
    args = ap.parse_args()
    halves = [int(x) for x in args.halves.split(",")]
    rng = np.random.default_rng(SEED)

    sizes = pd.read_csv("inputs/sacCer3.chrom.sizes", sep="\t", header=None,
                        names=["chr", "len"])
    length = int(sizes.loc[sizes.chr == args.chrom, "len"].iloc[0])
    ref = S_.load_abf1(MACISAAC)
    ref = sorted(ref.loc[ref.chr == args.chrom, "center"].tolist())

    ns = fiber_params_lib.load(verbose=False)
    bed = pd.read_csv(BED, sep="\t")
    pile = Pileup(PILEUP)
    counts = dense_counts(pile, args.chrom, length)
    y, phat = residuals(counts, length)
    ntot = counts["watson"][1] + counts["crick"][1]
    valid = np.flatnonzero(ntot > 0)

    saved = np.load("abf1_slide_tracks.npz")
    out = {"chrom": args.chrom, "tol": args.tol, "nperm": args.nperm, "ref": ref,
           "n_scored": int(valid.size), "rows": []}

    for half in halves:
        W = 2 * half + 1
        sites = bed[(bed.TF == TF) & (bed.chr != args.chrom)]
        p, _, _ = fit_group(ns, pile, sites, half, pseudo=True)
        pw = np.asarray(p["p"]["watson_signal"]["A"], float)
        pc = np.asarray(p["p"]["crick_signal"]["A"], float)
        ww, wc = pw - pw.mean(), pc - pc.mean()

        s, r, dnorm = score_tracks(y, ww, wc)
        ok = np.max(np.abs(s - saved["s_half%d" % half]))
        print("half=%2d (%2d bp): reproduces saved track, max|diff| = %.2e" % (half, W, ok))

        # --- local read depth per window (mean A trials per column) ---
        depth = uniform_filter1d(ntot.astype(float), W, mode="nearest")

        # --- does the NULL depend on depth? permuted template, by depth quintile ---
        nulls = {}
        for kind in ("perm", "phase"):
            arr = np.zeros((args.nperm, length))
            for i in range(args.nperm):
                if kind == "perm":
                    a_, b_ = rng.permutation(ww), rng.permutation(wc)
                else:
                    a_, b_ = phase_randomize(ww, rng), phase_randomize(wc, rng)
                arr[i] = score_tracks(y, a_, b_)[0]
            nulls[kind] = arr
        perm = nulls["perm"]
        pv = perm[:, valid].ravel()

        # per-position permutation z: how much better does the REAL ABF1 shape fit here
        # than a random shape carrying the same weights? Depth- and autocorrelation-free.
        zgrid = np.arange(0.0, 8.01, 0.25)
        zres, ztrack = {}, {}
        for kind, arr in nulls.items():
            mu, sd = arr.mean(0), arr.std(0)
            zz = np.where(sd > 0, (s - mu) / np.maximum(sd, 1e-9), 0.0)
            zn_ = np.where(sd > 0, (arr - mu) / np.maximum(sd, 1e-9), 0.0)
            zv_, zpv_ = zz[valid], zn_[:, valid].ravel()
            zo = np.array([(zv_ >= t_).sum() for t_ in zgrid], float)
            znull = np.array([(zpv_ >= t_).sum() / args.nperm for t_ in zgrid], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                zf = stepup(np.where(zo > 0, znull / zo, np.nan))
            cut = {}
            for target in (0.50, 0.20, 0.10, 0.05, 0.01):
                i_ = np.flatnonzero(zf <= target)
                cut["fdr%.2f" % target] = (round(float(zgrid[i_[0]]), 2) if i_.size else None)
            zres[kind] = dict(cuts=cut, obs=[int(v) for v in zo],
                              null=[round(float(v), 2) for v in znull],
                              fdr=[round(float(v), 4) for v in zf])
            ztrack[kind] = zz
        z = ztrack["phase"]          # the shape-isolating statistic
        zv = z[valid]
        zcuts = zres["phase"]["cuts"]

        dv, sv, rv = depth[valid], s[valid], r[valid]
        qs = np.quantile(dv[dv > 0], [0, .2, .4, .6, .8, 1.0])
        strata = []
        for i in range(5):
            m = (dv >= qs[i]) & (dv <= qs[i + 1])
            pm = np.tile(m, args.nperm)
            strata.append(dict(depth_lo=round(float(qs[i]), 1),
                               depth_hi=round(float(qs[i + 1]), 1),
                               n=int(m.sum()),
                               obs_mean=round(float(sv[m].mean()), 3),
                               obs_sd=round(float(sv[m].std()), 3),
                               obs_p999=round(float(np.percentile(sv[m], 99.9)), 3),
                               null_mean=round(float(pv[pm].mean()), 3),
                               null_sd=round(float(pv[pm].std()), 3),
                               null_p999=round(float(np.percentile(pv[pm], 99.9)), 3),
                               r_mean=round(float(rv[m].mean()), 3),
                               r_p999=round(float(np.percentile(rv[m], 99.9)), 3)))

        # --- FDR calibration: expected null calls / observed calls at threshold t ---
        grid = np.arange(2.0, 16.01, 0.25)
        n_obs = np.array([(sv >= t).sum() for t in grid], float)
        n_null = np.array([(pv >= t).sum() / args.nperm for t in grid], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            fdr = np.where(n_obs > 0, n_null / n_obs, np.nan)
        fdr = stepup(fdr)

        cuts = {}
        for target in (0.50, 0.20, 0.10, 0.05, 0.01):
            i = np.flatnonzero(fdr <= target)
            cuts["fdr%.2f" % target] = (round(float(grid[i[0]]), 2) if i.size else None)

        site = {}
        for c in ref:
            lo, hi = max(0, c - args.tol), min(length, c + args.tol + 1)
            j = lo + int(np.argmax(s[lo:hi]))
            k = int(np.searchsorted(grid, s[j]) - 1)
            site[str(c)] = dict(S=round(float(s[j]), 3), r=round(float(r[j]), 3),
                                evidence=round(float(dnorm[j]), 2),
                                depth=round(float(depth[j]), 1),
                                z=round(float(z[j]), 3),
                                z_pct=round(100.0 * float(np.mean(zv >= z[j])), 4),
                                pct=round(100.0 * float(np.mean(sv >= s[j])), 4),
                                fdr=(round(float(fdr[k]), 3) if 0 <= k < len(fdr) else None))

        out["rows"].append(dict(half=half, width=W, reproduces=float(ok),
                                strata=strata, cuts=cuts, sites=site,
                                zcuts=zcuts, z_by_null=zres,
                                zgrid=[round(float(g), 2) for g in zgrid],
                                grid=[round(float(g), 2) for g in grid],
                                n_obs=[int(v) for v in n_obs],
                                n_null=[round(float(v), 2) for v in n_null],
                                fdr=[round(float(v), 4) for v in fdr]))
        np.savez_compressed("%s_half%d.npz" % (args.out, half), s=s, r=r, z=z,
                            evidence=dnorm, depth=depth,
                            z_perm=ztrack["perm"], z_phase=ztrack["phase"])

    with open(args.out + ".json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote %s.json" % args.out)


if __name__ == "__main__":
    main()
