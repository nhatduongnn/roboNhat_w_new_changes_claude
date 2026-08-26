"""Same FDR question, but counting PEAKS instead of base pairs.

abf1_slide_calibrate.py counts a 'hit' as one base pair clearing the cutoff. Neighbouring
base pairs overlap, so a single real footprint contributes ~width adjacent hits. This
re-runs the real-vs-scrambled comparison using slide_abf1_profile.call_peaks (local maxima,
min separation = template width) so each footprint counts once, and checks the FDR verdict
is not an artefact of the counting rule.
"""
import json

import numpy as np
import pandas as pd

import fiber_params_lib
from make_params_pm50 import BED, PILEUP, Pileup, fit_group
from slide_abf1_profile import MACISAAC, TF, call_peaks, dense_counts, residuals
from abf1_slide_calibrate import phase_randomize, score_tracks

NPERM = 40
GRID = np.arange(2.0, 16.01, 0.5)
ZGRID = np.arange(0.0, 6.01, 0.25)


def peak_counts(track, valid_mask, width, grid):
    t = np.where(valid_mask, track, -np.inf)
    return np.array([len(call_peaks(t, width, g)) for g in grid], float)


def main():
    rng = np.random.default_rng(0)
    sizes = pd.read_csv("inputs/sacCer3.chrom.sizes", sep="\t", header=None,
                        names=["chr", "len"])
    length = int(sizes.loc[sizes.chr == "chrI", "len"].iloc[0])
    ns = fiber_params_lib.load(verbose=False)
    bed = pd.read_csv(BED, sep="\t")
    pile = Pileup(PILEUP)
    counts = dense_counts(pile, "chrI", length)
    y, _ = residuals(counts, length)
    ntot = counts["watson"][1] + counts["crick"][1]
    vmask = ntot > 0

    out = {"nperm": NPERM, "grid": GRID.tolist(), "zgrid": ZGRID.tolist(), "rows": []}
    for half in (7, 12):
        W = 2 * half + 1
        sites = bed[(bed.TF == TF) & (bed.chr != "chrI")]
        p, _, _ = fit_group(ns, pile, sites, half, pseudo=True)
        pw = np.asarray(p["p"]["watson_signal"]["A"], float)
        pc = np.asarray(p["p"]["crick_signal"]["A"], float)
        ww, wc = pw - pw.mean(), pc - pc.mean()

        s = score_tracks(y, ww, wc)[0]
        nulls = np.zeros((NPERM, length))
        for i in range(NPERM):
            nulls[i] = score_tracks(y, phase_randomize(ww, rng),
                                    phase_randomize(wc, rng))[0]

        obs_s = peak_counts(s, vmask, W, GRID)
        null_s = np.mean([peak_counts(nulls[i], vmask, W, GRID) for i in range(NPERM)], 0)

        mu, sd = nulls.mean(0), nulls.std(0)
        z = np.where(sd > 0, (s - mu) / np.maximum(sd, 1e-9), 0.0)
        zn = np.where(sd > 0, (nulls - mu) / np.maximum(sd, 1e-9), 0.0)
        obs_z = peak_counts(z, vmask, W, ZGRID)
        null_z = np.mean([peak_counts(zn[i], vmask, W, ZGRID) for i in range(NPERM)], 0)

        out["rows"].append(dict(half=half, width=W,
                                obs_s=obs_s.tolist(), null_s=null_s.tolist(),
                                obs_z=obs_z.tolist(), null_z=null_z.tolist()))
        print("half=%2d done" % half, flush=True)

    json.dump(out, open("abf1_slide_peakfdr.json", "w"), indent=2)

    print("\nHITS = PEAKS (local maxima, min separation = template width)\n")
    for row in out["rows"]:
        print("=" * 68); print("WIDTH %d bp" % row["width"])
        for name, gr, o, n in (("raw S", GRID, row["obs_s"], row["null_s"]),
                               ("z (phase null)", ZGRID, row["obs_z"], row["null_z"])):
            print("\n  %s" % name)
            print("  %-8s %10s %12s %8s" % ("cutoff", "real peaks", "fake peaks", "FDR"))
            for t in ([4, 6, 8, 10, 12] if name == "raw S" else [1, 2, 3, 4, 5]):
                i = int(np.argmin(abs(np.array(gr) - t)))
                f = n[i] / o[i] if o[i] > 0 else float("nan")
                print("  %-8.1f %10d %12.1f %8s" % (
                    gr[i], o[i], n[i], "%.2f" % f if np.isfinite(f) else "-"))
        print()


if __name__ == "__main__":
    main()
