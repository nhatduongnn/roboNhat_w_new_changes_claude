"""
For each Chereji +1/-1 reference dyad on chrI, compute the distance to the NEAREST
predicted dyad (nuc_center peak), reusing score_robocop's exact peak-calling and region
logic. Answers: are the ~20% "missed" dyads near-misses (just past the 20 bp tolerance)
or genuinely absent? Writes a histogram + CDF PNG and prints a distance breakdown.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import score_robocop as S

RUNS = [
    ("robocop_chrI_maskon",  "chrI mask-ON (Fiber-only)"),
    ("robocop_chrI_maskoff", "chrI mask-OFF (Fiber-only)"),
]
TOL = 20

def nearest_dists(outDir):
    dec = S.load_decode(outDir)
    regions = S.merge_regions(dec["coords"])
    chereji = S.load_chereji(S.DEFAULT_CHEREJI)
    ref_all, dmin_all = [], []
    for (chrm, start, end) in regions:
        optable, covered, _ = S.region_optable(dec, chrm, start, end)
        if covered.sum() == 0:
            continue
        pos = np.arange(start, end + 1)
        track = optable[S.NUC_DYAD_COL].values
        dh = max(0.02, 0.20 * np.nanmax(track)) if np.nanmax(track) > 0 else 0.02
        pk = S.call_peaks(track, height=dh, distance=120)
        pred = np.array([int(pos[i]) for i in pk])
        ref = chereji[(chereji["chr"] == chrm) &
                      (chereji["dyad"] >= start) & (chereji["dyad"] <= end)]["dyad"].tolist()
        for r in ref:
            ref_all.append(r)
            dmin_all.append(int(np.min(np.abs(pred - r))) if len(pred) else 10**9)
    return np.array(dmin_all)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for col, (outDir, title) in enumerate(RUNS):
    d = nearest_dists(outDir)
    n = len(d)
    bins = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 10**9]
    labels = ["0-5","5-10","10-20","20-30","30-50","50-75","75-100","100-150",
              "150-200","200-300","300-500","500-1k",">1k"]
    counts = [int(((d >= bins[i]) & (d < bins[i+1])).sum()) for i in range(len(bins)-1)]

    print(f"\n=== {title}  (n={n} Chereji dyads) ===")
    within = (d <= TOL).sum()
    print(f"  within tol {TOL}bp (matched/recall): {within}/{n} = {within/n:.3f}")
    print(f"  median nearest-dist: {np.median(d):.1f} bp   (all dyads, matched+missed)")
    miss = d[d > TOL]
    if len(miss):
        print(f"  MISSED dyads (>{TOL}bp): {len(miss)}  ->  median miss dist {np.median(miss):.0f} bp, "
              f"min {miss.min()} bp, max {miss.max() if miss.max()<10**9 else 'no-pred'} bp")
        for lo, hi, lab in [(20,30,'20-30 (near-miss)'),(30,50,'30-50'),(50,100,'50-100'),
                            (100,500,'100-500'),(500,10**9,'>500 / none')]:
            c = int(((miss>=lo)&(miss<hi)).sum())
            print(f"     {lab:22s}: {c}")

    # histogram (top row)
    ax = axes[0, col]
    ax.bar(range(len(counts)), counts, color="#4C78A8")
    ax.axvline(2.5, color="crimson", ls="--", lw=1.5, label=f"tol={TOL}bp")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("# Chereji dyads"); ax.set_title(title + f"\nnearest predicted dyad (n={n})")
    ax.legend()
    # CDF (bottom row)
    ax2 = axes[1, col]
    ds = np.sort(d[d < 10**9])
    ax2.plot(ds, np.arange(1, len(ds)+1)/n, color="#4C78A8")
    ax2.axvline(TOL, color="crimson", ls="--", lw=1.5, label=f"tol={TOL}bp")
    ax2.set_xlim(0, 300); ax2.set_ylim(0, 1)
    ax2.set_xlabel("nearest predicted dyad distance (bp)"); ax2.set_ylabel("cumulative fraction")
    ax2.set_title("CDF (zoom 0-300bp)"); ax2.grid(alpha=0.3); ax2.legend()

plt.tight_layout()
out = "dyad_distance_chrI.png"
plt.savefig(out, dpi=140)
print(f"\nchart written: {os.path.abspath(out)}")
