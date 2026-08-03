"""List the ABF1 reference sites on chrI and, per run, which are recovered.
Reuses score_robocop internals so the matching is identical to the recall number."""
import numpy as np
import score_robocop as S

RUNS = [
    ("robocop_chrI_maskon",      "Fiber / Abf1 only"),
    ("robocop_chrI_maskoff",     "Fiber / all TFs"),
    ("robocop_chrI_seq_maskon",  "Fiber+seq / Abf1 only"),
    ("robocop_chrI_seq_maskoff", "Fiber+seq / all TFs"),
]
TOL = 20
ABF1_COL = S.ABF1_COL

abf1 = S.load_abf1(S.DEFAULT_ABF1)
abf1 = abf1[abf1["chr"] == "chrI"].sort_values("center")
ref_centers = abf1["center"].tolist()
print("ABF1 reference sites on chrI (MacIsaac match_PWM), n=%d:" % len(ref_centers))
for i, r in enumerate(abf1.itertuples(), 1):
    print("  #%d  chrI:%d-%d  center=%d  strand=%s  score=%s"
          % (i, r.start, r.end, r.center, r.strand, r.score))
print()

def predicted_abf1(dec):
    """Return sorted list of predicted ABF1 peak genomic coords over all merged regions."""
    preds = []
    for chrm, start, end in S.merge_regions(dec["coords"]):
        if chrm != "chrI":
            continue
        optable, covered, _ = S.region_optable(dec, chrm, start, end)
        if ABF1_COL not in optable.columns:
            continue
        track = optable[ABF1_COL].values
        mx = np.nanmax(track)
        if not (mx > 0):
            continue
        h = max(0.10, 0.30 * mx)
        # Option A: footprint-center anchor (matches score_robocop), not find_peaks
        for pk in S.footprint_centers(track, height=h):
            preds.append(start + pk)
    return sorted(preds)

for outDir, label in RUNS:
    dec = S.load_decode(outDir)
    preds = predicted_abf1(dec)
    print("=== %-24s (%s) ===" % (label, outDir))
    print("   predicted ABF1 peaks: n=%d" % len(preds))
    # greedy nearest match, ref -> nearest unused pred within TOL (same as match_peaks)
    used = [False] * len(preds)
    recovered = 0
    for i, r in enumerate(ref_centers, 1):
        best, bd = None, TOL + 1
        for pi, p in enumerate(preds):
            if used[pi]:
                continue
            d = abs(p - r)
            if d < bd:
                bd, best = d, pi
        if best is not None and bd <= TOL:
            used[best] = True
            recovered += 1
            print("   site #%d center=%-7d  RECOVERED  pred=%d  dist=%dbp" % (i, r, preds[best], bd))
        else:
            # report nearest pred even if outside tol, for context
            if preds:
                nd = min(abs(p - r) for p in preds)
                nearest = min(preds, key=lambda p: abs(p - r))
                print("   site #%d center=%-7d  missed     (nearest pred=%d, %dbp away)" % (i, r, nearest, nd))
            else:
                print("   site #%d center=%-7d  missed     (no ABF1 peaks predicted)" % (i, r))
    print("   recall = %d/%d = %.2f" % (recovered, len(ref_centers), recovered / len(ref_centers)))
    print()
