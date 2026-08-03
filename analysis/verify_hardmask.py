"""
Verify the hard-forbid mask: in robocop_chrI_seq_maskon_hard, NO TF except ABF1 should
carry any posterior, and there must be no NaNs. Compare against the soft-mask seq run.
Lists, per run, every TF column whose max posterior over chrI exceeds a small threshold.
"""
import numpy as np
import score_robocop as S

THRESH = 1e-3
RUNS = [
    ("robocop_chrI_seq_maskon",      "SOFT mask (1e-30 floor)"),
    ("robocop_chrI_seq_maskon_hard", "HARD forbid (== 0)"),
]

for outDir, label in RUNS:
    print("\n" + "=" * 70)
    print(f"{label}   [{outDir}]")
    print("=" * 70)
    dec = S.load_decode(outDir)
    regions = S.merge_regions(dec["coords"])
    # accumulate max posterior per named column across all regions
    colmax = {}
    nan_hit = False
    for (chrm, start, end) in regions:
        optable, covered, _ = S.region_optable(dec, chrm, start, end)
        if covered.sum() == 0:
            continue
        for c in optable.columns:
            vals = optable[c].values
            if np.isnan(vals).any():
                nan_hit = True
            m = float(np.nanmax(vals)) if len(vals) else 0.0
            colmax[c] = max(colmax.get(c, 0.0), m)
    # TF columns = everything that's not the nucleosome/dyad/structural columns
    struct = {S.NUC_OCC_COL, S.NUC_DYAD_COL, "unknown"}
    tf_cols = {c: v for c, v in colmax.items()
               if c not in struct and v > THRESH}
    print(f"  NaN present: {nan_hit}")
    print(f"  nucleosome max={colmax.get(S.NUC_OCC_COL,0):.3f}  "
          f"nuc_center max={colmax.get(S.NUC_DYAD_COL,0):.3f}  "
          f"unknown max={colmax.get('unknown',0):.3f}")
    print(f"  TF columns with max posterior > {THRESH}:  ({len(tf_cols)})")
    for c, v in sorted(tf_cols.items(), key=lambda kv: -kv[1]):
        print(f"      {c:16s} {v:.4f}")
print("\nDONE")
