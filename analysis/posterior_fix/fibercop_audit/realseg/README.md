# realseg: every decoder variant on one real chrXII rDNA segment (2026-10-13)

Question: on real rDNA data, does the audit's C1 fix (bward rescales emitting cells before the
silent loop) remove fix B's divergence from log space? If yes, the divergence is the C1 mechanism;
if no, it is the DBL_MIN padding (POSTERIOR_FIX_PLAN.md "Results, 2026-10-06").

## What was run

Segment chrXII:458,001-463,000 (`posterior_fix/coord_rdna_chrXII_seg2.tsv`), tree
`pkgvar/seq_maskoff_tune75` ("both" layers, linear emission floored to 1e-30), trainDir
`robocop_train_tw_bv75_49` (harness default). Captured emission (7, 5000, 3485), ssb 3330.

```bash
cd analysis
SP=<scratchpad>/realseg
python posterior_fix/harness_capture.py pkgvar/seq_maskoff_tune75 posterior_fix/coord_rdna_chrXII_seg2.tsv $SP/cap_out $SP/cap_seg2_tune75.npz   # 5 min
python posterior_fix/fibercop_audit/realseg/compare_variants.py $SP/cap_seg2_tune75.npz \
       posterior_fix/fibercop_audit/realseg/results.json --save $SP/posteriors_seg2.npz              # ~40 s
```

The capture npz (~1 GB) and posteriors are in the session scratchpad, not the repo.
All variants get the identical emission array in-process. md5 of every .so checked before and after (unchanged).

| variant | library (md5) |
|---|---|
| ORIG | pkg/robocop/librobocop_prenorm.so (6a0724bf), 7-arg posterior_decoding |
| ORIG+B | ORIG posterior / row sum over [:ssb] (Python) |
| FIXB | pkg/robocop/librobocop.so (bf219a43), posterior_decoding_v2 |
| C1FIX | robocop_c_audit/walkthrough/variants/librobocop_fixed.so (6dbd5e6d) |
| C1FIX+B | C1FIX posterior / row sum over [:ssb] (Python) |
| LOG-D0 | pkg/robocop/librobocop_logfull.so (88f86a35), log(E) summed over layers, silent 0 |

## Results (emitting columns [:ssb], vs LOG-D0)

| variant | max abs | rows > 0.01 | row-sum range | loglik − D0 | silent max |
|---|---|---|---|---|---|
| ORIG | 1.2e153 | 2949 | 1.0 – 1.2e153 | +2318.09 | 3.0e128 |
| ORIG+B | 1.0 | 441 | 1 ± 3e-16 | +2318.09 | 0.0072 |
| FIXB | 1.0 | 441 | 1 ± 3e-15 | +2318.09 | 0.0072 |
| C1FIX | 1.0 | 414 | 1 ± 5e-14 | +2318.09 | 0.0072 |
| C1FIX+B | 1.0 | 414 | 1 ± 2e-16 | +2318.09 | 0.0072 |
| LOG-D0 | 0 | 0 | 1 ± 1e-13 | 0 (−624800.117) | 0 |

Pairwise rows > 0.01: FIXB vs ORIG+B 0 (max 2.8e-15); FIXB vs C1FIX+B 58; C1FIX+B vs LOG-D0 414.
Loglik is calc_sr's return (sum log forward offsets); fward is the same code in all three linear
libraries, so it is identical (−622482.028). Backward sum log offsets: ORIG/FIXB −621844.38,
C1FIX −622196.83, LOG-D0 −624514.92.

## Answer

C1FIX(+B) does **not** match LOG-D0: 414 of 5000 rows still differ by > 0.01 (FIXB: 441), and the
whole 2318-nat loglik gap is unchanged because it sits in fward, which C1 does not touch. So the
rDNA divergence of fix B is not the C1 mechanism; C1 accounts for 58 rows at most. This is
consistent with the 80-bit replica proof that it is the DBL_MIN padding (normalize adds DBL_MIN
to every cell, in fward and bward). The D0 − B loglik here (−2318.09) matches the value recorded
for this segment in POSTERIOR_FIX_PLAN.md (−2318).

On this segment no silent column exceeds 1 in any variant (defect 4 was seen on other segments).
