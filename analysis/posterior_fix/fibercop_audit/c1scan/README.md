# c1scan: what C1 is in FiberCOP, and how to detect it (2026-10-20)

Case: bo09_48, chrXIV segment 307 (412,001-417,000), tree pkgvar/seq_maskoff_fit9_bgopen,
trainDir robocop_train_tw_bo09_48. Emissions captured with posterior_fix/harness_capture.py.

    python posterior_fix/fibercop_audit/c1scan/c1_trace.py <cap.npz> 1383 1412

Findings (numpy replay of both bward orderings reproduces the C tables to <1e-14):
- Original (librobocop_prenorm.so) and C1-fixed bward differ at ONE position, idx 1385
  (chrXIV:413,386), by 29.96 nats = the 1.03e13 row-sum step.
- The MNase criterion (max emission at t x max emission at t+1 < DBL_MIN) does NOT flag it:
  the smallest pair product in the whole segment is 1e-120.
- Mechanism: normalize() (pkg/robocop/algo.c:10-30) adds DBL_MIN to EVERY cell, silent included,
  then divides the whole row by the emitting sum S. The original computes silent cells before that
  rescale, so they are S times too small and the padding swamps them: every silent cell becomes
  DBL_MIN/S (all 155 of 155 at idx 1406). With the 1e-30 floor on each unscored strand, S is
  1e-31..1e-99 here, so the fake mass is 1e-277..1e-209 per row (MNase: S ~ 1e-2, fake ~ 1e-306).
  It does damage where a later row's true emitting mass falls below it (idx 1385: S = 1.07e-99).
- Proof: original ordering WITHOUT DBL_MIN on silent cells gives scale 1.069e-99 at idx 1385
  (fixed C: 1.068e-99); with it, 1.097e-86 (original C).

## Genome scan (2026-10-20), Slurm 13070845 / 13070895 / 13070896, all 144 tasks COMPLETED

`scan_genome.py` per segment of `coord_genome_full.tsv` (3,021 windows, chrM excluded): emissions
rebuilt by the model's own tree, then original (librobocop_prenorm.so) and C1-fixed bward/posterior
on the same arrays. "C1 rows" = posterior rows with |emitting sum - 1| > 1e-4 in the original but
not in the fixed library. `summarize.py` -> `summary_by_chrom.tsv`; table below in `summary_c1_rows.tsv`.

| model | segments with C1 rows / C1 rows, whole genome | of which chrXII rDNA |
|---|---|---|
| bo09_48 (tree seq_maskoff_fit9_bgopen) | 217 / 122,382 | 5 / 13,408 |
| bu01_11 (seq_maskoff_all153) | 5 / 13,798 | all of it |
| bv75_49 (seq_maskoff_tune75) | 59 / 17,689 | 5 / 13,794 |

Cross-check: bo09 chrXIV + chrII bad rows = 6,804 + 485 = 7,289, identical to the stored bo09_48 decode
(extent/rowsums_bo09_bu01.tsv, rowsum > 1+1e-6). bu01 stored decodes: 0, scan: 0 on the same chromosomes.
Caveats: (1) "fixed" only removes C1; in rDNA its rows sum to 1 but are still wrong (DBL_MIN in fward,
emission floor; realseg/README.md). (2) bo09 chrVI:208,001-213,000: 2,763 rows bad in BOTH libraries,
cause not established. (3) The backward-scale "event" column misses short prefix breaks at segment starts
(e.g. bv75 2-15 rows reaching 1e42); use the row counts.
