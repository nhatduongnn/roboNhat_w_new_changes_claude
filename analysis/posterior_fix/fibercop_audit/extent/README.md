# extent: how far the un-normalised posterior reached in bo09 / bu01 (2026-10-20)

All 65 bo09/bu01 decodes (49 bo09 tune, 12 bu01 tune, r00 + final holdout each) predate the
fix B cutover. Scripts are read-only on the decodes.

    python posterior_fix/fibercop_audit/extent/scan_rowsums.py extent/rowsums_bo09_bu01.tsv <dirs...>   # 3 min
    python posterior_fix/fibercop_audit/extent/reader_impact.py <decode dir> <out.tsv.gz>              # ~15 min

- `rowsums_bo09_bu01.tsv`: one row per segment (25,932): bad rows (|rowsum-1|>0.01), where they sit, max.
- `impact_{tune,holdout}_bo09_48.tsv.gz`: per position, what each reader showed vs the normalised value
  (correct / viewer = clip+average / old = raw average, with count_calls' exclusion).

Findings:
- bu01: 0 bad rows in every round (one row at 0.99 in holdout r00).
- bo09: first bad rows at round 9, growing with the tuning (r48: 7,289 rows in 21 of 400 tune segments,
  5,316 rows in 29 of 383 holdout chrIV segments, counting rowsum > 1+1e-6).
- Shape: in every bad segment the bad rows are a PREFIX (segment start up to one step). Not the tail.
- "Correct" = row-normalised old output; equal to log space verified on chrXIV seg 307 only (4.8e-9).
