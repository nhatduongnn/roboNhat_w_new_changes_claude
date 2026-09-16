# Overnight results (2026-09-15 launch)

_generated 2026-09-15 11:18 by `analysis/overnight_summary.py`; anything not finished is marked **INCOMPLETE** — re-run `python overnight_summary.py` to refresh._

### Rule 7 — what differs

- **A**: only the layer configuration differs from sw01_05 (same weights).
- **C**: only the decode scope (whole genome) differs from each campaign's final round.
- **B**: differs from bw01 only in the fiber temper φ (2/5/10); everything else identical (weights start, rule, caps, targets, groups, chromosomes, rounds).

## A — fixed-prior fiber veto test (sw01 round-05 weights)

Rule 7: **only the layer configuration differs from sw01_05** (same trainDir `robocop_train_tw_sw01_05`, same weights). seq-only = sw01's own round-05 decode; +fiber = `run_split_revfix_seq_maskoff_mr58.py`; fiber-only = `run_split_revfix_fiber_maskoff_mr58.py`. 58 groups, calls = posterior >= 0.10, 30 bp greedy one-to-one match (tw_validate code). tune = chrXIV+chrII, holdout = chrIV.

_scored 2026-09-15 01:37_

**tuning chromosomes chrXIV+chrII** — P / R / F1

| config | calls | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted |
|---|---|---|---|---|---|---|---|
| seq-only (sw01_05) | 2833 | 0.067 / 0.132 / 0.089 | 0.136 / 0.278 / 0.183 | 0.051 / 0.100 / 0.068 | 0.064 / 0.126 / 0.085 | 0.234 / 0.198 / 0.215 | 0.025 / 0.071 / 0.037 |
| seq + fiber | 15048 | 0.017 / 0.177 / 0.031 | 0.013 / 0.402 / 0.026 | 0.021 / 0.128 / 0.036 | 0.018 / 0.189 / 0.033 | 0.025 / 0.318 / 0.047 | 0.010 / 0.091 / 0.018 |
| fiber-only | 30338 | 0.007 / 0.144 / 0.013 | 0.005 / 0.402 / 0.010 | 0.010 / 0.087 / 0.018 | 0.007 / 0.156 / 0.014 | 0.010 / 0.315 / 0.019 | 0.003 / 0.034 / 0.005 |

**holdout chrIV** — P / R / F1

| config | calls | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted |
|---|---|---|---|---|---|---|---|
| seq-only (sw01_05) | 2613 | 0.059 / 0.117 / 0.078 | 0.133 / 0.283 / 0.181 | 0.042 / 0.083 / 0.056 | 0.066 / 0.136 / 0.089 | 0.249 / 0.236 / 0.242 | 0.024 / 0.069 / 0.036 |
| seq + fiber | 14459 | 0.015 / 0.170 / 0.028 | 0.011 / 0.367 / 0.021 | 0.021 / 0.129 / 0.035 | 0.018 / 0.208 / 0.033 | 0.023 / 0.344 / 0.043 | 0.013 / 0.117 / 0.023 |
| fiber-only | 29466 | 0.006 / 0.129 / 0.011 | 0.005 / 0.412 / 0.009 | 0.008 / 0.070 / 0.014 | 0.008 / 0.179 / 0.015 | 0.010 / 0.375 / 0.019 | 0.003 / 0.046 / 0.006 |

**Fiber veto / fiber-added calls vs seq-only** (30 bp, not one-to-one; `supp` = a MacIsaac / Rossi site within 30 bp)

| comparison | set | seq-only calls | compared calls | veto (MacIsaac supp / Rossi supp) | added (MacIsaac supp / Rossi supp) | kept seq-only calls supp (Mac / Rossi) | groups with veto | groups with added |
|---|---|---|---|---|---|---|---|---|
| both_vs_seqonly | tune | 2833 | 15048 | 2379 (102 / 73) | 14547 (179 / 188) | 103 / 118 | 49 | 58 |
| both_vs_seqonly | holdout | 2613 | 14459 | 2280 (86 / 83) | 14079 (171 / 210) | 75 / 93 | 46 | 58 |
| fib_vs_seqonly | tune | 2833 | 30338 | 2608 (130 / 106) | 30004 (160 / 195) | 75 / 85 | 49 | 33 |
| fib_vs_seqonly | holdout | 2613 | 29466 | 2419 (113 / 99) | 29174 (150 / 218) | 48 / 77 | 49 | 33 |

Per group, seq + fiber vs seq-only (groups with any veto or added call; full table `overnight/A_veto_groups.tsv`, call list `overnight/A_veto_calls.tsv`):

| set | group | fitted | seq-only calls | +fiber calls | veto (Mac/Rossi supp) | added (Mac/Rossi supp) |
|---|---|---|---|---|---|---|
| tune | ABF1 | 1 | 95 | 3188 | 70 (5/9) | 3161 (27/39) |
| tune | REB1 | 1 | 114 | 1582 | 42 (3/4) | 1472 (16/52) |
| tune | FKH1 | 1 | 55 | 1125 | 35 (2/1) | 1102 (7/25) |
| tune | ROX1 | 0 | 0 | 866 | 0 (0/0) | 866 (4/3) |
| tune | SWI5 | 0 | 341 | 531 | 312 (4/0) | 498 (17/0) |
| tune | MCM1 | 1 | 17 | 704 | 16 (2/1) | 703 (4/14) |
| tune | MSN2 | 0 | 102 | 643 | 76 (10/0) | 615 (15/0) |
| tune | STE12 | 0 | 429 | 261 | 399 (18/2) | 234 (11/0) |
| tune | CIN5 | 1 | 101 | 505 | 88 (9/12) | 492 (9/2) |
| tune | PHD1 | 0 | 172 | 342 | 152 (0/1) | 321 (6/8) |
| tune | SWI4 | 0 | 107 | 374 | 82 (3/1) | 347 (7/1) |
| tune | GCR1 | 0 | 24 | 378 | 22 (0/0) | 376 (2/1) |
| tune | SKN7 | 0 | 100 | 256 | 82 (1/0) | 235 (10/3) |
| tune | SOK2 | 0 | 61 | 266 | 56 (0/2) | 261 (2/5) |
| tune | FHL1 | 1 | 44 | 281 | 39 (0/0) | 276 (0/2) |
| tune | HSF1 | 0 | 0 | 291 | 0 (0/0) | 291 (2/2) |
| tune | AFT2 | 0 | 94 | 205 | 80 (2/0) | 193 (4/0) |
| tune | UME6 | 1 | 53 | 246 | 40 (7/14) | 233 (2/4) |
| tune | GCN4 | 0 | 89 | 196 | 78 (7/2) | 185 (6/0) |
| tune | CBF1 | 0 | 58 | 200 | 40 (6/6) | 183 (2/1) |
| tune | RFX1 | 0 | 31 | 177 | 31 (0/0) | 177 (0/0) |
| tune | GLN3 | 0 | 17 | 176 | 15 (0/0) | 174 (1/3) |
| tune | ACE2 | 0 | 91 | 72 | 86 (2/1) | 66 (1/0) |
| tune | RTG3 | 0 | 51 | 121 | 41 (0/0) | 110 (2/0) |
| tune | NRG1 | 0 | 78 | 81 | 73 (8/2) | 76 (6/1) |
| tune | SFP1 | 0 | 22 | 129 | 21 (0/0) | 128 (0/0) |
| tune | RPN4 | 0 | 31 | 121 | 28 (0/1) | 118 (0/1) |
| tune | SUM1 | 0 | 0 | 142 | 0 (0/0) | 142 (0/0) |
| tune | CHA4 | 0 | 14 | 132 | 12 (0/0) | 129 (0/1) |
| tune | FKH2 | 0 | 83 | 59 | 80 (1/2) | 56 (1/0) |
| tune | ARO80 | 0 | 12 | 123 | 11 (0/0) | 122 (0/0) |
| tune | RAP1 | 1 | 38 | 130 | 17 (2/3) | 115 (4/7) |
| tune | MBP1 | 0 | 62 | 117 | 37 (4/1) | 91 (5/1) |
| tune | MET31 | 0 | 54 | 70 | 46 (1/2) | 62 (0/0) |
| tune | SUT1 | 0 | 51 | 71 | 44 (1/0) | 64 (0/0) |
| tune | SKO1 | 1 | 12 | 87 | 12 (0/2) | 87 (1/0) |
| tune | LEU3 | 0 | 2 | 94 | 2 (0/0) | 94 (0/1) |
| tune | BAS1 | 0 | 14 | 70 | 14 (0/0) | 70 (1/1) |
| tune | MET32 | 0 | 0 | 81 | 0 (0/0) | 81 (0/0) |
| tune | STP1 | 0 | 14 | 60 | 13 (0/0) | 59 (1/2) |
| tune | AZF1 | 0 | 0 | 64 | 0 (0/0) | 64 (0/2) |
| tune | HAP1 | 0 | 43 | 22 | 39 (1/1) | 18 (1/0) |
| tune | PDR1 | 0 | 2 | 54 | 2 (0/0) | 54 (0/1) |
| tune | GAL4 | 0 | 9 | 52 | 6 (0/0) | 49 (1/0) |
| tune | YAP1 | 0 | 11 | 36 | 11 (0/0) | 36 (0/0) |
| tune | GZF3 | 0 | 18 | 34 | 15 (0/0) | 31 (0/0) |
| tune | STB4 | 0 | 0 | 46 | 0 (0/0) | 46 (0/0) |
| tune | ZAP1 | 0 | 3 | 25 | 3 (0/0) | 25 (0/0) |
| tune | CAD1 | 0 | 4 | 25 | 3 (1/2) | 24 (0/0) |
| tune | PUT3 | 0 | 0 | 24 | 0 (0/0) | 24 (0/0) |
| tune | RPH1 | 0 | 1 | 23 | 1 (0/0) | 23 (0/0) |
| tune | MIG1 | 0 | 1 | 21 | 1 (0/0) | 21 (0/1) |
| tune | STB5 | 0 | 1 | 16 | 1 (0/0) | 16 (0/1) |
| tune | STP4 | 0 | 0 | 17 | 0 (0/0) | 17 (0/2) |
| tune | YDR520C | 0 | 1 | 13 | 1 (0/0) | 13 (1/1) |
| tune | PDR3 | 0 | 5 | 10 | 3 (2/1) | 8 (0/0) |
| tune | YML081W | 0 | 0 | 8 | 0 (0/0) | 8 (0/0) |
| tune | YRR1 | 0 | 1 | 5 | 1 (0/0) | 5 (0/0) |
| holdout | ABF1 | 1 | 93 | 3127 | 75 (7/14) | 3109 (19/29) |
| holdout | REB1 | 1 | 92 | 1609 | 36 (3/8) | 1516 (18/57) |
| holdout | FKH1 | 1 | 41 | 1078 | 30 (1/2) | 1064 (16/39) |
| holdout | ROX1 | 0 | 0 | 817 | 0 (0/0) | 817 (5/10) |
| holdout | SWI5 | 0 | 315 | 481 | 300 (2/0) | 464 (11/0) |
| holdout | MCM1 | 1 | 11 | 650 | 7 (0/0) | 646 (1/4) |
| holdout | MSN2 | 0 | 114 | 562 | 85 (2/0) | 532 (6/0) |
| holdout | STE12 | 0 | 382 | 246 | 367 (15/2) | 230 (5/2) |
| holdout | CIN5 | 1 | 110 | 484 | 100 (11/8) | 473 (2/2) |
| holdout | PHD1 | 0 | 176 | 315 | 161 (0/1) | 297 (6/5) |
| holdout | SWI4 | 0 | 87 | 392 | 65 (0/2) | 369 (17/6) |
| holdout | GCR1 | 0 | 30 | 351 | 29 (0/1) | 350 (0/0) |
| holdout | FHL1 | 1 | 51 | 268 | 47 (0/0) | 264 (1/2) |
| holdout | SKN7 | 0 | 72 | 250 | 62 (1/0) | 240 (14/2) |
| holdout | SOK2 | 0 | 41 | 262 | 35 (2/2) | 256 (4/6) |
| holdout | HSF1 | 0 | 0 | 272 | 0 (0/0) | 272 (0/1) |
| holdout | UME6 | 1 | 37 | 239 | 33 (4/10) | 235 (2/4) |
| holdout | AFT2 | 0 | 88 | 197 | 73 (1/0) | 184 (2/1) |
| holdout | GCN4 | 0 | 80 | 184 | 76 (6/0) | 180 (5/2) |
| holdout | CBF1 | 0 | 70 | 196 | 60 (11/8) | 186 (4/3) |
| holdout | RFX1 | 0 | 34 | 183 | 33 (0/0) | 182 (2/2) |
| holdout | GLN3 | 0 | 15 | 184 | 15 (0/0) | 184 (2/1) |
| holdout | SFP1 | 0 | 26 | 151 | 26 (0/0) | 151 (1/3) |
| holdout | ACE2 | 0 | 98 | 77 | 95 (1/0) | 73 (1/0) |
| holdout | SUM1 | 0 | 0 | 143 | 0 (0/0) | 143 (0/1) |
| holdout | ARO80 | 0 | 7 | 128 | 7 (0/0) | 128 (1/1) |
| holdout | CHA4 | 0 | 8 | 130 | 6 (0/0) | 128 (0/0) |
| holdout | RPN4 | 0 | 28 | 105 | 25 (2/1) | 101 (1/2) |
| holdout | NRG1 | 0 | 59 | 72 | 56 (2/2) | 69 (7/1) |
| holdout | FKH2 | 0 | 71 | 50 | 69 (6/2) | 48 (0/0) |
| holdout | MBP1 | 0 | 55 | 98 | 35 (3/1) | 78 (8/4) |
| holdout | RTG3 | 0 | 41 | 77 | 37 (0/0) | 73 (0/0) |
| holdout | RAP1 | 1 | 29 | 109 | 13 (4/6) | 95 (2/3) |
| holdout | SKO1 | 1 | 18 | 90 | 17 (0/5) | 89 (0/2) |
| holdout | SUT1 | 0 | 47 | 59 | 41 (0/0) | 53 (1/2) |
| holdout | MET31 | 0 | 47 | 49 | 41 (0/1) | 43 (2/2) |
| holdout | MET32 | 0 | 0 | 84 | 0 (0/0) | 84 (0/0) |
| holdout | BAS1 | 0 | 11 | 64 | 10 (0/0) | 63 (0/0) |
| holdout | GZF3 | 0 | 33 | 32 | 33 (0/1) | 32 (0/0) |
| holdout | GAL4 | 0 | 8 | 57 | 7 (0/0) | 56 (0/0) |
| holdout | LEU3 | 0 | 2 | 61 | 2 (0/0) | 61 (0/0) |
| holdout | PDR1 | 0 | 1 | 64 | 0 (0/0) | 63 (0/3) |
| holdout | AZF1 | 0 | 0 | 60 | 0 (0/0) | 60 (0/0) |
| holdout | STP1 | 0 | 12 | 50 | 8 (0/1) | 46 (2/5) |
| holdout | STB4 | 0 | 0 | 52 | 0 (0/0) | 52 (0/0) |
| holdout | YAP1 | 0 | 19 | 32 | 18 (0/0) | 31 (2/1) |
| holdout | HAP1 | 0 | 23 | 25 | 21 (1/0) | 23 (1/0) |
| holdout | CAD1 | 0 | 2 | 34 | 2 (0/0) | 34 (0/0) |
| holdout | MIG1 | 0 | 1 | 28 | 0 (0/0) | 27 (0/2) |
| holdout | YDR520C | 0 | 5 | 22 | 5 (0/0) | 22 (0/0) |
| holdout | STB5 | 0 | 9 | 15 | 9 (1/2) | 15 (0/0) |
| holdout | ZAP1 | 0 | 2 | 19 | 2 (0/0) | 19 (0/0) |
| holdout | PUT3 | 0 | 0 | 17 | 0 (0/0) | 17 (0/0) |
| holdout | STP4 | 0 | 0 | 16 | 0 (0/0) | 16 (0/0) |
| holdout | PDR3 | 0 | 10 | 15 | 5 (0/3) | 10 (0/0) |
| holdout | RPH1 | 0 | 0 | 14 | 0 (0/0) | 14 (0/0) |
| holdout | YML081W | 0 | 1 | 8 | 0 (0/0) | 7 (0/0) |
| holdout | YRR1 | 0 | 1 | 5 | 1 (0/0) | 5 (0/0) |

## C — genome-wide validation of the tuned models

Rule 7: **only the decode scope (whole genome, 48-task split of `coord_genome_full.tsv`) differs from each campaign's final round**: sw01_05 with seqonly_mr58, fw01_07 with fiber_mr58, bw01_07 with seq_mr58. Legacy u001/m001 round 7 re-scored from the existing `calls_ct_*_07` (no new decode) on the same 58 groups. n_excl = invalid-posterior positions dropped by count_calls (chrXII rDNA, chrVIII end, ...).

_scored 2026-09-15 02:46_

**genome (16 chromosomes)** — P / R / F1

| model | calls | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted | n_excl |
|---|---|---|---|---|---|---|---|---|
| tw_sw01_05 | 21622 | 0.063 / 0.112 / 0.080 | 0.134 / 0.267 / 0.178 | 0.046 / 0.079 / 0.058 | 0.067 / 0.124 / 0.087 | 0.249 / 0.208 / 0.227 | 0.024 / 0.062 / 0.035 | 1 |
| tw_fw01_07 | 86869 | 0.006 / 0.045 / 0.011 | 0.010 / 0.106 / 0.019 | 0.005 / 0.032 / 0.008 | 0.007 / 0.049 / 0.012 | 0.021 / 0.087 / 0.033 | 0.002 / 0.022 / 0.004 | 15684 |
| tw_bw01_07 | 32945 | 0.030 / 0.082 / 0.044 | 0.037 / 0.108 / 0.055 | 0.029 / 0.077 / 0.042 | 0.027 / 0.076 / 0.040 | 0.073 / 0.089 / 0.080 | 0.017 / 0.066 / 0.027 | 19677 |
| u001_07 | 26044 | 0.043 / 0.092 / 0.059 | 0.046 / 0.106 / 0.064 | 0.042 / 0.089 / 0.057 | 0.039 / 0.086 / 0.053 | 0.097 / 0.094 / 0.095 | 0.026 / 0.080 / 0.039 | 19436 |
| m001_07 | 28385 | 0.035 / 0.083 / 0.049 | 0.035 / 0.124 / 0.054 | 0.036 / 0.074 / 0.048 | 0.035 / 0.085 / 0.050 | 0.070 / 0.106 / 0.084 | 0.023 / 0.070 / 0.035 | 19671 |

**tuned chromosomes chrXIV+chrII** — P / R / F1

| model | calls | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted | n_excl |
|---|---|---|---|---|---|---|---|---|
| tw_sw01_05 | 2833 | 0.067 / 0.132 / 0.089 | 0.136 / 0.278 / 0.183 | 0.051 / 0.100 / 0.068 | 0.064 / 0.126 / 0.085 | 0.234 / 0.198 / 0.215 | 0.025 / 0.071 / 0.037 | 1 |
| tw_fw01_07 | 10832 | 0.008 / 0.058 / 0.014 | 0.011 / 0.108 / 0.020 | 0.007 / 0.047 / 0.012 | 0.006 / 0.047 / 0.011 | 0.022 / 0.086 / 0.035 | 0.002 / 0.017 / 0.003 | 0 |
| tw_bw01_07 | 4118 | 0.035 / 0.101 / 0.052 | 0.052 / 0.139 / 0.075 | 0.032 / 0.092 / 0.047 | 0.026 / 0.074 / 0.038 | 0.091 / 0.101 / 0.095 | 0.013 / 0.053 / 0.020 | 0 |
| u001_07 | 3371 | 0.045 / 0.106 / 0.064 | 0.058 / 0.131 / 0.080 | 0.043 / 0.101 / 0.060 | 0.036 / 0.083 / 0.050 | 0.106 / 0.099 / 0.102 | 0.021 / 0.071 / 0.032 | 0 |
| m001_07 | 3512 | 0.040 / 0.098 / 0.057 | 0.050 / 0.166 / 0.077 | 0.037 / 0.083 / 0.051 | 0.034 / 0.083 / 0.048 | 0.085 / 0.115 / 0.097 | 0.018 / 0.058 / 0.027 | 0 |

**never-tuned chromosomes (other 14)** — P / R / F1

| model | calls | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted | n_excl |
|---|---|---|---|---|---|---|---|---|
| tw_sw01_05 | 18789 | 0.062 / 0.109 / 0.079 | 0.134 / 0.265 / 0.178 | 0.045 / 0.077 / 0.057 | 0.068 / 0.123 / 0.087 | 0.251 / 0.210 / 0.229 | 0.024 / 0.061 / 0.034 | 0 |
| tw_fw01_07 | 76037 | 0.006 / 0.043 / 0.011 | 0.010 / 0.106 / 0.019 | 0.005 / 0.030 / 0.008 | 0.007 / 0.050 / 0.012 | 0.020 / 0.088 / 0.033 | 0.002 / 0.022 / 0.004 | 15684 |
| tw_bw01_07 | 28827 | 0.030 / 0.080 / 0.043 | 0.035 / 0.103 / 0.052 | 0.028 / 0.075 / 0.041 | 0.027 / 0.076 / 0.040 | 0.071 / 0.088 / 0.078 | 0.017 / 0.067 / 0.027 | 19677 |
| u001_07 | 22673 | 0.043 / 0.090 / 0.058 | 0.045 / 0.102 / 0.062 | 0.042 / 0.088 / 0.057 | 0.039 / 0.086 / 0.054 | 0.096 / 0.093 / 0.094 | 0.026 / 0.081 / 0.040 | 19436 |
| m001_07 | 24873 | 0.035 / 0.080 / 0.048 | 0.033 / 0.118 / 0.051 | 0.035 / 0.073 / 0.048 | 0.036 / 0.085 / 0.050 | 0.068 / 0.104 / 0.082 | 0.024 / 0.072 / 0.036 | 19671 |

**Per chromosome** — F1 vs MacIsaac / F1 vs Rossi (all 58 groups; `*` = tuned chromosome; [n] = excluded positions)

| chrom | tw_sw01_05 | tw_fw01_07 | tw_bw01_07 | u001_07 | m001_07 |
|---|---|---|---|---|---|
| chrI | 0.053 / 0.068 | 0.013 / 0.016 | 0.027 / 0.039 | 0.040 / 0.044 | 0.029 / 0.044 |
| chrII* | 0.080 / 0.077 [1] | 0.015 / 0.012 | 0.044 / 0.033 | 0.054 / 0.043 | 0.047 / 0.038 |
| chrIII | 0.089 / 0.093 | 0.011 / 0.013 | 0.039 / 0.037 | 0.046 / 0.056 | 0.042 / 0.048 |
| chrIV | 0.078 / 0.089 | 0.011 / 0.011 | 0.047 / 0.040 | 0.060 / 0.056 | 0.052 / 0.054 |
| chrV | 0.098 / 0.091 | 0.011 / 0.014 | 0.044 / 0.047 [6] | 0.072 / 0.063 | 0.050 / 0.058 |
| chrVI | 0.081 / 0.101 | 0.013 / 0.011 | 0.041 / 0.035 | 0.058 / 0.050 | 0.050 / 0.043 |
| chrVII | 0.089 / 0.097 | 0.010 / 0.014 | 0.051 / 0.045 | 0.062 / 0.056 | 0.052 / 0.053 |
| chrVIII | 0.080 / 0.078 | 0.012 / 0.010 [1] | 0.039 / 0.037 [8] | 0.047 / 0.049 | 0.041 / 0.044 [8] |
| chrIX | 0.063 / 0.095 | 0.010 / 0.016 | 0.046 / 0.042 | 0.064 / 0.060 | 0.051 / 0.057 |
| chrX | 0.083 / 0.091 | 0.013 / 0.009 | 0.045 / 0.039 | 0.066 / 0.049 | 0.053 / 0.046 |
| chrXI | 0.078 / 0.081 | 0.008 / 0.008 | 0.038 / 0.036 | 0.063 / 0.062 | 0.042 / 0.049 |
| chrXII | 0.054 / 0.076 | 0.008 / 0.011 [15683] | 0.040 / 0.037 [19663] | 0.050 / 0.043 [19436] | 0.045 / 0.044 [19663] |
| chrXIII | 0.078 / 0.087 | 0.010 / 0.015 | 0.046 / 0.046 | 0.062 / 0.060 | 0.053 / 0.060 |
| chrXIV* | 0.099 / 0.094 | 0.011 / 0.010 | 0.061 / 0.045 | 0.075 / 0.058 | 0.068 / 0.060 |
| chrXV | 0.081 / 0.091 | 0.012 / 0.012 | 0.038 / 0.036 | 0.052 / 0.049 | 0.046 / 0.045 |
| chrXVI | 0.089 / 0.085 | 0.011 / 0.011 | 0.043 / 0.040 | 0.053 / 0.054 | 0.050 / 0.048 |

## B — fiber-tempering tuning sweep (bt02 / bt05 / bt10)

Rule 7: **differs from bw01 only in the fiber temper φ (2 / 5 / 10)**: Fiber-seq layers 5 and 6 raised to 1/φ for every state, baked into `pkgvar/seq_maskoff_mr58_phi{2,5,10}` (after the 1e-30 floor, before the mr58 mask). Everything else identical to bw01: weight start, update rule, caps, targets, 58 groups, tune chrXIV+chrII, holdout chrIV (round 0 and final), MAX_ROUNDS 8.

**Campaign summary** (final round; holdout chrIV P / R / F1; Chereji +1/-1 recall on chrXIV at the final round)

| run | config | state | within-2× (T≥5) | nucleosome copies vs r0 | holdout r0 MacIsaac | holdout final MacIsaac | holdout final Rossi | Chereji recall chrXIV |
|---|---|---|---|---|---|---|---|---|
| bt02 | φ=2 | stopped: iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) | 41/41 | 8942 (+0.0%) | 0.026 / 0.142 / 0.044 | 0.053 / 0.129 / 0.075 | 0.048 / 0.121 / 0.068 | 0.791 (n_ref 626) |
| bt05 | φ=5 | stopped: iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) | 41/41 | 8937 (+0.0%) | 0.041 / 0.159 / 0.065 | 0.077 / 0.177 / 0.108 | 0.081 / 0.194 / 0.115 | 0.735 (n_ref 626) |
| bt10 | φ=10 | stopped: iter 4: converged: every tuned group took a zero step (deadband or held at the per-bp cap) | 41/41 | 8934 (-0.0%) | 0.049 / 0.152 / 0.075 | 0.093 / 0.206 / 0.128 | 0.094 / 0.217 / 0.131 | 0.687 (n_ref 626) |
| bw01 | φ=1 (untempered both-layers reference) | stopped: iter 7: reached the 8-round limit | 39/41 | 8944 (+0.3%) | 0.014 / 0.123 / 0.024 | 0.031 / 0.095 / 0.047 | 0.026 / 0.084 / 0.040 | 0.812 (n_ref 626) |
| sw01 | seq-only reference | stopped: iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) | 41/41 | 9192 (-0.7%) | 0.021 / 0.030 / 0.025 | 0.059 / 0.117 / 0.078 | 0.066 / 0.136 / 0.089 | 0.137 (n_ref 626) |

**bt02 (φ=2)** — per round, tuning chromosomes chrXIV+chrII

| round | within-2× | calls | MacIsaac all | MacIsaac nonfitted | Rossi all | Rossi nonfitted | nucleosome copies (vs r0) | capped_under |
|---|---|---|---|---|---|---|---|---|
| r00 | 11/41 | 7488 | 0.030 / 0.154 / 0.050 | 0.024 / 0.127 / 0.041 | 0.033 / 0.173 / 0.056 | 0.015 / 0.112 / 0.026 | 8941 (+0.0%) | – |
| r01 | 30/41 | 4703 | 0.048 / 0.158 / 0.074 | 0.043 / 0.136 / 0.065 | 0.048 / 0.156 / 0.073 | 0.022 / 0.102 / 0.036 | 8942 (+0.0%) | – |
| r02 | 38/41 | 3745 | 0.060 / 0.157 / 0.087 | 0.054 / 0.140 / 0.078 | 0.052 / 0.135 / 0.075 | 0.026 / 0.096 / 0.040 | 8942 (+0.0%) | – |
| r03 | 40/41 | 3481 | 0.064 / 0.156 / 0.091 | 0.057 / 0.142 / 0.082 | 0.052 / 0.125 / 0.073 | 0.026 / 0.093 / 0.041 | 8942 (+0.0%) | – |
| r04 | 41/41 | 3418 | 0.065 / 0.154 / 0.091 | 0.057 / 0.142 / 0.081 | 0.049 / 0.115 / 0.068 | 0.026 / 0.093 / 0.041 | 8942 (+0.0%) | – |
| r05 | 41/41 | 3398 | 0.065 / 0.154 / 0.092 | 0.057 / 0.142 / 0.081 | 0.049 / 0.117 / 0.069 | 0.026 / 0.094 / 0.041 | 8942 (+0.0%) | – |

holdout chrIV:

| round | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted | nucleosome copies (vs r0) |
|---|---|---|---|---|---|---|---|
| r00 | 0.026 / 0.142 / 0.044 | 0.051 / 0.288 / 0.086 | 0.021 / 0.111 / 0.035 | 0.033 / 0.185 / 0.056 | 0.095 / 0.238 / 0.135 | 0.019 / 0.150 / 0.034 | 8591 (0.0%) |
| r05 | 0.053 / 0.129 / 0.075 | 0.075 / 0.159 / 0.102 | 0.049 / 0.122 / 0.070 | 0.048 / 0.121 / 0.068 | 0.141 / 0.134 / 0.137 | 0.031 / 0.113 / 0.049 | 8592 (0.0%) |

**bt05 (φ=5)** — per round, tuning chromosomes chrXIV+chrII

| round | within-2× | calls | MacIsaac all | MacIsaac nonfitted | Rossi all | Rossi nonfitted | nucleosome copies (vs r0) | capped_under |
|---|---|---|---|---|---|---|---|---|
| r00 | 13/41 | 5375 | 0.044 / 0.165 / 0.070 | 0.036 / 0.152 / 0.059 | 0.043 / 0.160 / 0.068 | 0.021 / 0.129 / 0.037 | 8936 (+0.0%) | – |
| r01 | 30/41 | 3393 | 0.080 / 0.190 / 0.113 | 0.070 / 0.175 / 0.100 | 0.069 / 0.162 / 0.097 | 0.035 / 0.128 / 0.055 | 8937 (+0.0%) | – |
| r02 | 39/41 | 3173 | 0.091 / 0.200 / 0.125 | 0.077 / 0.178 / 0.107 | 0.077 / 0.169 / 0.106 | 0.038 / 0.128 / 0.059 | 8937 (+0.0%) | – |
| r03 | 41/41 | 3222 | 0.091 / 0.203 / 0.125 | 0.076 / 0.178 / 0.107 | 0.077 / 0.171 / 0.106 | 0.038 / 0.129 / 0.059 | 8937 (+0.0%) | – |
| r04 | 41/41 | 3243 | 0.091 / 0.205 / 0.126 | 0.076 / 0.178 / 0.107 | 0.078 / 0.175 / 0.108 | 0.038 / 0.129 / 0.059 | 8937 (+0.0%) | – |
| r05 | 41/41 | 3249 | 0.091 / 0.206 / 0.126 | 0.077 / 0.179 / 0.107 | 0.078 / 0.175 / 0.107 | 0.038 / 0.129 / 0.059 | 8937 (+0.0%) | – |

holdout chrIV:

| round | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted | nucleosome copies (vs r0) |
|---|---|---|---|---|---|---|---|
| r00 | 0.041 / 0.159 / 0.065 | 0.134 / 0.252 / 0.175 | 0.032 / 0.140 / 0.052 | 0.048 / 0.194 / 0.077 | 0.269 / 0.224 / 0.244 | 0.028 / 0.174 / 0.048 | 8591 (0.0%) |
| r05 | 0.077 / 0.177 / 0.108 | 0.159 / 0.350 / 0.218 | 0.061 / 0.142 / 0.085 | 0.081 / 0.194 / 0.115 | 0.269 / 0.263 / 0.266 | 0.044 / 0.147 / 0.068 | 8591 (-0.0%) |

**bt10 (φ=10)** — per round, tuning chromosomes chrXIV+chrII

| round | within-2× | calls | MacIsaac all | MacIsaac nonfitted | Rossi all | Rossi nonfitted | nucleosome copies (vs r0) | capped_under |
|---|---|---|---|---|---|---|---|---|
| r00 | 12/41 | 4217 | 0.054 / 0.157 / 0.080 | 0.043 / 0.143 / 0.066 | 0.051 / 0.151 / 0.077 | 0.027 / 0.128 / 0.044 | 8934 (+0.0%) | – |
| r01 | 34/41 | 2942 | 0.096 / 0.195 / 0.128 | 0.086 / 0.188 / 0.118 | 0.077 / 0.157 / 0.103 | 0.042 / 0.133 / 0.063 | 8935 (+0.0%) | – |
| r02 | 38/41 | 2996 | 0.101 / 0.211 / 0.137 | 0.086 / 0.191 / 0.119 | 0.082 / 0.171 / 0.111 | 0.040 / 0.129 / 0.061 | 8935 (+0.0%) | – |
| r03 | 41/41 | 3100 | 0.105 / 0.226 / 0.143 | 0.086 / 0.192 / 0.119 | 0.087 / 0.187 / 0.118 | 0.040 / 0.129 / 0.061 | 8934 (-0.0%) | – |
| r04 | 41/41 | 3118 | 0.106 / 0.231 / 0.146 | 0.087 / 0.193 / 0.120 | 0.089 / 0.192 / 0.121 | 0.040 / 0.129 / 0.061 | 8934 (-0.0%) | – |

holdout chrIV:

| round | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted | nucleosome copies (vs r0) |
|---|---|---|---|---|---|---|---|
| r00 | 0.049 / 0.152 / 0.075 | 0.168 / 0.212 / 0.188 | 0.040 / 0.140 / 0.063 | 0.059 / 0.188 / 0.090 | 0.375 / 0.210 / 0.270 | 0.035 / 0.174 / 0.058 | 8594 (0.0%) |
| r04 | 0.093 / 0.206 / 0.128 | 0.193 / 0.389 / 0.258 | 0.075 / 0.168 / 0.104 | 0.094 / 0.217 / 0.131 | 0.330 / 0.297 / 0.313 | 0.050 / 0.163 / 0.077 | 8594 (-0.0%) |

**bw01 (φ=1 (untempered both-layers reference))** — per round, tuning chromosomes chrXIV+chrII

| round | within-2× | calls | MacIsaac all | MacIsaac nonfitted | Rossi all | Rossi nonfitted | nucleosome copies (vs r0) | capped_under |
|---|---|---|---|---|---|---|---|---|
| r00 | 9/41 | 12174 | 0.016 / 0.136 / 0.029 | 0.015 / 0.100 / 0.027 | 0.019 / 0.160 / 0.034 | 0.008 / 0.079 / 0.015 | 8919 (+0.0%) | – |
| r01 | 24/41 | 8802 | 0.022 / 0.138 / 0.039 | 0.023 / 0.105 / 0.038 | 0.023 / 0.141 / 0.040 | 0.010 / 0.065 / 0.017 | 8926 (+0.1%) | – |
| r02 | 31/41 | 6697 | 0.026 / 0.122 / 0.043 | 0.028 / 0.095 / 0.043 | 0.027 / 0.124 / 0.044 | 0.012 / 0.058 / 0.019 | 8931 (+0.1%) | – |
| r03 | 35/41 | 5594 | 0.028 / 0.108 / 0.044 | 0.028 / 0.086 / 0.043 | 0.028 / 0.108 / 0.044 | 0.012 / 0.052 / 0.019 | 8935 (+0.2%) | – |
| r04 | 37/41 | 4885 | 0.031 / 0.106 / 0.048 | 0.031 / 0.089 / 0.046 | 0.028 / 0.096 / 0.044 | 0.012 / 0.049 / 0.019 | 8939 (+0.2%) | – |
| r05 | 37/41 | 4523 | 0.034 / 0.107 / 0.052 | 0.032 / 0.092 / 0.047 | 0.029 / 0.090 / 0.043 | 0.012 / 0.050 / 0.020 | 8942 (+0.3%) | – |
| r06 | 37/41 | 4309 | 0.035 / 0.105 / 0.053 | 0.032 / 0.092 / 0.047 | 0.027 / 0.081 / 0.041 | 0.012 / 0.050 / 0.019 | 8943 (+0.3%) | – |
| r07 | 39/41 | 4118 | 0.035 / 0.101 / 0.052 | 0.032 / 0.092 / 0.047 | 0.026 / 0.074 / 0.038 | 0.013 / 0.053 / 0.020 | 8944 (+0.3%) | – |

holdout chrIV:

| round | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted | nucleosome copies (vs r0) |
|---|---|---|---|---|---|---|---|
| r00 | 0.014 / 0.123 / 0.024 | 0.015 / 0.310 / 0.029 | 0.013 / 0.084 / 0.022 | 0.020 / 0.185 / 0.036 | 0.032 / 0.293 / 0.058 | 0.012 / 0.113 / 0.021 | 8570 (0.0%) |
| r07 | 0.031 / 0.095 / 0.047 | 0.039 / 0.124 / 0.060 | 0.029 / 0.089 / 0.044 | 0.026 / 0.084 / 0.040 | 0.074 / 0.104 / 0.087 | 0.016 / 0.070 / 0.026 | 8592 (0.3%) |

**sw01 (seq-only reference)** — per round, tuning chromosomes chrXIV+chrII

| round | within-2× | calls | MacIsaac all | MacIsaac nonfitted | Rossi all | Rossi nonfitted | nucleosome copies (vs r0) | capped_under |
|---|---|---|---|---|---|---|---|---|
| r00 | 8/41 | 1982 | 0.026 / 0.036 / 0.030 | 0.023 / 0.037 / 0.028 | 0.020 / 0.027 / 0.023 | 0.011 / 0.026 / 0.015 | 9259 (+0.0%) | – |
| r01 | 29/41 | 2023 | 0.062 / 0.087 / 0.072 | 0.046 / 0.069 / 0.055 | 0.060 / 0.084 / 0.070 | 0.023 / 0.049 / 0.031 | 9256 (-0.0%) | – |
| r02 | 37/41 | 2586 | 0.065 / 0.117 / 0.083 | 0.051 / 0.097 / 0.067 | 0.059 / 0.105 / 0.075 | 0.025 / 0.069 / 0.037 | 9213 (-0.5%) | – |
| r03 | 39/41 | 2763 | 0.066 / 0.126 / 0.087 | 0.051 / 0.097 / 0.067 | 0.063 / 0.121 / 0.083 | 0.025 / 0.070 / 0.037 | 9198 (-0.7%) | – |
| r04 | 41/41 | 2833 | 0.067 / 0.132 / 0.089 | 0.051 / 0.100 / 0.068 | 0.064 / 0.126 / 0.085 | 0.025 / 0.071 / 0.037 | 9192 (-0.7%) | – |
| r05 | 41/41 | 2833 | 0.067 / 0.132 / 0.089 | 0.051 / 0.100 / 0.068 | 0.064 / 0.126 / 0.085 | 0.025 / 0.071 / 0.037 | 9192 (-0.7%) | – |

holdout chrIV:

| round | MacIsaac all | MacIsaac fitted | MacIsaac nonfitted | Rossi all | Rossi fitted | Rossi nonfitted | nucleosome copies (vs r0) |
|---|---|---|---|---|---|---|---|
| r00 | 0.021 / 0.030 / 0.025 | 0.044 / 0.009 / 0.015 | 0.020 / 0.035 / 0.026 | 0.016 / 0.025 / 0.019 | 0.156 / 0.014 / 0.025 | 0.013 / 0.032 / 0.018 | 8884 (0.0%) |
| r05 | 0.059 / 0.117 / 0.078 | 0.133 / 0.283 / 0.181 | 0.042 / 0.083 / 0.056 | 0.066 / 0.136 / 0.089 | 0.249 / 0.236 / 0.242 | 0.024 / 0.069 / 0.036 | 8826 (-0.6%) |

### Jobs

```
{
 "A": {
  "robocop_chrIV_twS05_both": {
   "coords": "coord_tw_chrIV.tsv",
   "count": "12596113",
   "decode": "12596112",
   "driver": "run_split_revfix_seq_maskoff_mr58.py",
   "traindir": "robocop_train_tw_sw01_05"
  },
  "robocop_chrIV_twS05_fib": {
   "coords": "coord_tw_chrIV.tsv",
   "count": "12596115",
   "decode": "12596114",
   "driver": "run_split_revfix_fiber_maskoff_mr58.py",
   "traindir": "robocop_train_tw_sw01_05"
  },
  "robocop_chrXIV_chrII_twS05_both": {
   "coords": "coord_tw_chrXIV_chrII.tsv",
   "count": "12596109",
   "decode": "12596108",
   "driver": "run_split_revfix_seq_maskoff_mr58.py",
   "traindir": "robocop_train_tw_sw01_05"
  },
  "robocop_chrXIV_chrII_twS05_fib": {
   "coords": "coord_tw_chrXIV_chrII.tsv",
   "count": "12596111",
   "decode": "12596110",
   "driver": "run_split_revfix_fiber_maskoff_mr58.py",
   "traindir": "robocop_train_tw_sw01_05"
  }
 },
 "B_refresh": {
  "bt05": {
   "after_holdout_validate": "12597735",
   "chereji": "12597859",
   "stopped": "stopped after iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap)",
   "summary": "12597862"
  }
 },
 "B_watch": {
  "bt02": "12597684",
  "bt05": "12597278",
  "bt10": "12597686"
 },
 "C": {
  "robocop_genome_tw_bw01_07": {
   "coords": "coord_genome_full.tsv",
   "count": "12596121",
   "decode": "12596120",
   "driver": "run_split_revfix_seq_maskoff_mr58.py",
   "traindir": "robocop_train_tw_bw01_07"
  },
  "robocop_genome_tw_fw01_07": {
   "coords": "coord_genome_full.tsv",
   "count": "12596119",
   "decode": "12596118",
   "driver": "run_split_revfix_fiber_maskoff_mr58.py",
   "traindir": "robocop_train_tw_fw01_07"
  },
  "robocop_genome_tw_sw01_05": {
   "coords": "coord_genome_full.tsv",
   "count": "12596117",
   "decode": "12596116",
   "driver": "run_split_revfix_seqonly_maskoff_mr58.py",
   "traindir": "robocop_train_tw_sw01_05"
  }
 },
 "chereji": {
  "tw_bt10_04": "12597861",
  "tw_bw01_07": "12596329",
  "tw_sw01_05": "12596330"
 },
 "score": {
  "A": "12596326",
  "C": "12596327",
  "summary": "12596328",
  "summary_after_A": "12596338"
 }
}
```

