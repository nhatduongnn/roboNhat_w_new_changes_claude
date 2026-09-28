# ABF1 vs a same-length decoy: tuner-v2 campaigns bd14/fd14/sd14 and bd72/fd72/sd72

_generated 2026-09-23 18:30 by `analysis/decoy_summary.py` (decoy campaigns from `conc_tuning`); campaigns still running are marked **INCOMPLETE** — re-run `python decoy_summary.py` to refresh._

### Rule 7 — what differs from the comparators (ba01/fa01/sa01 and bp72/fp72/sp72)

**Under test:** a decoy `Zz_decoy_abf1` of ABF1's length (14 or 23 bp) and equal prior weight (w_decoy = w_ABF1 every round), with a flat Fiber-seq footprint at ABF1's own mean rate (14 bp: W 0.0843 / C 0.0797; 23 bp: W 0.0892 / C 0.0960) and no sequence motif (PWM = background, Kd 1.0; the reverse-strand block is the reverse complement of background, see the build notes).

**Also different, all stated:** each run's trainDir is retrained with one extra motif (the untuned shared-root tf_prob shifts, but every tuner round rebuilds the priors from weights); the plain-width runs have a 50-round cap (ba01/fa01/sa01 had 25); bracket expiry is on (bracket_max_age 3; the plain comparators never had it); rpy2 is imported lazily in the plain-width decoy trees (no numeric effect).

**Identical:** targets (MacIsaac ABF1 58 on chrXIV+chrII), chromosomes, chrIV holdout, step cap, ρ cap (core basis for 7/2), w_nuc 35, w_unknown 1e-3 (masked), no φ, deadband 1.1×, untuned start.

**Seq-only is a control:** there the decoy's sequence emission is (forward block) exactly background and the fiber layer is off, so it only reflects its prior.

## Campaigns (comparators first)

| run | layers | width | block bp | comparator | state | rounds | ABF1 E / T (final) | λ (final round) | decoy occ / calls (final) | tune MacIsaac P/R/F1 (final) | tune Rossi _CX P/R/F1 (final) | holdout chrIV MacIsaac P/R/F1 r00 | holdout chrIV MacIsaac P/R/F1 final | nucleosome copies chrXIV+chrII (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ba01 | both layers | plain 14 bp (no decoy) | 14 | — | iter 8: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r08 | 63.7 / 58 = 1.1 | ABF1 1.803e-08 | – | 0.073 / 0.138 / 0.095 (110 calls) | 0.091 / 0.102 / 0.096 (110 calls) | 0.012 / 0.432 / 0.023 (1580 calls) | 0.026 / 0.091 / 0.041 (153 calls) | 8950 (+0.24% vs r0) |
| fa01 | fiber only | plain 14 bp (no decoy) | 14 | — | iter 16: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r16 | 61.3 / 58 = 1.06 | ABF1 2.419e-16 | – | 0.019 / 0.034 / 0.024 (106 calls) | 0.038 / 0.041 / 0.039 (106 calls) | 0.004 / 0.455 / 0.008 (4929 calls) | 0.014 / 0.045 / 0.021 (146 calls) | 8948 (+1.46% vs r0) |
| sa01 | sequence only | plain 14 bp (no decoy) | 14 | — | iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r05 | 54.4 / 58 = 0.938 | ABF1 3648 | – | 0.162 / 0.276 / 0.204 (99 calls) | 0.253 / 0.255 / 0.254 (99 calls) | – / 0.000 / – (0 calls) | 0.124 / 0.295 / 0.174 (105 calls) | 9433 (-0.12% vs r0) |
| bp72 | both layers | 7/2 23 bp (no decoy) | 23 | — | iter 13: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r13 | 60.4 / 58 = 1.04 | ABF1 2.438e-13 | – | 0.092 / 0.155 / 0.115 (98 calls) | 0.112 / 0.112 / 0.112 (98 calls) | 0.016 / 0.523 / 0.030 (1470 calls) | 0.036 / 0.091 / 0.051 (112 calls) | 8948 (+0.43% vs r0) |
| fp72 | fiber only | 7/2 23 bp (no decoy) | 23 | — | iter 20: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r20 | 63.3 / 58 = 1.09 | ABF1 1.82e-20 | – | 0.010 / 0.017 / 0.013 (98 calls) | 0.041 / 0.041 / 0.041 (98 calls) | 0.007 / 0.545 / 0.014 (3445 calls) | 0.016 / 0.045 / 0.024 (125 calls) | 8946 (+1.51% vs r0) |
| sp72 | sequence only | 7/2 23 bp (no decoy) | 23 | — | iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r05 | 36.1 / 58 = 0.622 | ABF1 1.434e+04 | – | 0.176 / 0.224 / 0.197 (74 calls) | 0.297 / 0.224 / 0.256 (74 calls) | – / 0.000 / – (0 calls) | 0.162 / 0.273 / 0.203 (74 calls) | 9427 (-0.10% vs r0) |
| bd14 | both layers | plain 14 bp | 14 | ba01 | iter 4: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r04 | 53.2 / 58 = 0.917 | ABF1 24.99 | occ 5041.9, calls 5030 | 0.131 / 0.224 / 0.166 (99 calls) | 0.182 / 0.184 / 0.183 (99 calls) | 0.106 / 0.159 / 0.127 (66 calls) | 0.087 / 0.205 / 0.122 (103 calls) | 8788 (-0.59% vs r0) |
| fd14 | fiber only | plain 14 bp | 14 | fa01 | iter 16: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r16 | 60.0 / 58 = 1.03 | ABF1 3.853e-16 | occ 8.5, calls 23 | 0.018 / 0.034 / 0.024 (112 calls) | 0.036 / 0.041 / 0.038 (112 calls) | 0.004 / 0.455 / 0.008 (4876 calls) | 0.013 / 0.045 / 0.020 (154 calls) | 8947 (+1.52% vs r0) |
| sd14 | sequence only | plain 14 bp | 14 | sa01 | iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r05 | 54.5 / 58 = 0.94 | ABF1 3609 | occ 114.1, calls 0 | 0.162 / 0.276 / 0.204 (99 calls) | 0.253 / 0.255 / 0.254 (99 calls) | – / 0.000 / – (0 calls) | 0.127 / 0.295 / 0.178 (102 calls) | 9397 (-0.50% vs r0) |
| bd72 | both layers | 7/2 23 bp | 23 | bp72 | iter 7: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r07 | 62.3 / 58 = 1.07 | ABF1 2.765e-06 | occ 885.6, calls 1129 | 0.217 / 0.345 / 0.267 (92 calls) | 0.304 / 0.286 / 0.295 (92 calls) | 0.095 / 0.432 / 0.156 (200 calls) | 0.156 / 0.273 / 0.198 (77 calls) | 8918 (+0.86% vs r0) |
| fd72 | fiber only | 7/2 23 bp | 23 | fp72 | iter 20: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r20 | 62.9 / 58 = 1.08 | ABF1 2.19e-20 | occ 2.3, calls 6 | 0.010 / 0.017 / 0.013 (97 calls) | 0.041 / 0.041 / 0.041 (97 calls) | 0.007 / 0.545 / 0.014 (3280 calls) | 0.016 / 0.045 / 0.024 (123 calls) | 8945 (+1.57% vs r0) |
| sd72 | sequence only | 7/2 23 bp | 23 | sp72 | iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r05 | 36.2 / 58 = 0.624 | ABF1 1.434e+04 | occ 33.4, calls 0 | 0.178 / 0.224 / 0.198 (73 calls) | 0.301 / 0.224 / 0.257 (73 calls) | – / 0.000 / – (0 calls) | 0.162 / 0.273 / 0.203 (74 calls) | 9403 (-0.36% vs r0) |

## Spurious ABF1 calls: where do they go?

Comparator ABF1 calls with no MacIsaac ABF1 site within 30 bp, checked against the decoy campaign's calls in the same round (r00: both untuned, λ = 1 on the same ABF1 Kd; final: each campaign's own last round). 'gone' = neither an ABF1 nor a decoy call within 30 bp.

| run | comparator | set / rounds | comparator spurious / all ABF1 calls | moved to decoy | kept as ABF1 | gone | decoy calls (all) | decoy calls on a MacIsaac ABF1 site |
|---|---|---|---|---|---|---|---|---|
| bd14 | ba01 | tune r00 | 1369 / 1399 | 1308 (95.5%) | 51 (3.7%) | 50 (3.7%) | 4140 | 29 |
| bd14 | ba01 | holdout r00 | 1561 / 1580 | 1489 (95.4%) | 62 (4.0%) | 58 (3.7%) | 4172 | 20 |
| bd14 | ba01 | tune final (r08 vs r04) | 102 / 110 | 100 (98.0%) | 2 (2.0%) | 2 (2.0%) | 5030 | 32 |
| fd14 | fa01 | tune r00 | 4807 / 4841 | 3069 (63.8%) | 4663 (97.0%) | 45 (0.9%) | 2906 | 22 |
| fd14 | fa01 | holdout r00 | 4907 / 4929 | 3071 (62.6%) | 4749 (96.8%) | 48 (1.0%) | 2921 | 12 |
| fd14 | fa01 | tune final (r16 vs r16) | 104 / 106 | 22 (21.2%) | 104 (100.0%) | 0 (0.0%) | 23 | 0 |
| sd14 | sa01 | tune r00 | 0 / 1 | 0 | 0 | 0 | 0 | 0 |
| sd14 | sa01 | holdout r00 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| sd14 | sa01 | tune final (r05 vs r05) | 83 / 99 | 0 (0.0%) | 83 (100.0%) | 0 (0.0%) | 0 | 0 |
| bd72 | bp72 | tune r00 | 1408 / 1442 | 1256 (89.2%) | 183 (13.0%) | 61 (4.3%) | 2652 | 8 |
| bd72 | bp72 | holdout r00 | 1447 / 1470 | 1297 (89.6%) | 188 (13.0%) | 64 (4.4%) | 2686 | 10 |
| bd72 | bp72 | tune final (r13 vs r07) | 89 / 98 | 78 (87.6%) | 20 (22.5%) | 1 (1.1%) | 1129 | 5 |
| fd72 | fp72 | tune r00 | 3419 / 3454 | 1363 (39.9%) | 3193 (93.4%) | 33 (1.0%) | 1371 | 2 |
| fd72 | fp72 | holdout r00 | 3420 / 3445 | 1426 (41.7%) | 3219 (94.1%) | 40 (1.2%) | 1425 | 3 |
| fd72 | fp72 | tune final (r20 vs r20) | 97 / 98 | 5 (5.2%) | 96 (99.0%) | 0 (0.0%) | 6 | 0 |
| sd72 | sp72 | tune r00 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| sd72 | sp72 | holdout r00 | 0 / 0 | 0 | 0 | 0 | 0 | 0 |
| sd72 | sp72 | tune final (r05 vs r05) | 61 / 74 | 0 (0.0%) | 60 (98.4%) | 1 (1.6%) | 0 | 0 |

## bd14 — both layers, plain 14 bp (vs ba01)

src trainDir `robocop_train_decoy14`, driver `run_split_revfix_seq_maskoff_decoy14.py`, cap basis `own motif`, tied `{"Zz_decoy_abf1": "ABF1 x1"}`, max_rounds 50, deadband 1.1x

### ABF1 P / R / F1 — tuning chrXIV+chrII (bd14 r00 / final; ba01 r00 / final)

| ref | bd14 r00 | bd14 final | ba01 r00 | ba01 final |
|---|---|---|---|---|
| MacIsaac | 0.136 / 0.138 / 0.137 (59 calls) | 0.131 / 0.224 / 0.166 (99 calls) | 0.021 / 0.517 / 0.041 (1399 calls) | 0.073 / 0.138 / 0.095 (110 calls) |
| Rossi _CX | 0.220 / 0.133 / 0.166 (59 calls) | 0.182 / 0.184 / 0.183 (99 calls) | 0.030 / 0.429 / 0.056 (1399 calls) | 0.091 / 0.102 / 0.096 (110 calls) |

### ABF1 P / R / F1 — holdout chrIV (bd14 r00 / final; ba01 r00 / final)

| ref | bd14 r00 | bd14 final | ba01 r00 | ba01 final |
|---|---|---|---|---|
| MacIsaac | 0.106 / 0.159 / 0.127 (66 calls) | 0.087 / 0.205 / 0.122 (103 calls) | 0.012 / 0.432 / 0.023 (1580 calls) | 0.026 / 0.091 / 0.041 (153 calls) |
| Rossi _CX | 0.212 / 0.136 / 0.166 (66 calls) | 0.165 / 0.165 / 0.165 (103 calls) | 0.021 / 0.320 / 0.039 (1580 calls) | 0.039 / 0.058 / 0.047 (153 calls) |

### Per round (bd14; T = ABF1 58)

| round | ABF1 E | ABF1 calls | decoy occ | decoy calls | w_decoy = w_ABF1 (→ next) | λ_ABF1 → next | step | ABF1 MacIsaac P/R/F1 | ABF1 Rossi P/R/F1 | nucleosome copies |
|---|---|---|---|---|---|---|---|---|---|---|
| r00 | 31.5 | 59 | 3833.9 | 4140 | 4.729e-07 → 1.091e-06 | 1 → 2.306 | secant | 0.136 / 0.138 / 0.137 (59 calls) | 0.220 / 0.133 / 0.166 (59 calls) | 8840 (+0.00%) |
| r01 | 36.5 | 70 | 4122.2 | 4391 | 1.091e-06 → 2.794e-06 | 2.306 → 5.909 | secant | 0.129 / 0.155 / 0.141 (70 calls) | 0.200 / 0.143 / 0.167 (70 calls) | 8829 (-0.13%) |
| r02 | 42.5 | 82 | 4455.4 | 4695 | 2.794e-06 → 6.62e-06 | 5.909 → 14 | secant | 0.122 / 0.172 / 0.143 (82 calls) | 0.183 / 0.153 / 0.167 (82 calls) | 8816 (-0.28%) |
| r03 | 48.9 | 92 | 4804.5 | 4884 | 6.62e-06 → 1.182e-05 | 14 → 24.99 | secant | 0.130 / 0.207 / 0.160 (92 calls) | 0.185 / 0.173 / 0.179 (92 calls) | 8799 (-0.47%) |
| r04 | 53.2 | 99 | 5041.9 | 5030 | 1.182e-05 → 1.182e-05 | 24.99 → 24.99 | deadband | 0.131 / 0.224 / 0.166 (99 calls) | 0.182 / 0.184 / 0.183 (99 calls) | 8788 (-0.59%) |

state: stopped after iter 4: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12670845", "decode": "12670844", "holdout_count": "12671656", "holdout_decode": "12671655", "holdout_validate": "12671657", "next": "12670846"}, "1": {"count": "12671624", "decode": "12671623", "next": "12671625"}, "2": {"count": "12672462", "decode": "12672461", "next": "12672463"}, "3": {"count": "12675822", "decode": "12675821", "next": "12675823"}, "4": {"chereji": "12676773", "count": "12676166", "decode": "12676165", "holdout_count": "12676771", "holdout_decode": "12676770", "holdout_validate": "12676772", "next": "12676167", "summary": "12676774"}}`

## fd14 — fiber only, plain 14 bp (vs fa01)

src trainDir `robocop_train_decoy14`, driver `run_split_revfix_fiber_maskoff_decoy14.py`, cap basis `own motif`, tied `{"Zz_decoy_abf1": "ABF1 x1"}`, max_rounds 50, deadband 1.1x

### ABF1 P / R / F1 — tuning chrXIV+chrII (fd14 r00 / final; fa01 r00 / final)

| ref | fd14 r00 | fd14 final | fa01 r00 | fa01 final |
|---|---|---|---|---|
| MacIsaac | 0.007 / 0.569 / 0.014 (4777 calls) | 0.018 / 0.034 / 0.024 (112 calls) | 0.007 / 0.569 / 0.013 (4841 calls) | 0.019 / 0.034 / 0.024 (106 calls) |
| Rossi _CX | 0.009 / 0.449 / 0.018 (4777 calls) | 0.036 / 0.041 / 0.038 (112 calls) | 0.009 / 0.449 / 0.018 (4841 calls) | 0.038 / 0.041 / 0.039 (106 calls) |

### ABF1 P / R / F1 — holdout chrIV (fd14 r00 / final; fa01 r00 / final)

| ref | fd14 r00 | fd14 final | fa01 r00 | fa01 final |
|---|---|---|---|---|
| MacIsaac | 0.004 / 0.455 / 0.008 (4876 calls) | 0.013 / 0.045 / 0.020 (154 calls) | 0.004 / 0.455 / 0.008 (4929 calls) | 0.014 / 0.045 / 0.021 (146 calls) |
| Rossi _CX | 0.008 / 0.369 / 0.015 (4876 calls) | 0.013 / 0.019 / 0.016 (154 calls) | 0.008 / 0.369 / 0.015 (4929 calls) | 0.014 / 0.019 / 0.016 (146 calls) |

### Per round (fd14; T = ABF1 58)

| round | ABF1 E | ABF1 calls | decoy occ | decoy calls | w_decoy = w_ABF1 (→ next) | λ_ABF1 → next | step | ABF1 MacIsaac P/R/F1 | ABF1 Rossi P/R/F1 | nucleosome copies |
|---|---|---|---|---|---|---|---|---|---|---|
| r00 | 3345.4 | 4777 | 1304.6 | 2906 | 4.729e-07 → 4.729e-08 | 1 → 0.1 | secant+step-cap | 0.007 / 0.569 / 0.014 (4777 calls) | 0.009 / 0.449 / 0.018 (4777 calls) | 8813 (+0.00%) |
| r01 | 2795.2 | 4089 | 1048.0 | 2405 | 4.729e-08 → 4.729e-09 | 0.1 → 0.01 | secant+step-cap | 0.008 / 0.534 / 0.015 (4089 calls) | 0.010 / 0.418 / 0.020 (4089 calls) | 8842 (+0.32%) |
| r02 | 2316.8 | 3475 | 834.8 | 1948 | 4.729e-09 → 4.729e-10 | 0.01 → 0.001 | secant+step-cap | 0.008 / 0.500 / 0.016 (3475 calls) | 0.011 / 0.388 / 0.021 (3475 calls) | 8863 (+0.56%) |
| r03 | 1886.9 | 2861 | 647.0 | 1536 | 4.729e-10 → 4.729e-11 | 0.001 → 0.0001 | secant+step-cap | 0.009 / 0.466 / 0.018 (2861 calls) | 0.013 / 0.367 / 0.024 (2861 calls) | 8884 (+0.81%) |
| r04 | 1532.3 | 2383 | 494.1 | 1189 | 4.729e-11 → 4.729e-12 | 0.0001 → 1e-05 | secant+step-cap | 0.010 / 0.431 / 0.020 (2383 calls) | 0.014 / 0.337 / 0.027 (2383 calls) | 8897 (+0.95%) |
| r05 | 1219.6 | 1921 | 378.7 | 908 | 4.729e-12 → 4.729e-13 | 1e-05 → 1e-06 | secant+step-cap | 0.012 / 0.397 / 0.023 (1921 calls) | 0.016 / 0.316 / 0.031 (1921 calls) | 8911 (+1.11%) |
| r06 | 962.5 | 1550 | 287.7 | 703 | 4.729e-13 → 4.729e-14 | 1e-06 → 1e-07 | secant+step-cap | 0.012 / 0.328 / 0.024 (1550 calls) | 0.017 / 0.265 / 0.032 (1550 calls) | 8921 (+1.22%) |
| r07 | 750.6 | 1219 | 214.8 | 539 | 4.729e-14 → 4.729e-15 | 1e-07 → 1e-08 | secant+step-cap | 0.014 / 0.293 / 0.027 (1219 calls) | 0.017 / 0.214 / 0.032 (1219 calls) | 8928 (+1.30%) |
| r08 | 588.8 | 976 | 158.6 | 400 | 4.729e-15 → 4.729e-16 | 1e-08 → 1e-09 | secant+step-cap | 0.015 / 0.259 / 0.029 (976 calls) | 0.019 / 0.194 / 0.035 (976 calls) | 8931 (+1.34%) |
| r09 | 449.9 | 758 | 114.2 | 293 | 4.729e-16 → 4.729e-17 | 1e-09 → 1e-10 | secant+step-cap | 0.017 / 0.224 / 0.032 (758 calls) | 0.021 / 0.163 / 0.037 (758 calls) | 8935 (+1.38%) |
| r10 | 334.4 | 579 | 78.8 | 213 | 4.729e-17 → 4.729e-18 | 1e-10 → 1e-11 | secant+step-cap | 0.021 / 0.207 / 0.038 (579 calls) | 0.026 / 0.153 / 0.044 (579 calls) | 8938 (+1.42%) |
| r11 | 250.1 | 431 | 54.7 | 149 | 4.729e-18 → 4.729e-19 | 1e-11 → 1e-12 | secant+step-cap | 0.028 / 0.207 / 0.049 (431 calls) | 0.032 / 0.143 / 0.053 (431 calls) | 8940 (+1.44%) |
| r12 | 187.2 | 322 | 38.1 | 100 | 4.729e-19 → 4.729e-20 | 1e-12 → 1e-13 | secant+step-cap | 0.031 / 0.172 / 0.053 (322 calls) | 0.040 / 0.133 / 0.062 (322 calls) | 8941 (+1.45%) |
| r13 | 135.8 | 232 | 25.0 | 70 | 4.729e-20 → 4.729e-21 | 1e-13 → 1e-14 | secant+step-cap | 0.026 / 0.103 / 0.041 (232 calls) | 0.039 / 0.092 / 0.055 (232 calls) | 8943 (+1.47%) |
| r14 | 98.7 | 175 | 16.3 | 44 | 4.729e-21 → 4.729e-22 | 1e-14 → 1e-15 | secant+step-cap | 0.029 / 0.086 / 0.043 (175 calls) | 0.046 / 0.082 / 0.059 (175 calls) | 8945 (+1.49%) |
| r15 | 70.4 | 123 | 10.3 | 28 | 4.729e-22 → 1.822e-22 | 1e-15 → 3.853e-16 | secant | 0.016 / 0.034 / 0.022 (123 calls) | 0.033 / 0.041 / 0.036 (123 calls) | 8946 (+1.51%) |
| r16 | 60.0 | 112 | 8.5 | 23 | 1.822e-22 → 1.822e-22 | 3.853e-16 → 3.853e-16 | deadband | 0.018 / 0.034 / 0.024 (112 calls) | 0.036 / 0.041 / 0.038 (112 calls) | 8947 (+1.52%) |

state: stopped after iter 16: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12670887", "decode": "12670886", "holdout_count": "12671848", "holdout_decode": "12671847", "holdout_validate": "12671849", "next": "12670888"}, "1": {"count": "12671842", "decode": "12671841", "next": "12671843"}, "10": {"count": "12679404", "decode": "12679403", "next": "12679405"}, "11": {"count": "12679910", "decode": "12679909", "next": "12679911"}, "12": {"count": "12680219", "decode": "12680218", "next": "12680220"}, "13": {"count": "12680581", "decode": "12680580", "next": "12680582"}, "14": {"count": "12680934", "decode": "12680933", "next": "12680935"}, "15": {"count": "12681462", "decode": "12681461", "next": "12681463"}, "16": {"chereji": "12682635", "count": "12682323", "decode": "12682322", "holdout_count": "12682633", "holdout_decode": "12682632", "holdout_validate": "12682634", "next": "12682324", "summary": "12682636"}, "2": {"count": "12672687", "decode": "12672686", "next": "12672688"}, "3": {"count": "12675829", "decode": "12675828", "next": "12675830"}, "4": {"count": "12676237", "decode": "12676236", "next": "12676238"}, "5": {"count": "12676846", "decode": "12676845", "next": "12676847"}, "6": {"count": "12677349", "decode": "12677348", "next": "12677350"}, "7": {"count": "12678251", "decode": "12678250", "next": "12678252"}, "8": {"count": "12678680", "decode": "12678679", "next": "12678681"}, "9": {"count": "12679030", "decode": "12679029", "next": "12679031"}}`

## sd14 — sequence only, plain 14 bp (vs sa01)

src trainDir `robocop_train_decoy14`, driver `run_split_revfix_seqonly_maskoff_decoy14.py`, cap basis `own motif`, tied `{"Zz_decoy_abf1": "ABF1 x1"}`, max_rounds 50, deadband 1.1x

### ABF1 P / R / F1 — tuning chrXIV+chrII (sd14 r00 / final; sa01 r00 / final)

| ref | sd14 r00 | sd14 final | sa01 r00 | sa01 final |
|---|---|---|---|---|
| MacIsaac | 1.000 / 0.017 / 0.034 (1 calls) | 0.162 / 0.276 / 0.204 (99 calls) | 1.000 / 0.017 / 0.034 (1 calls) | 0.162 / 0.276 / 0.204 (99 calls) |
| Rossi _CX | 1.000 / 0.010 / 0.020 (1 calls) | 0.253 / 0.255 / 0.254 (99 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 0.253 / 0.255 / 0.254 (99 calls) |

### ABF1 P / R / F1 — holdout chrIV (sd14 r00 / final; sa01 r00 / final)

| ref | sd14 r00 | sd14 final | sa01 r00 | sa01 final |
|---|---|---|---|---|
| MacIsaac | – / 0.000 / – (0 calls) | 0.127 / 0.295 / 0.178 (102 calls) | – / 0.000 / – (0 calls) | 0.124 / 0.295 / 0.174 (105 calls) |
| Rossi _CX | – / 0.000 / – (0 calls) | 0.265 / 0.262 / 0.263 (102 calls) | – / 0.000 / – (0 calls) | 0.257 / 0.262 / 0.260 (105 calls) |

### Per round (sd14; T = ABF1 58)

| round | ABF1 E | ABF1 calls | decoy occ | decoy calls | w_decoy = w_ABF1 (→ next) | λ_ABF1 → next | step | ABF1 MacIsaac P/R/F1 | ABF1 Rossi P/R/F1 | nucleosome copies |
|---|---|---|---|---|---|---|---|---|---|---|
| r00 | 0.2 | 1 | 0.0 | 0 | 4.729e-07 → 4.729e-06 | 1 → 10 | no-slope | 1.000 / 0.017 / 0.034 (1 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 9444 (+0.00%) |
| r01 | 1.1 | 2 | 0.0 | 0 | 4.729e-06 → 4.729e-05 | 10 → 100 | secant+step-cap | 0.500 / 0.017 / 0.033 (2 calls) | 0.500 / 0.010 / 0.020 (2 calls) | 9444 (-0.00%) |
| r02 | 5.4 | 11 | 0.0 | 0 | 4.729e-05 → 0.0004729 | 100 → 1000 | secant+step-cap | 0.182 / 0.034 / 0.058 (11 calls) | 0.545 / 0.061 / 0.110 (11 calls) | 9442 (-0.02%) |
| r03 | 25.1 | 55 | 3.2 | 0 | 0.0004729 → 0.001332 | 1000 → 2818 | secant | 0.164 / 0.155 / 0.159 (55 calls) | 0.291 / 0.163 / 0.209 (55 calls) | 9429 (-0.16%) |
| r04 | 47.2 | 93 | 71.3 | 0 | 0.001332 → 0.001707 | 2818 → 3609 | secant | 0.161 / 0.259 / 0.199 (93 calls) | 0.258 / 0.245 / 0.251 (93 calls) | 9406 (-0.40%) |
| r05 | 54.5 | 99 | 114.1 | 0 | 0.001707 → 0.001707 | 3609 → 3609 | deadband | 0.162 / 0.276 / 0.204 (99 calls) | 0.253 / 0.255 / 0.254 (99 calls) | 9397 (-0.50%) |

state: stopped after iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12670929", "decode": "12670928", "holdout_count": "12671864", "holdout_decode": "12671863", "holdout_validate": "12671865", "next": "12670930"}, "1": {"count": "12671851", "decode": "12671850", "next": "12671852"}, "2": {"count": "12672695", "decode": "12672694", "next": "12672696"}, "3": {"count": "12675852", "decode": "12675851", "next": "12675853"}, "4": {"count": "12676226", "decode": "12676225", "next": "12676227"}, "5": {"chereji": "12677371", "count": "12676813", "decode": "12676812", "holdout_count": "12677369", "holdout_decode": "12677368", "holdout_validate": "12677370", "next": "12676814", "summary": "12677372"}}`

## bd72 — both layers, 7/2 23 bp (vs bp72)

src trainDir `robocop_train_wide_decoy23`, driver `run_split_revfix_seq_maskoff_decoy72.py`, cap basis `robocop_train_fiberonly`, tied `{"Zz_decoy_abf1": "ABF1 x1"}`, max_rounds 50, deadband 1.1x

### ABF1 P / R / F1 — tuning chrXIV+chrII (bd72 r00 / final; bp72 r00 / final)

| ref | bd72 r00 | bd72 final | bp72 r00 | bp72 final |
|---|---|---|---|---|
| MacIsaac | 0.151 / 0.552 / 0.237 (212 calls) | 0.217 / 0.345 / 0.267 (92 calls) | 0.024 / 0.586 / 0.045 (1442 calls) | 0.092 / 0.155 / 0.115 (98 calls) |
| Rossi _CX | 0.208 / 0.449 / 0.284 (212 calls) | 0.304 / 0.286 / 0.295 (92 calls) | 0.033 / 0.490 / 0.062 (1442 calls) | 0.112 / 0.112 / 0.112 (98 calls) |

### ABF1 P / R / F1 — holdout chrIV (bd72 r00 / final; bp72 r00 / final)

| ref | bd72 r00 | bd72 final | bp72 r00 | bp72 final |
|---|---|---|---|---|
| MacIsaac | 0.095 / 0.432 / 0.156 (200 calls) | 0.156 / 0.273 / 0.198 (77 calls) | 0.016 / 0.523 / 0.030 (1470 calls) | 0.036 / 0.091 / 0.051 (112 calls) |
| Rossi _CX | 0.155 / 0.301 / 0.205 (200 calls) | 0.247 / 0.184 / 0.211 (77 calls) | 0.025 / 0.359 / 0.047 (1470 calls) | 0.054 / 0.058 / 0.056 (112 calls) |

### Per round (bd72; T = ABF1 58)

| round | ABF1 E | ABF1 calls | decoy occ | decoy calls | w_decoy = w_ABF1 (→ next) | λ_ABF1 → next | step | ABF1 MacIsaac P/R/F1 | ABF1 Rossi P/R/F1 | nucleosome copies |
|---|---|---|---|---|---|---|---|---|---|---|
| r00 | 148.1 | 212 | 2422.8 | 2652 | 9.239e-08 → 2.522e-08 | 1 → 0.273 | secant | 0.151 / 0.552 / 0.237 (212 calls) | 0.208 / 0.449 / 0.284 (212 calls) | 8842 (+0.00%) |
| r01 | 135.7 | 200 | 2217.2 | 2471 | 2.522e-08 → 3.175e-09 | 0.273 → 0.03436 | secant | 0.160 / 0.552 / 0.248 (200 calls) | 0.220 / 0.449 / 0.295 (200 calls) | 8853 (+0.12%) |
| r02 | 117.7 | 173 | 1924.7 | 2227 | 3.175e-09 → 3.175e-10 | 0.03436 → 0.003436 | secant+step-cap | 0.168 / 0.500 / 0.251 (173 calls) | 0.231 / 0.408 / 0.295 (173 calls) | 8863 (+0.23%) |
| r03 | 101.2 | 144 | 1611.7 | 1925 | 3.175e-10 → 3.175e-11 | 0.003436 → 0.0003436 | secant+step-cap | 0.174 / 0.431 / 0.248 (144 calls) | 0.257 / 0.378 / 0.306 (144 calls) | 8878 (+0.40%) |
| r04 | 86.7 | 123 | 1335.2 | 1589 | 3.175e-11 → 3.175e-12 | 0.0003436 → 3.436e-05 | secant+step-cap | 0.187 / 0.397 / 0.254 (123 calls) | 0.260 / 0.327 / 0.290 (123 calls) | 8894 (+0.59%) |
| r05 | 73.8 | 108 | 1115.4 | 1373 | 3.175e-12 → 6.024e-13 | 3.436e-05 → 6.52e-06 | secant | 0.194 / 0.362 / 0.253 (108 calls) | 0.278 / 0.306 / 0.291 (108 calls) | 8903 (+0.69%) |
| r06 | 65.7 | 96 | 956.1 | 1223 | 6.024e-13 → 2.555e-13 | 6.52e-06 → 2.765e-06 | secant | 0.208 / 0.345 / 0.260 (96 calls) | 0.292 / 0.286 / 0.289 (96 calls) | 8914 (+0.82%) |
| r07 | 62.3 | 92 | 885.6 | 1129 | 2.555e-13 → 2.555e-13 | 2.765e-06 → 2.765e-06 | deadband | 0.217 / 0.345 / 0.267 (92 calls) | 0.304 / 0.286 / 0.295 (92 calls) | 8918 (+0.86%) |

state: stopped after iter 7: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12670933", "decode": "12670932", "holdout_count": "12671872", "holdout_decode": "12671871", "holdout_validate": "12671873", "next": "12670934"}, "1": {"count": "12671857", "decode": "12671856", "next": "12671858"}, "2": {"count": "12672713", "decode": "12672712", "next": "12672714"}, "3": {"count": "12675855", "decode": "12675854", "next": "12675856"}, "4": {"count": "12676338", "decode": "12676337", "next": "12676339"}, "5": {"count": "12676889", "decode": "12676888", "next": "12676890"}, "6": {"count": "12677421", "decode": "12677420", "next": "12677422"}, "7": {"chereji": "12678754", "count": "12678312", "decode": "12678311", "holdout_count": "12678752", "holdout_decode": "12678751", "holdout_validate": "12678753", "next": "12678313", "summary": "12678755"}}`

## fd72 — fiber only, 7/2 23 bp (vs fp72)

src trainDir `robocop_train_wide_decoy23`, driver `run_split_revfix_fiber_maskoff_decoy72.py`, cap basis `robocop_train_fiberonly`, tied `{"Zz_decoy_abf1": "ABF1 x1"}`, max_rounds 50, deadband 1.1x

### ABF1 P / R / F1 — tuning chrXIV+chrII (fd72 r00 / final; fp72 r00 / final)

| ref | fd72 r00 | fd72 final | fp72 r00 | fp72 final |
|---|---|---|---|---|
| MacIsaac | 0.010 / 0.586 / 0.020 (3270 calls) | 0.010 / 0.017 / 0.013 (97 calls) | 0.010 / 0.586 / 0.019 (3454 calls) | 0.010 / 0.017 / 0.013 (98 calls) |
| Rossi _CX | 0.014 / 0.469 / 0.027 (3270 calls) | 0.041 / 0.041 / 0.041 (97 calls) | 0.014 / 0.490 / 0.027 (3454 calls) | 0.041 / 0.041 / 0.041 (98 calls) |

### ABF1 P / R / F1 — holdout chrIV (fd72 r00 / final; fp72 r00 / final)

| ref | fd72 r00 | fd72 final | fp72 r00 | fp72 final |
|---|---|---|---|---|
| MacIsaac | 0.007 / 0.545 / 0.014 (3280 calls) | 0.016 / 0.045 / 0.024 (123 calls) | 0.007 / 0.545 / 0.014 (3445 calls) | 0.016 / 0.045 / 0.024 (125 calls) |
| Rossi _CX | 0.013 / 0.408 / 0.025 (3280 calls) | 0.016 / 0.019 / 0.018 (123 calls) | 0.012 / 0.408 / 0.024 (3445 calls) | 0.016 / 0.019 / 0.018 (125 calls) |

### Per round (fd72; T = ABF1 58)

| round | ABF1 E | ABF1 calls | decoy occ | decoy calls | w_decoy = w_ABF1 (→ next) | λ_ABF1 → next | step | ABF1 MacIsaac P/R/F1 | ABF1 Rossi P/R/F1 | nucleosome copies |
|---|---|---|---|---|---|---|---|---|---|---|
| r00 | 2574.8 | 3270 | 693.7 | 1371 | 9.239e-08 → 9.239e-09 | 1 → 0.1 | secant+step-cap | 0.010 / 0.586 / 0.020 (3270 calls) | 0.014 / 0.469 / 0.027 (3270 calls) | 8807 (+0.00%) |
| r01 | 2268.1 | 2893 | 564.9 | 1166 | 9.239e-09 → 9.239e-10 | 0.1 → 0.01 | secant+step-cap | 0.010 / 0.517 / 0.020 (2893 calls) | 0.014 / 0.418 / 0.027 (2893 calls) | 8830 (+0.26%) |
| r02 | 1984.2 | 2565 | 444.2 | 962 | 9.239e-10 → 9.239e-11 | 0.01 → 0.001 | secant+step-cap | 0.011 / 0.500 / 0.022 (2565 calls) | 0.015 / 0.398 / 0.029 (2565 calls) | 8850 (+0.48%) |
| r03 | 1726.8 | 2276 | 347.6 | 750 | 9.239e-11 → 9.239e-12 | 0.001 → 0.0001 | secant+step-cap | 0.011 / 0.414 / 0.021 (2276 calls) | 0.014 / 0.337 / 0.028 (2276 calls) | 8866 (+0.66%) |
| r04 | 1500.0 | 1975 | 271.8 | 598 | 9.239e-12 → 9.239e-13 | 0.0001 → 1e-05 | secant+step-cap | 0.011 / 0.379 / 0.022 (1975 calls) | 0.015 / 0.306 / 0.029 (1975 calls) | 8880 (+0.82%) |
| r05 | 1289.4 | 1719 | 208.5 | 461 | 9.239e-13 → 9.239e-14 | 1e-05 → 1e-06 | secant+step-cap | 0.013 / 0.379 / 0.025 (1719 calls) | 0.017 / 0.306 / 0.033 (1719 calls) | 8892 (+0.96%) |
| r06 | 1097.7 | 1485 | 158.4 | 359 | 9.239e-14 → 9.239e-15 | 1e-06 → 1e-07 | secant+step-cap | 0.013 / 0.345 / 0.026 (1485 calls) | 0.018 / 0.276 / 0.034 (1485 calls) | 8902 (+1.07%) |
| r07 | 928.3 | 1258 | 118.6 | 272 | 9.239e-15 → 9.239e-16 | 1e-07 → 1e-08 | secant+step-cap | 0.015 / 0.328 / 0.029 (1258 calls) | 0.021 / 0.276 / 0.040 (1258 calls) | 8911 (+1.18%) |
| r08 | 782.3 | 1067 | 90.6 | 202 | 9.239e-16 → 9.239e-17 | 1e-08 → 1e-09 | secant+step-cap | 0.016 / 0.293 / 0.030 (1067 calls) | 0.022 / 0.235 / 0.039 (1067 calls) | 8921 (+1.29%) |
| r09 | 657.6 | 920 | 69.4 | 166 | 9.239e-17 → 9.239e-18 | 1e-09 → 1e-10 | secant+step-cap | 0.016 / 0.259 / 0.031 (920 calls) | 0.022 / 0.204 / 0.039 (920 calls) | 8928 (+1.37%) |
| r10 | 544.5 | 763 | 52.0 | 131 | 9.239e-18 → 9.239e-19 | 1e-10 → 1e-11 | secant+step-cap | 0.018 / 0.241 / 0.034 (763 calls) | 0.025 / 0.194 / 0.044 (763 calls) | 8932 (+1.42%) |
| r11 | 446.0 | 634 | 36.9 | 94 | 9.239e-19 → 9.239e-20 | 1e-11 → 1e-12 | secant+step-cap | 0.022 / 0.241 / 0.040 (634 calls) | 0.028 / 0.184 / 0.049 (634 calls) | 8934 (+1.43%) |
| r12 | 363.0 | 512 | 26.3 | 62 | 9.239e-20 → 9.239e-21 | 1e-12 → 1e-13 | secant+step-cap | 0.027 / 0.241 / 0.049 (512 calls) | 0.035 / 0.184 / 0.059 (512 calls) | 8937 (+1.47%) |
| r13 | 292.4 | 421 | 18.4 | 45 | 9.239e-21 → 9.239e-22 | 1e-13 → 1e-14 | secant+step-cap | 0.029 / 0.207 / 0.050 (421 calls) | 0.038 / 0.163 / 0.062 (421 calls) | 8939 (+1.49%) |
| r14 | 234.5 | 338 | 12.3 | 32 | 9.239e-22 → 9.239e-23 | 1e-14 → 1e-15 | secant+step-cap | 0.033 / 0.190 / 0.056 (338 calls) | 0.044 / 0.153 / 0.069 (338 calls) | 8940 (+1.51%) |
| r15 | 191.5 | 275 | 8.5 | 22 | 9.239e-23 → 9.239e-24 | 1e-15 → 1e-16 | secant+step-cap | 0.033 / 0.155 / 0.054 (275 calls) | 0.047 / 0.133 / 0.070 (275 calls) | 8941 (+1.51%) |
| r16 | 155.3 | 236 | 6.4 | 18 | 9.239e-24 → 9.239e-25 | 1e-16 → 1e-17 | secant+step-cap | 0.030 / 0.121 / 0.048 (236 calls) | 0.047 / 0.112 / 0.066 (236 calls) | 8941 (+1.52%) |
| r17 | 120.7 | 190 | 4.8 | 10 | 9.239e-25 → 9.239e-26 | 1e-17 → 1e-18 | secant+step-cap | 0.021 / 0.069 / 0.032 (190 calls) | 0.042 / 0.082 / 0.056 (190 calls) | 8943 (+1.54%) |
| r18 | 93.4 | 142 | 3.8 | 9 | 9.239e-26 → 9.239e-27 | 1e-18 → 1e-19 | secant+step-cap | 0.021 / 0.052 / 0.030 (142 calls) | 0.049 / 0.071 / 0.058 (142 calls) | 8944 (+1.55%) |
| r19 | 73.2 | 117 | 3.0 | 7 | 9.239e-27 → 2.023e-27 | 1e-19 → 2.19e-20 | secant | 0.017 / 0.034 / 0.023 (117 calls) | 0.043 / 0.051 / 0.047 (117 calls) | 8945 (+1.56%) |
| r20 | 62.9 | 97 | 2.3 | 6 | 2.023e-27 → 2.023e-27 | 2.19e-20 → 2.19e-20 | deadband | 0.010 / 0.017 / 0.013 (97 calls) | 0.041 / 0.041 / 0.041 (97 calls) | 8945 (+1.57%) |

state: stopped after iter 20: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12670936", "decode": "12670935", "holdout_count": "12672032", "holdout_decode": "12672031", "holdout_validate": "12672033", "next": "12670937"}, "1": {"count": "12672024", "decode": "12672023", "next": "12672025"}, "10": {"count": "12679435", "decode": "12679434", "next": "12679436"}, "11": {"count": "12679969", "decode": "12679968", "next": "12679970"}, "12": {"count": "12680308", "decode": "12680307", "next": "12680309"}, "13": {"count": "12680650", "decode": "12680649", "next": "12680651"}, "14": {"count": "12681028", "decode": "12681027", "next": "12681029"}, "15": {"count": "12682096", "decode": "12682095", "next": "12682097"}, "16": {"count": "12682400", "decode": "12682399", "next": "12682401"}, "17": {"count": "12682735", "decode": "12682734", "next": "12682736"}, "18": {"count": "12682991", "decode": "12682990", "next": "12682992"}, "19": {"count": "12683190", "decode": "12683189", "next": "12683191"}, "2": {"count": "12673467", "decode": "12673466", "next": "12673468"}, "20": {"chereji": "12683346", "count": "12683272", "decode": "12683271", "holdout_count": "12683344", "holdout_decode": "12683343", "holdout_validate": "12683345", "next": "12683273", "summary": "12683347"}, "3": {"count": "12675893", "decode": "12675892", "next": "12675894"}, "4": {"count": "12676434", "decode": "12676433", "next": "12676435"}, "5": {"count": "12676938", "decode": "12676937", "next": "12676939"}, "6": {"count": "12677759", "decode": "12677758", "next": "12677760"}, "7": {"count": "12678374", "decode": "12678373", "next": "12678375"}, "8": {"count": "12678773", "decode": "12678772", "next": "12678774"}, "9": {"count": "12679119", "decode": "12679118", "next": "12679120"}}`

## sd72 — sequence only, 7/2 23 bp (vs sp72)

src trainDir `robocop_train_wide_decoy23`, driver `run_split_revfix_seqonly_maskoff_decoy72.py`, cap basis `robocop_train_fiberonly`, tied `{"Zz_decoy_abf1": "ABF1 x1"}`, max_rounds 50, deadband 1.1x

### ABF1 P / R / F1 — tuning chrXIV+chrII (sd72 r00 / final; sp72 r00 / final)

| ref | sd72 r00 | sd72 final | sp72 r00 | sp72 final |
|---|---|---|---|---|
| MacIsaac | – / 0.000 / – (0 calls) | 0.178 / 0.224 / 0.198 (73 calls) | – / 0.000 / – (0 calls) | 0.176 / 0.224 / 0.197 (74 calls) |
| Rossi _CX | – / 0.000 / – (0 calls) | 0.301 / 0.224 / 0.257 (73 calls) | – / 0.000 / – (0 calls) | 0.297 / 0.224 / 0.256 (74 calls) |

### ABF1 P / R / F1 — holdout chrIV (sd72 r00 / final; sp72 r00 / final)

| ref | sd72 r00 | sd72 final | sp72 r00 | sp72 final |
|---|---|---|---|---|
| MacIsaac | – / 0.000 / – (0 calls) | 0.162 / 0.273 / 0.203 (74 calls) | – / 0.000 / – (0 calls) | 0.162 / 0.273 / 0.203 (74 calls) |
| Rossi _CX | – / 0.000 / – (0 calls) | 0.338 / 0.243 / 0.282 (74 calls) | – / 0.000 / – (0 calls) | 0.338 / 0.243 / 0.282 (74 calls) |

### Per round (sd72; T = ABF1 58)

| round | ABF1 E | ABF1 calls | decoy occ | decoy calls | w_decoy = w_ABF1 (→ next) | λ_ABF1 → next | step | ABF1 MacIsaac P/R/F1 | ABF1 Rossi P/R/F1 | nucleosome copies |
|---|---|---|---|---|---|---|---|---|---|---|
| r00 | 0.0 | 0 | 0.0 | 0 | 9.239e-08 → 9.239e-07 | 1 → 10 | no-slope | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9437 (+0.00%) |
| r01 | 0.4 | 1 | 0.0 | 0 | 9.239e-07 → 9.239e-06 | 10 → 100 | no-slope | 1.000 / 0.017 / 0.034 (1 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 9437 (-0.00%) |
| r02 | 1.5 | 2 | 0.0 | 0 | 9.239e-06 → 9.239e-05 | 100 → 1000 | secant+step-cap | 0.500 / 0.017 / 0.033 (2 calls) | 0.500 / 0.010 / 0.020 (2 calls) | 9436 (-0.01%) |
| r03 | 6.1 | 11 | 0.0 | 0 | 9.239e-05 → 0.0009239 | 1000 → 1e+04 | secant+step-cap | 0.182 / 0.034 / 0.058 (11 calls) | 0.455 / 0.051 / 0.092 (11 calls) | 9434 (-0.04%) |
| r04 | 28.9 | 60 | 13.4 | 0 | 0.0009239 → 0.001325 | 1e+04 → 1.434e+04 | secant+rho-cap | 0.200 / 0.207 / 0.203 (60 calls) | 0.350 / 0.214 / 0.266 (60 calls) | 9412 (-0.26%) |
| r05 | 36.2 | 73 | 33.4 | 0 | 0.001325 → 0.001325 | 1.434e+04 → 1.434e+04 | secant+rho-cap | 0.178 / 0.224 / 0.198 (73 calls) | 0.301 / 0.224 / 0.257 (73 calls) | 9403 (-0.36%) |

state: stopped after iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12670939", "decode": "12670938", "holdout_count": "12672063", "holdout_decode": "12672062", "holdout_validate": "12672064", "next": "12670940"}, "1": {"count": "12672054", "decode": "12672053", "next": "12672055"}, "2": {"count": "12675224", "decode": "12675223", "next": "12675225"}, "3": {"count": "12675950", "decode": "12675949", "next": "12675951"}, "4": {"count": "12676500", "decode": "12676499", "next": "12676501"}, "5": {"chereji": "12677849", "count": "12676971", "decode": "12676970", "holdout_count": "12677847", "holdout_decode": "12677846", "holdout_validate": "12677848", "next": "12676972", "summary": "12677850"}}`

