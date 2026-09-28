# Widened-ABF1 tuner-v2 campaigns: 7/2, ±20, ±100 × both / fiber / sequence

_generated 2026-09-23 18:32 by `analysis/wide_summary.py` (widened campaigns from `conc_tuning`); campaigns still running are marked **INCOMPLETE** — re-run `python wide_summary.py` to refresh._

### Rule 7 — what differs from ba01 / fa01 / sa01

**Under test:** ABF1's width. Each width brings its own trainDir (built from a widened MEME whose pad columns are the flank composition of the 341 Rossi ABF1 sites; the 14 bp core rows are the shipped ones verbatim) and a widened ABF1 Fiber-seq p-vector sliced from the Rossi refit (pm50 for 7/2 and ±20, pm200 for ±100).

**Also different, all stated:** max_rounds 50 (baselines 25); bracket expiry on (bracket_max_age 3; the baselines never bisected); rpy2 imported lazily in the new pkgvar trees (no numeric effect); the per-bp cap is judged on the 14 bp **core** (cap_delta 9.571, numerically the baselines' cap, but not the widened motif's own ρ).

**Identical:** untuned start (λ = 1 on the widened model's own Kd), no φ, `unknown` hard-masked (w 1e-3 gauge), w_nuc 35, no nucleosome hold, step cap 1 decade, deadband 1.1×, β seed, MacIsaac target 58, tune chrXIV+chrII, holdout chrIV.

**Caveats.** Circularity: sequence and fiber pads are estimated at Rossi sites, so Rossi scoring is fully circular and MacIsaac partly (77.5% of MacIsaac ABF1 sites lie inside Rossi, per the 2026-09-19 plan). A call is the centre of a posterior run ≥ 0.10, so ±100 (214 bp) plateaus can merge nearby sites: compare calls with E.

## Campaigns (baselines first)

| run | layers | ABF1 pad | block bp | baseline | state | rounds | E / T (final) | λ (final round) | tune MacIsaac P/R/F1 (final) | tune Rossi _CX P/R/F1 (final) | holdout chrIV MacIsaac P/R/F1 r00 | holdout chrIV MacIsaac P/R/F1 final | nucleosome copies chrXIV+chrII (final) | Chereji +1/-1 recall chrXIV (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ba01 | both layers | none (14 bp core) | 14 | — | iter 8: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r08 | 63.7 / 58 = 1.1 | ABF1 1.803e-08 | 0.073 / 0.138 / 0.095 (110 calls) | 0.091 / 0.102 / 0.096 (110 calls) | 0.012 / 0.432 / 0.023 (1580 calls) | 0.026 / 0.091 / 0.041 (153 calls) | 8950 (+0.24% vs r0) | 0.802 (n_ref 626) |
| fa01 | fiber only | none (14 bp core) | 14 | — | iter 16: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r16 | 61.3 / 58 = 1.06 | ABF1 2.419e-16 | 0.019 / 0.034 / 0.024 (106 calls) | 0.038 / 0.041 / 0.039 (106 calls) | 0.004 / 0.455 / 0.008 (4929 calls) | 0.014 / 0.045 / 0.021 (146 calls) | 8948 (+1.46% vs r0) | 0.804 (n_ref 626) |
| sa01 | sequence only | none (14 bp core) | 14 | — | iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r05 | 54.4 / 58 = 0.938 | ABF1 3648 | 0.162 / 0.276 / 0.204 (99 calls) | 0.253 / 0.255 / 0.254 (99 calls) | – / 0.000 / – (0 calls) | 0.124 / 0.295 / 0.174 (105 calls) | 9433 (-0.12% vs r0) | 0.117 (n_ref 626) |
| bp72 | both layers | 7/2 | 23 | ba01 | iter 13: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r13 | 60.4 / 58 = 1.04 | ABF1 2.438e-13 | 0.092 / 0.155 / 0.115 (98 calls) | 0.112 / 0.112 / 0.112 (98 calls) | 0.016 / 0.523 / 0.030 (1470 calls) | 0.036 / 0.091 / 0.051 (112 calls) | 8948 (+0.43% vs r0) | 0.808 (n_ref 626) |
| fp72 | fiber only | 7/2 | 23 | fa01 | iter 20: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r20 | 63.3 / 58 = 1.09 | ABF1 1.82e-20 | 0.010 / 0.017 / 0.013 (98 calls) | 0.041 / 0.041 / 0.041 (98 calls) | 0.007 / 0.545 / 0.014 (3445 calls) | 0.016 / 0.045 / 0.024 (125 calls) | 8946 (+1.51% vs r0) | 0.804 (n_ref 626) |
| sp72 | sequence only | 7/2 | 23 | sa01 | iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r05 | 36.1 / 58 = 0.622 | ABF1 1.434e+04 | 0.176 / 0.224 / 0.197 (74 calls) | 0.297 / 0.224 / 0.256 (74 calls) | – / 0.000 / – (0 calls) | 0.162 / 0.273 / 0.203 (74 calls) | 9427 (-0.10% vs r0) | 0.123 (n_ref 626) |
| bp20 | both layers | ±20 | 54 | ba01 | iter 49: reached the 50-round limit [deadband 1.1x] | r00–r49 | 141.2 / 58 = 2.43 | ABF1 1e-49 | 0.111 / 0.293 / 0.161 (153 calls) | 0.131 / 0.204 / 0.159 (153 calls) | 0.033 / 0.795 / 0.063 (1060 calls) | 0.098 / 0.318 / 0.150 (143 calls) | 8949 (+0.04% vs r0) | 0.802 (n_ref 626) |
| fp20 | fiber only | ±20 | 54 | fa01 | iter 49: reached the 50-round limit [deadband 1.1x] | r00–r49 | 229.2 / 58 = 3.95 | ABF1 1e-49 | 0.075 / 0.310 / 0.121 (240 calls) | 0.083 / 0.204 / 0.118 (240 calls) | 0.030 / 0.818 / 0.058 (1203 calls) | 0.072 / 0.364 / 0.121 (221 calls) | 8948 (+0.05% vs r0) | 0.805 (n_ref 626) |
| sp20 | sequence only | ±20 | 54 | sa01 | iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r05 | 6.3 / 58 = 0.109 | ABF1 1.434e+04 | 0.250 / 0.069 / 0.108 (16 calls) | 0.438 / 0.071 / 0.123 (16 calls) | – / 0.000 / – (0 calls) | 0.182 / 0.045 / 0.073 (11 calls) | 9436 (-0.03% vs r0) | 0.118 (n_ref 626) |
| bp100 | both layers | ±100 | 214 | ba01 | iter 49: reached the 50-round limit [deadband 1.1x] | r00–r49 | 171.7 / 58 = 2.96 | ABF1 1.279e-49 | 0.045 / 0.138 / 0.068 (179 calls) | 0.073 / 0.133 / 0.094 (179 calls) | 0.042 / 0.227 / 0.071 (237 calls) | 0.042 / 0.159 / 0.067 (165 calls) | 8923 (-0.06% vs r0) | 0.791 (n_ref 626) |
| fp100 | fiber only | ±100 | 214 | fa01 | iter 49: reached the 50-round limit [deadband 1.1x] | r00–r49 | 182.0 / 58 = 3.14 | ABF1 1.211e-49 | 0.043 / 0.138 / 0.065 (187 calls) | 0.070 / 0.133 / 0.091 (187 calls) | 0.040 / 0.227 / 0.068 (250 calls) | 0.040 / 0.159 / 0.064 (175 calls) | 8921 (-0.06% vs r0) | 0.789 (n_ref 626) |
| sp100 | sequence only | ±100 | 214 | sa01 | iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r05 | 0.0 / 58 = 0 | ABF1 1.434e+04 | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9445 (+0.00% vs r0) | 0.099 (n_ref 626) |

## bp72 — both layers, ABF1 7/2 (23 bp; vs ba01)

src trainDir `robocop_train_widememe`, driver `run_split_revfix_seq_maskoff_abf1w72.py`, cap basis `robocop_train_fiberonly`, max_rounds 50, deadband 1.1x

### P / R / F1 — tuning chrXIV+chrII (bp72 r00 / final; ba01 r00 / final)

| ref | bp72 r00 | bp72 final | ba01 r00 | ba01 final |
|---|---|---|---|---|
| MacIsaac | 0.024 / 0.586 / 0.045 (1442 calls) | 0.092 / 0.155 / 0.115 (98 calls) | 0.021 / 0.517 / 0.041 (1399 calls) | 0.073 / 0.138 / 0.095 (110 calls) |
| Rossi _CX | 0.033 / 0.490 / 0.062 (1442 calls) | 0.112 / 0.112 / 0.112 (98 calls) | 0.030 / 0.429 / 0.056 (1399 calls) | 0.091 / 0.102 / 0.096 (110 calls) |

### P / R / F1 — holdout chrIV (bp72 r00 / final; ba01 r00 / final)

| ref | bp72 r00 | bp72 final | ba01 r00 | ba01 final |
|---|---|---|---|---|
| MacIsaac | 0.016 / 0.523 / 0.030 (1470 calls) | 0.036 / 0.091 / 0.051 (112 calls) | 0.012 / 0.432 / 0.023 (1580 calls) | 0.026 / 0.091 / 0.041 (153 calls) |
| Rossi _CX | 0.025 / 0.359 / 0.047 (1470 calls) | 0.054 / 0.058 / 0.056 (112 calls) | 0.021 / 0.320 / 0.039 (1580 calls) | 0.039 / 0.058 / 0.047 (153 calls) |

### Per round (bp72; T = ABF1 58)

| round | E | calls (MacIsaac rows) | λ → λ next | step | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | unknown occ (masked: expect 0) |
|---|---|---|---|---|---|---|---|---|
| r00 | 1115.0 | 1442 | 1 → 0.1 | secant+step-cap | 0.024 / 0.586 / 0.045 (1442 calls) | 0.033 / 0.490 / 0.062 (1442 calls) | 8909 (+0.00%) | 0 |
| r01 | 911.5 | 1215 | 0.1 → 0.01 | secant+step-cap | 0.026 / 0.552 / 0.050 (1215 calls) | 0.037 / 0.459 / 0.069 (1215 calls) | 8919 (+0.11%) | 0 |
| r02 | 740.5 | 1001 | 0.01 → 0.001 | secant+step-cap | 0.029 / 0.500 / 0.055 (1001 calls) | 0.041 / 0.418 / 0.075 (1001 calls) | 8927 (+0.20%) | 0 |
| r03 | 598.0 | 837 | 0.001 → 0.0001 | secant+step-cap | 0.030 / 0.431 / 0.056 (837 calls) | 0.044 / 0.378 / 0.079 (837 calls) | 8931 (+0.25%) | 0 |
| r04 | 479.4 | 672 | 0.0001 → 1e-05 | secant+step-cap | 0.036 / 0.414 / 0.066 (672 calls) | 0.051 / 0.347 / 0.088 (672 calls) | 8935 (+0.28%) | 0 |
| r05 | 379.5 | 543 | 1e-05 → 1e-06 | secant+step-cap | 0.041 / 0.379 / 0.073 (543 calls) | 0.057 / 0.316 / 0.097 (543 calls) | 8938 (+0.32%) | 0 |
| r06 | 302.2 | 437 | 1e-06 → 1e-07 | secant+step-cap | 0.048 / 0.362 / 0.085 (437 calls) | 0.069 / 0.306 / 0.112 (437 calls) | 8940 (+0.35%) | 0 |
| r07 | 238.6 | 348 | 1e-07 → 1e-08 | secant+step-cap | 0.055 / 0.328 / 0.094 (348 calls) | 0.083 / 0.296 / 0.130 (348 calls) | 8942 (+0.37%) | 0 |
| r08 | 189.5 | 279 | 1e-08 → 1e-09 | secant+step-cap | 0.061 / 0.293 / 0.101 (279 calls) | 0.086 / 0.245 / 0.127 (279 calls) | 8943 (+0.37%) | 0 |
| r09 | 149.4 | 224 | 1e-09 → 1e-10 | secant+step-cap | 0.062 / 0.241 / 0.099 (224 calls) | 0.085 / 0.194 / 0.118 (224 calls) | 8944 (+0.38%) | 0 |
| r10 | 118.1 | 170 | 1e-10 → 1e-11 | secant+step-cap | 0.076 / 0.224 / 0.114 (170 calls) | 0.100 / 0.173 / 0.127 (170 calls) | 8945 (+0.40%) | 0 |
| r11 | 94.1 | 142 | 1e-11 → 1e-12 | secant+step-cap | 0.077 / 0.190 / 0.110 (142 calls) | 0.106 / 0.153 / 0.125 (142 calls) | 8946 (+0.41%) | 0 |
| r12 | 72.2 | 119 | 1e-12 → 2.438e-13 | secant | 0.092 / 0.190 / 0.124 (119 calls) | 0.118 / 0.143 / 0.129 (119 calls) | 8947 (+0.43%) | 0 |
| r13 | 60.4 | 98 | 2.438e-13 → 2.438e-13 | deadband | 0.092 / 0.155 / 0.115 (98 calls) | 0.112 / 0.112 / 0.112 (98 calls) | 8948 (+0.43%) | 0 |

state: stopped after iter 13: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12644146", "decode": "12644145", "holdout_count": "12644536", "holdout_decode": "12644535", "holdout_validate": "12644537", "next": "12644147"}, "1": {"count": "12644507", "decode": "12644506", "next": "12644508"}, "10": {"count": "12649543", "decode": "12649542", "next": "12649544"}, "11": {"count": "12649915", "decode": "12649914", "next": "12649916"}, "12": {"count": "12650396", "decode": "12650395", "next": "12650397"}, "13": {"chereji": "12651368", "count": "12650889", "decode": "12650888", "holdout_count": "12651366", "holdout_decode": "12651365", "holdout_validate": "12651367", "next": "12650890", "summary": "12651369"}, "2": {"count": "12645131", "decode": "12645130", "next": "12645132"}, "3": {"count": "12646162", "decode": "12646161", "next": "12646163"}, "4": {"count": "12646695", "decode": "12646694", "next": "12646696"}, "5": {"count": "12647165", "decode": "12647164", "next": "12647166"}, "6": {"count": "12647574", "decode": "12647573", "next": "12647575"}, "7": {"count": "12647930", "decode": "12647929", "next": "12647931"}, "8": {"count": "12648324", "decode": "12648323", "next": "12648325"}, "9": {"count": "12648591", "decode": "12648590", "next": "12648592"}}`

## fp72 — fiber only, ABF1 7/2 (23 bp; vs fa01)

src trainDir `robocop_train_widememe`, driver `run_split_revfix_fiber_maskoff_abf1w72.py`, cap basis `robocop_train_fiberonly`, max_rounds 50, deadband 1.1x

### P / R / F1 — tuning chrXIV+chrII (fp72 r00 / final; fa01 r00 / final)

| ref | fp72 r00 | fp72 final | fa01 r00 | fa01 final |
|---|---|---|---|---|
| MacIsaac | 0.010 / 0.586 / 0.019 (3454 calls) | 0.010 / 0.017 / 0.013 (98 calls) | 0.007 / 0.569 / 0.013 (4841 calls) | 0.019 / 0.034 / 0.024 (106 calls) |
| Rossi _CX | 0.014 / 0.490 / 0.027 (3454 calls) | 0.041 / 0.041 / 0.041 (98 calls) | 0.009 / 0.449 / 0.018 (4841 calls) | 0.038 / 0.041 / 0.039 (106 calls) |

### P / R / F1 — holdout chrIV (fp72 r00 / final; fa01 r00 / final)

| ref | fp72 r00 | fp72 final | fa01 r00 | fa01 final |
|---|---|---|---|---|
| MacIsaac | 0.007 / 0.545 / 0.014 (3445 calls) | 0.016 / 0.045 / 0.024 (125 calls) | 0.004 / 0.455 / 0.008 (4929 calls) | 0.014 / 0.045 / 0.021 (146 calls) |
| Rossi _CX | 0.012 / 0.408 / 0.024 (3445 calls) | 0.016 / 0.019 / 0.018 (125 calls) | 0.008 / 0.369 / 0.015 (4929 calls) | 0.014 / 0.019 / 0.016 (146 calls) |

### Per round (fp72; T = ABF1 58)

| round | E | calls (MacIsaac rows) | λ → λ next | step | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | unknown occ (masked: expect 0) |
|---|---|---|---|---|---|---|---|---|
| r00 | 3116.2 | 3454 | 1 → 0.1 | secant+step-cap | 0.010 / 0.586 / 0.019 (3454 calls) | 0.014 / 0.490 / 0.027 (3454 calls) | 8812 (+0.00%) | 0 |
| r01 | 2690.1 | 3054 | 0.1 → 0.01 | secant+step-cap | 0.010 / 0.517 / 0.019 (3054 calls) | 0.014 / 0.439 / 0.027 (3054 calls) | 8835 (+0.26%) | 0 |
| r02 | 2326.4 | 2681 | 0.01 → 0.001 | secant+step-cap | 0.011 / 0.500 / 0.021 (2681 calls) | 0.015 / 0.408 / 0.029 (2681 calls) | 8852 (+0.45%) | 0 |
| r03 | 1993.7 | 2366 | 0.001 → 0.0001 | secant+step-cap | 0.010 / 0.414 / 0.020 (2366 calls) | 0.014 / 0.347 / 0.028 (2366 calls) | 8868 (+0.64%) | 0 |
| r04 | 1703.4 | 2054 | 0.0001 → 1e-05 | secant+step-cap | 0.011 / 0.379 / 0.021 (2054 calls) | 0.015 / 0.316 / 0.029 (2054 calls) | 8882 (+0.79%) | 0 |
| r05 | 1447.8 | 1768 | 1e-05 → 1e-06 | secant+step-cap | 0.012 / 0.379 / 0.024 (1768 calls) | 0.018 / 0.316 / 0.033 (1768 calls) | 8893 (+0.92%) | 0 |
| r06 | 1216.2 | 1522 | 1e-06 → 1e-07 | secant+step-cap | 0.013 / 0.345 / 0.025 (1522 calls) | 0.018 / 0.276 / 0.033 (1522 calls) | 8903 (+1.03%) | 0 |
| r07 | 1017.3 | 1292 | 1e-07 → 1e-08 | secant+step-cap | 0.015 / 0.328 / 0.028 (1292 calls) | 0.021 / 0.276 / 0.039 (1292 calls) | 8912 (+1.13%) | 0 |
| r08 | 850.9 | 1093 | 1e-08 → 1e-09 | secant+step-cap | 0.016 / 0.293 / 0.030 (1093 calls) | 0.021 / 0.235 / 0.039 (1093 calls) | 8923 (+1.26%) | 0 |
| r09 | 712.2 | 938 | 1e-09 → 1e-10 | secant+step-cap | 0.016 / 0.259 / 0.030 (938 calls) | 0.021 / 0.204 / 0.039 (938 calls) | 8929 (+1.32%) | 0 |
| r10 | 582.6 | 789 | 1e-10 → 1e-11 | secant+step-cap | 0.018 / 0.241 / 0.033 (789 calls) | 0.024 / 0.194 / 0.043 (789 calls) | 8933 (+1.37%) | 0 |
| r11 | 472.2 | 647 | 1e-11 → 1e-12 | secant+step-cap | 0.022 / 0.241 / 0.040 (647 calls) | 0.028 / 0.184 / 0.048 (647 calls) | 8934 (+1.38%) | 0 |
| r12 | 380.8 | 521 | 1e-12 → 1e-13 | secant+step-cap | 0.027 / 0.241 / 0.048 (521 calls) | 0.035 / 0.184 / 0.058 (521 calls) | 8937 (+1.41%) | 0 |
| r13 | 304.6 | 425 | 1e-13 → 1e-14 | secant+step-cap | 0.028 / 0.207 / 0.050 (425 calls) | 0.038 / 0.163 / 0.061 (425 calls) | 8939 (+1.44%) | 0 |
| r14 | 242.9 | 342 | 1e-14 → 1e-15 | secant+step-cap | 0.032 / 0.190 / 0.055 (342 calls) | 0.044 / 0.153 / 0.068 (342 calls) | 8940 (+1.45%) | 0 |
| r15 | 198.0 | 277 | 1e-15 → 1e-16 | secant+step-cap | 0.032 / 0.155 / 0.054 (277 calls) | 0.047 / 0.133 / 0.069 (277 calls) | 8941 (+1.46%) | 0 |
| r16 | 160.5 | 238 | 1e-16 → 1e-17 | secant+step-cap | 0.029 / 0.121 / 0.047 (238 calls) | 0.046 / 0.112 / 0.065 (238 calls) | 8941 (+1.47%) | 0 |
| r17 | 124.6 | 190 | 1e-17 → 1e-18 | secant+step-cap | 0.021 / 0.069 / 0.032 (190 calls) | 0.042 / 0.082 / 0.056 (190 calls) | 8943 (+1.48%) | 0 |
| r18 | 96.6 | 143 | 1e-18 → 1e-19 | secant+step-cap | 0.021 / 0.052 / 0.030 (143 calls) | 0.049 / 0.071 / 0.058 (143 calls) | 8944 (+1.49%) | 0 |
| r19 | 75.5 | 118 | 1e-19 → 1.82e-20 | secant | 0.017 / 0.034 / 0.023 (118 calls) | 0.042 / 0.051 / 0.046 (118 calls) | 8945 (+1.51%) | 0 |
| r20 | 63.3 | 98 | 1.82e-20 → 1.82e-20 | deadband | 0.010 / 0.017 / 0.013 (98 calls) | 0.041 / 0.041 / 0.041 (98 calls) | 8946 (+1.51%) | 0 |

state: stopped after iter 20: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12644186", "decode": "12644185", "holdout_count": "12644558", "holdout_decode": "12644557", "holdout_validate": "12644559", "next": "12644187"}, "1": {"count": "12644551", "decode": "12644550", "next": "12644552"}, "10": {"count": "12649546", "decode": "12649545", "next": "12649547"}, "11": {"count": "12649971", "decode": "12649970", "next": "12649972"}, "12": {"count": "12650436", "decode": "12650435", "next": "12650437"}, "13": {"count": "12650937", "decode": "12650936", "next": "12650938"}, "14": {"count": "12651459", "decode": "12651458", "next": "12651460"}, "15": {"count": "12651931", "decode": "12651930", "next": "12651932"}, "16": {"count": "12652263", "decode": "12652262", "next": "12652264"}, "17": {"count": "12652612", "decode": "12652611", "next": "12652613"}, "18": {"count": "12652943", "decode": "12652942", "next": "12652944"}, "19": {"count": "12653343", "decode": "12653342", "next": "12653344"}, "2": {"count": "12645290", "decode": "12645289", "next": "12645291"}, "20": {"chereji": "12665233", "count": "12664724", "decode": "12664723", "holdout_count": "12665231", "holdout_decode": "12665230", "holdout_validate": "12665232", "next": "12664725", "summary": "12665234"}, "3": {"count": "12646207", "decode": "12646206", "next": "12646208"}, "4": {"count": "12646701", "decode": "12646700", "next": "12646702"}, "5": {"count": "12647227", "decode": "12647226", "next": "12647228"}, "6": {"count": "12647620", "decode": "12647619", "next": "12647621"}, "7": {"count": "12648042", "decode": "12648041", "next": "12648043"}, "8": {"count": "12648368", "decode": "12648367", "next": "12648369"}, "9": {"count": "12648647", "decode": "12648646", "next": "12648648"}}`

## sp72 — sequence only, ABF1 7/2 (23 bp; vs sa01)

src trainDir `robocop_train_widememe`, driver `run_split_revfix_seqonly_maskoff_abf1w72.py`, cap basis `robocop_train_fiberonly`, max_rounds 50, deadband 1.1x

### P / R / F1 — tuning chrXIV+chrII (sp72 r00 / final; sa01 r00 / final)

| ref | sp72 r00 | sp72 final | sa01 r00 | sa01 final |
|---|---|---|---|---|
| MacIsaac | – / 0.000 / – (0 calls) | 0.176 / 0.224 / 0.197 (74 calls) | 1.000 / 0.017 / 0.034 (1 calls) | 0.162 / 0.276 / 0.204 (99 calls) |
| Rossi _CX | – / 0.000 / – (0 calls) | 0.297 / 0.224 / 0.256 (74 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 0.253 / 0.255 / 0.254 (99 calls) |

### P / R / F1 — holdout chrIV (sp72 r00 / final; sa01 r00 / final)

| ref | sp72 r00 | sp72 final | sa01 r00 | sa01 final |
|---|---|---|---|---|
| MacIsaac | – / 0.000 / – (0 calls) | 0.162 / 0.273 / 0.203 (74 calls) | – / 0.000 / – (0 calls) | 0.124 / 0.295 / 0.174 (105 calls) |
| Rossi _CX | – / 0.000 / – (0 calls) | 0.338 / 0.243 / 0.282 (74 calls) | – / 0.000 / – (0 calls) | 0.257 / 0.262 / 0.260 (105 calls) |

### Per round (sp72; T = ABF1 58)

| round | E | calls (MacIsaac rows) | λ → λ next | step | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | unknown occ (masked: expect 0) |
|---|---|---|---|---|---|---|---|---|
| r00 | 0.0 | 0 | 1 → 10 | no-slope | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9437 (+0.00%) | 0 |
| r01 | 0.4 | 1 | 10 → 100 | no-slope | 1.000 / 0.017 / 0.034 (1 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 9437 (-0.00%) | 0 |
| r02 | 1.5 | 2 | 100 → 1000 | secant+step-cap | 0.500 / 0.017 / 0.033 (2 calls) | 0.500 / 0.010 / 0.020 (2 calls) | 9437 (-0.00%) | 0 |
| r03 | 6.1 | 11 | 1000 → 1e+04 | secant+step-cap | 0.182 / 0.034 / 0.058 (11 calls) | 0.455 / 0.051 / 0.092 (11 calls) | 9435 (-0.02%) | 0 |
| r04 | 28.8 | 60 | 1e+04 → 1.434e+04 | secant+rho-cap | 0.200 / 0.207 / 0.203 (60 calls) | 0.350 / 0.214 / 0.266 (60 calls) | 9429 (-0.08%) | 0 |
| r05 | 36.1 | 74 | 1.434e+04 → 1.434e+04 | secant+rho-cap | 0.176 / 0.224 / 0.197 (74 calls) | 0.297 / 0.224 / 0.256 (74 calls) | 9427 (-0.10%) | 0 |

state: stopped after iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12644191", "decode": "12644190", "holdout_count": "12644632", "holdout_decode": "12644631", "holdout_validate": "12644633", "next": "12644192"}, "1": {"count": "12644626", "decode": "12644625", "next": "12644627"}, "2": {"count": "12645380", "decode": "12645379", "next": "12645381"}, "3": {"count": "12646224", "decode": "12646223", "next": "12646225"}, "4": {"count": "12646755", "decode": "12646754", "next": "12646756"}, "5": {"chereji": "12647689", "count": "12647274", "decode": "12647273", "holdout_count": "12647687", "holdout_decode": "12647686", "holdout_validate": "12647688", "next": "12647275", "summary": "12647690"}}`

## bp20 — both layers, ABF1 ±20 (54 bp; vs ba01)

src trainDir `robocop_train_abf1_pm20`, driver `run_split_revfix_seq_maskoff_abf1w20.py`, cap basis `robocop_train_fiberonly`, max_rounds 50, deadband 1.1x

### P / R / F1 — tuning chrXIV+chrII (bp20 r00 / final; ba01 r00 / final)

| ref | bp20 r00 | bp20 final | ba01 r00 | ba01 final |
|---|---|---|---|---|
| MacIsaac | 0.042 / 0.810 / 0.080 (1111 calls) | 0.111 / 0.293 / 0.161 (153 calls) | 0.021 / 0.517 / 0.041 (1399 calls) | 0.073 / 0.138 / 0.095 (110 calls) |
| Rossi _CX | 0.058 / 0.653 / 0.106 (1111 calls) | 0.131 / 0.204 / 0.159 (153 calls) | 0.030 / 0.429 / 0.056 (1399 calls) | 0.091 / 0.102 / 0.096 (110 calls) |

### P / R / F1 — holdout chrIV (bp20 r00 / final; ba01 r00 / final)

| ref | bp20 r00 | bp20 final | ba01 r00 | ba01 final |
|---|---|---|---|---|
| MacIsaac | 0.033 / 0.795 / 0.063 (1060 calls) | 0.098 / 0.318 / 0.150 (143 calls) | 0.012 / 0.432 / 0.023 (1580 calls) | 0.026 / 0.091 / 0.041 (153 calls) |
| Rossi _CX | 0.061 / 0.631 / 0.112 (1060 calls) | 0.161 / 0.223 / 0.187 (143 calls) | 0.021 / 0.320 / 0.039 (1580 calls) | 0.039 / 0.058 / 0.047 (153 calls) |

### Per round (bp20; T = ABF1 58)

| round | E | calls (MacIsaac rows) | λ → λ next | step | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | unknown occ (masked: expect 0) |
|---|---|---|---|---|---|---|---|---|
| r00 | 1151.5 | 1111 | 1 → 0.1 | secant+step-cap | 0.042 / 0.810 / 0.080 (1111 calls) | 0.058 / 0.653 / 0.106 (1111 calls) | 8946 (+0.00%) | 0 |
| r01 | 1123.2 | 1092 | 0.1 → 0.01 | secant+step-cap | 0.043 / 0.810 / 0.082 (1092 calls) | 0.059 / 0.653 / 0.108 (1092 calls) | 8946 (+0.00%) | 0 |
| r02 | 1092.7 | 1065 | 0.01 → 0.001 | secant+step-cap | 0.041 / 0.759 / 0.078 (1065 calls) | 0.056 / 0.612 / 0.103 (1065 calls) | 8947 (+0.00%) | 0 |
| r03 | 1063.8 | 1029 | 0.001 → 0.0001 | secant+step-cap | 0.042 / 0.741 / 0.079 (1029 calls) | 0.058 / 0.612 / 0.106 (1029 calls) | 8947 (+0.01%) | 0 |
| r04 | 1032.0 | 999 | 0.0001 → 1e-05 | secant+step-cap | 0.043 / 0.741 / 0.081 (999 calls) | 0.059 / 0.602 / 0.108 (999 calls) | 8947 (+0.01%) | 0 |
| r05 | 1000.9 | 984 | 1e-05 → 1e-06 | secant+step-cap | 0.045 / 0.759 / 0.084 (984 calls) | 0.059 / 0.592 / 0.107 (984 calls) | 8947 (+0.01%) | 0 |
| r06 | 970.2 | 959 | 1e-06 → 1e-07 | secant+step-cap | 0.046 / 0.759 / 0.087 (959 calls) | 0.060 / 0.592 / 0.110 (959 calls) | 8947 (+0.01%) | 0 |
| r07 | 943.2 | 937 | 1e-07 → 1e-08 | secant+step-cap | 0.047 / 0.759 / 0.088 (937 calls) | 0.061 / 0.582 / 0.110 (937 calls) | 8948 (+0.02%) | 0 |
| r08 | 921.8 | 910 | 1e-08 → 1e-09 | secant+step-cap | 0.047 / 0.741 / 0.089 (910 calls) | 0.060 / 0.561 / 0.109 (910 calls) | 8948 (+0.02%) | 0 |
| r09 | 900.0 | 890 | 1e-09 → 1e-10 | secant+step-cap | 0.048 / 0.741 / 0.091 (890 calls) | 0.062 / 0.561 / 0.111 (890 calls) | 8948 (+0.02%) | 0 |
| r10 | 879.3 | 871 | 1e-10 → 1e-11 | secant+step-cap | 0.049 / 0.741 / 0.093 (871 calls) | 0.063 / 0.561 / 0.114 (871 calls) | 8949 (+0.03%) | 0 |
| r11 | 857.1 | 852 | 1e-11 → 1e-12 | secant+step-cap | 0.050 / 0.741 / 0.095 (852 calls) | 0.063 / 0.551 / 0.114 (852 calls) | 8949 (+0.03%) | 0 |
| r12 | 831.3 | 830 | 1e-12 → 1e-13 | secant+step-cap | 0.052 / 0.741 / 0.097 (830 calls) | 0.065 / 0.551 / 0.116 (830 calls) | 8949 (+0.03%) | 0 |
| r13 | 805.5 | 806 | 1e-13 → 1e-14 | secant+step-cap | 0.053 / 0.741 / 0.100 (806 calls) | 0.067 / 0.551 / 0.119 (806 calls) | 8949 (+0.03%) | 0 |
| r14 | 780.2 | 783 | 1e-14 → 1e-15 | secant+step-cap | 0.055 / 0.741 / 0.102 (783 calls) | 0.066 / 0.531 / 0.118 (783 calls) | 8949 (+0.03%) | 0 |
| r15 | 755.4 | 760 | 1e-15 → 1e-16 | secant+step-cap | 0.057 / 0.741 / 0.105 (760 calls) | 0.067 / 0.520 / 0.119 (760 calls) | 8949 (+0.03%) | 0 |
| r16 | 733.4 | 730 | 1e-16 → 1e-17 | secant+step-cap | 0.059 / 0.741 / 0.109 (730 calls) | 0.070 / 0.520 / 0.123 (730 calls) | 8949 (+0.03%) | 0 |
| r17 | 712.3 | 708 | 1e-17 → 1e-18 | secant+step-cap | 0.059 / 0.724 / 0.110 (708 calls) | 0.072 / 0.520 / 0.127 (708 calls) | 8949 (+0.03%) | 0 |
| r18 | 690.8 | 692 | 1e-18 → 1e-19 | secant+step-cap | 0.061 / 0.724 / 0.112 (692 calls) | 0.072 / 0.510 / 0.127 (692 calls) | 8949 (+0.03%) | 0 |
| r19 | 669.4 | 668 | 1e-19 → 1e-20 | secant+step-cap | 0.060 / 0.690 / 0.110 (668 calls) | 0.073 / 0.500 / 0.128 (668 calls) | 8949 (+0.03%) | 0 |
| r20 | 646.3 | 653 | 1e-20 → 1e-21 | secant+step-cap | 0.060 / 0.672 / 0.110 (653 calls) | 0.075 / 0.500 / 0.130 (653 calls) | 8949 (+0.03%) | 0 |
| r21 | 622.0 | 628 | 1e-21 → 1e-22 | secant+step-cap | 0.061 / 0.655 / 0.111 (628 calls) | 0.075 / 0.480 / 0.129 (628 calls) | 8949 (+0.03%) | 0 |
| r22 | 596.6 | 606 | 1e-22 → 1e-23 | secant+step-cap | 0.061 / 0.638 / 0.111 (606 calls) | 0.076 / 0.469 / 0.131 (606 calls) | 8949 (+0.03%) | 0 |
| r23 | 573.1 | 582 | 1e-23 → 1e-24 | secant+step-cap | 0.064 / 0.638 / 0.116 (582 calls) | 0.079 / 0.469 / 0.135 (582 calls) | 8949 (+0.03%) | 0 |
| r24 | 551.7 | 563 | 1e-24 → 1e-25 | secant+step-cap | 0.066 / 0.638 / 0.119 (563 calls) | 0.082 / 0.469 / 0.139 (563 calls) | 8949 (+0.03%) | 0 |
| r25 | 534.3 | 541 | 1e-25 → 1e-26 | secant+step-cap | 0.067 / 0.621 / 0.120 (541 calls) | 0.085 / 0.469 / 0.144 (541 calls) | 8949 (+0.03%) | 0 |
| r26 | 517.6 | 529 | 1e-26 → 1e-27 | secant+step-cap | 0.066 / 0.603 / 0.119 (529 calls) | 0.085 / 0.459 / 0.144 (529 calls) | 8949 (+0.03%) | 0 |
| r27 | 499.0 | 511 | 1e-27 → 1e-28 | secant+step-cap | 0.067 / 0.586 / 0.120 (511 calls) | 0.084 / 0.439 / 0.141 (511 calls) | 8949 (+0.03%) | 0 |
| r28 | 477.3 | 497 | 1e-28 → 1e-29 | secant+step-cap | 0.068 / 0.586 / 0.123 (497 calls) | 0.087 / 0.439 / 0.145 (497 calls) | 8949 (+0.03%) | 0 |
| r29 | 453.3 | 473 | 1e-29 → 1e-30 | secant+step-cap | 0.070 / 0.569 / 0.124 (473 calls) | 0.087 / 0.418 / 0.144 (473 calls) | 8949 (+0.03%) | 0 |
| r30 | 429.8 | 450 | 1e-30 → 1e-31 | secant+step-cap | 0.071 / 0.552 / 0.126 (450 calls) | 0.089 / 0.408 / 0.146 (450 calls) | 8949 (+0.03%) | 0 |
| r31 | 409.0 | 425 | 1e-31 → 1e-32 | secant+step-cap | 0.075 / 0.552 / 0.133 (425 calls) | 0.094 / 0.408 / 0.153 (425 calls) | 8949 (+0.03%) | 0 |
| r32 | 384.7 | 409 | 1e-32 → 1e-33 | secant+step-cap | 0.078 / 0.552 / 0.137 (409 calls) | 0.095 / 0.398 / 0.154 (409 calls) | 8949 (+0.03%) | 0 |
| r33 | 364.9 | 378 | 1e-33 → 1e-34 | secant+step-cap | 0.082 / 0.534 / 0.142 (378 calls) | 0.098 / 0.378 / 0.155 (378 calls) | 8949 (+0.03%) | 0 |
| r34 | 350.8 | 360 | 1e-34 → 1e-35 | secant+step-cap | 0.086 / 0.534 / 0.148 (360 calls) | 0.100 / 0.367 / 0.157 (360 calls) | 8949 (+0.03%) | 0 |
| r35 | 332.6 | 350 | 1e-35 → 1e-36 | secant+step-cap | 0.089 / 0.534 / 0.152 (350 calls) | 0.100 / 0.357 / 0.156 (350 calls) | 8949 (+0.03%) | 0 |
| r36 | 312.7 | 327 | 1e-36 → 1e-37 | secant+step-cap | 0.089 / 0.500 / 0.151 (327 calls) | 0.101 / 0.337 / 0.155 (327 calls) | 8949 (+0.03%) | 0 |
| r37 | 294.8 | 311 | 1e-37 → 1e-38 | secant+step-cap | 0.090 / 0.483 / 0.152 (311 calls) | 0.100 / 0.316 / 0.152 (311 calls) | 8949 (+0.03%) | 0 |
| r38 | 280.7 | 291 | 1e-38 → 1e-39 | secant+step-cap | 0.089 / 0.448 / 0.149 (291 calls) | 0.100 / 0.296 / 0.149 (291 calls) | 8949 (+0.03%) | 0 |
| r39 | 267.4 | 280 | 1e-39 → 1e-40 | secant+step-cap | 0.089 / 0.431 / 0.148 (280 calls) | 0.100 / 0.286 / 0.148 (280 calls) | 8949 (+0.03%) | 0 |
| r40 | 251.8 | 266 | 1e-40 → 1e-41 | secant+step-cap | 0.090 / 0.414 / 0.148 (266 calls) | 0.102 / 0.276 / 0.148 (266 calls) | 8949 (+0.03%) | 0 |
| r41 | 234.1 | 249 | 1e-41 → 1e-42 | secant+step-cap | 0.096 / 0.414 / 0.156 (249 calls) | 0.108 / 0.276 / 0.156 (249 calls) | 8949 (+0.03%) | 0 |
| r42 | 219.6 | 229 | 1e-42 → 1e-43 | secant+step-cap | 0.096 / 0.379 / 0.153 (229 calls) | 0.109 / 0.255 / 0.153 (229 calls) | 8949 (+0.03%) | 0 |
| r43 | 209.1 | 213 | 1e-43 → 1e-44 | secant+step-cap | 0.099 / 0.362 / 0.155 (213 calls) | 0.113 / 0.245 / 0.154 (213 calls) | 8949 (+0.03%) | 0 |
| r44 | 199.6 | 205 | 1e-44 → 1e-45 | secant+step-cap | 0.102 / 0.362 / 0.160 (205 calls) | 0.117 / 0.245 / 0.158 (205 calls) | 8949 (+0.03%) | 0 |
| r45 | 191.6 | 196 | 1e-45 → 1e-46 | secant+step-cap | 0.097 / 0.328 / 0.150 (196 calls) | 0.112 / 0.224 / 0.150 (196 calls) | 8949 (+0.04%) | 0 |
| r46 | 181.0 | 192 | 1e-46 → 1e-47 | secant+step-cap | 0.099 / 0.328 / 0.152 (192 calls) | 0.115 / 0.224 / 0.152 (192 calls) | 8949 (+0.04%) | 0 |
| r47 | 167.5 | 181 | 1e-47 → 1e-48 | secant+step-cap | 0.105 / 0.328 / 0.159 (181 calls) | 0.122 / 0.224 / 0.158 (181 calls) | 8949 (+0.04%) | 0 |
| r48 | 154.3 | 167 | 1e-48 → 1e-49 | secant+step-cap | 0.102 / 0.293 / 0.151 (167 calls) | 0.120 / 0.204 / 0.151 (167 calls) | 8949 (+0.04%) | 0 |
| r49 | 141.2 | 153 | 1e-49 → 1e-50 | secant+step-cap | 0.111 / 0.293 / 0.161 (153 calls) | 0.131 / 0.204 / 0.159 (153 calls) | 8949 (+0.04%) | 0 |

state: stopped after iter 49: reached the 50-round limit [deadband 1.1x]

jobs: `{"0": {"count": "12644196", "decode": "12644195", "holdout_count": "12644676", "holdout_decode": "12644675", "holdout_validate": "12644677", "next": "12644197"}, "1": {"count": "12644673", "decode": "12644672", "next": "12644674"}, "10": {"count": "12649561", "decode": "12649560", "next": "12649562"}, "11": {"count": "12650238", "decode": "12650237", "next": "12650239"}, "12": {"count": "12650651", "decode": "12650650", "next": "12650652"}, "13": {"count": "12651202", "decode": "12651201", "next": "12651203"}, "14": {"count": "12651767", "decode": "12651766", "next": "12651768"}, "15": {"count": "12652155", "decode": "12652154", "next": "12652156"}, "16": {"count": "12652509", "decode": "12652508", "next": "12652510"}, "17": {"count": "12652820", "decode": "12652819", "next": "12652821"}, "18": {"count": "12653206", "decode": "12653205", "next": "12653207"}, "19": {"count": "12653589", "decode": "12653588", "next": "12653590"}, "2": {"count": "12645442", "decode": "12645441", "next": "12645443"}, "20": {"count": "12664736", "decode": "12664735", "next": "12664737"}, "21": {"count": "12665341", "decode": "12665340", "next": "12665342"}, "22": {"count": "12665973", "decode": "12665972", "next": "12665974"}, "23": {"count": "12666465", "decode": "12666464", "next": "12666466"}, "24": {"count": "12666765", "decode": "12666764", "next": "12666766"}, "25": {"count": "12667089", "decode": "12667088", "next": "12667090"}, "26": {"count": "12667309", "decode": "12667308", "next": "12667310"}, "27": {"count": "12668653", "decode": "12668652", "next": "12668654"}, "28": {"count": "12669247", "decode": "12669246", "next": "12669248"}, "29": {"count": "12669478", "decode": "12669477", "next": "12669479"}, "3": {"count": "12646265", "decode": "12646264", "next": "12646266"}, "30": {"count": "12669704", "decode": "12669703", "next": "12669705"}, "31": {"count": "12670019", "decode": "12670018", "next": "12670020"}, "32": {"count": "12670395", "decode": "12670394", "next": "12670396"}, "33": {"count": "12670798", "decode": "12670797", "next": "12670799"}, "34": {"count": "12671366", "decode": "12671365", "next": "12671367"}, "35": {"count": "12672455", "decode": "12672454", "next": "12672456"}, "36": {"count": "12675805", "decode": "12675804", "next": "12675806"}, "37": {"count": "12676169", "decode": "12676168", "next": "12676170"}, "38": {"count": "12676762", "decode": "12676761", "next": "12676763"}, "39": {"count": "12677182", "decode": "12677181", "next": "12677183"}, "4": {"count": "12646814", "decode": "12646813", "next": "12646815"}, "40": {"count": "12678207", "decode": "12678206", "next": "12678208"}, "41": {"count": "12678658", "decode": "12678657", "next": "12678659"}, "42": {"count": "12679001", "decode": "12679000", "next": "12679002"}, "43": {"count": "12679345", "decode": "12679344", "next": "12679346"}, "44": {"count": "12679877", "decode": "12679876", "next": "12679878"}, "45": {"count": "12680190", "decode": "12680189", "next": "12680191"}, "46": {"count": "12680566", "decode": "12680565", "next": "12680567"}, "47": {"count": "12680910", "decode": "12680909", "next": "12680911"}, "48": {"count": "12681417", "decode": "12681416", "next": "12681418"}, "49": {"chereji": "12682626", "count": "12682311", "decode": "12682310", "holdout_count": "12682624", "holdout_decode": "12682623", "holdout_validate": "12682625", "next": "12682312", "summary": "12682627"}, "5": {"count": "12647320", "decode": "12647319", "next": "12647321"}, "6": {"count": "12647760", "decode": "12647759", "next": "12647761"}, "7": {"count": "12648165", "decode": "12648164", "next": "12648166"}, "8": {"count": "12648500", "decode": "12648499", "next": "12648501"}, "9": {"count": "12648786", "decode": "12648785", "next": "12648787"}}`

## fp20 — fiber only, ABF1 ±20 (54 bp; vs fa01)

src trainDir `robocop_train_abf1_pm20`, driver `run_split_revfix_fiber_maskoff_abf1w20.py`, cap basis `robocop_train_fiberonly`, max_rounds 50, deadband 1.1x

### P / R / F1 — tuning chrXIV+chrII (fp20 r00 / final; fa01 r00 / final)

| ref | fp20 r00 | fp20 final | fa01 r00 | fa01 final |
|---|---|---|---|---|
| MacIsaac | 0.038 / 0.828 / 0.073 (1266 calls) | 0.075 / 0.310 / 0.121 (240 calls) | 0.007 / 0.569 / 0.013 (4841 calls) | 0.019 / 0.034 / 0.024 (106 calls) |
| Rossi _CX | 0.051 / 0.653 / 0.094 (1266 calls) | 0.083 / 0.204 / 0.118 (240 calls) | 0.009 / 0.449 / 0.018 (4841 calls) | 0.038 / 0.041 / 0.039 (106 calls) |

### P / R / F1 — holdout chrIV (fp20 r00 / final; fa01 r00 / final)

| ref | fp20 r00 | fp20 final | fa01 r00 | fa01 final |
|---|---|---|---|---|
| MacIsaac | 0.030 / 0.818 / 0.058 (1203 calls) | 0.072 / 0.364 / 0.121 (221 calls) | 0.004 / 0.455 / 0.008 (4929 calls) | 0.014 / 0.045 / 0.021 (146 calls) |
| Rossi _CX | 0.058 / 0.680 / 0.107 (1203 calls) | 0.118 / 0.252 / 0.160 (221 calls) | 0.008 / 0.369 / 0.015 (4929 calls) | 0.014 / 0.019 / 0.016 (146 calls) |

### Per round (fp20; T = ABF1 58)

| round | E | calls (MacIsaac rows) | λ → λ next | step | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | unknown occ (masked: expect 0) |
|---|---|---|---|---|---|---|---|---|
| r00 | 1346.3 | 1266 | 1 → 0.1 | secant+step-cap | 0.038 / 0.828 / 0.073 (1266 calls) | 0.051 / 0.653 / 0.094 (1266 calls) | 8943 (+0.00%) | 0 |
| r01 | 1321.6 | 1253 | 0.1 → 0.01 | secant+step-cap | 0.038 / 0.810 / 0.072 (1253 calls) | 0.050 / 0.643 / 0.093 (1253 calls) | 8943 (+0.00%) | 0 |
| r02 | 1296.9 | 1234 | 0.01 → 0.001 | secant+step-cap | 0.038 / 0.810 / 0.073 (1234 calls) | 0.051 / 0.643 / 0.095 (1234 calls) | 8944 (+0.00%) | 0 |
| r03 | 1271.8 | 1209 | 0.001 → 0.0001 | secant+step-cap | 0.038 / 0.793 / 0.073 (1209 calls) | 0.051 / 0.633 / 0.095 (1209 calls) | 8944 (+0.00%) | 0 |
| r04 | 1247.0 | 1188 | 0.0001 → 1e-05 | secant+step-cap | 0.039 / 0.793 / 0.074 (1188 calls) | 0.052 / 0.633 / 0.096 (1188 calls) | 8944 (+0.01%) | 0 |
| r05 | 1220.4 | 1171 | 1e-05 → 1e-06 | secant+step-cap | 0.038 / 0.776 / 0.073 (1171 calls) | 0.052 / 0.622 / 0.096 (1171 calls) | 8944 (+0.01%) | 0 |
| r06 | 1191.5 | 1151 | 1e-06 → 1e-07 | secant+step-cap | 0.039 / 0.776 / 0.074 (1151 calls) | 0.053 / 0.622 / 0.098 (1151 calls) | 8945 (+0.01%) | 0 |
| r07 | 1162.6 | 1123 | 1e-07 → 1e-08 | secant+step-cap | 0.039 / 0.759 / 0.075 (1123 calls) | 0.053 / 0.612 / 0.098 (1123 calls) | 8945 (+0.01%) | 0 |
| r08 | 1135.0 | 1093 | 1e-08 → 1e-09 | secant+step-cap | 0.040 / 0.759 / 0.076 (1093 calls) | 0.054 / 0.602 / 0.099 (1093 calls) | 8945 (+0.02%) | 0 |
| r09 | 1107.9 | 1074 | 1e-09 → 1e-10 | secant+step-cap | 0.041 / 0.759 / 0.078 (1074 calls) | 0.055 / 0.602 / 0.101 (1074 calls) | 8945 (+0.02%) | 0 |
| r10 | 1079.9 | 1047 | 1e-10 → 1e-11 | secant+step-cap | 0.041 / 0.741 / 0.078 (1047 calls) | 0.054 / 0.582 / 0.100 (1047 calls) | 8945 (+0.02%) | 0 |
| r11 | 1050.4 | 1018 | 1e-11 → 1e-12 | secant+step-cap | 0.041 / 0.724 / 0.078 (1018 calls) | 0.054 / 0.561 / 0.099 (1018 calls) | 8946 (+0.02%) | 0 |
| r12 | 1023.0 | 991 | 1e-12 → 1e-13 | secant+step-cap | 0.042 / 0.724 / 0.080 (991 calls) | 0.054 / 0.551 / 0.099 (991 calls) | 8946 (+0.03%) | 0 |
| r13 | 995.5 | 976 | 1e-13 → 1e-14 | secant+step-cap | 0.043 / 0.724 / 0.081 (976 calls) | 0.054 / 0.541 / 0.099 (976 calls) | 8946 (+0.03%) | 0 |
| r14 | 968.2 | 950 | 1e-14 → 1e-15 | secant+step-cap | 0.044 / 0.724 / 0.083 (950 calls) | 0.056 / 0.541 / 0.101 (950 calls) | 8947 (+0.04%) | 0 |
| r15 | 937.2 | 925 | 1e-15 → 1e-16 | secant+step-cap | 0.044 / 0.707 / 0.083 (925 calls) | 0.056 / 0.531 / 0.102 (925 calls) | 8947 (+0.04%) | 0 |
| r16 | 906.1 | 898 | 1e-16 → 1e-17 | secant+step-cap | 0.046 / 0.707 / 0.086 (898 calls) | 0.057 / 0.520 / 0.102 (898 calls) | 8948 (+0.05%) | 0 |
| r17 | 883.3 | 871 | 1e-17 → 1e-18 | secant+step-cap | 0.048 / 0.724 / 0.090 (871 calls) | 0.059 / 0.520 / 0.105 (871 calls) | 8948 (+0.05%) | 0 |
| r18 | 860.9 | 854 | 1e-18 → 1e-19 | secant+step-cap | 0.049 / 0.724 / 0.092 (854 calls) | 0.059 / 0.510 / 0.105 (854 calls) | 8948 (+0.05%) | 0 |
| r19 | 838.8 | 830 | 1e-19 → 1e-20 | secant+step-cap | 0.051 / 0.724 / 0.095 (830 calls) | 0.059 / 0.500 / 0.106 (830 calls) | 8948 (+0.05%) | 0 |
| r20 | 815.9 | 814 | 1e-20 → 1e-21 | secant+step-cap | 0.052 / 0.724 / 0.096 (814 calls) | 0.059 / 0.490 / 0.105 (814 calls) | 8948 (+0.05%) | 0 |
| r21 | 791.9 | 790 | 1e-21 → 1e-22 | secant+step-cap | 0.053 / 0.724 / 0.099 (790 calls) | 0.061 / 0.490 / 0.108 (790 calls) | 8948 (+0.05%) | 0 |
| r22 | 767.6 | 763 | 1e-22 → 1e-23 | secant+step-cap | 0.055 / 0.724 / 0.102 (763 calls) | 0.063 / 0.490 / 0.111 (763 calls) | 8948 (+0.05%) | 0 |
| r23 | 742.6 | 745 | 1e-23 → 1e-24 | secant+step-cap | 0.052 / 0.672 / 0.097 (745 calls) | 0.062 / 0.469 / 0.109 (745 calls) | 8948 (+0.05%) | 0 |
| r24 | 717.0 | 719 | 1e-24 → 1e-25 | secant+step-cap | 0.054 / 0.672 / 0.100 (719 calls) | 0.063 / 0.459 / 0.110 (719 calls) | 8948 (+0.05%) | 0 |
| r25 | 694.9 | 693 | 1e-25 → 1e-26 | secant+step-cap | 0.056 / 0.672 / 0.104 (693 calls) | 0.065 / 0.459 / 0.114 (693 calls) | 8948 (+0.05%) | 0 |
| r26 | 676.4 | 670 | 1e-26 → 1e-27 | secant+step-cap | 0.057 / 0.655 / 0.104 (670 calls) | 0.066 / 0.449 / 0.115 (670 calls) | 8948 (+0.05%) | 0 |
| r27 | 656.3 | 655 | 1e-27 → 1e-28 | secant+step-cap | 0.058 / 0.655 / 0.107 (655 calls) | 0.067 / 0.449 / 0.117 (655 calls) | 8948 (+0.05%) | 0 |
| r28 | 633.6 | 635 | 1e-28 → 1e-29 | secant+step-cap | 0.060 / 0.655 / 0.110 (635 calls) | 0.069 / 0.449 / 0.120 (635 calls) | 8948 (+0.05%) | 0 |
| r29 | 608.0 | 619 | 1e-29 → 1e-30 | secant+step-cap | 0.057 / 0.603 / 0.103 (619 calls) | 0.068 / 0.429 / 0.117 (619 calls) | 8948 (+0.05%) | 0 |
| r30 | 585.3 | 590 | 1e-30 → 1e-31 | secant+step-cap | 0.058 / 0.586 / 0.105 (590 calls) | 0.071 / 0.429 / 0.122 (590 calls) | 8948 (+0.05%) | 0 |
| r31 | 563.5 | 567 | 1e-31 → 1e-32 | secant+step-cap | 0.060 / 0.586 / 0.109 (567 calls) | 0.074 / 0.429 / 0.126 (567 calls) | 8948 (+0.05%) | 0 |
| r32 | 540.2 | 550 | 1e-32 → 1e-33 | secant+step-cap | 0.060 / 0.569 / 0.109 (550 calls) | 0.075 / 0.418 / 0.127 (550 calls) | 8948 (+0.05%) | 0 |
| r33 | 520.3 | 533 | 1e-33 → 1e-34 | secant+step-cap | 0.064 / 0.586 / 0.115 (533 calls) | 0.077 / 0.418 / 0.130 (533 calls) | 8948 (+0.05%) | 0 |
| r34 | 501.7 | 513 | 1e-34 → 1e-35 | secant+step-cap | 0.062 / 0.552 / 0.112 (513 calls) | 0.076 / 0.398 / 0.128 (513 calls) | 8948 (+0.05%) | 0 |
| r35 | 480.4 | 498 | 1e-35 → 1e-36 | secant+step-cap | 0.064 / 0.552 / 0.115 (498 calls) | 0.078 / 0.398 / 0.131 (498 calls) | 8948 (+0.05%) | 0 |
| r36 | 454.1 | 480 | 1e-36 → 1e-37 | secant+step-cap | 0.065 / 0.534 / 0.115 (480 calls) | 0.079 / 0.388 / 0.131 (480 calls) | 8948 (+0.05%) | 0 |
| r37 | 429.2 | 450 | 1e-37 → 1e-38 | secant+step-cap | 0.069 / 0.534 / 0.122 (450 calls) | 0.080 / 0.367 / 0.131 (450 calls) | 8948 (+0.05%) | 0 |
| r38 | 408.8 | 426 | 1e-38 → 1e-39 | secant+step-cap | 0.063 / 0.466 / 0.112 (426 calls) | 0.075 / 0.327 / 0.122 (426 calls) | 8948 (+0.05%) | 0 |
| r39 | 388.4 | 406 | 1e-39 → 1e-40 | secant+step-cap | 0.064 / 0.448 / 0.112 (406 calls) | 0.074 / 0.306 / 0.119 (406 calls) | 8948 (+0.05%) | 0 |
| r40 | 368.4 | 385 | 1e-40 → 1e-41 | secant+step-cap | 0.065 / 0.431 / 0.113 (385 calls) | 0.075 / 0.296 / 0.120 (385 calls) | 8948 (+0.05%) | 0 |
| r41 | 349.9 | 366 | 1e-41 → 1e-42 | secant+step-cap | 0.068 / 0.431 / 0.118 (366 calls) | 0.079 / 0.296 / 0.125 (366 calls) | 8948 (+0.05%) | 0 |
| r42 | 332.6 | 350 | 1e-42 → 1e-43 | secant+step-cap | 0.066 / 0.397 / 0.113 (350 calls) | 0.077 / 0.276 / 0.121 (350 calls) | 8948 (+0.05%) | 0 |
| r43 | 316.5 | 328 | 1e-43 → 1e-44 | secant+step-cap | 0.067 / 0.379 / 0.114 (328 calls) | 0.076 / 0.255 / 0.117 (328 calls) | 8948 (+0.05%) | 0 |
| r44 | 299.6 | 311 | 1e-44 → 1e-45 | secant+step-cap | 0.068 / 0.362 / 0.114 (311 calls) | 0.077 / 0.245 / 0.117 (311 calls) | 8948 (+0.05%) | 0 |
| r45 | 283.9 | 296 | 1e-45 → 1e-46 | secant+step-cap | 0.064 / 0.328 / 0.107 (296 calls) | 0.074 / 0.224 / 0.112 (296 calls) | 8948 (+0.05%) | 0 |
| r46 | 271.1 | 280 | 1e-46 → 1e-47 | secant+step-cap | 0.064 / 0.310 / 0.107 (280 calls) | 0.075 / 0.214 / 0.111 (280 calls) | 8948 (+0.05%) | 0 |
| r47 | 257.4 | 269 | 1e-47 → 1e-48 | secant+step-cap | 0.067 / 0.310 / 0.110 (269 calls) | 0.078 / 0.214 / 0.114 (269 calls) | 8948 (+0.05%) | 0 |
| r48 | 242.2 | 257 | 1e-48 → 1e-49 | secant+step-cap | 0.070 / 0.310 / 0.114 (257 calls) | 0.078 / 0.204 / 0.113 (257 calls) | 8948 (+0.05%) | 0 |
| r49 | 229.2 | 240 | 1e-49 → 1e-50 | secant+step-cap | 0.075 / 0.310 / 0.121 (240 calls) | 0.083 / 0.204 / 0.118 (240 calls) | 8948 (+0.05%) | 0 |

state: stopped after iter 49: reached the 50-round limit [deadband 1.1x]

jobs: `{"0": {"count": "12644200", "decode": "12644199", "holdout_count": "12644713", "holdout_decode": "12644712", "holdout_validate": "12644714", "next": "12644201"}, "1": {"count": "12644703", "decode": "12644702", "next": "12644704"}, "10": {"count": "12649551", "decode": "12649550", "next": "12649552"}, "11": {"count": "12650046", "decode": "12650045", "next": "12650047"}, "12": {"count": "12650479", "decode": "12650478", "next": "12650480"}, "13": {"count": "12650978", "decode": "12650977", "next": "12650979"}, "14": {"count": "12651511", "decode": "12651510", "next": "12651512"}, "15": {"count": "12651981", "decode": "12651980", "next": "12651982"}, "16": {"count": "12652320", "decode": "12652319", "next": "12652321"}, "17": {"count": "12652655", "decode": "12652654", "next": "12652656"}, "18": {"count": "12653002", "decode": "12653001", "next": "12653003"}, "19": {"count": "12653396", "decode": "12653395", "next": "12653397"}, "2": {"count": "12645555", "decode": "12645554", "next": "12645556"}, "20": {"count": "12664746", "decode": "12664745", "next": "12664747"}, "21": {"count": "12665412", "decode": "12665411", "next": "12665413"}, "22": {"count": "12666221", "decode": "12666220", "next": "12666222"}, "23": {"count": "12666487", "decode": "12666486", "next": "12666488"}, "24": {"count": "12666775", "decode": "12666774", "next": "12666776"}, "25": {"count": "12667132", "decode": "12667131", "next": "12667133"}, "26": {"count": "12667337", "decode": "12667336", "next": "12667338"}, "27": {"count": "12668697", "decode": "12668696", "next": "12668698"}, "28": {"count": "12669289", "decode": "12669288", "next": "12669290"}, "29": {"count": "12669563", "decode": "12669562", "next": "12669564"}, "3": {"count": "12646248", "decode": "12646247", "next": "12646249"}, "30": {"count": "12669855", "decode": "12669854", "next": "12669856"}, "31": {"count": "12670116", "decode": "12670115", "next": "12670117"}, "32": {"count": "12670541", "decode": "12670540", "next": "12670542"}, "33": {"count": "12671127", "decode": "12671126", "next": "12671128"}, "34": {"count": "12672381", "decode": "12672380", "next": "12672382"}, "35": {"count": "12675778", "decode": "12675777", "next": "12675779"}, "36": {"count": "12676111", "decode": "12676110", "next": "12676112"}, "37": {"count": "12676676", "decode": "12676675", "next": "12676677"}, "38": {"count": "12677099", "decode": "12677098", "next": "12677100"}, "39": {"count": "12678056", "decode": "12678055", "next": "12678057"}, "4": {"count": "12646741", "decode": "12646740", "next": "12646742"}, "40": {"count": "12678551", "decode": "12678550", "next": "12678552"}, "41": {"count": "12678896", "decode": "12678895", "next": "12678897"}, "42": {"count": "12679263", "decode": "12679262", "next": "12679264"}, "43": {"count": "12679698", "decode": "12679697", "next": "12679699"}, "44": {"count": "12680041", "decode": "12680040", "next": "12680042"}, "45": {"count": "12680421", "decode": "12680420", "next": "12680422"}, "46": {"count": "12680773", "decode": "12680772", "next": "12680774"}, "47": {"count": "12681262", "decode": "12681261", "next": "12681263"}, "48": {"count": "12682198", "decode": "12682197", "next": "12682199"}, "49": {"chereji": "12682833", "count": "12682517", "decode": "12682516", "holdout_count": "12682831", "holdout_decode": "12682830", "holdout_validate": "12682832", "next": "12682518", "summary": "12682834"}, "5": {"count": "12647246", "decode": "12647245", "next": "12647247"}, "6": {"count": "12647666", "decode": "12647665", "next": "12647667"}, "7": {"count": "12648046", "decode": "12648045", "next": "12648047"}, "8": {"count": "12648455", "decode": "12648454", "next": "12648486"}, "9": {"count": "12648739", "decode": "12648738", "next": "12648740"}}`

## sp20 — sequence only, ABF1 ±20 (54 bp; vs sa01)

src trainDir `robocop_train_abf1_pm20`, driver `run_split_revfix_seqonly_maskoff_abf1w20.py`, cap basis `robocop_train_fiberonly`, max_rounds 50, deadband 1.1x

### P / R / F1 — tuning chrXIV+chrII (sp20 r00 / final; sa01 r00 / final)

| ref | sp20 r00 | sp20 final | sa01 r00 | sa01 final |
|---|---|---|---|---|
| MacIsaac | – / 0.000 / – (0 calls) | 0.250 / 0.069 / 0.108 (16 calls) | 1.000 / 0.017 / 0.034 (1 calls) | 0.162 / 0.276 / 0.204 (99 calls) |
| Rossi _CX | – / 0.000 / – (0 calls) | 0.438 / 0.071 / 0.123 (16 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 0.253 / 0.255 / 0.254 (99 calls) |

### P / R / F1 — holdout chrIV (sp20 r00 / final; sa01 r00 / final)

| ref | sp20 r00 | sp20 final | sa01 r00 | sa01 final |
|---|---|---|---|---|
| MacIsaac | – / 0.000 / – (0 calls) | 0.182 / 0.045 / 0.073 (11 calls) | – / 0.000 / – (0 calls) | 0.124 / 0.295 / 0.174 (105 calls) |
| Rossi _CX | – / 0.000 / – (0 calls) | 0.636 / 0.068 / 0.123 (11 calls) | – / 0.000 / – (0 calls) | 0.257 / 0.262 / 0.260 (105 calls) |

### Per round (sp20; T = ABF1 58)

| round | E | calls (MacIsaac rows) | λ → λ next | step | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | unknown occ (masked: expect 0) |
|---|---|---|---|---|---|---|---|---|
| r00 | 0.0 | 0 | 1 → 10 | no-slope | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9439 (+0.00%) | 0 |
| r01 | 0.0 | 0 | 10 → 100 | no-slope | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9439 (-0.00%) | 0 |
| r02 | 0.2 | 1 | 100 → 1000 | no-slope | 1.000 / 0.017 / 0.034 (1 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 9439 (-0.00%) | 0 |
| r03 | 1.2 | 2 | 1000 → 1e+04 | secant+step-cap | 0.500 / 0.017 / 0.033 (2 calls) | 0.500 / 0.010 / 0.020 (2 calls) | 9438 (-0.01%) | 0 |
| r04 | 5.0 | 10 | 1e+04 → 1.434e+04 | secant+step-cap+rho-cap | 0.200 / 0.034 / 0.059 (10 calls) | 0.500 / 0.051 / 0.093 (10 calls) | 9437 (-0.02%) | 0 |
| r05 | 6.3 | 16 | 1.434e+04 → 1.434e+04 | secant+step-cap+rho-cap | 0.250 / 0.069 / 0.108 (16 calls) | 0.438 / 0.071 / 0.123 (16 calls) | 9436 (-0.03%) | 0 |

state: stopped after iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12644205", "decode": "12644204", "holdout_count": "12644793", "holdout_decode": "12644792", "holdout_validate": "12644794", "next": "12644206"}, "1": {"count": "12644777", "decode": "12644776", "next": "12644778"}, "2": {"count": "12645621", "decode": "12645620", "next": "12645622"}, "3": {"count": "12646413", "decode": "12646412", "next": "12646414"}, "4": {"count": "12646858", "decode": "12646857", "next": "12646859"}, "5": {"chereji": "12647811", "count": "12647400", "decode": "12647399", "holdout_count": "12647809", "holdout_decode": "12647808", "holdout_validate": "12647810", "next": "12647401", "summary": "12647812"}}`

## bp100 — both layers, ABF1 ±100 (214 bp; vs ba01)

src trainDir `robocop_train_abf1_pm100`, driver `run_split_revfix_seq_maskoff_abf1w100.py`, cap basis `robocop_train_fiberonly`, max_rounds 50, deadband 1.1x

### P / R / F1 — tuning chrXIV+chrII (bp100 r00 / final; ba01 r00 / final)

| ref | bp100 r00 | bp100 final | ba01 r00 | ba01 final |
|---|---|---|---|---|
| MacIsaac | 0.053 / 0.241 / 0.087 (263 calls) | 0.045 / 0.138 / 0.068 (179 calls) | 0.021 / 0.517 / 0.041 (1399 calls) | 0.073 / 0.138 / 0.095 (110 calls) |
| Rossi _CX | 0.080 / 0.214 / 0.116 (263 calls) | 0.073 / 0.133 / 0.094 (179 calls) | 0.030 / 0.429 / 0.056 (1399 calls) | 0.091 / 0.102 / 0.096 (110 calls) |

### P / R / F1 — holdout chrIV (bp100 r00 / final; ba01 r00 / final)

| ref | bp100 r00 | bp100 final | ba01 r00 | ba01 final |
|---|---|---|---|---|
| MacIsaac | 0.042 / 0.227 / 0.071 (237 calls) | 0.042 / 0.159 / 0.067 (165 calls) | 0.012 / 0.432 / 0.023 (1580 calls) | 0.026 / 0.091 / 0.041 (153 calls) |
| Rossi _CX | 0.084 / 0.194 / 0.118 (237 calls) | 0.079 / 0.126 / 0.097 (165 calls) | 0.021 / 0.320 / 0.039 (1580 calls) | 0.039 / 0.058 / 0.047 (153 calls) |

### Per round (bp100; T = ABF1 58)

| round | E | calls (MacIsaac rows) | λ → λ next | step | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | unknown occ (masked: expect 0) |
|---|---|---|---|---|---|---|---|---|
| r00 | 255.3 | 263 | 1 → 0.1279 | secant | 0.053 / 0.241 / 0.087 (263 calls) | 0.080 / 0.214 / 0.116 (263 calls) | 8928 (+0.00%) | 0 |
| r01 | 254.7 | 262 | 0.1279 → 0.01279 | secant+step-cap | 0.053 / 0.241 / 0.087 (262 calls) | 0.080 / 0.214 / 0.117 (262 calls) | 8928 (+0.00%) | 0 |
| r02 | 254.0 | 261 | 0.01279 → 0.001279 | secant+step-cap | 0.054 / 0.241 / 0.088 (261 calls) | 0.080 / 0.214 / 0.117 (261 calls) | 8928 (+0.00%) | 0 |
| r03 | 252.8 | 260 | 0.001279 → 0.0001279 | secant+step-cap | 0.054 / 0.241 / 0.088 (260 calls) | 0.077 / 0.204 / 0.112 (260 calls) | 8928 (+0.00%) | 0 |
| r04 | 250.7 | 258 | 0.0001279 → 1.279e-05 | secant+step-cap | 0.054 / 0.241 / 0.089 (258 calls) | 0.078 / 0.204 / 0.112 (258 calls) | 8928 (+0.00%) | 0 |
| r05 | 248.9 | 255 | 1.279e-05 → 1.279e-06 | secant+step-cap | 0.055 / 0.241 / 0.089 (255 calls) | 0.078 / 0.204 / 0.113 (255 calls) | 8928 (+0.00%) | 0 |
| r06 | 248.0 | 254 | 1.279e-06 → 1.279e-07 | secant+step-cap | 0.055 / 0.241 / 0.090 (254 calls) | 0.079 / 0.204 / 0.114 (254 calls) | 8928 (+0.00%) | 0 |
| r07 | 246.5 | 253 | 1.279e-07 → 1.279e-08 | secant+step-cap | 0.055 / 0.241 / 0.090 (253 calls) | 0.079 / 0.204 / 0.114 (253 calls) | 8929 (+0.01%) | 0 |
| r08 | 244.5 | 251 | 1.279e-08 → 1.279e-09 | secant+step-cap | 0.056 / 0.241 / 0.091 (251 calls) | 0.080 / 0.204 / 0.115 (251 calls) | 8929 (+0.01%) | 0 |
| r09 | 243.3 | 251 | 1.279e-09 → 1.279e-10 | secant+step-cap | 0.056 / 0.241 / 0.091 (251 calls) | 0.080 / 0.204 / 0.115 (251 calls) | 8929 (+0.01%) | 0 |
| r10 | 241.5 | 250 | 1.279e-10 → 1.279e-11 | secant+step-cap | 0.056 / 0.241 / 0.091 (250 calls) | 0.080 / 0.204 / 0.115 (250 calls) | 8929 (+0.01%) | 0 |
| r11 | 238.3 | 248 | 1.279e-11 → 1.279e-12 | secant+step-cap | 0.056 / 0.241 / 0.092 (248 calls) | 0.081 / 0.204 / 0.116 (248 calls) | 8930 (+0.02%) | 0 |
| r12 | 234.9 | 243 | 1.279e-12 → 1.279e-13 | secant+step-cap | 0.058 / 0.241 / 0.093 (243 calls) | 0.082 / 0.204 / 0.117 (243 calls) | 8931 (+0.03%) | 0 |
| r13 | 233.0 | 239 | 1.279e-13 → 1.279e-14 | secant+step-cap | 0.059 / 0.241 / 0.094 (239 calls) | 0.084 / 0.204 / 0.119 (239 calls) | 8931 (+0.04%) | 0 |
| r14 | 231.5 | 238 | 1.279e-14 → 1.279e-15 | secant+step-cap | 0.059 / 0.241 / 0.095 (238 calls) | 0.084 / 0.204 / 0.119 (238 calls) | 8932 (+0.05%) | 0 |
| r15 | 230.3 | 236 | 1.279e-15 → 1.279e-16 | secant+step-cap | 0.055 / 0.224 / 0.088 (236 calls) | 0.081 / 0.194 / 0.114 (236 calls) | 8933 (+0.05%) | 0 |
| r16 | 229.7 | 235 | 1.279e-16 → 1.279e-17 | secant+step-cap | 0.055 / 0.224 / 0.089 (235 calls) | 0.081 / 0.194 / 0.114 (235 calls) | 8933 (+0.06%) | 0 |
| r17 | 228.7 | 235 | 1.279e-17 → 1.279e-18 | secant+step-cap | 0.055 / 0.224 / 0.089 (235 calls) | 0.081 / 0.194 / 0.114 (235 calls) | 8933 (+0.06%) | 0 |
| r18 | 227.2 | 234 | 1.279e-18 → 1.279e-19 | secant+step-cap | 0.056 / 0.224 / 0.089 (234 calls) | 0.081 / 0.194 / 0.114 (234 calls) | 8933 (+0.06%) | 0 |
| r19 | 224.7 | 232 | 1.279e-19 → 1.279e-20 | secant+step-cap | 0.052 / 0.207 / 0.083 (232 calls) | 0.078 / 0.184 / 0.109 (232 calls) | 8933 (+0.06%) | 0 |
| r20 | 221.4 | 230 | 1.279e-20 → 1.279e-21 | secant+step-cap | 0.052 / 0.207 / 0.083 (230 calls) | 0.078 / 0.184 / 0.110 (230 calls) | 8933 (+0.06%) | 0 |
| r21 | 219.4 | 225 | 1.279e-21 → 1.279e-22 | secant+step-cap | 0.049 / 0.190 / 0.078 (225 calls) | 0.071 / 0.163 / 0.099 (225 calls) | 8931 (+0.03%) | 0 |
| r22 | 217.3 | 224 | 1.279e-22 → 1.279e-23 | secant+step-cap | 0.049 / 0.190 / 0.078 (224 calls) | 0.071 / 0.163 / 0.099 (224 calls) | 8918 (-0.12%) | 0 |
| r23 | 215.2 | 222 | 1.279e-23 → 1.279e-24 | secant+step-cap | 0.050 / 0.190 / 0.079 (222 calls) | 0.072 / 0.163 / 0.100 (222 calls) | 8917 (-0.12%) | 0 |
| r24 | 213.7 | 219 | 1.279e-24 → 1.279e-25 | secant+step-cap | 0.050 / 0.190 / 0.079 (219 calls) | 0.073 / 0.163 / 0.101 (219 calls) | 8917 (-0.13%) | 0 |
| r25 | 212.5 | 218 | 1.279e-25 → 1.279e-26 | secant+step-cap | 0.046 / 0.172 / 0.072 (218 calls) | 0.069 / 0.153 / 0.095 (218 calls) | 8915 (-0.14%) | 0 |
| r26 | 211.5 | 217 | 1.279e-26 → 1.279e-27 | secant+step-cap | 0.046 / 0.172 / 0.073 (217 calls) | 0.069 / 0.153 / 0.095 (217 calls) | 8915 (-0.14%) | 0 |
| r27 | 209.7 | 215 | 1.279e-27 → 1.279e-28 | secant+step-cap | 0.047 / 0.172 / 0.073 (215 calls) | 0.070 / 0.153 / 0.096 (215 calls) | 8915 (-0.14%) | 0 |
| r28 | 205.9 | 214 | 1.279e-28 → 1.279e-29 | secant+step-cap | 0.047 / 0.172 / 0.074 (214 calls) | 0.070 / 0.153 / 0.096 (214 calls) | 8916 (-0.14%) | 0 |
| r29 | 203.1 | 209 | 1.279e-29 → 1.279e-30 | secant+step-cap | 0.048 / 0.172 / 0.075 (209 calls) | 0.072 / 0.153 / 0.098 (209 calls) | 8916 (-0.13%) | 0 |
| r30 | 201.5 | 206 | 1.279e-30 → 1.279e-31 | secant+step-cap | 0.049 / 0.172 / 0.076 (206 calls) | 0.073 / 0.153 / 0.099 (206 calls) | 8916 (-0.13%) | 0 |
| r31 | 200.1 | 205 | 1.279e-31 → 1.279e-32 | secant+step-cap | 0.049 / 0.172 / 0.076 (205 calls) | 0.073 / 0.153 / 0.099 (205 calls) | 8916 (-0.13%) | 0 |
| r32 | 198.6 | 204 | 1.279e-32 → 1.279e-33 | secant+step-cap | 0.049 / 0.172 / 0.076 (204 calls) | 0.074 / 0.153 / 0.099 (204 calls) | 8917 (-0.13%) | 0 |
| r33 | 197.3 | 203 | 1.279e-33 → 1.279e-34 | secant+step-cap | 0.049 / 0.172 / 0.077 (203 calls) | 0.074 / 0.153 / 0.100 (203 calls) | 8917 (-0.12%) | 0 |
| r34 | 195.7 | 200 | 1.279e-34 → 1.279e-35 | secant+step-cap | 0.050 / 0.172 / 0.078 (200 calls) | 0.075 / 0.153 / 0.101 (200 calls) | 8919 (-0.11%) | 0 |
| r35 | 193.8 | 198 | 1.279e-35 → 1.279e-36 | secant+step-cap | 0.045 / 0.155 / 0.070 (198 calls) | 0.076 / 0.153 / 0.101 (198 calls) | 8920 (-0.09%) | 0 |
| r36 | 192.5 | 197 | 1.279e-36 → 1.279e-37 | secant+step-cap | 0.046 / 0.155 / 0.071 (197 calls) | 0.076 / 0.153 / 0.102 (197 calls) | 8920 (-0.09%) | 0 |
| r37 | 190.3 | 196 | 1.279e-37 → 1.279e-38 | secant+step-cap | 0.046 / 0.155 / 0.071 (196 calls) | 0.071 / 0.143 / 0.095 (196 calls) | 8920 (-0.09%) | 0 |
| r38 | 187.9 | 194 | 1.279e-38 → 1.279e-39 | secant+step-cap | 0.046 / 0.155 / 0.071 (194 calls) | 0.072 / 0.143 / 0.096 (194 calls) | 8921 (-0.08%) | 0 |
| r39 | 186.2 | 192 | 1.279e-39 → 1.279e-40 | secant+step-cap | 0.047 / 0.155 / 0.072 (192 calls) | 0.073 / 0.143 / 0.097 (192 calls) | 8921 (-0.08%) | 0 |
| r40 | 184.9 | 191 | 1.279e-40 → 1.279e-41 | secant+step-cap | 0.047 / 0.155 / 0.072 (191 calls) | 0.073 / 0.143 / 0.097 (191 calls) | 8921 (-0.08%) | 0 |
| r41 | 183.4 | 189 | 1.279e-41 → 1.279e-42 | secant+step-cap | 0.048 / 0.155 / 0.073 (189 calls) | 0.074 / 0.143 / 0.098 (189 calls) | 8921 (-0.07%) | 0 |
| r42 | 182.4 | 189 | 1.279e-42 → 1.279e-43 | secant+step-cap | 0.048 / 0.155 / 0.073 (189 calls) | 0.074 / 0.143 / 0.098 (189 calls) | 8922 (-0.07%) | 0 |
| r43 | 181.4 | 188 | 1.279e-43 → 1.279e-44 | secant+step-cap | 0.048 / 0.155 / 0.073 (188 calls) | 0.074 / 0.143 / 0.098 (188 calls) | 8922 (-0.07%) | 0 |
| r44 | 179.9 | 185 | 1.279e-44 → 1.279e-45 | secant+step-cap | 0.049 / 0.155 / 0.074 (185 calls) | 0.076 / 0.143 / 0.099 (185 calls) | 8923 (-0.06%) | 0 |
| r45 | 177.4 | 183 | 1.279e-45 → 1.279e-46 | secant+step-cap | 0.049 / 0.155 / 0.075 (183 calls) | 0.077 / 0.143 / 0.100 (183 calls) | 8923 (-0.06%) | 0 |
| r46 | 175.3 | 183 | 1.279e-46 → 1.279e-47 | secant+step-cap | 0.049 / 0.155 / 0.075 (183 calls) | 0.077 / 0.143 / 0.100 (183 calls) | 8923 (-0.06%) | 0 |
| r47 | 173.7 | 181 | 1.279e-47 → 1.279e-48 | secant+step-cap | 0.050 / 0.155 / 0.075 (181 calls) | 0.077 / 0.143 / 0.100 (181 calls) | 8923 (-0.06%) | 0 |
| r48 | 172.4 | 180 | 1.279e-48 → 1.279e-49 | secant+step-cap | 0.050 / 0.155 / 0.076 (180 calls) | 0.078 / 0.143 / 0.101 (180 calls) | 8923 (-0.06%) | 0 |
| r49 | 171.7 | 179 | 1.279e-49 → 1.279e-50 | secant+step-cap | 0.045 / 0.138 / 0.068 (179 calls) | 0.073 / 0.133 / 0.094 (179 calls) | 8923 (-0.06%) | 0 |

state: stopped after iter 49: reached the 50-round limit [deadband 1.1x]

jobs: `{"0": {"count": "12644209", "decode": "12644208", "holdout_count": "12644861", "holdout_decode": "12644860", "holdout_validate": "12644862", "next": "12644210"}, "1": {"count": "12644857", "decode": "12644856", "next": "12644858"}, "10": {"count": "12649554", "decode": "12649553", "next": "12649555"}, "11": {"count": "12650102", "decode": "12650101", "next": "12650103"}, "12": {"count": "12650539", "decode": "12650538", "next": "12650540"}, "13": {"count": "12651029", "decode": "12651028", "next": "12651030"}, "14": {"count": "12651642", "decode": "12651641", "next": "12651643"}, "15": {"count": "12652040", "decode": "12652039", "next": "12652041"}, "16": {"count": "12652386", "decode": "12652385", "next": "12652387"}, "17": {"count": "12652724", "decode": "12652723", "next": "12652725"}, "18": {"count": "12653071", "decode": "12653070", "next": "12653072"}, "19": {"count": "12653470", "decode": "12653469", "next": "12653471"}, "2": {"count": "12645727", "decode": "12645726", "next": "12645728"}, "20": {"count": "12664762", "decode": "12664761", "next": "12664763"}, "21": {"count": "12665699", "decode": "12665698", "next": "12665700"}, "22": {"count": "12666328", "decode": "12666327", "next": "12666329"}, "23": {"count": "12666614", "decode": "12666613", "next": "12666615"}, "24": {"count": "12666914", "decode": "12666913", "next": "12666915"}, "25": {"count": "12667223", "decode": "12667222", "next": "12667224"}, "26": {"count": "12668084", "decode": "12668083", "next": "12668085"}, "27": {"count": "12668874", "decode": "12668873", "next": "12668875"}, "28": {"count": "12669402", "decode": "12669401", "next": "12669403"}, "29": {"count": "12669659", "decode": "12669658", "next": "12669660"}, "3": {"count": "12646421", "decode": "12646420", "next": "12646422"}, "30": {"count": "12669974", "decode": "12669973", "next": "12669975"}, "31": {"count": "12670398", "decode": "12670397", "next": "12670399"}, "32": {"count": "12670942", "decode": "12670941", "next": "12670943"}, "33": {"count": "12672132", "decode": "12672131", "next": "12672133"}, "34": {"count": "12675535", "decode": "12675534", "next": "12675536"}, "35": {"count": "12676042", "decode": "12676041", "next": "12676043"}, "36": {"count": "12676573", "decode": "12676572", "next": "12676574"}, "37": {"count": "12677064", "decode": "12677063", "next": "12677065"}, "38": {"count": "12677942", "decode": "12677941", "next": "12677943"}, "39": {"count": "12678506", "decode": "12678505", "next": "12678507"}, "4": {"count": "12646926", "decode": "12646925", "next": "12646927"}, "40": {"count": "12678880", "decode": "12678879", "next": "12678881"}, "41": {"count": "12679187", "decode": "12679186", "next": "12679188"}, "42": {"count": "12679618", "decode": "12679617", "next": "12679619"}, "43": {"count": "12679994", "decode": "12679993", "next": "12679995"}, "44": {"count": "12680349", "decode": "12680348", "next": "12680350"}, "45": {"count": "12680714", "decode": "12680713", "next": "12680715"}, "46": {"count": "12681106", "decode": "12681105", "next": "12681107"}, "47": {"count": "12682171", "decode": "12682170", "next": "12682172"}, "48": {"count": "12682421", "decode": "12682420", "next": "12682422"}, "49": {"chereji": "12683030", "count": "12682747", "decode": "12682746", "holdout_count": "12683028", "holdout_decode": "12683027", "holdout_validate": "12683029", "next": "12682748", "summary": "12683031"}, "5": {"count": "12647436", "decode": "12647435", "next": "12647437"}, "6": {"count": "12647826", "decode": "12647825", "next": "12647827"}, "7": {"count": "12648251", "decode": "12648250", "next": "12648252"}, "8": {"count": "12648542", "decode": "12648541", "next": "12648543"}, "9": {"count": "12648836", "decode": "12648835", "next": "12648837"}}`

## fp100 — fiber only, ABF1 ±100 (214 bp; vs fa01)

src trainDir `robocop_train_abf1_pm100`, driver `run_split_revfix_fiber_maskoff_abf1w100.py`, cap basis `robocop_train_fiberonly`, max_rounds 50, deadband 1.1x

### P / R / F1 — tuning chrXIV+chrII (fp100 r00 / final; fa01 r00 / final)

| ref | fp100 r00 | fp100 final | fa01 r00 | fa01 final |
|---|---|---|---|---|
| MacIsaac | 0.048 / 0.224 / 0.080 (269 calls) | 0.043 / 0.138 / 0.065 (187 calls) | 0.007 / 0.569 / 0.013 (4841 calls) | 0.019 / 0.034 / 0.024 (106 calls) |
| Rossi _CX | 0.074 / 0.204 / 0.109 (269 calls) | 0.070 / 0.133 / 0.091 (187 calls) | 0.009 / 0.449 / 0.018 (4841 calls) | 0.038 / 0.041 / 0.039 (106 calls) |

### P / R / F1 — holdout chrIV (fp100 r00 / final; fa01 r00 / final)

| ref | fp100 r00 | fp100 final | fa01 r00 | fa01 final |
|---|---|---|---|---|
| MacIsaac | 0.040 / 0.227 / 0.068 (250 calls) | 0.040 / 0.159 / 0.064 (175 calls) | 0.004 / 0.455 / 0.008 (4929 calls) | 0.014 / 0.045 / 0.021 (146 calls) |
| Rossi _CX | 0.076 / 0.184 / 0.108 (250 calls) | 0.074 / 0.126 / 0.094 (175 calls) | 0.008 / 0.369 / 0.015 (4929 calls) | 0.014 / 0.019 / 0.016 (146 calls) |

### Per round (fp100; T = ABF1 58)

| round | E | calls (MacIsaac rows) | λ → λ next | step | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | unknown occ (masked: expect 0) |
|---|---|---|---|---|---|---|---|---|
| r00 | 265.5 | 269 | 1 → 0.1211 | secant | 0.048 / 0.224 / 0.080 (269 calls) | 0.074 / 0.204 / 0.109 (269 calls) | 8927 (+0.00%) | 0 |
| r01 | 263.8 | 268 | 0.1211 → 0.01211 | secant+step-cap | 0.049 / 0.224 / 0.080 (268 calls) | 0.075 / 0.204 / 0.109 (268 calls) | 8927 (+0.00%) | 0 |
| r02 | 262.1 | 267 | 0.01211 → 0.001211 | secant+step-cap | 0.049 / 0.224 / 0.080 (267 calls) | 0.075 / 0.204 / 0.110 (267 calls) | 8927 (+0.00%) | 0 |
| r03 | 260.6 | 266 | 0.001211 → 0.0001211 | secant+step-cap | 0.049 / 0.224 / 0.080 (266 calls) | 0.075 / 0.204 / 0.110 (266 calls) | 8927 (+0.00%) | 0 |
| r04 | 259.4 | 265 | 0.0001211 → 1.211e-05 | secant+step-cap | 0.049 / 0.224 / 0.080 (265 calls) | 0.075 / 0.204 / 0.110 (265 calls) | 8927 (+0.00%) | 0 |
| r05 | 258.2 | 264 | 1.211e-05 → 1.211e-06 | secant+step-cap | 0.049 / 0.224 / 0.081 (264 calls) | 0.076 / 0.204 / 0.110 (264 calls) | 8927 (+0.01%) | 0 |
| r06 | 257.1 | 262 | 1.211e-06 → 1.211e-07 | secant+step-cap | 0.050 / 0.224 / 0.081 (262 calls) | 0.076 / 0.204 / 0.111 (262 calls) | 8928 (+0.01%) | 0 |
| r07 | 256.1 | 261 | 1.211e-07 → 1.211e-08 | secant+step-cap | 0.050 / 0.224 / 0.082 (261 calls) | 0.077 / 0.204 / 0.111 (261 calls) | 8928 (+0.01%) | 0 |
| r08 | 255.1 | 261 | 1.211e-08 → 1.211e-09 | secant+step-cap | 0.050 / 0.224 / 0.082 (261 calls) | 0.077 / 0.204 / 0.111 (261 calls) | 8928 (+0.01%) | 0 |
| r09 | 254.4 | 259 | 1.211e-09 → 1.211e-10 | secant+step-cap | 0.050 / 0.224 / 0.082 (259 calls) | 0.077 / 0.204 / 0.112 (259 calls) | 8928 (+0.01%) | 0 |
| r10 | 253.7 | 258 | 1.211e-10 → 1.211e-11 | secant+step-cap | 0.050 / 0.224 / 0.082 (258 calls) | 0.078 / 0.204 / 0.112 (258 calls) | 8928 (+0.01%) | 0 |
| r11 | 252.9 | 258 | 1.211e-11 → 1.211e-12 | secant+step-cap | 0.050 / 0.224 / 0.082 (258 calls) | 0.078 / 0.204 / 0.112 (258 calls) | 8928 (+0.01%) | 0 |
| r12 | 251.1 | 256 | 1.211e-12 → 1.211e-13 | secant+step-cap | 0.051 / 0.224 / 0.083 (256 calls) | 0.078 / 0.204 / 0.113 (256 calls) | 8928 (+0.01%) | 0 |
| r13 | 249.2 | 254 | 1.211e-13 → 1.211e-14 | secant+step-cap | 0.051 / 0.224 / 0.083 (254 calls) | 0.079 / 0.204 / 0.114 (254 calls) | 8928 (+0.01%) | 0 |
| r14 | 247.2 | 254 | 1.211e-14 → 1.211e-15 | secant+step-cap | 0.051 / 0.224 / 0.083 (254 calls) | 0.079 / 0.204 / 0.114 (254 calls) | 8928 (+0.01%) | 0 |
| r15 | 245.3 | 252 | 1.211e-15 → 1.211e-16 | secant+step-cap | 0.052 / 0.224 / 0.084 (252 calls) | 0.079 / 0.204 / 0.114 (252 calls) | 8928 (+0.01%) | 0 |
| r16 | 243.8 | 249 | 1.211e-16 → 1.211e-17 | secant+step-cap | 0.052 / 0.224 / 0.085 (249 calls) | 0.080 / 0.204 / 0.115 (249 calls) | 8928 (+0.02%) | 0 |
| r17 | 242.3 | 249 | 1.211e-17 → 1.211e-18 | secant+step-cap | 0.048 / 0.207 / 0.078 (249 calls) | 0.076 / 0.194 / 0.110 (249 calls) | 8929 (+0.02%) | 0 |
| r18 | 239.8 | 248 | 1.211e-18 → 1.211e-19 | secant+step-cap | 0.048 / 0.207 / 0.078 (248 calls) | 0.077 / 0.194 / 0.110 (248 calls) | 8929 (+0.03%) | 0 |
| r19 | 238.0 | 244 | 1.211e-19 → 1.211e-20 | secant+step-cap | 0.049 / 0.207 / 0.079 (244 calls) | 0.078 / 0.194 / 0.111 (244 calls) | 8930 (+0.04%) | 0 |
| r20 | 237.1 | 243 | 1.211e-20 → 1.211e-21 | secant+step-cap | 0.049 / 0.207 / 0.080 (243 calls) | 0.078 / 0.194 / 0.111 (243 calls) | 8930 (+0.04%) | 0 |
| r21 | 236.0 | 241 | 1.211e-21 → 1.211e-22 | secant+step-cap | 0.050 / 0.207 / 0.080 (241 calls) | 0.079 / 0.194 / 0.112 (241 calls) | 8931 (+0.05%) | 0 |
| r22 | 234.1 | 241 | 1.211e-22 → 1.211e-23 | secant+step-cap | 0.050 / 0.207 / 0.080 (241 calls) | 0.079 / 0.194 / 0.112 (241 calls) | 8931 (+0.05%) | 0 |
| r23 | 232.1 | 238 | 1.211e-23 → 1.211e-24 | secant+step-cap | 0.046 / 0.190 / 0.074 (238 calls) | 0.076 / 0.184 / 0.107 (238 calls) | 8932 (+0.06%) | 0 |
| r24 | 230.7 | 236 | 1.211e-24 → 1.211e-25 | secant+step-cap | 0.047 / 0.190 / 0.075 (236 calls) | 0.076 / 0.184 / 0.108 (236 calls) | 8932 (+0.06%) | 0 |
| r25 | 229.4 | 235 | 1.211e-25 → 1.211e-26 | secant+step-cap | 0.047 / 0.190 / 0.075 (235 calls) | 0.077 / 0.184 / 0.108 (235 calls) | 8932 (+0.06%) | 0 |
| r26 | 227.5 | 233 | 1.211e-26 → 1.211e-27 | secant+step-cap | 0.047 / 0.190 / 0.076 (233 calls) | 0.077 / 0.184 / 0.109 (233 calls) | 8932 (+0.06%) | 0 |
| r27 | 225.9 | 232 | 1.211e-27 → 1.211e-28 | secant+step-cap | 0.047 / 0.190 / 0.076 (232 calls) | 0.078 / 0.184 / 0.109 (232 calls) | 8917 (-0.11%) | 0 |
| r28 | 224.9 | 229 | 1.211e-28 → 1.211e-29 | secant+step-cap | 0.048 / 0.190 / 0.077 (229 calls) | 0.079 / 0.184 / 0.110 (229 calls) | 8916 (-0.12%) | 0 |
| r29 | 223.4 | 226 | 1.211e-29 → 1.211e-30 | secant+step-cap | 0.049 / 0.190 / 0.077 (226 calls) | 0.080 / 0.184 / 0.111 (226 calls) | 8915 (-0.13%) | 0 |
| r30 | 221.2 | 226 | 1.211e-30 → 1.211e-31 | secant+step-cap | 0.044 / 0.172 / 0.070 (226 calls) | 0.075 / 0.173 / 0.105 (226 calls) | 8913 (-0.15%) | 0 |
| r31 | 217.9 | 225 | 1.211e-31 → 1.211e-32 | secant+step-cap | 0.044 / 0.172 / 0.071 (225 calls) | 0.076 / 0.173 / 0.105 (225 calls) | 8914 (-0.14%) | 0 |
| r32 | 214.1 | 221 | 1.211e-32 → 1.211e-33 | secant+step-cap | 0.041 / 0.155 / 0.065 (221 calls) | 0.068 / 0.153 / 0.094 (221 calls) | 8914 (-0.14%) | 0 |
| r33 | 210.6 | 218 | 1.211e-33 → 1.211e-34 | secant+step-cap | 0.041 / 0.155 / 0.065 (218 calls) | 0.069 / 0.153 / 0.095 (218 calls) | 8915 (-0.13%) | 0 |
| r34 | 208.7 | 213 | 1.211e-34 → 1.211e-35 | secant+step-cap | 0.042 / 0.155 / 0.066 (213 calls) | 0.070 / 0.153 / 0.096 (213 calls) | 8915 (-0.12%) | 0 |
| r35 | 206.6 | 213 | 1.211e-35 → 1.211e-36 | secant+step-cap | 0.042 / 0.155 / 0.066 (213 calls) | 0.070 / 0.153 / 0.096 (213 calls) | 8915 (-0.12%) | 0 |
| r36 | 205.1 | 210 | 1.211e-36 → 1.211e-37 | secant+step-cap | 0.043 / 0.155 / 0.067 (210 calls) | 0.071 / 0.153 / 0.097 (210 calls) | 8916 (-0.12%) | 0 |
| r37 | 204.3 | 208 | 1.211e-37 → 1.211e-38 | secant+step-cap | 0.043 / 0.155 / 0.068 (208 calls) | 0.072 / 0.153 / 0.098 (208 calls) | 8916 (-0.12%) | 0 |
| r38 | 202.7 | 208 | 1.211e-38 → 1.211e-39 | secant+step-cap | 0.043 / 0.155 / 0.068 (208 calls) | 0.072 / 0.153 / 0.098 (208 calls) | 8916 (-0.12%) | 0 |
| r39 | 201.1 | 206 | 1.211e-39 → 1.211e-40 | secant+step-cap | 0.044 / 0.155 / 0.068 (206 calls) | 0.073 / 0.153 / 0.099 (206 calls) | 8916 (-0.12%) | 0 |
| r40 | 200.3 | 204 | 1.211e-40 → 1.211e-41 | secant+step-cap | 0.044 / 0.155 / 0.069 (204 calls) | 0.074 / 0.153 / 0.099 (204 calls) | 8916 (-0.12%) | 0 |
| r41 | 199.2 | 203 | 1.211e-41 → 1.211e-42 | secant+step-cap | 0.044 / 0.155 / 0.069 (203 calls) | 0.074 / 0.153 / 0.100 (203 calls) | 8916 (-0.12%) | 0 |
| r42 | 197.5 | 202 | 1.211e-42 → 1.211e-43 | secant+step-cap | 0.045 / 0.155 / 0.069 (202 calls) | 0.074 / 0.153 / 0.100 (202 calls) | 8916 (-0.12%) | 0 |
| r43 | 195.0 | 201 | 1.211e-43 → 1.211e-44 | secant+step-cap | 0.045 / 0.155 / 0.069 (201 calls) | 0.075 / 0.153 / 0.100 (201 calls) | 8917 (-0.10%) | 0 |
| r44 | 193.4 | 198 | 1.211e-44 → 1.211e-45 | secant+step-cap | 0.045 / 0.155 / 0.070 (198 calls) | 0.076 / 0.153 / 0.101 (198 calls) | 8918 (-0.09%) | 0 |
| r45 | 191.6 | 198 | 1.211e-45 → 1.211e-46 | secant+step-cap | 0.045 / 0.155 / 0.070 (198 calls) | 0.076 / 0.153 / 0.101 (198 calls) | 8919 (-0.09%) | 0 |
| r46 | 189.6 | 195 | 1.211e-46 → 1.211e-47 | secant+step-cap | 0.046 / 0.155 / 0.071 (195 calls) | 0.077 / 0.153 / 0.102 (195 calls) | 8919 (-0.08%) | 0 |
| r47 | 187.3 | 194 | 1.211e-47 → 1.211e-48 | secant+step-cap | 0.046 / 0.155 / 0.071 (194 calls) | 0.077 / 0.153 / 0.103 (194 calls) | 8919 (-0.08%) | 0 |
| r48 | 184.5 | 191 | 1.211e-48 → 1.211e-49 | secant+step-cap | 0.042 / 0.138 / 0.064 (191 calls) | 0.073 / 0.143 / 0.097 (191 calls) | 8920 (-0.07%) | 0 |
| r49 | 182.0 | 187 | 1.211e-49 → 1.211e-50 | secant+step-cap | 0.043 / 0.138 / 0.065 (187 calls) | 0.070 / 0.133 / 0.091 (187 calls) | 8921 (-0.06%) | 0 |

state: stopped after iter 49: reached the 50-round limit [deadband 1.1x]

jobs: `{"0": {"count": "12644217", "decode": "12644216", "holdout_count": "12644920", "holdout_decode": "12644919", "holdout_validate": "12644921", "next": "12644218"}, "1": {"count": "12644917", "decode": "12644916", "next": "12644918"}, "10": {"count": "12649557", "decode": "12649556", "next": "12649558"}, "11": {"count": "12650164", "decode": "12650163", "next": "12650165"}, "12": {"count": "12650601", "decode": "12650600", "next": "12650602"}, "13": {"count": "12651141", "decode": "12651140", "next": "12651142"}, "14": {"count": "12651696", "decode": "12651695", "next": "12651697"}, "15": {"count": "12652095", "decode": "12652094", "next": "12652096"}, "16": {"count": "12652470", "decode": "12652469", "next": "12652471"}, "17": {"count": "12652792", "decode": "12652791", "next": "12652793"}, "18": {"count": "12653162", "decode": "12653161", "next": "12653163"}, "19": {"count": "12653535", "decode": "12653534", "next": "12653536"}, "2": {"count": "12645885", "decode": "12645884", "next": "12645886"}, "20": {"count": "12664769", "decode": "12664768", "next": "12664770"}, "21": {"count": "12665858", "decode": "12665857", "next": "12665859"}, "22": {"count": "12666371", "decode": "12666370", "next": "12666372"}, "23": {"count": "12666664", "decode": "12666663", "next": "12666665"}, "24": {"count": "12667001", "decode": "12667000", "next": "12667002"}, "25": {"count": "12667265", "decode": "12667264", "next": "12667266"}, "26": {"count": "12668326", "decode": "12668325", "next": "12668327"}, "27": {"count": "12669201", "decode": "12669200", "next": "12669202"}, "28": {"count": "12669520", "decode": "12669519", "next": "12669521"}, "29": {"count": "12669746", "decode": "12669745", "next": "12669747"}, "3": {"count": "12646549", "decode": "12646548", "next": "12646550"}, "30": {"count": "12670069", "decode": "12670068", "next": "12670070"}, "31": {"count": "12670496", "decode": "12670495", "next": "12670497"}, "32": {"count": "12671109", "decode": "12671108", "next": "12671110"}, "33": {"count": "12672372", "decode": "12672371", "next": "12672373"}, "34": {"count": "12675710", "decode": "12675709", "next": "12675711"}, "35": {"count": "12676067", "decode": "12676066", "next": "12676068"}, "36": {"count": "12676680", "decode": "12676679", "next": "12676681"}, "37": {"count": "12677176", "decode": "12677175", "next": "12677177"}, "38": {"count": "12678169", "decode": "12678168", "next": "12678170"}, "39": {"count": "12678606", "decode": "12678605", "next": "12678607"}, "4": {"count": "12646996", "decode": "12646995", "next": "12646997"}, "40": {"count": "12678987", "decode": "12678986", "next": "12678988"}, "41": {"count": "12679342", "decode": "12679341", "next": "12679343"}, "42": {"count": "12679801", "decode": "12679800", "next": "12679802"}, "43": {"count": "12680122", "decode": "12680121", "next": "12680123"}, "44": {"count": "12680495", "decode": "12680494", "next": "12680496"}, "45": {"count": "12680845", "decode": "12680844", "next": "12680846"}, "46": {"count": "12681353", "decode": "12681352", "next": "12681354"}, "47": {"count": "12682263", "decode": "12682262", "next": "12682264"}, "48": {"count": "12682573", "decode": "12682572", "next": "12682574"}, "49": {"chereji": "12683123", "count": "12682920", "decode": "12682919", "holdout_count": "12683121", "holdout_decode": "12683120", "holdout_validate": "12683122", "next": "12682921", "summary": "12683124"}, "5": {"count": "12647522", "decode": "12647521", "next": "12647523"}, "6": {"count": "12647933", "decode": "12647932", "next": "12647934"}, "7": {"count": "12648402", "decode": "12648401", "next": "12648403"}, "8": {"count": "12648694", "decode": "12648693", "next": "12648695"}, "9": {"count": "12648907", "decode": "12648906", "next": "12648908"}}`

## sp100 — sequence only, ABF1 ±100 (214 bp; vs sa01)

src trainDir `robocop_train_abf1_pm100`, driver `run_split_revfix_seqonly_maskoff_abf1w100.py`, cap basis `robocop_train_fiberonly`, max_rounds 50, deadband 1.1x

### P / R / F1 — tuning chrXIV+chrII (sp100 r00 / final; sa01 r00 / final)

| ref | sp100 r00 | sp100 final | sa01 r00 | sa01 final |
|---|---|---|---|---|
| MacIsaac | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 1.000 / 0.017 / 0.034 (1 calls) | 0.162 / 0.276 / 0.204 (99 calls) |
| Rossi _CX | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 0.253 / 0.255 / 0.254 (99 calls) |

### P / R / F1 — holdout chrIV (sp100 r00 / final; sa01 r00 / final)

| ref | sp100 r00 | sp100 final | sa01 r00 | sa01 final |
|---|---|---|---|---|
| MacIsaac | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 0.124 / 0.295 / 0.174 (105 calls) |
| Rossi _CX | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 0.257 / 0.262 / 0.260 (105 calls) |

### Per round (sp100; T = ABF1 58)

| round | E | calls (MacIsaac rows) | λ → λ next | step | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | unknown occ (masked: expect 0) |
|---|---|---|---|---|---|---|---|---|
| r00 | 0.0 | 0 | 1 → 10 | no-slope | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9445 (+0.00%) | 0 |
| r01 | 0.0 | 0 | 10 → 100 | no-slope | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9445 (+0.00%) | 0 |
| r02 | 0.0 | 0 | 100 → 1000 | no-slope | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9445 (+0.00%) | 0 |
| r03 | 0.0 | 0 | 1000 → 1e+04 | no-slope | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9445 (+0.00%) | 0 |
| r04 | 0.0 | 0 | 1e+04 → 1.434e+04 | no-slope+rho-cap | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9445 (+0.00%) | 0 |
| r05 | 0.0 | 0 | 1.434e+04 → 1.434e+04 | no-slope+rho-cap | – / 0.000 / – (0 calls) | – / 0.000 / – (0 calls) | 9445 (+0.00%) | 0 |

state: stopped after iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12644222", "decode": "12644221", "holdout_count": "12644983", "holdout_decode": "12644982", "holdout_validate": "12644984", "next": "12644223"}, "1": {"count": "12644977", "decode": "12644976", "next": "12644978"}, "2": {"count": "12645963", "decode": "12645962", "next": "12645964"}, "3": {"count": "12646648", "decode": "12646647", "next": "12646649"}, "4": {"count": "12647137", "decode": "12647136", "next": "12647138"}, "5": {"chereji": "12647924", "count": "12647532", "decode": "12647531", "holdout_count": "12647922", "holdout_decode": "12647921", "holdout_validate": "12647923", "next": "12647533", "summary": "12647925"}}`

