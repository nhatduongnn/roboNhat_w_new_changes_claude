# Masked tuner-v2 campaigns: ABF1 only and the 9 fitted TFs vs the unmasked runs

_generated 2026-10-02 15:47 by `analysis/masked_summary.py` (masked campaigns from `conc_tuning`); campaigns still running are marked **INCOMPLETE** — re-run `python masked_summary.py` to refresh._

### Rule 7 — what differs

Versus bw01 / fw01 / sw01 at the same layer configuration, **only the live factor set differs** (ABF1 only = `Abf1_murphy`; or the 9 fitted TFs `Abf1_murphy, Cin5_murphy, Fhl1_zhu, Fkh1_zhu, Mcm1_zhu, Rap1_telomeric, Reb1_badis, Sko1_murphy, Ume6_zhu`; every other motif **and `unknown`** hard-masked) **and the stop settings are 1.1× deadband / 25 rounds from round 0** (the counterparts started at 1.25× / 8 rounds and were continued; their stop-setting history is listed below). Everything else is identical: untuned start (round-0 weights identical to `robocop_train_tw_bw01_00`), no φ (fiber untempered), no nucleosome hold (w_nuc 35), step cap 10×/round, per-bp cap 0.70, bisection, β seed, MacIsaac targets, tune chrXIV+chrII, holdout chrIV.

Definitions: within 1.1× = |ln((E+1)/(T+1))| ≤ ln 1.1 (the tuner's deadband test). E = summed occupancy of the group's live motifs. P / R / F1 = calls at posterior ≥ 0.10 matched one-to-one within 30 bp (tw_validate.py). **Counterpart rows are restricted to the same groups**; their RAP1 group pools the motifs listed below, the fit9 campaigns' RAP1 is `Rap1_telomeric` only.

- counterpart bw01: stopped after iter 24: reached the 25-round limit [deadband 1.1x]; stop settings 1.25x/8 → 1.1x/15 (from r07) → 1.1x/25 (from r14); RAP1 = `Rap1_zhu+Rap1_motif1+Rap1_motif2+Rap1_telomeric`; last round 24; holdout rounds [0, 7, 14, 24]
- counterpart fw01: stopped after iter 24: reached the 25-round limit [deadband 1.1x]; stop settings 1.25x/8 → 1.1x/15 (from r07) → 1.1x/25 (from r14); RAP1 = `Rap1_zhu+Rap1_motif1+Rap1_motif2+Rap1_telomeric`; last round 24; holdout rounds [0, 7, 14, 24]
- counterpart sw01: stopped after iter 7: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]; stop settings 1.25x/8 → 1.1x/15 (from r05); RAP1 = `Rap1_zhu+Rap1_motif1+Rap1_motif2+Rap1_telomeric`; last round 7; holdout rounds [0, 5, 7]

## Campaigns

| run | layers | factor set | counterpart | state | rounds | within 1.1× (final) | counterpart within 1.1× (same groups) | nucleosome copies chrXIV+chrII (final) | counterpart nucleosome copies | holdout chrIV nucleosome copies | Chereji +1/-1 recall chrXIV (final) | counterpart Chereji recall (latest) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ba01 | both layers | abf1only | bw01 | iter 8: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r08 | 1/1 | 1/1 (r24) | 8950 (+0.24% vs r0) | 8945 (+0.29%, r24) | r00 8573 → r08 8596 | 0.802 (n_ref 626) | 0.812 (r24) |
| bf09 | both layers | fit9 | bw01 | iter 40: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r40 | 8/9 | 9/9 (r24) | 8948 (+0.26% vs r0) | 8945 (+0.29%, r24) | r00 8571 → r40 8594 | 0.802 (n_ref 626) | 0.812 (r24) |
| sa01 | sequence only | abf1only | sw01 | iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r05 | 1/1 | 1/1 (r07) | 9433 (-0.12% vs r0) | 9186 (-0.78%, r07) | r00 9059 → r05 9047 | 0.117 (n_ref 626) | 0.137 (r07) |
| sf09 | sequence only | fit9 | sw01 | iter 10: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r10 | 8/9 | 9/9 (r07) | 9401 (-0.42% vs r0) | 9186 (-0.78%, r07) | r00 9055 → r10 9018 | 0.144 (n_ref 626) | 0.137 (r07) |
| fa01 | fiber only | abf1only | fw01 | iter 16: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r16 | 1/1 | 1/1 (r24) | 8948 (+1.46% vs r0) | 8941 (+1.60%, r24) | r00 8457 → r16 8595 | 0.804 (n_ref 626) | 0.810 (r24) |
| ff09 | fiber only | fit9 | fw01 | iter 40: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | r00–r40 | 9/9 | 9/9 (r24) | 8944 (+1.60% vs r0) | 8941 (+1.60%, r24) | r00 8436 → r40 8593 | 0.805 (n_ref 626) | 0.810 (r24) |

## ba01 — both layers, abf1only (vs bw01)

### Counts (masked r00 → r08; counterpart bw01 r00 → r24)

| group | live motif(s) | T | E masked | (E+1)/(T+1) final | within 1.1× | λ next | last step | E bw01 | bw01 within 1.1× | bw01 motifs |
|---|---|---|---|---|---|---|---|---|---|---|
| ABF1 | Abf1_murphy | 58 | 1007.3 → 63.7 | 1.1x | yes | 1.803e-08 | deadband | 978.1 → 60.8 | yes | Abf1_murphy |

### P / R / F1 — tuning chrXIV+chrII (masked r00, masked r08; bw01 r00, bw01 r24)

| ref | group | masked r00 | masked final | bw01 r00 | bw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 0.021 / 0.517 / 0.041 (1399 calls) | 0.073 / 0.138 / 0.095 (110 calls) | 0.020 / 0.483 / 0.039 (1366 calls) | 0.075 / 0.138 / 0.098 (106 calls) |
| MacIsaac | set total (1 groups) | 0.021 / 0.517 / 0.041 (1399 calls) | 0.073 / 0.138 / 0.095 (110 calls) | 0.020 / 0.483 / 0.039 (1366 calls) | 0.075 / 0.138 / 0.098 (106 calls) |
| Rossi _CX | ABF1 | 0.030 / 0.429 / 0.056 (1399 calls) | 0.091 / 0.102 / 0.096 (110 calls) | 0.029 / 0.398 / 0.053 (1366 calls) | 0.085 / 0.092 / 0.088 (106 calls) |
| Rossi _CX | set total (1 groups) | 0.030 / 0.429 / 0.056 (1399 calls) | 0.091 / 0.102 / 0.096 (110 calls) | 0.029 / 0.398 / 0.053 (1366 calls) | 0.085 / 0.092 / 0.088 (106 calls) |

### P / R / F1 — holdout chrIV (masked r00, masked r08; bw01 r00, bw01 r24)

| ref | group | masked r00 | masked final | bw01 r00 | bw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 0.012 / 0.432 / 0.023 (1580 calls) | 0.026 / 0.091 / 0.041 (153 calls) | 0.012 / 0.432 / 0.024 (1531 calls) | 0.027 / 0.091 / 0.042 (147 calls) |
| MacIsaac | set total (1 groups) | 0.012 / 0.432 / 0.023 (1580 calls) | 0.026 / 0.091 / 0.041 (153 calls) | 0.012 / 0.432 / 0.024 (1531 calls) | 0.027 / 0.091 / 0.042 (147 calls) |
| Rossi _CX | ABF1 | 0.021 / 0.320 / 0.039 (1580 calls) | 0.039 / 0.058 / 0.047 (153 calls) | 0.020 / 0.301 / 0.038 (1531 calls) | 0.034 / 0.049 / 0.040 (147 calls) |
| Rossi _CX | set total (1 groups) | 0.021 / 0.320 / 0.039 (1580 calls) | 0.039 / 0.058 / 0.047 (153 calls) | 0.020 / 0.301 / 0.038 (1531 calls) | 0.034 / 0.049 / 0.040 (147 calls) |

### Per round (ba01)

| round | within 1.1× | MacIsaac P/R/F1 (set) | Rossi P/R/F1 (set) | nucleosome copies | unknown occ (masked: expect 0) | non-deadband steps |
|---|---|---|---|---|---|---|
| r00 | 0/1 | 0.021 / 0.517 / 0.041 (1399 calls) | 0.030 / 0.429 / 0.056 (1399 calls) | 8928 (+0.00%) | 0 | ABF1 secant+step-cap |
| r01 | 0/1 | 0.026 / 0.483 / 0.050 (1062 calls) | 0.036 / 0.388 / 0.066 (1062 calls) | 8934 (+0.06%) | 0 | ABF1 secant+step-cap |
| r02 | 0/1 | 0.028 / 0.397 / 0.052 (819 calls) | 0.039 / 0.327 / 0.070 (819 calls) | 8936 (+0.09%) | 0 | ABF1 secant+step-cap |
| r03 | 0/1 | 0.029 / 0.310 / 0.053 (616 calls) | 0.045 / 0.286 / 0.078 (616 calls) | 8940 (+0.13%) | 0 | ABF1 secant+step-cap |
| r04 | 0/1 | 0.030 / 0.224 / 0.053 (432 calls) | 0.046 / 0.204 / 0.075 (432 calls) | 8943 (+0.16%) | 0 | ABF1 secant+step-cap |
| r05 | 0/1 | 0.042 / 0.224 / 0.071 (309 calls) | 0.058 / 0.184 / 0.088 (309 calls) | 8946 (+0.20%) | 0 | ABF1 secant+step-cap |
| r06 | 0/1 | 0.055 / 0.190 / 0.085 (200 calls) | 0.075 / 0.153 / 0.101 (200 calls) | 8948 (+0.22%) | 0 | ABF1 secant+step-cap |
| r07 | 0/1 | 0.063 / 0.155 / 0.090 (142 calls) | 0.077 / 0.112 / 0.092 (142 calls) | 8949 (+0.23%) | 0 | ABF1 secant |
| r08 | 1/1 | 0.073 / 0.138 / 0.095 (110 calls) | 0.091 / 0.102 / 0.096 (110 calls) | 8950 (+0.24%) | 0 | all deadband |

state: stopped after iter 8: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12611322", "decode": "12611321", "holdout_count": "12611804", "holdout_decode": "12611803", "holdout_validate": "12611805", "next": "12611323"}, "1": {"count": "12611786", "decode": "12611785", "next": "12611787"}, "2": {"count": "12612331", "decode": "12612330", "next": "12612332"}, "3": {"count": "12612634", "decode": "12612633", "next": "12612635"}, "4": {"count": "12612911", "decode": "12612910", "next": "12612912"}, "5": {"count": "12613215", "decode": "12613214", "next": "12613216"}, "6": {"count": "12613502", "decode": "12613501", "next": "12613503"}, "7": {"count": "12613772", "decode": "12613770", "next": "12613774"}, "8": {"chereji": "12614747", "count": "12614034", "decode": "12614033", "failed_holdout_count": "12614745", "failed_holdout_decode": "12614744", "failed_holdout_validate": "12614746", "holdout_count": "12637877", "holdout_decode": "12637876", "holdout_validate": "12637878", "next": "12614035", "summary": "12614748"}}`

## bf09 — both layers, fit9 (vs bw01)

### Counts (masked r00 → r40; counterpart bw01 r00 → r24)

| group | live motif(s) | T | E masked | (E+1)/(T+1) final | within 1.1× | λ next | last step | E bw01 | bw01 within 1.1× | bw01 motifs |
|---|---|---|---|---|---|---|---|---|---|---|
| ABF1 | Abf1_murphy | 58 | 968.9 → 61.3 | 1.06x | yes | 1.44e-08 | deadband | 978.1 → 60.8 | yes | Abf1_murphy |
| CIN5 | Cin5_murphy | 49 | 358.5 → 53.6 | 1.09x | yes | 4.005e-24 | deadband | 79.3 → 53.4 | yes | Cin5_murphy |
| FHL1 | Fhl1_zhu | 21 | 1008.1 → 22.1 | 1.05x | yes | 2.832e-24 | deadband | 131.9 → 22.0 | yes | Fhl1_zhu |
| FKH1 | Fkh1_zhu | 23 | 928.6 → 23.6 | 1.02x | yes | 1.163e-25 | deadband | 611.3 → 24.2 | yes | Fkh1_zhu |
| MCM1 | Mcm1_zhu | 11 | 2037.9 → 11.5 | 1.04x | yes | 1e-37 | deadband | 246.1 → 11.7 | yes | Mcm1_zhu |
| RAP1 | Rap1_telomeric | 29 | 2.2 → 19.6 | 0.687x | no | 1.477e+08 | secant+step-cap+rho-cap | 13.5 → 31.0 | yes | Rap1_zhu+Rap1_motif1+Rap1_motif2+Rap1_telomeric |
| REB1 | Reb1_badis | 44 | 3465.7 → 46.5 | 1.06x | yes | 8.888e-38 | deadband | 990.5 → 45.3 | yes | Reb1_badis |
| SKO1 | Sko1_murphy | 5 | 283.0 → 5.3 | 1.04x | yes | 3.459e-30 | deadband | 59.1 → 5.5 | yes | Sko1_murphy |
| UME6 | Ume6_zhu | 19 | 325.6 → 20.7 | 1.09x | yes | 3.38e-22 | deadband | 125.3 → 19.3 | yes | Ume6_zhu |

### P / R / F1 — tuning chrXIV+chrII (masked r00, masked r40; bw01 r00, bw01 r24)

| ref | group | masked r00 | masked final | bw01 r00 | bw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 0.021 / 0.483 / 0.040 (1352 calls) | 0.075 / 0.138 / 0.097 (107 calls) | 0.020 / 0.483 / 0.039 (1366 calls) | 0.075 / 0.138 / 0.098 (106 calls) |
| MacIsaac | CIN5 | 0.009 / 0.102 / 0.016 (560 calls) | 0.000 / 0.000 / 0.000 (65 calls) | 0.007 / 0.020 / 0.011 (134 calls) | 0.011 / 0.020 / 0.015 (87 calls) |
| MacIsaac | FHL1 | 0.003 / 0.238 / 0.006 (1535 calls) | 0.000 / 0.000 / 0.000 (32 calls) | 0.000 / 0.000 / 0.000 (238 calls) | 0.000 / 0.000 / 0.000 (40 calls) |
| MacIsaac | FKH1 | 0.006 / 0.348 / 0.013 (1233 calls) | 0.034 / 0.043 / 0.038 (29 calls) | 0.009 / 0.304 / 0.017 (812 calls) | 0.059 / 0.087 / 0.070 (34 calls) |
| MacIsaac | MCM1 | 0.003 / 0.727 / 0.007 (2428 calls) | 0.000 / 0.000 / 0.000 (19 calls) | 0.008 / 0.273 / 0.016 (375 calls) | 0.000 / 0.000 / 0.000 (15 calls) |
| MacIsaac | RAP1 | 0.000 / 0.000 / 0.000 (1 calls) | 0.000 / 0.000 / 0.000 (22 calls) | 0.000 / 0.000 / 0.000 (29 calls) | 0.017 / 0.034 / 0.023 (58 calls) |
| MacIsaac | REB1 | 0.011 / 0.886 / 0.021 (3676 calls) | 0.036 / 0.045 / 0.040 (55 calls) | 0.028 / 0.795 / 0.054 (1253 calls) | 0.129 / 0.182 / 0.151 (62 calls) |
| MacIsaac | SKO1 | 0.004 / 0.400 / 0.009 (457 calls) | 0.000 / 0.000 / 0.000 (7 calls) | 0.000 / 0.000 / 0.000 (104 calls) | 0.000 / 0.000 / 0.000 (10 calls) |
| MacIsaac | UME6 | 0.014 / 0.368 / 0.027 (492 calls) | 0.071 / 0.105 / 0.085 (28 calls) | 0.019 / 0.211 / 0.036 (206 calls) | 0.000 / 0.000 / 0.000 (34 calls) |
| MacIsaac | set total (9 groups) | 0.009 / 0.394 / 0.017 (11734 calls) | 0.036 / 0.050 / 0.042 (364 calls) | 0.017 / 0.301 / 0.033 (4517 calls) | 0.045 / 0.077 / 0.057 (446 calls) |
| Rossi _CX | ABF1 | 0.030 / 0.408 / 0.055 (1352 calls) | 0.093 / 0.102 / 0.098 (107 calls) | 0.029 / 0.398 / 0.053 (1366 calls) | 0.085 / 0.092 / 0.088 (106 calls) |
| Rossi _CX | CIN5 | 0.013 / 0.104 / 0.022 (560 calls) | 0.015 / 0.015 / 0.015 (65 calls) | 0.015 / 0.030 / 0.020 (134 calls) | 0.000 / 0.000 / 0.000 (87 calls) |
| Rossi _CX | FHL1 | 0.015 / 0.295 / 0.029 (1535 calls) | 0.031 / 0.013 / 0.018 (32 calls) | 0.017 / 0.051 / 0.025 (238 calls) | 0.000 / 0.000 / 0.000 (40 calls) |
| Rossi _CX | FKH1 | 0.022 / 0.338 / 0.041 (1233 calls) | 0.069 / 0.025 / 0.037 (29 calls) | 0.022 / 0.225 / 0.040 (812 calls) | 0.176 / 0.075 / 0.105 (34 calls) |
| Rossi _CX | MCM1 | 0.011 / 0.510 / 0.021 (2428 calls) | 0.000 / 0.000 / 0.000 (19 calls) | 0.029 / 0.216 / 0.052 (375 calls) | 0.000 / 0.000 / 0.000 (15 calls) |
| Rossi _CX | RAP1 | 0.000 / 0.000 / 0.000 (1 calls) | 0.000 / 0.000 / 0.000 (22 calls) | 0.069 / 0.045 / 0.055 (29 calls) | 0.052 / 0.068 / 0.059 (58 calls) |
| Rossi _CX | REB1 | 0.026 / 0.703 / 0.051 (3676 calls) | 0.091 / 0.036 / 0.052 (55 calls) | 0.062 / 0.565 / 0.112 (1253 calls) | 0.242 / 0.109 / 0.150 (62 calls) |
| Rossi _CX | SKO1 | 0.009 / 0.148 / 0.017 (457 calls) | 0.000 / 0.000 / 0.000 (7 calls) | 0.019 / 0.074 / 0.031 (104 calls) | 0.000 / 0.000 / 0.000 (10 calls) |
| Rossi _CX | UME6 | 0.028 / 0.326 / 0.052 (492 calls) | 0.071 / 0.047 / 0.056 (28 calls) | 0.049 / 0.233 / 0.080 (206 calls) | 0.029 / 0.023 / 0.026 (34 calls) |
| Rossi _CX | set total (9 groups) | 0.020 / 0.380 / 0.039 (11734 calls) | 0.058 / 0.034 / 0.042 (364 calls) | 0.037 / 0.265 / 0.065 (4517 calls) | 0.076 / 0.054 / 0.063 (446 calls) |

### P / R / F1 — holdout chrIV (masked r00, masked r40; bw01 r00, bw01 r24)

| ref | group | masked r00 | masked final | bw01 r00 | bw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 0.012 / 0.409 / 0.023 (1538 calls) | 0.027 / 0.091 / 0.041 (150 calls) | 0.012 / 0.432 / 0.024 (1531 calls) | 0.027 / 0.091 / 0.042 (147 calls) |
| MacIsaac | CIN5 | 0.008 / 0.098 / 0.014 (522 calls) | 0.012 / 0.024 / 0.016 (82 calls) | 0.018 / 0.049 / 0.026 (110 calls) | 0.048 / 0.098 / 0.064 (84 calls) |
| MacIsaac | FHL1 | 0.006 / 0.450 / 0.012 (1471 calls) | 0.000 / 0.000 / 0.000 (18 calls) | 0.004 / 0.050 / 0.008 (225 calls) | 0.000 / 0.000 / 0.000 (42 calls) |
| MacIsaac | FKH1 | 0.014 / 0.600 / 0.028 (1262 calls) | 0.000 / 0.000 / 0.000 (30 calls) | 0.018 / 0.500 / 0.035 (830 calls) | 0.071 / 0.067 / 0.069 (28 calls) |
| MacIsaac | MCM1 | 0.002 / 0.667 / 0.003 (2371 calls) | 0.000 / 0.000 / 0.000 (23 calls) | 0.003 / 0.167 / 0.006 (357 calls) | 0.000 / 0.000 / 0.000 (17 calls) |
| MacIsaac | RAP1 | 0.000 / 0.000 / 0.000 (3 calls) | 0.000 / 0.000 / 0.000 (12 calls) | 0.000 / 0.000 / 0.000 (17 calls) | 0.021 / 0.033 / 0.026 (48 calls) |
| MacIsaac | REB1 | 0.009 / 1.000 / 0.018 (3507 calls) | 0.032 / 0.062 / 0.042 (63 calls) | 0.022 / 0.875 / 0.042 (1291 calls) | 0.109 / 0.188 / 0.138 (55 calls) |
| MacIsaac | SKO1 | 0.000 / 0.000 / 0.000 (451 calls) | 0.000 / 0.000 / 0.000 (8 calls) | 0.000 / 0.000 / 0.000 (90 calls) | 0.000 / 0.000 / 0.000 (6 calls) |
| MacIsaac | UME6 | 0.020 / 0.556 / 0.039 (490 calls) | 0.053 / 0.056 / 0.054 (19 calls) | 0.021 / 0.222 / 0.038 (192 calls) | 0.000 / 0.000 / 0.000 (21 calls) |
| MacIsaac | set total (9 groups) | 0.008 / 0.420 / 0.016 (11615 calls) | 0.020 / 0.035 / 0.025 (405 calls) | 0.015 / 0.310 / 0.029 (4643 calls) | 0.038 / 0.075 / 0.050 (448 calls) |
| Rossi _CX | ABF1 | 0.020 / 0.301 / 0.038 (1538 calls) | 0.033 / 0.049 / 0.040 (150 calls) | 0.020 / 0.301 / 0.038 (1531 calls) | 0.034 / 0.049 / 0.040 (147 calls) |
| Rossi _CX | CIN5 | 0.011 / 0.125 / 0.021 (522 calls) | 0.024 / 0.042 / 0.031 (82 calls) | 0.018 / 0.042 / 0.025 (110 calls) | 0.036 / 0.062 / 0.045 (84 calls) |
| Rossi _CX | FHL1 | 0.007 / 0.227 / 0.013 (1471 calls) | 0.000 / 0.000 / 0.000 (18 calls) | 0.004 / 0.023 / 0.007 (225 calls) | 0.024 / 0.023 / 0.023 (42 calls) |
| Rossi _CX | FKH1 | 0.027 / 0.557 / 0.051 (1262 calls) | 0.033 / 0.016 / 0.022 (30 calls) | 0.037 / 0.508 / 0.070 (830 calls) | 0.107 / 0.049 / 0.067 (28 calls) |
| Rossi _CX | MCM1 | 0.007 / 0.486 / 0.014 (2371 calls) | 0.000 / 0.000 / 0.000 (23 calls) | 0.008 / 0.086 / 0.015 (357 calls) | 0.000 / 0.000 / 0.000 (17 calls) |
| Rossi _CX | RAP1 | 0.000 / 0.000 / 0.000 (3 calls) | 0.083 / 0.023 / 0.036 (12 calls) | 0.059 / 0.023 / 0.033 (17 calls) | 0.083 / 0.091 / 0.087 (48 calls) |
| Rossi _CX | REB1 | 0.026 / 0.800 / 0.051 (3507 calls) | 0.048 / 0.026 / 0.034 (63 calls) | 0.055 / 0.617 / 0.101 (1291 calls) | 0.218 / 0.104 / 0.141 (55 calls) |
| Rossi _CX | SKO1 | 0.011 / 0.161 / 0.021 (451 calls) | 0.000 / 0.000 / 0.000 (8 calls) | 0.022 / 0.065 / 0.033 (90 calls) | 0.000 / 0.000 / 0.000 (6 calls) |
| Rossi _CX | UME6 | 0.029 / 0.500 / 0.054 (490 calls) | 0.053 / 0.036 / 0.043 (19 calls) | 0.036 / 0.250 / 0.064 (192 calls) | 0.048 / 0.036 / 0.041 (21 calls) |
| Rossi _CX | set total (9 groups) | 0.018 / 0.411 / 0.034 (11615 calls) | 0.032 / 0.026 / 0.028 (405 calls) | 0.032 / 0.293 / 0.058 (4643 calls) | 0.065 / 0.057 / 0.061 (448 calls) |

### Per round (bf09)

| round | within 1.1× | MacIsaac P/R/F1 (set) | Rossi P/R/F1 (set) | nucleosome copies | unknown occ (masked: expect 0) | non-deadband steps |
|---|---|---|---|---|---|---|
| r00 | 0/9 | 0.009 / 0.394 / 0.017 (11734 calls) | 0.020 / 0.380 / 0.039 (11734 calls) | 8925 (+0.00%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r01 | 0/9 | 0.009 / 0.375 / 0.018 (10427 calls) | 0.022 / 0.361 / 0.041 (10427 calls) | 8931 (+0.07%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r02 | 0/9 | 0.010 / 0.347 / 0.019 (9292 calls) | 0.023 / 0.339 / 0.043 (9292 calls) | 8934 (+0.10%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r03 | 0/9 | 0.010 / 0.328 / 0.020 (8362 calls) | 0.024 / 0.319 / 0.045 (8362 calls) | 8937 (+0.13%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r04 | 0/9 | 0.010 / 0.286 / 0.019 (7485 calls) | 0.024 / 0.286 / 0.044 (7485 calls) | 8940 (+0.17%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r05 | 0/9 | 0.011 / 0.282 / 0.021 (6730 calls) | 0.025 / 0.270 / 0.046 (6730 calls) | 8943 (+0.21%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r06 | 0/9 | 0.011 / 0.266 / 0.022 (6112 calls) | 0.027 / 0.260 / 0.048 (6112 calls) | 8946 (+0.24%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r07 | 0/9 | 0.012 / 0.266 / 0.024 (5575 calls) | 0.028 / 0.249 / 0.050 (5575 calls) | 8947 (+0.25%) | 0 | ABF1 secant, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r08 | 1/9 | 0.013 / 0.255 / 0.025 (5054 calls) | 0.029 / 0.235 / 0.052 (5054 calls) | 8948 (+0.26%) | 0 | CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r09 | 0/9 | 0.013 / 0.239 / 0.025 (4624 calls) | 0.030 / 0.224 / 0.053 (4624 calls) | 8948 (+0.26%) | 0 | ABF1 secant, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r10 | 1/9 | 0.014 / 0.228 / 0.026 (4248 calls) | 0.032 / 0.217 / 0.056 (4248 calls) | 8948 (+0.26%) | 0 | CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r11 | 1/9 | 0.014 / 0.212 / 0.027 (3834 calls) | 0.033 / 0.200 / 0.056 (3834 calls) | 8948 (+0.26%) | 0 | CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r12 | 1/9 | 0.015 / 0.197 / 0.027 (3467 calls) | 0.034 / 0.187 / 0.057 (3467 calls) | 8948 (+0.26%) | 0 | CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r13 | 1/9 | 0.015 / 0.181 / 0.028 (3137 calls) | 0.034 / 0.173 / 0.057 (3137 calls) | 8948 (+0.26%) | 0 | CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r14 | 1/9 | 0.017 / 0.181 / 0.030 (2823 calls) | 0.037 / 0.166 / 0.060 (2823 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r15 | 1/9 | 0.016 / 0.158 / 0.029 (2535 calls) | 0.036 / 0.147 / 0.058 (2535 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r16 | 1/9 | 0.017 / 0.147 / 0.030 (2303 calls) | 0.038 / 0.141 / 0.060 (2303 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r17 | 1/9 | 0.017 / 0.135 / 0.030 (2094 calls) | 0.040 / 0.134 / 0.062 (2094 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r18 | 1/9 | 0.018 / 0.135 / 0.032 (1915 calls) | 0.041 / 0.126 / 0.062 (1915 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r19 | 1/9 | 0.018 / 0.120 / 0.031 (1725 calls) | 0.041 / 0.112 / 0.060 (1725 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r20 | 1/9 | 0.019 / 0.116 / 0.033 (1542 calls) | 0.042 / 0.104 / 0.060 (1542 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r21 | 1/9 | 0.021 / 0.112 / 0.036 (1364 calls) | 0.046 / 0.101 / 0.063 (1364 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r22 | 1/9 | 0.024 / 0.112 / 0.039 (1234 calls) | 0.049 / 0.096 / 0.065 (1234 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r23 | 1/9 | 0.024 / 0.104 / 0.039 (1121 calls) | 0.050 / 0.089 / 0.064 (1121 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r24 | 1/9 | 0.027 / 0.104 / 0.043 (995 calls) | 0.049 / 0.078 / 0.060 (995 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r25 | 1/9 | 0.029 / 0.100 / 0.045 (894 calls) | 0.050 / 0.072 / 0.059 (894 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant, UME6 secant |
| r26 | 2/9 | 0.030 / 0.097 / 0.046 (821 calls) | 0.055 / 0.072 / 0.062 (821 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant, UME6 secant |
| r27 | 1/9 | 0.032 / 0.093 / 0.048 (740 calls) | 0.061 / 0.072 / 0.066 (740 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant, UME6 secant |
| r28 | 3/9 | 0.036 / 0.093 / 0.051 (675 calls) | 0.064 / 0.069 / 0.066 (675 calls) | 8948 (+0.26%) | 0 | CIN5 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant, UME6 secant |
| r29 | 3/9 | 0.032 / 0.077 / 0.045 (623 calls) | 0.061 / 0.061 / 0.061 (623 calls) | 8948 (+0.26%) | 0 | FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant |
| r30 | 3/9 | 0.032 / 0.069 / 0.044 (567 calls) | 0.063 / 0.058 / 0.060 (567 calls) | 8948 (+0.26%) | 0 | CIN5 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant, UME6 secant |
| r31 | 3/9 | 0.031 / 0.062 / 0.042 (512 calls) | 0.057 / 0.046 / 0.051 (512 calls) | 8948 (+0.26%) | 0 | FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant |
| r32 | 3/9 | 0.033 / 0.062 / 0.043 (482 calls) | 0.056 / 0.043 / 0.049 (482 calls) | 8948 (+0.26%) | 0 | CIN5 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, SKO1 secant, UME6 secant |
| r33 | 5/9 | 0.034 / 0.058 / 0.043 (445 calls) | 0.056 / 0.040 / 0.047 (445 calls) | 8948 (+0.26%) | 0 | FHL1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap |
| r34 | 3/9 | 0.035 / 0.058 / 0.044 (425 calls) | 0.059 / 0.040 / 0.048 (425 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap, UME6 secant |
| r35 | 6/9 | 0.036 / 0.054 / 0.043 (393 calls) | 0.059 / 0.037 / 0.045 (393 calls) | 8948 (+0.26%) | 0 | MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant+step-cap |
| r36 | 3/9 | 0.034 / 0.050 / 0.041 (377 calls) | 0.056 / 0.034 / 0.042 (377 calls) | 8948 (+0.26%) | 0 | CIN5 secant, FHL1 secant, MCM1 secant+step-cap, RAP1 secant+step-cap+rho-cap, REB1 secant, SKO1 secant |
| r37 | 7/9 | 0.035 / 0.050 / 0.041 (369 calls) | 0.057 / 0.034 / 0.042 (369 calls) | 8948 (+0.26%) | 0 | RAP1 secant+step-cap+rho-cap, REB1 secant |
| r38 | 7/9 | 0.035 / 0.050 / 0.042 (367 calls) | 0.057 / 0.034 / 0.042 (367 calls) | 8948 (+0.26%) | 0 | FKH1 secant, RAP1 secant+step-cap+rho-cap |
| r39 | 7/9 | 0.036 / 0.050 / 0.042 (364 calls) | 0.058 / 0.034 / 0.042 (364 calls) | 8948 (+0.26%) | 0 | RAP1 secant+step-cap+rho-cap, SKO1 secant |
| r40 | 8/9 | 0.036 / 0.050 / 0.042 (364 calls) | 0.058 / 0.034 / 0.042 (364 calls) | 8948 (+0.26%) | 0 | RAP1 secant+step-cap+rho-cap |

state: stopped after iter 40: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12611365", "decode": "12611364", "holdout_count": "12611744", "holdout_decode": "12611743", "holdout_validate": "12611745", "next": "12611366"}, "1": {"count": "12611700", "decode": "12611699", "next": "12611701"}, "10": {"count": "12626957", "decode": "12626956", "next": "12626958"}, "11": {"count": "12627136", "decode": "12627135", "next": "12627137"}, "12": {"count": "12628270", "decode": "12628269", "next": "12628271"}, "13": {"count": "12628408", "decode": "12628407", "next": "12628409"}, "14": {"count": "12628546", "decode": "12628545", "next": "12628547"}, "15": {"count": "12629557", "decode": "12629556", "next": "12629558"}, "16": {"count": "12630561", "decode": "12630560", "next": "12630562"}, "17": {"count": "12631590", "decode": "12631589", "next": "12631591"}, "18": {"count": "12634692", "decode": "12634691", "next": "12634693"}, "19": {"count": "12636021", "decode": "12636020", "next": "12636022"}, "2": {"count": "12612080", "decode": "12612079", "next": "12612081"}, "20": {"count": "12636247", "decode": "12636246", "next": "12636248"}, "21": {"count": "12636736", "decode": "12636735", "next": "12636737"}, "22": {"count": "12636978", "decode": "12636977", "next": "12636979"}, "23": {"count": "12637223", "decode": "12637191", "next": "12637224"}, "24": {"chereji": "12638005", "count": "12637351", "decode": "12637350", "holdout_count": "12638003", "holdout_decode": "12638002", "holdout_validate": "12638004", "next": "12637352", "summary": "12638006"}, "25": {"count": "12640134", "decode": "12640133", "next": "12640135"}, "26": {"count": "12640247", "decode": "12640246", "next": "12640248"}, "27": {"count": "12640335", "decode": "12640334", "next": "12640336"}, "28": {"count": "12640441", "decode": "12640440", "next": "12640442"}, "29": {"count": "12640531", "decode": "12640530", "next": "12640532"}, "3": {"count": "12612578", "decode": "12612577", "next": "12612579"}, "30": {"count": "12640665", "decode": "12640664", "next": "12640666"}, "31": {"count": "12640790", "decode": "12640789", "next": "12640791"}, "32": {"count": "12640884", "decode": "12640883", "next": "12640885"}, "33": {"count": "12640974", "decode": "12640973", "next": "12640975"}, "34": {"count": "12643653", "decode": "12643652", "next": "12643654"}, "35": {"count": "12643754", "decode": "12643753", "next": "12643755"}, "36": {"count": "12643855", "decode": "12643854", "next": "12643856"}, "37": {"count": "12643966", "decode": "12643965", "next": "12643967"}, "38": {"count": "12644272", "decode": "12644271", "next": "12644273"}, "39": {"count": "12645007", "decode": "12645006", "next": "12645008"}, "4": {"count": "12612864", "decode": "12612863", "next": "12612865"}, "40": {"chereji": "12646631", "count": "12646069", "decode": "12646068", "holdout_count": "12646629", "holdout_decode": "12646628", "holdout_validate": "12646630", "next": "12646070", "summary": "12646632"}, "5": {"count": "12613163", "decode": "12613162", "next": "12613164"}, "6": {"count": "12613437", "decode": "12613436", "next": "12613438"}, "7": {"count": "12613704", "decode": "12613703", "next": "12613705"}, "8": {"count": "12613946", "decode": "12613945", "next": "12613947"}, "9": {"count": "12626770", "decode": "12626769", "next": "12626771"}}`

## sa01 — sequence only, abf1only (vs sw01)

### Counts (masked r00 → r05; counterpart sw01 r00 → r07)

| group | live motif(s) | T | E masked | (E+1)/(T+1) final | within 1.1× | λ next | last step | E sw01 | sw01 within 1.1× | sw01 motifs |
|---|---|---|---|---|---|---|---|---|---|---|
| ABF1 | Abf1_murphy | 58 | 0.2 → 54.4 | 0.939x | yes | 3648 | deadband | 0.2 → 54.7 | yes | Abf1_murphy |

### P / R / F1 — tuning chrXIV+chrII (masked r00, masked r05; sw01 r00, sw01 r07)

| ref | group | masked r00 | masked final | sw01 r00 | sw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 1.000 / 0.017 / 0.034 (1 calls) | 0.162 / 0.276 / 0.204 (99 calls) | 1.000 / 0.017 / 0.034 (1 calls) | 0.163 / 0.293 / 0.210 (104 calls) |
| MacIsaac | set total (1 groups) | 1.000 / 0.017 / 0.034 (1 calls) | 0.162 / 0.276 / 0.204 (99 calls) | 1.000 / 0.017 / 0.034 (1 calls) | 0.163 / 0.293 / 0.210 (104 calls) |
| Rossi _CX | ABF1 | 1.000 / 0.010 / 0.020 (1 calls) | 0.253 / 0.255 / 0.254 (99 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 0.250 / 0.265 / 0.257 (104 calls) |
| Rossi _CX | set total (1 groups) | 1.000 / 0.010 / 0.020 (1 calls) | 0.253 / 0.255 / 0.254 (99 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 0.250 / 0.265 / 0.257 (104 calls) |

### P / R / F1 — holdout chrIV (masked r00, masked r05; sw01 r00, sw01 r07)

| ref | group | masked r00 | masked final | sw01 r00 | sw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | – / 0.000 / – (0 calls) | 0.124 / 0.295 / 0.174 (105 calls) | – / 0.000 / – (0 calls) | 0.137 / 0.318 / 0.192 (102 calls) |
| MacIsaac | set total (1 groups) | – / 0.000 / – (0 calls) | 0.124 / 0.295 / 0.174 (105 calls) | – / 0.000 / – (0 calls) | 0.137 / 0.318 / 0.192 (102 calls) |
| Rossi _CX | ABF1 | – / 0.000 / – (0 calls) | 0.257 / 0.262 / 0.260 (105 calls) | – / 0.000 / – (0 calls) | 0.275 / 0.272 / 0.273 (102 calls) |
| Rossi _CX | set total (1 groups) | – / 0.000 / – (0 calls) | 0.257 / 0.262 / 0.260 (105 calls) | – / 0.000 / – (0 calls) | 0.275 / 0.272 / 0.273 (102 calls) |

### Per round (sa01)

| round | within 1.1× | MacIsaac P/R/F1 (set) | Rossi P/R/F1 (set) | nucleosome copies | unknown occ (masked: expect 0) | non-deadband steps |
|---|---|---|---|---|---|---|
| r00 | 0/1 | 1.000 / 0.017 / 0.034 (1 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 9444 (+0.00%) | 0 | ABF1 no-slope |
| r01 | 0/1 | 0.500 / 0.017 / 0.033 (2 calls) | 0.500 / 0.010 / 0.020 (2 calls) | 9444 (-0.00%) | 0 | ABF1 secant+step-cap |
| r02 | 0/1 | 0.182 / 0.034 / 0.058 (11 calls) | 0.545 / 0.061 / 0.110 (11 calls) | 9443 (-0.01%) | 0 | ABF1 secant+step-cap |
| r03 | 0/1 | 0.164 / 0.155 / 0.159 (55 calls) | 0.291 / 0.163 / 0.209 (55 calls) | 9439 (-0.06%) | 0 | ABF1 secant |
| r04 | 0/1 | 0.152 / 0.241 / 0.187 (92 calls) | 0.250 / 0.235 / 0.242 (92 calls) | 9434 (-0.11%) | 0 | ABF1 secant |
| r05 | 1/1 | 0.162 / 0.276 / 0.204 (99 calls) | 0.253 / 0.255 / 0.254 (99 calls) | 9433 (-0.12%) | 0 | all deadband |

state: stopped after iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12611408", "decode": "12611407", "holdout_count": "12611826", "holdout_decode": "12611825", "holdout_validate": "12611827", "next": "12611409"}, "1": {"count": "12611811", "decode": "12611810", "next": "12611812"}, "2": {"count": "12612374", "decode": "12612373", "next": "12612375"}, "3": {"count": "12612677", "decode": "12612676", "next": "12612678"}, "4": {"count": "12612960", "decode": "12612959", "next": "12612961"}, "5": {"chereji": "12613590", "count": "12613303", "decode": "12613302", "holdout_count": "12613588", "holdout_decode": "12613587", "holdout_validate": "12613589", "next": "12613304", "summary": "12613591"}}`

## sf09 — sequence only, fit9 (vs sw01)

### Counts (masked r00 → r10; counterpart sw01 r00 → r07)

| group | live motif(s) | T | E masked | (E+1)/(T+1) final | within 1.1× | λ next | last step | E sw01 | sw01 within 1.1× | sw01 motifs |
|---|---|---|---|---|---|---|---|---|---|---|
| ABF1 | Abf1_murphy | 58 | 0.2 → 54.3 | 0.937x | yes | 3587 | deadband | 0.2 → 54.7 | yes | Abf1_murphy |
| CIN5 | Cin5_murphy | 49 | 0.2 → 47.6 | 0.972x | yes | 519.8 | deadband | 0.2 → 47.9 | yes | Cin5_murphy |
| FHL1 | Fhl1_zhu | 21 | 8.5 → 22.1 | 1.05x | yes | 2.814 | deadband | 9.1 → 22.3 | yes | Fhl1_zhu |
| FKH1 | Fkh1_zhu | 23 | 1.5 → 22.0 | 0.958x | yes | 16.78 | deadband | 1.6 → 23.2 | yes | Fkh1_zhu |
| MCM1 | Mcm1_zhu | 11 | 0.3 → 11.9 | 1.07x | yes | 27.8 | deadband | 0.3 → 11.9 | yes | Mcm1_zhu |
| RAP1 | Rap1_telomeric | 29 | 0.1 → 22.3 | 0.776x | no | 1.477e+08 | secant+rho-cap | 2.5 → 27.8 | yes | Rap1_zhu+Rap1_motif1+Rap1_motif2+Rap1_telomeric |
| REB1 | Reb1_badis | 44 | 5.2 → 42.4 | 0.964x | yes | 10 | deadband | 5.7 → 45.5 | yes | Reb1_badis |
| SKO1 | Sko1_murphy | 5 | 3.0 → 4.7 | 0.949x | yes | 1.793 | deadband | 3.2 → 4.6 | yes | Sko1_murphy |
| UME6 | Ume6_zhu | 19 | 5.0 → 20.8 | 1.09x | yes | 4.987 | deadband | 5.3 → 20.7 | yes | Ume6_zhu |

### P / R / F1 — tuning chrXIV+chrII (masked r00, masked r10; sw01 r00, sw01 r07)

| ref | group | masked r00 | masked final | sw01 r00 | sw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 1.000 / 0.017 / 0.034 (1 calls) | 0.170 / 0.293 / 0.215 (100 calls) | 1.000 / 0.017 / 0.034 (1 calls) | 0.163 / 0.293 / 0.210 (104 calls) |
| MacIsaac | CIN5 | – / 0.000 / – (0 calls) | 0.083 / 0.163 / 0.110 (96 calls) | – / 0.000 / – (0 calls) | 0.085 / 0.184 / 0.116 (106 calls) |
| MacIsaac | FHL1 | 0.000 / 0.000 / 0.000 (12 calls) | 0.000 / 0.000 / 0.000 (41 calls) | 0.000 / 0.000 / 0.000 (14 calls) | 0.000 / 0.000 / 0.000 (39 calls) |
| MacIsaac | FKH1 | – / 0.000 / – (0 calls) | 0.040 / 0.087 / 0.055 (50 calls) | – / 0.000 / – (0 calls) | 0.074 / 0.174 / 0.104 (54 calls) |
| MacIsaac | MCM1 | – / 0.000 / – (0 calls) | 0.071 / 0.091 / 0.080 (14 calls) | – / 0.000 / – (0 calls) | 0.000 / 0.000 / 0.000 (13 calls) |
| MacIsaac | RAP1 | – / 0.000 / – (0 calls) | 0.042 / 0.034 / 0.038 (24 calls) | 0.100 / 0.034 / 0.051 (10 calls) | 0.050 / 0.069 / 0.058 (40 calls) |
| MacIsaac | REB1 | 0.500 / 0.045 / 0.083 (4 calls) | 0.279 / 0.659 / 0.392 (104 calls) | 0.429 / 0.068 / 0.118 (7 calls) | 0.270 / 0.705 / 0.390 (115 calls) |
| MacIsaac | SKO1 | 0.000 / 0.000 / 0.000 (7 calls) | 0.000 / 0.000 / 0.000 (12 calls) | 0.000 / 0.000 / 0.000 (11 calls) | 0.000 / 0.000 / 0.000 (13 calls) |
| MacIsaac | UME6 | 0.273 / 0.158 / 0.200 (11 calls) | 0.217 / 0.526 / 0.308 (46 calls) | 0.250 / 0.158 / 0.194 (12 calls) | 0.200 / 0.526 / 0.290 (50 calls) |
| MacIsaac | set total (9 groups) | 0.171 / 0.023 / 0.041 (35 calls) | 0.140 / 0.263 / 0.182 (487 calls) | 0.145 / 0.031 / 0.051 (55 calls) | 0.137 / 0.282 / 0.184 (534 calls) |
| Rossi _CX | ABF1 | 1.000 / 0.010 / 0.020 (1 calls) | 0.260 / 0.265 / 0.263 (100 calls) | 1.000 / 0.010 / 0.020 (1 calls) | 0.250 / 0.265 / 0.257 (104 calls) |
| Rossi _CX | CIN5 | – / 0.000 / – (0 calls) | 0.104 / 0.149 / 0.123 (96 calls) | – / 0.000 / – (0 calls) | 0.142 / 0.224 / 0.173 (106 calls) |
| Rossi _CX | FHL1 | 0.000 / 0.000 / 0.000 (12 calls) | 0.000 / 0.000 / 0.000 (41 calls) | 0.000 / 0.000 / 0.000 (14 calls) | 0.000 / 0.000 / 0.000 (39 calls) |
| Rossi _CX | FKH1 | – / 0.000 / – (0 calls) | 0.080 / 0.050 / 0.062 (50 calls) | – / 0.000 / – (0 calls) | 0.130 / 0.087 / 0.104 (54 calls) |
| Rossi _CX | MCM1 | – / 0.000 / – (0 calls) | 0.143 / 0.039 / 0.062 (14 calls) | – / 0.000 / – (0 calls) | 0.077 / 0.020 / 0.031 (13 calls) |
| Rossi _CX | RAP1 | – / 0.000 / – (0 calls) | 0.042 / 0.023 / 0.029 (24 calls) | 0.100 / 0.023 / 0.037 (10 calls) | 0.125 / 0.114 / 0.119 (40 calls) |
| Rossi _CX | REB1 | 1.000 / 0.029 / 0.056 (4 calls) | 0.452 / 0.341 / 0.388 (104 calls) | 1.000 / 0.051 / 0.097 (7 calls) | 0.452 / 0.377 / 0.411 (115 calls) |
| Rossi _CX | SKO1 | 0.143 / 0.037 / 0.059 (7 calls) | 0.167 / 0.074 / 0.103 (12 calls) | 0.182 / 0.074 / 0.105 (11 calls) | 0.154 / 0.074 / 0.100 (13 calls) |
| Rossi _CX | UME6 | 0.636 / 0.163 / 0.259 (11 calls) | 0.435 / 0.465 / 0.449 (46 calls) | 0.583 / 0.163 / 0.255 (12 calls) | 0.400 / 0.465 / 0.430 (50 calls) |
| Rossi _CX | set total (9 groups) | 0.371 / 0.021 / 0.039 (35 calls) | 0.230 / 0.179 / 0.201 (487 calls) | 0.327 / 0.029 / 0.053 (55 calls) | 0.240 / 0.204 / 0.221 (534 calls) |

### P / R / F1 — holdout chrIV (masked r00, masked r10; sw01 r00, sw01 r07)

| ref | group | masked r00 | masked final | sw01 r00 | sw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | – / 0.000 / – (0 calls) | 0.120 / 0.273 / 0.167 (100 calls) | – / 0.000 / – (0 calls) | 0.137 / 0.318 / 0.192 (102 calls) |
| MacIsaac | CIN5 | – / 0.000 / – (0 calls) | 0.103 / 0.293 / 0.153 (116 calls) | – / 0.000 / – (0 calls) | 0.117 / 0.341 / 0.174 (120 calls) |
| MacIsaac | FHL1 | 0.000 / 0.000 / 0.000 (15 calls) | 0.000 / 0.000 / 0.000 (47 calls) | 0.000 / 0.000 / 0.000 (17 calls) | 0.000 / 0.000 / 0.000 (46 calls) |
| MacIsaac | FKH1 | – / 0.000 / – (0 calls) | 0.143 / 0.200 / 0.167 (42 calls) | – / 0.000 / – (0 calls) | 0.143 / 0.200 / 0.167 (42 calls) |
| MacIsaac | MCM1 | – / 0.000 / – (0 calls) | 0.000 / 0.000 / 0.000 (7 calls) | – / 0.000 / – (0 calls) | 0.000 / 0.000 / 0.000 (9 calls) |
| MacIsaac | RAP1 | – / 0.000 / – (0 calls) | 0.000 / 0.000 / 0.000 (8 calls) | 0.000 / 0.000 / 0.000 (4 calls) | 0.125 / 0.133 / 0.129 (32 calls) |
| MacIsaac | REB1 | – / 0.000 / – (0 calls) | 0.278 / 0.688 / 0.396 (79 calls) | 0.000 / 0.000 / 0.000 (1 calls) | 0.237 / 0.688 / 0.352 (93 calls) |
| MacIsaac | SKO1 | 0.000 / 0.000 / 0.000 (10 calls) | 0.000 / 0.000 / 0.000 (21 calls) | 0.000 / 0.000 / 0.000 (12 calls) | 0.000 / 0.000 / 0.000 (19 calls) |
| MacIsaac | UME6 | 0.200 / 0.111 / 0.143 (10 calls) | 0.121 / 0.222 / 0.157 (33 calls) | 0.182 / 0.111 / 0.138 (11 calls) | 0.171 / 0.333 / 0.226 (35 calls) |
| MacIsaac | set total (9 groups) | 0.057 / 0.009 / 0.015 (35 calls) | 0.124 / 0.248 / 0.165 (453 calls) | 0.044 / 0.009 / 0.015 (45 calls) | 0.133 / 0.292 / 0.182 (498 calls) |
| Rossi _CX | ABF1 | – / 0.000 / – (0 calls) | 0.260 / 0.252 / 0.256 (100 calls) | – / 0.000 / – (0 calls) | 0.275 / 0.272 / 0.273 (102 calls) |
| Rossi _CX | CIN5 | – / 0.000 / – (0 calls) | 0.069 / 0.167 / 0.098 (116 calls) | – / 0.000 / – (0 calls) | 0.075 / 0.188 / 0.107 (120 calls) |
| Rossi _CX | FHL1 | 0.000 / 0.000 / 0.000 (15 calls) | 0.000 / 0.000 / 0.000 (47 calls) | 0.000 / 0.000 / 0.000 (17 calls) | 0.000 / 0.000 / 0.000 (46 calls) |
| Rossi _CX | FKH1 | – / 0.000 / – (0 calls) | 0.167 / 0.115 / 0.136 (42 calls) | – / 0.000 / – (0 calls) | 0.167 / 0.115 / 0.136 (42 calls) |
| Rossi _CX | MCM1 | – / 0.000 / – (0 calls) | 0.143 / 0.029 / 0.048 (7 calls) | – / 0.000 / – (0 calls) | 0.111 / 0.029 / 0.045 (9 calls) |
| Rossi _CX | RAP1 | – / 0.000 / – (0 calls) | 0.000 / 0.000 / 0.000 (8 calls) | 0.000 / 0.000 / 0.000 (4 calls) | 0.219 / 0.159 / 0.184 (32 calls) |
| Rossi _CX | REB1 | – / 0.000 / – (0 calls) | 0.582 / 0.400 / 0.474 (79 calls) | 0.000 / 0.000 / 0.000 (1 calls) | 0.548 / 0.443 / 0.490 (93 calls) |
| Rossi _CX | SKO1 | 0.200 / 0.065 / 0.098 (10 calls) | 0.238 / 0.161 / 0.192 (21 calls) | 0.167 / 0.065 / 0.093 (12 calls) | 0.263 / 0.161 / 0.200 (19 calls) |
| Rossi _CX | UME6 | 0.500 / 0.179 / 0.263 (10 calls) | 0.303 / 0.357 / 0.328 (33 calls) | 0.455 / 0.179 / 0.256 (11 calls) | 0.371 / 0.464 / 0.413 (35 calls) |
| Rossi _CX | set total (9 groups) | 0.200 / 0.014 / 0.026 (35 calls) | 0.227 / 0.202 / 0.214 (453 calls) | 0.156 / 0.014 / 0.025 (45 calls) | 0.243 / 0.238 / 0.240 (498 calls) |

### Per round (sf09)

| round | within 1.1× | MacIsaac P/R/F1 (set) | Rossi P/R/F1 (set) | nucleosome copies | unknown occ (masked: expect 0) | non-deadband steps |
|---|---|---|---|---|---|---|
| r00 | 0/9 | 0.171 / 0.023 / 0.041 (35 calls) | 0.371 / 0.021 / 0.039 (35 calls) | 9440 (+0.00%) | 0 | ABF1 no-slope, CIN5 no-slope, FHL1 secant, FKH1 secant+step-cap, MCM1 no-slope, RAP1 no-slope, REB1 secant+step-cap, SKO1 secant, UME6 secant |
| r01 | 2/9 | 0.165 / 0.162 / 0.164 (254 calls) | 0.299 / 0.121 / 0.173 (254 calls) | 9426 (-0.15%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant, FKH1 secant, MCM1 secant, RAP1 secant+step-cap, UME6 secant |
| r02 | 4/9 | 0.156 / 0.185 / 0.170 (307 calls) | 0.274 / 0.134 / 0.180 (307 calls) | 9419 (-0.23%) | 0 | ABF1 secant+step-cap, CIN5 secant, FHL1 secant, MCM1 secant, RAP1 secant |
| r03 | 5/9 | 0.138 / 0.224 / 0.171 (421 calls) | 0.242 / 0.163 / 0.195 (421 calls) | 9409 (-0.33%) | 0 | ABF1 secant, CIN5 secant, MCM1 secant, RAP1 secant |
| r04 | 7/9 | 0.140 / 0.251 / 0.180 (463 calls) | 0.235 / 0.174 / 0.200 (463 calls) | 9404 (-0.39%) | 0 | ABF1 secant, RAP1 secant |
| r05 | 8/9 | 0.142 / 0.259 / 0.183 (472 calls) | 0.235 / 0.177 / 0.202 (472 calls) | 9402 (-0.41%) | 0 | RAP1 secant+step-cap |
| r06 | 8/9 | 0.142 / 0.259 / 0.183 (473 calls) | 0.235 / 0.177 / 0.202 (473 calls) | 9402 (-0.41%) | 0 | RAP1 secant+step-cap |
| r07 | 8/9 | 0.142 / 0.259 / 0.183 (473 calls) | 0.235 / 0.177 / 0.202 (473 calls) | 9402 (-0.41%) | 0 | RAP1 secant+step-cap |
| r08 | 8/9 | 0.139 / 0.259 / 0.181 (481 calls) | 0.231 / 0.177 / 0.201 (481 calls) | 9401 (-0.41%) | 0 | RAP1 secant+step-cap |
| r09 | 8/9 | 0.140 / 0.263 / 0.182 (487 calls) | 0.230 / 0.179 / 0.201 (487 calls) | 9401 (-0.42%) | 0 | RAP1 secant+rho-cap |
| r10 | 8/9 | 0.140 / 0.263 / 0.182 (487 calls) | 0.230 / 0.179 / 0.201 (487 calls) | 9401 (-0.42%) | 0 | RAP1 secant+rho-cap |

state: stopped after iter 10: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12611425", "decode": "12611424", "holdout_count": "12611829", "holdout_decode": "12611828", "holdout_validate": "12611830", "next": "12611426"}, "1": {"count": "12611816", "decode": "12611814", "next": "12611818"}, "10": {"chereji": "12627133", "count": "12626999", "decode": "12626998", "holdout_count": "12627131", "holdout_decode": "12627130", "holdout_validate": "12627132", "next": "12627000", "summary": "12627134"}, "2": {"count": "12612420", "decode": "12612419", "next": "12612421"}, "3": {"count": "12612732", "decode": "12612730", "next": "12612734"}, "4": {"count": "12613002", "decode": "12613001", "next": "12613003"}, "5": {"count": "12613261", "decode": "12613260", "next": "12613262"}, "6": {"count": "12613544", "decode": "12613543", "next": "12613545"}, "7": {"count": "12613773", "decode": "12613771", "next": "12613775"}, "8": {"count": "12613993", "decode": "12613992", "next": "12613994"}, "9": {"count": "12626773", "decode": "12626772", "next": "12626774"}}`

## fa01 — fiber only, abf1only (vs fw01)

### Counts (masked r00 → r16; counterpart fw01 r00 → r24)

| group | live motif(s) | T | E masked | (E+1)/(T+1) final | within 1.1× | λ next | last step | E fw01 | fw01 within 1.1× | fw01 motifs |
|---|---|---|---|---|---|---|---|---|---|---|
| ABF1 | Abf1_murphy | 58 | 4456.8 → 61.3 | 1.06x | yes | 2.419e-16 | deadband | 3068.8 → 59.3 | yes | Abf1_murphy |

### P / R / F1 — tuning chrXIV+chrII (masked r00, masked r16; fw01 r00, fw01 r24)

| ref | group | masked r00 | masked final | fw01 r00 | fw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 0.007 / 0.569 / 0.013 (4841 calls) | 0.019 / 0.034 / 0.024 (106 calls) | 0.006 / 0.397 / 0.011 (3946 calls) | 0.019 / 0.034 / 0.024 (106 calls) |
| MacIsaac | set total (1 groups) | 0.007 / 0.569 / 0.013 (4841 calls) | 0.019 / 0.034 / 0.024 (106 calls) | 0.006 / 0.397 / 0.011 (3946 calls) | 0.019 / 0.034 / 0.024 (106 calls) |
| Rossi _CX | ABF1 | 0.009 / 0.449 / 0.018 (4841 calls) | 0.038 / 0.041 / 0.039 (106 calls) | 0.009 / 0.347 / 0.017 (3946 calls) | 0.038 / 0.041 / 0.039 (106 calls) |
| Rossi _CX | set total (1 groups) | 0.009 / 0.449 / 0.018 (4841 calls) | 0.038 / 0.041 / 0.039 (106 calls) | 0.009 / 0.347 / 0.017 (3946 calls) | 0.038 / 0.041 / 0.039 (106 calls) |

### P / R / F1 — holdout chrIV (masked r00, masked r16; fw01 r00, fw01 r24)

| ref | group | masked r00 | masked final | fw01 r00 | fw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 0.004 / 0.455 / 0.008 (4929 calls) | 0.014 / 0.045 / 0.021 (146 calls) | 0.005 / 0.409 / 0.009 (3989 calls) | 0.014 / 0.045 / 0.022 (142 calls) |
| MacIsaac | set total (1 groups) | 0.004 / 0.455 / 0.008 (4929 calls) | 0.014 / 0.045 / 0.021 (146 calls) | 0.005 / 0.409 / 0.009 (3989 calls) | 0.014 / 0.045 / 0.022 (142 calls) |
| Rossi _CX | ABF1 | 0.008 / 0.369 / 0.015 (4929 calls) | 0.014 / 0.019 / 0.016 (146 calls) | 0.008 / 0.320 / 0.016 (3989 calls) | 0.014 / 0.019 / 0.016 (142 calls) |
| Rossi _CX | set total (1 groups) | 0.008 / 0.369 / 0.015 (4929 calls) | 0.014 / 0.019 / 0.016 (146 calls) | 0.008 / 0.320 / 0.016 (3989 calls) | 0.014 / 0.019 / 0.016 (142 calls) |

### Per round (fa01)

| round | within 1.1× | MacIsaac P/R/F1 (set) | Rossi P/R/F1 (set) | nucleosome copies | unknown occ (masked: expect 0) | non-deadband steps |
|---|---|---|---|---|---|---|
| r00 | 0/1 | 0.007 / 0.569 / 0.013 (4841 calls) | 0.009 / 0.449 / 0.018 (4841 calls) | 8819 (+0.00%) | 0 | ABF1 secant+step-cap |
| r01 | 0/1 | 0.007 / 0.534 / 0.015 (4176 calls) | 0.010 / 0.408 / 0.019 (4176 calls) | 8845 (+0.30%) | 0 | ABF1 secant+step-cap |
| r02 | 0/1 | 0.008 / 0.500 / 0.016 (3551 calls) | 0.011 / 0.388 / 0.021 (3551 calls) | 8866 (+0.54%) | 0 | ABF1 secant+step-cap |
| r03 | 0/1 | 0.009 / 0.466 / 0.018 (2923 calls) | 0.012 / 0.367 / 0.024 (2923 calls) | 8886 (+0.77%) | 0 | ABF1 secant+step-cap |
| r04 | 0/1 | 0.010 / 0.431 / 0.020 (2408 calls) | 0.014 / 0.337 / 0.026 (2408 calls) | 8899 (+0.91%) | 0 | ABF1 secant+step-cap |
| r05 | 0/1 | 0.012 / 0.397 / 0.023 (1957 calls) | 0.016 / 0.316 / 0.030 (1957 calls) | 8912 (+1.06%) | 0 | ABF1 secant+step-cap |
| r06 | 0/1 | 0.012 / 0.328 / 0.023 (1581 calls) | 0.017 / 0.276 / 0.032 (1581 calls) | 8922 (+1.17%) | 0 | ABF1 secant+step-cap |
| r07 | 0/1 | 0.014 / 0.310 / 0.028 (1242 calls) | 0.018 / 0.224 / 0.033 (1242 calls) | 8929 (+1.25%) | 0 | ABF1 secant+step-cap |
| r08 | 0/1 | 0.015 / 0.259 / 0.028 (999 calls) | 0.019 / 0.194 / 0.035 (999 calls) | 8932 (+1.28%) | 0 | ABF1 secant+step-cap |
| r09 | 0/1 | 0.017 / 0.224 / 0.031 (771 calls) | 0.021 / 0.163 / 0.037 (771 calls) | 8936 (+1.32%) | 0 | ABF1 secant+step-cap |
| r10 | 0/1 | 0.021 / 0.207 / 0.037 (585 calls) | 0.026 / 0.153 / 0.044 (585 calls) | 8938 (+1.36%) | 0 | ABF1 secant+step-cap |
| r11 | 0/1 | 0.027 / 0.207 / 0.048 (439 calls) | 0.032 / 0.143 / 0.052 (439 calls) | 8940 (+1.37%) | 0 | ABF1 secant+step-cap |
| r12 | 0/1 | 0.031 / 0.172 / 0.052 (327 calls) | 0.040 / 0.133 / 0.061 (327 calls) | 8941 (+1.38%) | 0 | ABF1 secant+step-cap |
| r13 | 0/1 | 0.025 / 0.103 / 0.041 (236 calls) | 0.038 / 0.092 / 0.054 (236 calls) | 8943 (+1.41%) | 0 | ABF1 secant+step-cap |
| r14 | 0/1 | 0.028 / 0.086 / 0.043 (177 calls) | 0.045 / 0.082 / 0.058 (177 calls) | 8945 (+1.43%) | 0 | ABF1 secant+step-cap |
| r15 | 0/1 | 0.016 / 0.034 / 0.022 (123 calls) | 0.033 / 0.041 / 0.036 (123 calls) | 8946 (+1.45%) | 0 | ABF1 secant |
| r16 | 1/1 | 0.019 / 0.034 / 0.024 (106 calls) | 0.038 / 0.041 / 0.039 (106 calls) | 8948 (+1.46%) | 0 | all deadband |

state: stopped after iter 16: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12611429", "decode": "12611428", "holdout_count": "12611822", "holdout_decode": "12611821", "holdout_validate": "12611823", "next": "12611430"}, "1": {"count": "12611817", "decode": "12611815", "next": "12611819"}, "10": {"count": "12627041", "decode": "12627040", "next": "12627042"}, "11": {"count": "12627217", "decode": "12627216", "next": "12627218"}, "12": {"count": "12628294", "decode": "12628293", "next": "12628295"}, "13": {"count": "12628450", "decode": "12628449", "next": "12628451"}, "14": {"count": "12628588", "decode": "12628587", "next": "12628589"}, "15": {"count": "12629599", "decode": "12629598", "next": "12629600"}, "16": {"chereji": "12631649", "count": "12630603", "decode": "12630602", "holdout_count": "12631647", "holdout_decode": "12631646", "holdout_validate": "12631648", "next": "12630604", "summary": "12631650"}, "2": {"count": "12612457", "decode": "12612456", "next": "12612458"}, "3": {"count": "12612733", "decode": "12612731", "next": "12612735"}, "4": {"count": "12613055", "decode": "12613054", "next": "12613057"}, "5": {"count": "12613348", "decode": "12613347", "next": "12613349"}, "6": {"count": "12613633", "decode": "12613632", "next": "12613634"}, "7": {"count": "12613858", "decode": "12613857", "next": "12613859"}, "8": {"count": "12614081", "decode": "12614080", "next": "12614082"}, "9": {"count": "12626854", "decode": "12626853", "next": "12626855"}}`

## ff09 — fiber only, fit9 (vs fw01)

### Counts (masked r00 → r40; counterpart fw01 r00 → r24)

| group | live motif(s) | T | E masked | (E+1)/(T+1) final | within 1.1× | λ next | last step | E fw01 | fw01 within 1.1× | fw01 motifs |
|---|---|---|---|---|---|---|---|---|---|---|
| ABF1 | Abf1_murphy | 58 | 3053.9 → 59.0 | 1.02x | yes | 7.128e-16 | deadband | 3068.8 → 59.3 | yes | Abf1_murphy |
| CIN5 | Cin5_murphy | 49 | 743.0 → 50.7 | 1.03x | yes | 1.147e-29 | deadband | 262.5 → 50.1 | yes | Cin5_murphy |
| FHL1 | Fhl1_zhu | 21 | 1535.8 → 22.0 | 1.05x | yes | 3.469e-28 | deadband | 346.6 → 23.2 | yes | Fhl1_zhu |
| FKH1 | Fkh1_zhu | 23 | 1142.1 → 24.7 | 1.07x | yes | 1.207e-28 | deadband | 771.9 → 23.3 | yes | Fkh1_zhu |
| MCM1 | Mcm1_zhu | 11 | 801.7 → 11.2 | 1.02x | yes | 1.636e-39 | deadband | 173.6 → 11.8 | yes | Mcm1_zhu |
| RAP1 | Rap1_telomeric | 29 | 479.6 → 28.6 | 0.988x | yes | 1.578e-21 | deadband | 492.6 → 28.7 | yes | Rap1_zhu+Rap1_motif1+Rap1_motif2+Rap1_telomeric |
| REB1 | Reb1_badis | 44 | 4678.8 → 48.1 | 1.09x | yes | 4.176e-40 | deadband | 2226.0 → 46.2 | yes | Reb1_badis |
| SKO1 | Sko1_murphy | 5 | 2051.0 → 5.1 | 1.01x | yes | 4.272e-37 | deadband | 1103.1 → 5.3 | yes | Sko1_murphy |
| UME6 | Ume6_zhu | 19 | 2276.8 → 20.5 | 1.08x | yes | 7.186e-26 | deadband | 2115.3 → 19.5 | yes | Ume6_zhu |

### P / R / F1 — tuning chrXIV+chrII (masked r00, masked r40; fw01 r00, fw01 r24)

| ref | group | masked r00 | masked final | fw01 r00 | fw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 0.006 / 0.397 / 0.012 (3917 calls) | 0.020 / 0.034 / 0.025 (102 calls) | 0.006 / 0.397 / 0.011 (3946 calls) | 0.019 / 0.034 / 0.024 (106 calls) |
| MacIsaac | CIN5 | 0.004 / 0.102 / 0.008 (1230 calls) | 0.015 / 0.020 / 0.017 (67 calls) | 0.007 / 0.061 / 0.012 (432 calls) | 0.000 / 0.000 / 0.000 (74 calls) |
| MacIsaac | FHL1 | 0.004 / 0.476 / 0.007 (2720 calls) | 0.032 / 0.048 / 0.038 (31 calls) | 0.007 / 0.238 / 0.014 (689 calls) | 0.000 / 0.000 / 0.000 (44 calls) |
| MacIsaac | FKH1 | 0.004 / 0.348 / 0.008 (1888 calls) | 0.000 / 0.000 / 0.000 (31 calls) | 0.006 / 0.348 / 0.012 (1293 calls) | 0.000 / 0.000 / 0.000 (38 calls) |
| MacIsaac | MCM1 | 0.005 / 0.545 / 0.009 (1316 calls) | 0.000 / 0.000 / 0.000 (19 calls) | 0.006 / 0.182 / 0.012 (312 calls) | 0.000 / 0.000 / 0.000 (15 calls) |
| MacIsaac | RAP1 | 0.006 / 0.172 / 0.012 (802 calls) | 0.000 / 0.000 / 0.000 (47 calls) | 0.009 / 0.241 / 0.017 (811 calls) | 0.000 / 0.000 / 0.000 (48 calls) |
| MacIsaac | REB1 | 0.008 / 0.864 / 0.015 (4993 calls) | 0.000 / 0.000 / 0.000 (57 calls) | 0.012 / 0.750 / 0.023 (2858 calls) | 0.045 / 0.068 / 0.055 (66 calls) |
| MacIsaac | SKO1 | 0.000 / 0.200 / 0.001 (3195 calls) | 0.000 / 0.000 / 0.000 (9 calls) | 0.001 / 0.200 / 0.001 (1759 calls) | 0.000 / 0.000 / 0.000 (10 calls) |
| MacIsaac | UME6 | 0.002 / 0.421 / 0.004 (3764 calls) | 0.036 / 0.053 / 0.043 (28 calls) | 0.002 / 0.421 / 0.005 (3497 calls) | 0.000 / 0.000 / 0.000 (33 calls) |
| MacIsaac | set total (9 groups) | 0.004 / 0.402 / 0.009 (23825 calls) | 0.013 / 0.019 / 0.015 (391 calls) | 0.006 / 0.347 / 0.011 (15597 calls) | 0.012 / 0.019 / 0.014 (434 calls) |
| Rossi _CX | ABF1 | 0.008 / 0.337 / 0.016 (3917 calls) | 0.039 / 0.041 / 0.040 (102 calls) | 0.009 / 0.347 / 0.017 (3946 calls) | 0.038 / 0.041 / 0.039 (106 calls) |
| Rossi _CX | CIN5 | 0.007 / 0.134 / 0.014 (1230 calls) | 0.015 / 0.015 / 0.015 (67 calls) | 0.005 / 0.030 / 0.008 (432 calls) | 0.000 / 0.000 / 0.000 (74 calls) |
| Rossi _CX | FHL1 | 0.011 / 0.385 / 0.021 (2720 calls) | 0.000 / 0.000 / 0.000 (31 calls) | 0.010 / 0.090 / 0.018 (689 calls) | 0.000 / 0.000 / 0.000 (44 calls) |
| Rossi _CX | FKH1 | 0.009 / 0.212 / 0.017 (1888 calls) | 0.032 / 0.013 / 0.018 (31 calls) | 0.013 / 0.212 / 0.025 (1293 calls) | 0.053 / 0.025 / 0.034 (38 calls) |
| Rossi _CX | MCM1 | 0.012 / 0.314 / 0.023 (1316 calls) | 0.053 / 0.020 / 0.029 (19 calls) | 0.019 / 0.118 / 0.033 (312 calls) | 0.067 / 0.020 / 0.030 (15 calls) |
| Rossi _CX | RAP1 | 0.010 / 0.182 / 0.019 (802 calls) | 0.000 / 0.000 / 0.000 (47 calls) | 0.012 / 0.227 / 0.023 (811 calls) | 0.000 / 0.000 / 0.000 (48 calls) |
| Rossi _CX | REB1 | 0.020 / 0.732 / 0.039 (4993 calls) | 0.053 / 0.022 / 0.031 (57 calls) | 0.031 / 0.652 / 0.060 (2858 calls) | 0.106 / 0.051 / 0.069 (66 calls) |
| Rossi _CX | SKO1 | 0.005 / 0.556 / 0.009 (3195 calls) | 0.000 / 0.000 / 0.000 (9 calls) | 0.004 / 0.259 / 0.008 (1759 calls) | 0.000 / 0.000 / 0.000 (10 calls) |
| Rossi _CX | UME6 | 0.003 / 0.302 / 0.007 (3764 calls) | 0.036 / 0.023 / 0.028 (28 calls) | 0.003 / 0.279 / 0.007 (3497 calls) | 0.000 / 0.000 / 0.000 (33 calls) |
| Rossi _CX | set total (9 groups) | 0.010 / 0.387 / 0.020 (23825 calls) | 0.028 / 0.018 / 0.022 (391 calls) | 0.012 / 0.296 / 0.023 (15597 calls) | 0.032 / 0.022 / 0.026 (434 calls) |

### P / R / F1 — holdout chrIV (masked r00, masked r40; fw01 r00, fw01 r24)

| ref | group | masked r00 | masked final | fw01 r00 | fw01 final |
|---|---|---|---|---|---|
| MacIsaac | ABF1 | 0.005 / 0.409 / 0.009 (3987 calls) | 0.014 / 0.045 / 0.021 (146 calls) | 0.005 / 0.409 / 0.009 (3989 calls) | 0.014 / 0.045 / 0.022 (142 calls) |
| MacIsaac | CIN5 | 0.005 / 0.146 / 0.010 (1183 calls) | 0.011 / 0.024 / 0.016 (87 calls) | 0.004 / 0.049 / 0.008 (479 calls) | 0.000 / 0.000 / 0.000 (65 calls) |
| MacIsaac | FHL1 | 0.003 / 0.350 / 0.005 (2560 calls) | 0.000 / 0.000 / 0.000 (17 calls) | 0.001 / 0.050 / 0.003 (676 calls) | 0.000 / 0.000 / 0.000 (42 calls) |
| MacIsaac | FKH1 | 0.006 / 0.367 / 0.012 (1777 calls) | 0.000 / 0.000 / 0.000 (30 calls) | 0.006 / 0.267 / 0.012 (1287 calls) | 0.000 / 0.000 / 0.000 (31 calls) |
| MacIsaac | MCM1 | 0.002 / 0.333 / 0.003 (1266 calls) | 0.000 / 0.000 / 0.000 (25 calls) | 0.000 / 0.000 / 0.000 (304 calls) | 0.000 / 0.000 / 0.000 (9 calls) |
| MacIsaac | RAP1 | 0.011 / 0.333 / 0.022 (894 calls) | 0.000 / 0.000 / 0.000 (51 calls) | 0.010 / 0.300 / 0.020 (893 calls) | 0.000 / 0.000 / 0.000 (53 calls) |
| MacIsaac | REB1 | 0.007 / 1.000 / 0.013 (4740 calls) | 0.016 / 0.031 / 0.021 (64 calls) | 0.011 / 0.938 / 0.021 (2797 calls) | 0.029 / 0.062 / 0.040 (68 calls) |
| MacIsaac | SKO1 | 0.000 / 0.200 / 0.001 (3057 calls) | 0.000 / 0.000 / 0.000 (11 calls) | 0.001 / 0.400 / 0.002 (1729 calls) | 0.000 / 0.000 / 0.000 (7 calls) |
| MacIsaac | UME6 | 0.002 / 0.389 / 0.004 (3726 calls) | 0.056 / 0.056 / 0.056 (18 calls) | 0.002 / 0.333 / 0.003 (3499 calls) | 0.000 / 0.000 / 0.000 (37 calls) |
| MacIsaac | set total (9 groups) | 0.004 / 0.416 / 0.008 (23190 calls) | 0.011 / 0.022 / 0.015 (449 calls) | 0.005 / 0.336 / 0.010 (15653 calls) | 0.009 / 0.018 / 0.012 (454 calls) |
| Rossi _CX | ABF1 | 0.009 / 0.340 / 0.017 (3987 calls) | 0.014 / 0.019 / 0.016 (146 calls) | 0.008 / 0.320 / 0.016 (3989 calls) | 0.014 / 0.019 / 0.016 (142 calls) |
| Rossi _CX | CIN5 | 0.008 / 0.188 / 0.015 (1183 calls) | 0.034 / 0.062 / 0.044 (87 calls) | 0.002 / 0.021 / 0.004 (479 calls) | 0.000 / 0.000 / 0.000 (65 calls) |
| Rossi _CX | FHL1 | 0.005 / 0.295 / 0.010 (2560 calls) | 0.000 / 0.000 / 0.000 (17 calls) | 0.003 / 0.045 / 0.006 (676 calls) | 0.000 / 0.000 / 0.000 (42 calls) |
| Rossi _CX | FKH1 | 0.014 / 0.393 / 0.026 (1777 calls) | 0.000 / 0.000 / 0.000 (30 calls) | 0.016 / 0.328 / 0.030 (1287 calls) | 0.032 / 0.016 / 0.022 (31 calls) |
| Rossi _CX | MCM1 | 0.004 / 0.143 / 0.008 (1266 calls) | 0.000 / 0.000 / 0.000 (25 calls) | 0.007 / 0.057 / 0.012 (304 calls) | 0.000 / 0.000 / 0.000 (9 calls) |
| Rossi _CX | RAP1 | 0.011 / 0.227 / 0.021 (894 calls) | 0.000 / 0.000 / 0.000 (51 calls) | 0.009 / 0.182 / 0.017 (893 calls) | 0.000 / 0.000 / 0.000 (53 calls) |
| Rossi _CX | REB1 | 0.019 / 0.791 / 0.037 (4740 calls) | 0.016 / 0.009 / 0.011 (64 calls) | 0.029 / 0.704 / 0.056 (2797 calls) | 0.029 / 0.017 / 0.022 (68 calls) |
| Rossi _CX | SKO1 | 0.003 / 0.258 / 0.005 (3057 calls) | 0.000 / 0.000 / 0.000 (11 calls) | 0.003 / 0.194 / 0.007 (1729 calls) | 0.000 / 0.000 / 0.000 (7 calls) |
| Rossi _CX | UME6 | 0.005 / 0.643 / 0.010 (3726 calls) | 0.056 / 0.036 / 0.043 (18 calls) | 0.003 / 0.429 / 0.007 (3499 calls) | 0.000 / 0.000 / 0.000 (37 calls) |
| Rossi _CX | set total (9 groups) | 0.009 / 0.418 / 0.018 (23190 calls) | 0.016 / 0.014 / 0.015 (449 calls) | 0.011 / 0.324 / 0.020 (15653 calls) | 0.011 / 0.010 / 0.010 (454 calls) |

### Per round (ff09)

| round | within 1.1× | MacIsaac P/R/F1 (set) | Rossi P/R/F1 (set) | nucleosome copies | unknown occ (masked: expect 0) | non-deadband steps |
|---|---|---|---|---|---|---|
| r00 | 0/9 | 0.004 / 0.402 / 0.009 (23825 calls) | 0.010 / 0.387 / 0.020 (23825 calls) | 8804 (+0.00%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r01 | 0/9 | 0.005 / 0.378 / 0.009 (20598 calls) | 0.011 / 0.367 / 0.022 (20598 calls) | 8830 (+0.29%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r02 | 0/9 | 0.005 / 0.367 / 0.010 (18078 calls) | 0.012 / 0.348 / 0.023 (18078 calls) | 8853 (+0.56%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r03 | 0/9 | 0.006 / 0.363 / 0.012 (15906 calls) | 0.013 / 0.335 / 0.025 (15906 calls) | 8876 (+0.82%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r04 | 0/9 | 0.006 / 0.336 / 0.012 (14091 calls) | 0.014 / 0.323 / 0.027 (14091 calls) | 8888 (+0.95%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r05 | 0/9 | 0.007 / 0.332 / 0.014 (12354 calls) | 0.015 / 0.304 / 0.029 (12354 calls) | 8901 (+1.11%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r06 | 0/9 | 0.008 / 0.320 / 0.015 (10984 calls) | 0.017 / 0.294 / 0.032 (10984 calls) | 8910 (+1.21%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r07 | 0/9 | 0.008 / 0.301 / 0.016 (9761 calls) | 0.018 / 0.276 / 0.033 (9761 calls) | 8920 (+1.32%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r08 | 0/9 | 0.008 / 0.286 / 0.016 (8738 calls) | 0.019 / 0.262 / 0.035 (8738 calls) | 8925 (+1.37%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r09 | 0/9 | 0.009 / 0.270 / 0.017 (7782 calls) | 0.020 / 0.249 / 0.037 (7782 calls) | 8929 (+1.43%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r10 | 0/9 | 0.009 / 0.247 / 0.018 (6899 calls) | 0.020 / 0.222 / 0.037 (6899 calls) | 8932 (+1.46%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r11 | 0/9 | 0.010 / 0.232 / 0.019 (6158 calls) | 0.022 / 0.216 / 0.040 (6158 calls) | 8936 (+1.50%) | 0 | ABF1 secant+step-cap, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r12 | 0/9 | 0.010 / 0.216 / 0.019 (5530 calls) | 0.024 / 0.208 / 0.042 (5530 calls) | 8938 (+1.53%) | 0 | ABF1 secant, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r13 | 0/9 | 0.011 / 0.212 / 0.021 (4959 calls) | 0.024 / 0.193 / 0.043 (4959 calls) | 8940 (+1.54%) | 0 | ABF1 secant, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r14 | 0/9 | 0.011 / 0.193 / 0.021 (4484 calls) | 0.025 / 0.179 / 0.044 (4484 calls) | 8940 (+1.55%) | 0 | ABF1 secant, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r15 | 1/9 | 0.012 / 0.189 / 0.023 (4021 calls) | 0.026 / 0.169 / 0.046 (4021 calls) | 8941 (+1.56%) | 0 | CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r16 | 0/9 | 0.013 / 0.174 / 0.023 (3593 calls) | 0.026 / 0.150 / 0.045 (3593 calls) | 8942 (+1.57%) | 0 | ABF1 secant, CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r17 | 1/9 | 0.013 / 0.154 / 0.023 (3158 calls) | 0.026 / 0.133 / 0.044 (3158 calls) | 8943 (+1.58%) | 0 | CIN5 secant+step-cap, FHL1 secant+step-cap, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant+step-cap |
| r18 | 0/9 | 0.014 / 0.147 / 0.025 (2801 calls) | 0.026 / 0.118 / 0.043 (2801 calls) | 8943 (+1.58%) | 0 | ABF1 secant, CIN5 secant+step-cap, FHL1 secant, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r19 | 1/9 | 0.014 / 0.131 / 0.025 (2472 calls) | 0.027 / 0.105 / 0.043 (2472 calls) | 8944 (+1.59%) | 0 | CIN5 secant+step-cap, FHL1 secant, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r20 | 0/9 | 0.015 / 0.124 / 0.026 (2204 calls) | 0.029 / 0.104 / 0.046 (2204 calls) | 8944 (+1.59%) | 0 | ABF1 secant, CIN5 secant+step-cap, FHL1 secant, FKH1 secant+step-cap, MCM1 secant+step-cap, RAP1 secant, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r21 | 2/9 | 0.013 / 0.100 / 0.023 (1956 calls) | 0.029 / 0.091 / 0.044 (1956 calls) | 8944 (+1.60%) | 0 | CIN5 secant+step-cap, FHL1 secant, FKH1 secant+step-cap, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r22 | 2/9 | 0.012 / 0.081 / 0.021 (1753 calls) | 0.029 / 0.081 / 0.043 (1753 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant+step-cap, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r23 | 2/9 | 0.013 / 0.077 / 0.022 (1537 calls) | 0.031 / 0.077 / 0.044 (1537 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r24 | 2/9 | 0.014 / 0.077 / 0.024 (1396 calls) | 0.034 / 0.077 / 0.047 (1396 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r25 | 3/9 | 0.014 / 0.069 / 0.024 (1265 calls) | 0.034 / 0.069 / 0.045 (1265 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap |
| r26 | 2/9 | 0.016 / 0.069 / 0.026 (1131 calls) | 0.033 / 0.059 / 0.042 (1131 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r27 | 3/9 | 0.018 / 0.069 / 0.029 (1001 calls) | 0.035 / 0.056 / 0.043 (1001 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r28 | 2/9 | 0.019 / 0.066 / 0.029 (918 calls) | 0.034 / 0.050 / 0.040 (918 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r29 | 2/9 | 0.018 / 0.058 / 0.028 (822 calls) | 0.034 / 0.045 / 0.039 (822 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r30 | 2/9 | 0.020 / 0.058 / 0.030 (735 calls) | 0.037 / 0.043 / 0.040 (735 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r31 | 2/9 | 0.018 / 0.046 / 0.026 (661 calls) | 0.033 / 0.035 / 0.034 (661 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FHL1 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r32 | 4/9 | 0.016 / 0.039 / 0.023 (607 calls) | 0.030 / 0.029 / 0.029 (607 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap |
| r33 | 4/9 | 0.018 / 0.039 / 0.024 (558 calls) | 0.029 / 0.026 / 0.027 (558 calls) | 8944 (+1.60%) | 0 | FHL1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap, UME6 secant |
| r34 | 4/9 | 0.015 / 0.031 / 0.020 (522 calls) | 0.027 / 0.022 / 0.024 (522 calls) | 8944 (+1.60%) | 0 | CIN5 secant, FKH1 secant, MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant+step-cap |
| r35 | 5/9 | 0.016 / 0.031 / 0.021 (489 calls) | 0.025 / 0.019 / 0.022 (489 calls) | 8944 (+1.60%) | 0 | MCM1 secant+step-cap, REB1 secant+step-cap, SKO1 secant, UME6 secant |
| r36 | 4/9 | 0.017 / 0.031 / 0.022 (461 calls) | 0.028 / 0.021 / 0.024 (461 calls) | 8944 (+1.60%) | 0 | FHL1 secant, FKH1 secant, MCM1 secant+step-cap, RAP1 secant, REB1 secant+step-cap |
| r37 | 5/9 | 0.016 / 0.027 / 0.020 (441 calls) | 0.029 / 0.021 / 0.024 (441 calls) | 8944 (+1.60%) | 0 | CIN5 secant, MCM1 secant+step-cap, REB1 secant+step-cap, UME6 secant |
| r38 | 5/9 | 0.014 / 0.023 / 0.018 (417 calls) | 0.029 / 0.019 / 0.023 (417 calls) | 8944 (+1.60%) | 0 | FHL1 secant, FKH1 secant, MCM1 secant, REB1 secant |
| r39 | 5/9 | 0.015 / 0.023 / 0.018 (405 calls) | 0.030 / 0.019 / 0.023 (405 calls) | 8944 (+1.60%) | 0 | CIN5 secant, MCM1 secant, REB1 secant, SKO1 secant |
| r40 | 9/9 | 0.013 / 0.019 / 0.015 (391 calls) | 0.028 / 0.018 / 0.022 (391 calls) | 8944 (+1.60%) | 0 | all deadband |

state: stopped after iter 40: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x]

jobs: `{"0": {"count": "12611433", "decode": "12611432", "holdout_count": "12611835", "holdout_decode": "12611834", "holdout_validate": "12611836", "next": "12611434"}, "1": {"count": "12611832", "decode": "12611831", "next": "12611833"}, "10": {"count": "12627083", "decode": "12627082", "next": "12627084"}, "11": {"count": "12627265", "decode": "12627264", "next": "12627266"}, "12": {"count": "12628301", "decode": "12628300", "next": "12628302"}, "13": {"count": "12628496", "decode": "12628495", "next": "12628497"}, "14": {"count": "12628630", "decode": "12628629", "next": "12628631"}, "15": {"count": "12629642", "decode": "12629641", "next": "12629643"}, "16": {"count": "12631519", "decode": "12631518", "next": "12631520"}, "17": {"count": "12631697", "decode": "12631696", "next": "12631698"}, "18": {"count": "12634699", "decode": "12634698", "next": "12634700"}, "19": {"count": "12636018", "decode": "12635977", "next": "12636019"}, "2": {"count": "12612526", "decode": "12612525", "next": "12612527"}, "20": {"count": "12636244", "decode": "12636243", "next": "12636245"}, "21": {"count": "12636769", "decode": "12636768", "next": "12636770"}, "22": {"count": "12637097", "decode": "12637077", "next": "12637098"}, "23": {"count": "12637266", "decode": "12637265", "next": "12637267"}, "24": {"chereji": "12638284", "count": "12637437", "decode": "12637436", "holdout_count": "12638282", "holdout_decode": "12638281", "holdout_validate": "12638283", "next": "12637438", "summary": "12638285"}, "25": {"count": "12640177", "decode": "12640176", "next": "12640178"}, "26": {"count": "12640290", "decode": "12640289", "next": "12640291"}, "27": {"count": "12640383", "decode": "12640382", "next": "12640384"}, "28": {"count": "12640483", "decode": "12640482", "next": "12640484"}, "29": {"count": "12640573", "decode": "12640572", "next": "12640574"}, "3": {"count": "12612787", "decode": "12612786", "next": "12612788"}, "30": {"count": "12640708", "decode": "12640707", "next": "12640709"}, "31": {"count": "12640835", "decode": "12640834", "next": "12640836"}, "32": {"count": "12640926", "decode": "12640925", "next": "12640927"}, "33": {"count": "12641016", "decode": "12641015", "next": "12641017"}, "34": {"count": "12643696", "decode": "12643695", "next": "12643697"}, "35": {"count": "12643797", "decode": "12643796", "next": "12643798"}, "36": {"count": "12643904", "decode": "12643903", "next": "12643905"}, "37": {"count": "12644049", "decode": "12644048", "next": "12644050"}, "38": {"count": "12644372", "decode": "12644371", "next": "12644373"}, "39": {"count": "12645048", "decode": "12645047", "next": "12645049"}, "4": {"count": "12613058", "decode": "12613056", "next": "12613059"}, "40": {"chereji": "12646642", "count": "12646093", "decode": "12646092", "holdout_count": "12646640", "holdout_decode": "12646639", "holdout_validate": "12646641", "next": "12646094", "summary": "12646643"}, "5": {"count": "12613393", "decode": "12613392", "next": "12613394"}, "6": {"count": "12613677", "decode": "12613676", "next": "12613678"}, "7": {"count": "12613903", "decode": "12613902", "next": "12613904"}, "8": {"count": "12614124", "decode": "12614123", "next": "12614125"}, "9": {"count": "12626896", "decode": "12626895", "next": "12626897"}}`

