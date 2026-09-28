# Tuner-v2 continuation: deadband 1.25× → 1.1×, up to 15 rounds

_generated 2026-09-17 13:34 by `analysis/continue_summary.py`; campaigns still running are marked **INCOMPLETE** — re-run `python continue_summary.py` to refresh._

### Rule 7 — what differs

Each campaign differs from its original run **only in the stop criteria, from the continuation round onward: deadband 1.25× → 1.1× and max rounds 8 → 15 (rounds 0..14)**. Everything else is identical: step cap 10×/round, zero-count ×10 step, per-bp weight cap 0.70, bisection bracket (unchanged), β seed/secant, α damping, unknown fixed w = 1e-3, w_nuc 35 with no hold, 58 groups with the mr58 mask, MacIsaac targets, tune chrXIV+chrII, holdout chrIV, same driver/pkgvar per campaign. The continuation re-ran the last round's update from its pre-update snapshot under 1.1× (the original post-update state is kept as `state_after_update_NN_deadband1p25.json`), so the old final's decode is shared by both columns below.

Definitions: within 1.1× / 1.25× = tuned groups (of 55) with |ln((E+1)/(T+1))| ≤ ln fold (the deadband test); within 2× = the tuner's metric (T ≥ 5, |ln E/T| ≤ ln 2). P / R / F1 pooled over all 58 groups (calls posterior ≥ 0.10, 30 bp one-to-one). Bracket-pinned = outside the deadband with the step set by the bisection bracket (bracket < 0.1 decade).

## Campaigns

| run | config | old final (1.25× stop) | new final | state | within 1.1× (of 55) | within 1.25× | within 2× (T≥5) | bisect-set outside 1.1× (old → new) |
|---|---|---|---|---|---|---|---|---|
| bt10 | both, fiber tempered φ=10 | r04 (iter 4: converged: every tuned group took a zero step (deadband or held at the per-bp cap)) | r05 | stopped: iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | 27 → 55 | 55 → 55 | 41/41 → 41/41 | 0 → 0 |
| bt05 | both, φ=5 | r05 (iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap)) | r08 | stopped: iter 8: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | 23 → 55 | 55 → 55 | 41/41 → 41/41 | 0 → 0 |
| sw01 | sequence only | r05 (iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap)) | r07 | stopped: iter 7: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | 20 → 55 | 55 → 55 | 41/41 → 41/41 | 0 → 0 |
| bw01 | both layers (φ=1) | r07 (iter 7: reached the 8-round limit) | r24 | stopped: iter 24: reached the 25-round limit [deadband 1.1x] | 23 → 52 | 47 → 52 | 39/41 → 40/41 | 3 → 3 |
| bt02 | both, φ=2 | r05 (iter 5: converged: every tuned group took a zero step (deadband or held at the per-bp cap)) | r08 | stopped: iter 8: converged: every tuned group took a zero step (deadband or held at the per-bp cap) [deadband 1.1x] | 21 → 55 | 55 → 55 | 41/41 → 41/41 | 0 → 0 |
| fw01 | fiber only | r07 (iter 7: reached the 8-round limit) | r24 | stopped: iter 24: reached the 25-round limit [deadband 1.1x] | 15 → 42 | 32 → 44 | 30/41 → 36/41 | 11 → 13 |

## P / R / F1, tuning chromosomes chrXIV+chrII — old final → new final

| run | ref | all old | all new | fitted old | fitted new | nonfitted old | nonfitted new | calls |
|---|---|---|---|---|---|---|---|---|
| bt10 | MacIsaac | 0.106 / 0.231 / 0.146 | 0.104 / 0.233 / 0.144 | 0.214 / 0.402 / 0.279 | 0.201 / 0.398 / 0.267 | 0.087 / 0.193 / 0.120 | 0.086 / 0.197 / 0.119 | 3118 → 3219 |
| bt10 | Rossi _CX | 0.089 / 0.192 / 0.121 | 0.087 / 0.194 / 0.120 | 0.351 / 0.273 / 0.307 | 0.340 / 0.278 / 0.306 | 0.040 / 0.129 / 0.061 | 0.039 / 0.130 / 0.060 | 3118 → 3219 |
| bt05 | MacIsaac | 0.091 / 0.206 / 0.126 | 0.092 / 0.206 / 0.127 | 0.172 / 0.328 / 0.226 | 0.176 / 0.340 / 0.232 | 0.077 / 0.179 / 0.107 | 0.076 / 0.177 / 0.106 | 3249 → 3244 |
| bt05 | Rossi _CX | 0.078 / 0.175 / 0.107 | 0.078 / 0.175 / 0.108 | 0.298 / 0.235 / 0.263 | 0.303 / 0.241 / 0.268 | 0.038 / 0.129 / 0.059 | 0.037 / 0.124 / 0.057 | 3249 → 3244 |
| sw01 | MacIsaac | 0.067 / 0.132 / 0.089 | 0.067 / 0.134 / 0.089 | 0.136 / 0.278 / 0.183 | 0.137 / 0.282 / 0.184 | 0.051 / 0.100 / 0.068 | 0.051 / 0.102 / 0.068 | 2833 → 2887 |
| sw01 | Rossi _CX | 0.064 / 0.126 / 0.085 | 0.064 / 0.128 / 0.085 | 0.234 / 0.198 / 0.215 | 0.240 / 0.204 / 0.221 | 0.025 / 0.071 / 0.037 | 0.024 / 0.070 / 0.036 | 2833 → 2887 |
| bw01 | MacIsaac | 0.035 / 0.101 / 0.052 | 0.034 / 0.089 / 0.050 | 0.052 / 0.139 / 0.075 | 0.045 / 0.077 / 0.057 | 0.032 / 0.092 / 0.047 | 0.033 / 0.092 / 0.048 | 4118 → 3726 |
| bw01 | Rossi _CX | 0.026 / 0.074 / 0.038 | 0.021 / 0.055 / 0.031 | 0.091 / 0.101 / 0.095 | 0.076 / 0.054 / 0.063 | 0.013 / 0.053 / 0.020 | 0.014 / 0.055 / 0.022 | 4118 → 3726 |
| bt02 | MacIsaac | 0.065 / 0.154 / 0.092 | 0.065 / 0.151 / 0.091 | 0.116 / 0.212 / 0.150 | 0.118 / 0.212 / 0.152 | 0.057 / 0.142 / 0.081 | 0.057 / 0.138 / 0.081 | 3398 → 3333 |
| bt02 | Rossi _CX | 0.049 / 0.117 / 0.069 | 0.050 / 0.115 / 0.069 | 0.192 / 0.145 / 0.166 | 0.193 / 0.144 / 0.165 | 0.026 / 0.094 / 0.041 | 0.026 / 0.092 / 0.041 | 3398 → 3333 |
| fw01 | MacIsaac | 0.008 / 0.058 / 0.014 | 0.007 / 0.045 / 0.012 | 0.011 / 0.108 / 0.020 | 0.012 / 0.019 / 0.014 | 0.007 / 0.047 / 0.012 | 0.007 / 0.051 / 0.012 | 10832 → 9184 |
| fw01 | Rossi _CX | 0.006 / 0.047 / 0.011 | 0.003 / 0.021 / 0.006 | 0.022 / 0.086 / 0.035 | 0.032 / 0.022 / 0.026 | 0.002 / 0.017 / 0.003 | 0.002 / 0.020 / 0.003 | 10832 → 9184 |

## P / R / F1, holdout chrIV — old final → new final

| run | ref | all old | all new | fitted old | fitted new | nonfitted old | nonfitted new | calls |
|---|---|---|---|---|---|---|---|---|
| bt10 | MacIsaac | 0.093 / 0.206 / 0.128 | 0.093 / 0.212 / 0.129 | 0.193 / 0.389 / 0.258 | 0.186 / 0.398 / 0.254 | 0.075 / 0.168 / 0.104 | 0.075 / 0.173 / 0.104 | 2906 → 3005 |
| bt10 | Rossi _CX | 0.094 / 0.217 / 0.131 | 0.092 / 0.218 / 0.129 | 0.330 / 0.297 / 0.313 | 0.315 / 0.299 / 0.306 | 0.050 / 0.163 / 0.077 | 0.049 / 0.164 / 0.076 | 2906 → 3005 |
| bt05 | MacIsaac | 0.077 / 0.177 / 0.108 | 0.078 / 0.178 / 0.108 | 0.159 / 0.350 / 0.218 | 0.157 / 0.350 / 0.216 | 0.061 / 0.142 / 0.085 | 0.062 / 0.143 / 0.086 | 3015 → 3013 |
| bt05 | Rossi _CX | 0.081 / 0.194 / 0.115 | 0.083 / 0.197 / 0.116 | 0.269 / 0.263 / 0.266 | 0.274 / 0.271 / 0.272 | 0.044 / 0.147 / 0.068 | 0.044 / 0.147 / 0.068 | 3015 → 3013 |
| sw01 | MacIsaac | 0.059 / 0.117 / 0.078 | 0.060 / 0.123 / 0.080 | 0.133 / 0.283 / 0.181 | 0.133 / 0.292 / 0.182 | 0.042 / 0.083 / 0.056 | 0.043 / 0.087 / 0.058 | 2613 → 2690 |
| sw01 | Rossi _CX | 0.066 / 0.136 / 0.089 | 0.066 / 0.140 / 0.090 | 0.249 / 0.236 / 0.242 | 0.243 / 0.238 / 0.240 | 0.024 / 0.069 / 0.036 | 0.026 / 0.074 / 0.038 | 2613 → 2690 |
| bw01 | MacIsaac | 0.031 / 0.095 / 0.047 | 0.032 / 0.088 / 0.047 | 0.039 / 0.124 / 0.060 | 0.038 / 0.075 / 0.050 | 0.029 / 0.089 / 0.044 | 0.031 / 0.091 / 0.046 | 4009 → 3642 |
| bw01 | Rossi _CX | 0.026 / 0.084 / 0.040 | 0.024 / 0.068 / 0.035 | 0.074 / 0.104 / 0.087 | 0.065 / 0.057 / 0.061 | 0.016 / 0.070 / 0.026 | 0.018 / 0.075 / 0.029 | 4009 → 3642 |
| bt02 | MacIsaac | 0.053 / 0.129 / 0.075 | 0.052 / 0.126 / 0.073 | 0.075 / 0.159 / 0.102 | 0.072 / 0.155 / 0.098 | 0.049 / 0.122 / 0.070 | 0.048 / 0.120 / 0.069 | 3217 → 3185 |
| bt02 | Rossi _CX | 0.048 / 0.121 / 0.068 | 0.046 / 0.117 / 0.067 | 0.141 / 0.134 / 0.137 | 0.136 / 0.130 / 0.133 | 0.031 / 0.113 / 0.049 | 0.030 / 0.109 / 0.047 | 3217 → 3185 |
| fw01 | MacIsaac | 0.006 / 0.049 / 0.011 | 0.004 / 0.030 / 0.008 | 0.012 / 0.142 / 0.023 | 0.009 / 0.018 / 0.012 | 0.004 / 0.029 / 0.007 | 0.004 / 0.032 / 0.007 | 10755 → 8989 |
| fw01 | Rossi _CX | 0.006 / 0.053 / 0.011 | 0.003 / 0.021 / 0.005 | 0.018 / 0.090 / 0.029 | 0.011 / 0.010 / 0.010 | 0.003 / 0.028 / 0.005 | 0.003 / 0.029 / 0.005 | 10755 → 8989 |

## Nucleosomes

| run | tune copies old (vs r0) | tune copies new (vs r0) | holdout chrIV copies old | holdout chrIV copies new | Chereji +1/-1 recall chrXIV old | Chereji recall new |
|---|---|---|---|---|---|---|
| bt10 | 8934 (-0.00%) | 8934 (-0.01%) | 8594 | 8593 | 0.687 (n_ref 626) | 0.687 (n_ref 626) |
| bt05 | 8937 (+0.01%) | 8936 (+0.01%) | 8591 | 8590 | 0.735 (n_ref 626) | 0.736 (n_ref 626) |
| sw01 | 9192 (-0.72%) | 9186 (-0.78%) | 8826 | 8821 | 0.137 (n_ref 626) | 0.137 (n_ref 626) |
| bw01 | 8944 (+0.27%) | 8945 (+0.29%) | 8592 | 8593 | 0.812 (n_ref 626) | 0.812 (n_ref 626) |
| bt02 | 8942 (+0.01%) | 8942 (+0.01%) | 8592 | 8592 | 0.791 (n_ref 626) | 0.791 (n_ref 626) |
| fw01 | 8918 (+1.35%) | 8941 (+1.60%) | 8561 | 8592 | 0.810 (n_ref 626) | 0.810 (n_ref 626) |

## bt10 (both, fiber tempered φ=10) — per round from the old final

| round | within 1.1× | within 1.25× | within 2× | zero steps | calls | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | bisect-set (outside 1.1×) | capped_under |
|---|---|---|---|---|---|---|---|---|---|---|
| r04 | 27 | 55 | 41/41 | 27 | 3118 | 0.106 / 0.231 / 0.146 | 0.089 / 0.192 / 0.121 | 8934 (-0.00%) | – | – |
| r05 | 55 | 55 | 41/41 | 55 | 3219 | 0.104 / 0.233 / 0.144 | 0.087 / 0.194 / 0.120 | 8934 (-0.01%) | – | – |

jobs: `{"4": {"count": "12597530", "decode": "12597529", "holdout_count": "12597688", "holdout_decode": "12597687", "holdout_validate": "12597689", "next": "12597531"}, "5": {"chereji": "12601402", "count": "12601098", "decode": "12601097", "holdout_count": "12601400", "holdout_decode": "12601399", "holdout_validate": "12601401", "next": "12601099", "summary": "12601403"}}`

## bt05 (both, φ=5) — per round from the old final

| round | within 1.1× | within 1.25× | within 2× | zero steps | calls | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | bisect-set (outside 1.1×) | capped_under |
|---|---|---|---|---|---|---|---|---|---|---|
| r05 | 23 | 55 | 41/41 | 23 | 3249 | 0.091 / 0.206 / 0.126 | 0.078 / 0.175 / 0.107 | 8937 (+0.01%) | – | – |
| r06 | 54 | 55 | 41/41 | 54 | 3245 | 0.092 / 0.206 / 0.127 | 0.078 / 0.175 / 0.108 | 8936 (+0.01%) | – | – |
| r07 | 54 | 55 | 41/41 | 54 | 3245 | 0.092 / 0.206 / 0.127 | 0.078 / 0.175 / 0.108 | 8936 (+0.01%) | – | – |
| r08 | 55 | 55 | 41/41 | 55 | 3244 | 0.092 / 0.206 / 0.127 | 0.078 / 0.175 / 0.108 | 8936 (+0.01%) | – | – |

jobs: `{"5": {"count": "12597584", "decode": "12597583", "holdout_count": "12597734", "holdout_decode": "12597733", "holdout_validate": "12597735", "next": "12597585"}, "6": {"count": "12601140", "decode": "12601139", "next": "12601141"}, "7": {"count": "12601405", "decode": "12601404", "next": "12601406"}, "8": {"chereji": "12601843", "count": "12601668", "decode": "12601667", "holdout_count": "12601841", "holdout_decode": "12601840", "holdout_validate": "12601842", "next": "12601669", "summary": "12601844"}}`

## sw01 (sequence only) — per round from the old final

| round | within 1.1× | within 1.25× | within 2× | zero steps | calls | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | bisect-set (outside 1.1×) | capped_under |
|---|---|---|---|---|---|---|---|---|---|---|
| r05 | 20 | 55 | 41/41 | 20 | 2833 | 0.067 / 0.132 / 0.089 | 0.064 / 0.126 / 0.085 | 9192 (-0.72%) | – | – |
| r06 | 47 | 55 | 41/41 | 47 | 2910 | 0.067 / 0.135 / 0.089 | 0.065 / 0.130 / 0.086 | 9185 (-0.80%) | – | – |
| r07 | 55 | 55 | 41/41 | 55 | 2887 | 0.067 / 0.134 / 0.089 | 0.064 / 0.128 / 0.085 | 9186 (-0.78%) | – | – |

jobs: `{"5": {"count": "12593758", "decode": "12593757", "holdout_count": "12593894", "holdout_decode": "12593893", "holdout_validate": "12593895", "next": "12593759"}, "6": {"count": "12601182", "decode": "12601181", "next": "12601183"}, "7": {"chereji": "12601624", "count": "12601356", "decode": "12601355", "holdout_count": "12601622", "holdout_decode": "12601621", "holdout_validate": "12601623", "next": "12601357", "summary": "12601625"}}`

## bw01 (both layers (φ=1)) — per round from the old final

| round | within 1.1× | within 1.25× | within 2× | zero steps | calls | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | bisect-set (outside 1.1×) | capped_under |
|---|---|---|---|---|---|---|---|---|---|---|
| r07 | 23 | 47 | 39/41 | 23 | 4118 | 0.035 / 0.101 / 0.052 | 0.026 / 0.074 / 0.038 | 8944 (+0.27%) | BAS1,RTG3,SKN7 | – |
| r08 | 34 | 50 | 39/41 | 34 | 3882 | 0.035 / 0.094 / 0.051 | 0.023 / 0.062 / 0.033 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r09 | 45 | 50 | 40/41 | 45 | 3789 | 0.035 / 0.093 / 0.051 | 0.022 / 0.059 / 0.033 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r10 | 48 | 52 | 40/41 | 48 | 3742 | 0.034 / 0.090 / 0.050 | 0.022 / 0.056 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r11 | 51 | 52 | 40/41 | 51 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r12 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r13 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r14 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r15 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r16 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r17 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r18 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r19 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r20 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r21 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r22 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r23 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |
| r24 | 52 | 52 | 40/41 | 52 | 3726 | 0.034 / 0.089 / 0.050 | 0.021 / 0.055 / 0.031 | 8945 (+0.29%) | BAS1,RTG3,SKN7 | – |

jobs: `{"10": {"count": "12601710", "decode": "12601709", "next": "12601711"}, "11": {"count": "12601886", "decode": "12601885", "next": "12601887"}, "12": {"count": "12602025", "decode": "12602024", "next": "12602026"}, "13": {"count": "12602167", "decode": "12602166", "next": "12602168"}, "14": {"chereji": "12603285", "count": "12602952", "decode": "12602951", "holdout_count": "12603283", "holdout_decode": "12603282", "holdout_validate": "12603284", "next": "12602953", "summary": "12603286"}, "15": {"count": "12606830", "decode": "12606829", "next": "12606831"}, "16": {"count": "12606968", "decode": "12606967", "next": "12606969"}, "17": {"count": "12607054", "decode": "12607053", "next": "12607055"}, "18": {"count": "12607162", "decode": "12607161", "next": "12607163"}, "19": {"count": "12607281", "decode": "12607280", "next": "12607282"}, "20": {"count": "12607425", "decode": "12607424", "next": "12607426"}, "21": {"count": "12607555", "decode": "12607554", "next": "12607556"}, "22": {"count": "12607714", "decode": "12607713", "next": "12607715"}, "23": {"count": "12608054", "decode": "12608053", "next": "12608055"}, "24": {"chereji": "12611445", "count": "12611176", "decode": "12611175", "holdout_count": "12611443", "holdout_decode": "12611442", "holdout_validate": "12611444", "next": "12611177", "summary": "12611446"}, "7": {"count": "12594077", "decode": "12594076", "holdout_count": "12594200", "holdout_decode": "12594199", "holdout_validate": "12594240", "next": "12594078"}, "8": {"count": "12601224", "decode": "12601223", "next": "12601225"}, "9": {"count": "12601486", "decode": "12601485", "next": "12601487"}}`

## bt02 (both, φ=2) — per round from the old final

| round | within 1.1× | within 1.25× | within 2× | zero steps | calls | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | bisect-set (outside 1.1×) | capped_under |
|---|---|---|---|---|---|---|---|---|---|---|
| r05 | 21 | 55 | 41/41 | 21 | 3398 | 0.065 / 0.154 / 0.092 | 0.049 / 0.117 / 0.069 | 8942 (+0.01%) | – | – |
| r06 | 50 | 55 | 41/41 | 50 | 3343 | 0.065 / 0.151 / 0.091 | 0.049 / 0.114 / 0.069 | 8942 (+0.01%) | – | – |
| r07 | 54 | 55 | 41/41 | 54 | 3339 | 0.065 / 0.151 / 0.091 | 0.049 / 0.115 / 0.069 | 8942 (+0.01%) | – | – |
| r08 | 55 | 55 | 41/41 | 55 | 3333 | 0.065 / 0.151 / 0.091 | 0.050 / 0.115 / 0.069 | 8942 (+0.01%) | – | – |

jobs: `{"5": {"count": "12597626", "decode": "12597625", "holdout_count": "12597776", "holdout_decode": "12597775", "holdout_validate": "12597777", "next": "12597627"}, "6": {"count": "12601231", "decode": "12601230", "next": "12601232"}, "7": {"count": "12601529", "decode": "12601528", "next": "12601530"}, "8": {"chereji": "12601930", "count": "12601752", "decode": "12601751", "holdout_count": "12601928", "holdout_decode": "12601927", "holdout_validate": "12601929", "next": "12601753", "summary": "12601931"}}`

## fw01 (fiber only) — per round from the old final

| round | within 1.1× | within 1.25× | within 2× | zero steps | calls | MacIsaac P/R/F1 | Rossi P/R/F1 | nucleosome copies | bisect-set (outside 1.1×) | capped_under |
|---|---|---|---|---|---|---|---|---|---|---|
| r07 | 15 | 32 | 30/41 | 15 | 10832 | 0.008 / 0.058 / 0.014 | 0.006 / 0.047 / 0.011 | 8918 (+1.35%) | ACE2,HAP1,MBP1,MET31,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r08 | 26 | 38 | 30/41 | 26 | 10417 | 0.008 / 0.056 / 0.013 | 0.006 / 0.041 / 0.010 | 8922 (+1.39%) | ACE2,HAP1,MBP1,MET31,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r09 | 29 | 39 | 31/41 | 29 | 10091 | 0.008 / 0.054 / 0.013 | 0.005 / 0.037 / 0.009 | 8926 (+1.44%) | ACE2,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r10 | 32 | 40 | 32/41 | 32 | 9800 | 0.008 / 0.055 / 0.014 | 0.005 / 0.032 / 0.008 | 8929 (+1.47%) | ACE2,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r11 | 33 | 41 | 33/41 | 33 | 9615 | 0.008 / 0.054 / 0.014 | 0.004 / 0.030 / 0.008 | 8932 (+1.50%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r12 | 35 | 41 | 35/41 | 35 | 9496 | 0.008 / 0.052 / 0.014 | 0.004 / 0.029 / 0.008 | 8935 (+1.54%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r13 | 39 | 43 | 35/41 | 39 | 9423 | 0.008 / 0.051 / 0.014 | 0.004 / 0.028 / 0.007 | 8937 (+1.56%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r14 | 40 | 43 | 35/41 | 40 | 9360 | 0.008 / 0.050 / 0.013 | 0.004 / 0.026 / 0.007 | 8938 (+1.57%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r15 | 41 | 43 | 35/41 | 41 | 9317 | 0.008 / 0.050 / 0.013 | 0.004 / 0.026 / 0.007 | 8939 (+1.58%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r16 | 40 | 42 | 35/41 | 40 | 9290 | 0.008 / 0.050 / 0.013 | 0.004 / 0.026 / 0.007 | 8939 (+1.58%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r17 | 40 | 43 | 35/41 | 40 | 9245 | 0.008 / 0.049 / 0.013 | 0.004 / 0.024 / 0.006 | 8940 (+1.59%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r18 | 40 | 43 | 35/41 | 40 | 9225 | 0.007 / 0.048 / 0.013 | 0.004 / 0.023 / 0.006 | 8940 (+1.59%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r19 | 41 | 43 | 36/41 | 41 | 9202 | 0.007 / 0.047 / 0.013 | 0.003 / 0.022 / 0.006 | 8940 (+1.60%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r20 | 40 | 44 | 36/41 | 40 | 9201 | 0.007 / 0.047 / 0.013 | 0.004 / 0.023 / 0.006 | 8940 (+1.60%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r21 | 41 | 44 | 36/41 | 41 | 9184 | 0.007 / 0.046 / 0.012 | 0.003 / 0.022 / 0.006 | 8941 (+1.60%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r22 | 42 | 44 | 36/41 | 42 | 9184 | 0.007 / 0.045 / 0.012 | 0.003 / 0.021 / 0.006 | 8941 (+1.60%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r23 | 42 | 44 | 36/41 | 42 | 9184 | 0.007 / 0.045 / 0.012 | 0.003 / 0.021 / 0.006 | 8941 (+1.60%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |
| r24 | 42 | 44 | 36/41 | 42 | 9184 | 0.007 / 0.045 / 0.012 | 0.003 / 0.021 / 0.006 | 8941 (+1.60%) | ACE2,CAD1,HAP1,MBP1,MET31,MSN2,NRG1,PDR3,PHD1,RTG3,SKN7,STE12,SUT1 | – |

jobs: `{"10": {"count": "12601795", "decode": "12601794", "next": "12601796"}, "11": {"count": "12601972", "decode": "12601971", "next": "12601973"}, "12": {"count": "12602068", "decode": "12602067", "next": "12602069"}, "13": {"count": "12602219", "decode": "12602218", "next": "12602220"}, "14": {"chereji": "12603390", "count": "12602994", "decode": "12602993", "holdout_count": "12603388", "holdout_decode": "12603387", "holdout_validate": "12603389", "next": "12602995", "summary": "12603391"}, "15": {"count": "12606872", "decode": "12606871", "next": "12606873"}, "16": {"count": "12606926", "decode": "12606925", "next": "12606927"}, "17": {"count": "12607012", "decode": "12607011", "next": "12607013"}, "18": {"count": "12607120", "decode": "12607119", "next": "12607121"}, "19": {"count": "12607237", "decode": "12607236", "next": "12607238"}, "20": {"count": "12607383", "decode": "12607382", "next": "12607384"}, "21": {"count": "12607506", "decode": "12607505", "next": "12607507"}, "22": {"count": "12607672", "decode": "12607671", "next": "12607673"}, "23": {"count": "12608010", "decode": "12608009", "next": "12608011"}, "24": {"chereji": "12611450", "count": "12611218", "decode": "12611217", "holdout_count": "12611448", "holdout_decode": "12611447", "holdout_validate": "12611449", "next": "12611219", "summary": "12611451"}, "7": {"count": "12594031", "decode": "12594030", "holdout_count": "12594127", "holdout_decode": "12594126", "holdout_validate": "12594128", "next": "12594032"}, "8": {"count": "12601234", "decode": "12601233", "next": "12601235"}, "9": {"count": "12601551", "decode": "12601550", "next": "12601552"}}`

