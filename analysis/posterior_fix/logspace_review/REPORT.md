# Log-space decoder review (2026-10-23)

**Verdict: SAFE TO DECODE** with `ROBOCOP_LOG_EMISSION=exact` and the trees `pkgvar/seq_maskoff_fit9_bgopen_logfull` (bo09),
`pkgvar/seq_maskoff_all153_logfull` (bu01) and `pkgvar/seq_maskoff_tune75_logfull` (bv75). No blockers were found. Three items
should be fixed eventually; none of them can fire with these models (all end_probs = 1, n_obs >= 1).

Reviewed by a subagent and spot-checked against the output files (`cv_412001.txt`, `cv_408001.txt`,
`check_decode.jsonl`, `replica_ld_exact_rdna2.json`, `bruteforce.txt`). Nothing in `pkg/`, the existing pkgvar trees or
`inputs/` was changed. The six capture/posterior npz files (~5.3 GB) are in `analysis/_trash/logspace_review_scratch_20261023/`.

## Short answer
- **Faithful log twin?** Yes. `algo_log.c` runs the same recursion as `algo.c`, with the same order, the same silent-state time
  index (audit C4) and the same end-probs handling (audit C5). It drops only the `+DBL_MIN` and cannot underflow.
  On normal segments it matches fix B to 6e-14 to 4.8e-9.
- **Correct HMM posterior?** Yes, for every emitting state at every position except the last, where end_probs is counted twice
  (C5). That is harmless with end_probs = 1. Silent-state f*b products are not a posterior (C4); log space writes them as 0 and
  nothing downstream uses them.

## Findings
| # | item | severity | evidence |
|---|---|---|---|
| 1 | Recursion order and indices match the linear code: forward does emitting then silent states ascending, backward does emitting then silent states descending; end_probs enters on the last row | correct | `algo_log.c:141-165` vs `algo.c:79-125`; `algo_log.c:212-234` vs `algo.c:171-204` |
| 2 | Silent order is valid: all 154 silent->silent edges go to a higher index | verified | `hmm_structure.py`, all 3 HMMconfig.pkl |
| 3 | initial_probs puts no mass on silent states | harmless | 0.0 in all 3 models |
| 4 | C4 time index: emitting f and b are each consistent, so emitting posteriors are exact; silent f*b is not a posterior | harmless (silent columns written 0) | brute force: log emitting error <= 1.7e-12; fix B silent columns off from textbook by up to 5.5 |
| 5 | C5: end_probs counted twice at the last position (`algo_log.c:148-149` and `:197-198`) | should-fix if end_probs is ever != 1; **not active now** | brute force with random end: last row = "end squared" to <= 6e-13, off from truth by up to 0.42. Fix: subtract log(end_probs) once on the last row in `posterior_decoding_log` |
| 6 | At n_obs = 1 the forward pass never applies end_probs, so the log-likelihood is off by log(end); the posterior is correct | harmless (end = 1; segments are 5000 long) | brute force n = 1, random end: |ll - Z| up to 2.2; end = 1: <= 4e-16 |
| 7 | n_obs = 0 writes out of bounds (`algo_log.c:129, 198, 208`) | should-fix (`if (n_obs < 1) return -1`); unreachable from robocop.py | `edgecases.txt`: crash with `free(): invalid next size`; robocop.py raises earlier (`:968`) |
| 8 | rc = -2 does not stop the recursion | harmless | robocop.py asserts every rc equals the ABI, so the decode aborts loudly (except under `python -O`) |
| 9 | 32-bit int index arithmetic | harmless | max index 17,425,000; overflow needs n_obs > 616,000 |
| 10 | `nuc_end` factor column always 0 | harmless, not a regression | `nuc_start+nuc_len` = 3330 = ssb (`robocop.py:1257`): it is the central silent state, an upstream off-by-one. Fix B already gave <= 3.2e-31 there |
| 11 | bo09 pkl lines | correct | logfull `robocop.py:635,643` = parent `:598,606` |
| 12 | Tree edits are only the intended ones | correct | only `robocop.py`, `robocopExtras.py`, `robocop_no_em.py` differ; masks -inf (bo09 145 motifs = 2598 states, bu01 none). Cosmetic: tune75_logfull's comment at `robocopExtras.py:169` is not in the new trees |
| 13 | Binary matches source | correct | rebuild gives md5 88f86a356aad55d466098d214b29ef06; no warnings with `-Wall -Wextra -pedantic -Wconversion` |
| 14 | Every transition edge has a usable log weight | harmless | smallest nonzero: bo09 1.4e-51, bu01 2.3e-15, bv75 1.6e-39 |

## Tests
**Brute force** (`bruteforce.py`, `bruteforce.txt`): 128 configurations x 30 seeds; RoboCOP-shaped and random silent-chain HMMs;
n_obs 1, 2, 3, 6; emissions normal, ~1e-300, ~1e-600, mixed; masks 0% / 25%; end 1 or random. Truth = exact path enumeration.

| quantity | log decoder | fix B |
|---|---|---|
| emitting posterior (end = 1) | <= 1.7e-12 | <= 4e-15 at normal emissions; up to 1.0 at tiny emissions |
| log-likelihood | <= 3.6e-12 | off by up to 4.1e3 with mixed tiny emissions |
| bad return codes | 0 of 3840 | n/a |

**Real bo09 segments** (linear tree `seq_maskoff_fit9_bgopen`, `compare_variants.py`):

| segment | LOG-D0 vs fix B | LOG-D0 vs C1FIX | loglik D0 - fix B |
|---|---|---|---|
| chrXIV:412001-417000 (C1 event) | 4.78e-9 | 1.1e-13 | 7.6e-10 |
| chrXIV:408001-413000 | 6.3e-14 | 1.2e-13 | 2.9e-10 |

New bo09 log tree in floored mode vs log of the linear capture (`compare_caps_412001.json`): layers 1-6 identical, layer 0
to 1.8e-15, every -inf cell matches a linear zero (0 mismatches), summed log emission to 2.8e-14, silent columns 0.

**Exact mode through the drivers** (`check_decode.jsonl`): no NaN, no row below 0.99, silent max 0.

| run | emitting row sums (after 1e-4 save floor) | log-likelihood |
|---|---|---|
| bo09 chrXIV:412001 | 0.99981 - 1.0 | -19850.63 |
| bu01 chrXIV:412001 | 0.99561 - 1.0 | -17959.07 |
| bo09 rDNA chrXII:458001 | 0.99991 - 1.0 | -200546.34 |

**80-bit reference, bo09 rDNA** (`replica_ld_exact_rdna2.json`): posterior agrees to 9.2e-14, log-likelihood identical; emissions
reach log -7675 per cell, beyond float64 linear range. The h5 posterior equals the in-process result (`h5_vs_inprocess_rdna2.txt`).
