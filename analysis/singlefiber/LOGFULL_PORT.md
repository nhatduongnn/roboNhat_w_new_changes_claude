# Single-fiber FiberCOP on the log-space decoder (2026-10-25)

## Verdict
The port works. On the test window (chrI:60,001-65,000, 439 reads, arm A, efficiency off) the log-space
per-read decoder gives the same posteriors as the old linear prototype. Every check gives the same
numbers. The pooled check agrees with the log-space aggregate decode to rounding error (9e-14). On this
window the DBL_MIN padding and the 1e-30 floor did no hidden damage. The port removes both anyway, and
adds loud failures for impossible data.

No model parameter changed. The trainDir is `robocop_train_fiberonly`, arm A uses the shipped `ps`, and
the pkls, masks and layers are the same as in `permol_seq_maskoff`. Only the decoder changed.

## Files
| file | what |
|---|---|
| `pkgvar/permol_seq_maskoff_logfull/` | copy of `permol_seq_maskoff` plus the logfull edits. Diff: `robocop_no_em.py` (cshared -> `librobocop_logfull.so`), `robocop/robocop.py` (the `seq_maskoff_tune75_logfull` log path, applied as the patch `seq_maskoff_all153` -> `_all153_logfull`; the PNG-dump hunk did not apply because that block was already removed in permol; the log Bernoulli emission), `robocop/utils/robocopExtras.py` (both 1e-30 floors deleted) |
| `pkgvar/seq_maskoff_logfull/` | copy of `seq_maskoff` (the parent of permol) plus the same three edits. Used for the aggregate reference. `seq_maskoff` has no hard mask, so no `= 0` line needed changing to `-inf` |
| `run_split_seq_maskoff_logfull.py` | aggregate driver (twin of `run_split_revfix_seq_maskoff.py`) |
| `singlefiber/run_permol_window_logfull.py` | per-read driver (old `run_permol_window.py` is untouched) |
| `singlefiber/pooled_emission_check_logfull.py` | pooled check in log space, plus a log-likelihood identity |
| `singlefiber/test_logfull_permol.py` | unit tests T1-T7 |
| `singlefiber/compare_permol_runs.py` | per-molecule comparison of two runs |
| `singlefiber/sbatch_permol_window_logfull.sh`, `sbatch_agg_baseline_logfull.sh` | slurm jobs |

Outputs: `/usr/project/xtmp/nd141/permol_proto_logfull/` (`agg_logfull_{exact,floored}`,
`armA_effoff_exact`, `armA_effoff_floored_first20`, `pooled_check/`, `checks_armA_effoff_exact.json`,
`compare_*.json`, `unit_tests_{exact,floored}.json`, `logs/`).

## No information vs forbidden vs impossible
Line numbers refer to `pkgvar/permol_seq_maskoff_logfull/robocop/robocop.py` unless another file is named.

| kind | value | where |
|---|---|---|
| no information: no call (ambiguous call, deletion, or a position the read does not cover) | 0.0 (log 1): the layer starts at 0.0 (`:300`) and the code never writes to a no-call cell | `:1569-1570` add terms only where the call is 1 or 0 |
| no information: base is not the layer's target base | exact: 0.0. floored: log(1e-30) on every emitting state, a per-position constant that cancels | `:1571` |
| silent states | 0.0. Asserted | `:979` |
| evidence | `+log(ps)` for a methylated call, `+log1p(-ps)` for an unmethylated call | `:1566-1570` |
| ps guard | `0 < ps < 1` and finite, else AssertionError. Calls must be -1, 0 or 1 | `:1563-1565` |
| forbidden | -inf in single cells. Here it comes only from PWM zeros (`build_log_emission_mat_from_pwm`, `:335`) | (the Bernoulli code never writes -inf) |
| impossible: every emitting state is -inf at one position | Python asserts before C (`:975-978`). C returns -2 (`pkg/robocop/algo_log.c:138,161-163,209,230-232,285`). Python asserts rc == ABI (`:1038,1048,1061`) | the driver does not catch it: `decode_one` has only try/finally, which closes the file and re-raises |
| other gates | no NaN (`:968`), no positive log value (`:970`, which catches a linear layer leaking in), posterior in [0,1] (`:1062`). The driver checks each read for no NaN, emitting rows summing to 1 within 1e-9, and silent columns exactly 0 | `run_permol_window_logfull.py:305-307` |

Fitted `ps` ranges (T7): arm A 0.0242-0.4593, arm C 0.0242-0.4593, arm B 0.001-0.999999 (pi = 0.3 reaches the
`PS_CEIL` clip of 1-1e-6). None is exactly 0 or 1.

## Validation
| | test | result |
|---|---|---|
| a | floored, first 20 reads, vs old linear arm A (`compare_floored20_vs_linear.json`) | 20/20 matched. Max \|d\| 3.7e-9 (unknown), 0 for nucleosome/background/ABF1. 0 cells > 0.01 out of 51,199 per column. Per-read loglik \|d\| <= 2.3e-10. 99.9997% of float32 cells are bit-equal |
| a' | exact, all 439 reads, vs old linear (`compare_exact_vs_linear.json`) | max \|d\| 3.0e-8 (unknown), 1.5e-8 (background), 0 for nucleosome/ABF1. 0 cells > 0.01. Loglik differs by up to 4.95e5, which is the state-independent log(1e-30) constant (exact vs floored on the same reads: posteriors max \|d\| 3.7e-9, loglik difference the same 4.95e5) |
| b | pooled check, exact (`pooled_check/pooled_check.json`) | r >= 0.99999992 on every column (nucleosome 0.999999993). Raw max \|d\| 1.3e-3 comes from the aggregate's 1e-4 save truncation. After the same truncation, max \|d\| <= 8.7e-14 (columns) and 9.0e-14 (states). loglik_agg - loglik_pooled - sum log C(n,k) = -1.1e-10. Reader vs aggregate (k, n): max \|d\| 0. Old linear version: r 0.99999996, max \|d\| 4.9e-4 |
| c | checks 1-4, exact, 439 reads vs `permol_proto/checks_armA_effoff.json` | identical: check 1 r 0.4440 (old 0.4441), FAIL as before (it is mis-specified, P4). Check 2: recall 4/4, median distance 0.5 bp, 434/439 distinct paths, dyad sd 27.2 bp. Check 3: ABF1 median 6.0e-6 / 8.1e-4, max 6.9e-6 / 1.06e-3, 0 reads > 0.10 at either site, Spearman -0.68 / -0.74, AUROC 1.0 (sequence-driven, P3). Check 4: 0.900 (controls 0.351) |
| d | unit tests (`unit_tests_{exact,floored}.json`) | 7/7 pass in both modes (detail below) |
| e | runtime | 0.79 s/read (348.6 s for 439 reads), peak RSS 1.54 GiB. Old: 0.69 s/read, 1.77 GiB. Different jobs and possibly different nodes, so the 15% gap is unverified as a like-for-like number |

Aggregate side note: on this window the old linear aggregate (`permol_proto/agg_baseline`), log floored
and log exact agree to max \|d\| <= 1.9e-13 per column. In the aggregate too, the numerics change nothing
here.

Unit tests (exact; floored gives the same results):
- T1: a real 5,000 bp Watson read with 1,500 bp (386 calls) set to no-call. Fiber layers in that stretch
  are exactly 0.0, with 0 -inf cells. The stretch's total emission equals the sequence layer alone.
  Row sum deviation 9.5e-15.
- T2: a read with no calls gives the same posterior as one with layers 1-6 zeroed by hand (max \|d\| 0,
  same loglik).
- T3: all 2,798 TF states set to -inf at one position. They get posterior exactly 0 there, and the row
  sums to 1.
- T4: one all -inf row in the per-read path raises AssertionError "1 positions have every emitting state
  at -inf". It is not swallowed.
- T5: the same row passed straight to C: fward_log and bward_log both return -2. The control returns the ABI.
- T6: ps = 0, ps = 1 and call = 2 are each refused.
- T7: the ps ranges above.

## Needs the user's decision
1. This port answers open decision 4 for `singlefiber/` only. `analysis/singlefiber/` now has a log-space
   path, and the linear one is kept. Whether log space becomes the default for per-read work depends on
   decision 1 (log space as production).
2. The port does not change the science. Per-read ABF1 is still at the data ceiling (0 reads > 0.10).
   The next steps in `singlefiber_review/REPORT.md` (site-conditioned per-read test, footprint fits)
   still need the user's choices there.
3. Arms B, C and efficiency-on were not re-run in log space. The same sbatch script with `--array=1-9`
   would run them. Given (a), the expected result is the same as the old arms, but that is unverified.
