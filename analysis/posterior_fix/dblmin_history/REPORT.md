# DBL_MIN in the RoboCOP decoder: why, when, and what log space does instead (2026-10-24)

Investigated by a subagent; key claims re-checked against git and the output files in this folder.

## Q1. Why was DBL_MIN there?
It prevents a divide-by-zero. `normalize()` (`pkg/robocop/algo.c:21-27`) divides each row by the sum of its
emitting cells. If that sum is 0 (every emission 0, e.g. an N base, PWM zero or hard mask; or no allowed state
reachable from the previous row), every cell becomes 0/0 = NaN and the NaN spreads through the segment. Adding
DBL_MIN (2.2e-308) to every cell keeps the sum non-zero. The cost: an impossible position becomes a uniform row
(forbidden states included) and the run continues silently.

Toy run through the real C (`toy_dblmin.py` -> `toy_dblmin.out`; `build/lib_nodblmin.so` = algo.c with `+ DBL_MIN` -> `+ 0.0`):

| case | with DBL_MIN | without |
|---|---|---|
| A: every emission at pos 3 is 0 | forward row 3 uniform [1/3,1/3,1/3,1/3]; loglik -711.1, mostly padding (ln DBL_MIN = -708.4) | rows 3-4 NaN, sf[3] = 0, loglik NaN |
| B: no valid path (m1 forbidden at pos 2, only m2 allowed at pos 3) | m1 at pos 2 gets posterior 0.132 although its emission is 0 | NaN from row 3 on |

Fix B would hide the NaN if the padding were removed: `posterior_decoding_v2` writes zeros when `rowsum > 0` is
false (`algo.c:300-309`), and NaN > 0 is false. Audit bug C2 (`ROBOCOP_C_CODE_AUDIT_REPORT.md:184-190`) recommends
failing loudly instead.

## Q2. When did DBL_MIN come in?
Upstream, not this fork. Sneha Mitra added it on 2021-08-31 in `c6a8727` (`#include <float.h>`, both `+ DBL_MIN`,
and `if(isinf(sr[i-1])) sr[i-1] = DBL_MAX;` in `calc_sr`), one day after tag v1.0.0 (`7c7fbf3`), whose algo.c has
no DBL. Upstream commented out the isinf guard in `17163cd` (2022-01-05). The public upstream master
(`79320ab`, github.com/HarteminkLab/RoboCOP) still has it, and `git diff 79320ab 5e6fa4e -- pkg/robocop/algo.c` is
empty: the fork's root copied upstream's file exactly. Why upstream added it is not recorded (unverified).

## Q3. Is DBL_MIN still needed in log space?
The padding is not; the case it guarded still must be handled, and log space fails loudly instead of papering
over it. In log space nothing underflows (small probabilities are large negative numbers). The only case left is
a row where no state is possible (all -inf):
- `log_normalize` (`algo_log.c:57-74`) leaves such a row at -inf; `fward_log` / `bward_log` / `posterior_decoding_log`
  return rc -2 ("no emitting mass at position i", `:138`, `:161-163`, `:209`, `:230-232`, `:270-285`).
- Python (`pkgvar/*_logfull/robocop/robocop.py:976-979`) asserts no position has every emitting state at -inf before
  the recursion, and asserts every rc equals the ABI after it.
- Toy: case A trips the Python pre-check; case B passes it but C returns -2 and the rc assert stops the run.
Log space still rescales every row (subtracting a log offset), which must stay (`POSTERIOR_FIX_PLAN.md:180-181`);
only the fake mass is gone.

## Q4. Defect 3, and how it relates to C1
One root cause, the DBL_MIN padding, acting in two places.
- **Defect 3** is the padding on every row of both passes: a state that should be exactly 0 (masked TF, forbidden
  nucleosome column, zero emission) gets DBL_MIN/S instead, S = the row's true emitting sum. With the 1e-30 floor,
  S is 1e-31 to 1e-80, so each such cell carries 1e-277 to 1e-228 of fake mass, and paths through forbidden cells
  exist. Emissions alternate between ~1e-31 and ~1e-60 per row, enough for such a path to grow until it beats the
  real ones.
- **C1** is the same padding meeting an ordering bug in the backward pass (silent cells built before rescaling, so
  the padding swamps them). The audit's C1 fix reorders but keeps the padding, so it does not fix defect 3.

bo09 chrIV:1,200,001-1,205,000 (seg 300), all arms on identical captured inputs (`seg300/`):

| arm | loglik | rows off log space by > 0.01 |
|---|---|---|
| fix B (`librobocop.so`, with DBL_MIN) | -486,260.97 | 147 (rows 1682-1828), max 1.0 |
| same algo.c without DBL_MIN, float64 | -486,267.708292105 | 0 |
| log space (`librobocop_logfull.so`, floored emission) | -486,267.708292105 | reference |

- The C1 fix also gives 147 rows (`fibercop_audit/logcompare/seg300/compare_variants.json`); the 80-bit replica
  without DBL_MIN equals log space (`replica_longdouble.json`).
- In those rows log space puts one nucleosome (occupancy 1.000); fix B calls background (p = 1.0 at 1682-1686).
- Fix B's loglik is 6.74 nats higher: padding only adds mass, so padding-only paths carry ~850x the mass of all
  real paths (inference; not traced with Viterbi).
- Mixing tables (`fb_split.out`): forward A + backward L -> 139 bad rows; forward L + backward A -> 16; both A -> 147.
  On this segment the damage is mostly in the forward pass.
- Not rDNA: seg 300 is ordinary chrIV. How often this happens genome-wide is unverified (check: run the no-DBL_MIN
  arm against log space on every segment, as c1scan/scan_genome.py does).
