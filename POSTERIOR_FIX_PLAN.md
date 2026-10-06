# Plan: fix B to production, log-space reference, viewer confirmation, refit

Approved 2026-10-05. **Status: Phase B landed locally 2026-10-06 (gate passed on the pre-rename build; re-confirm on `bf219a43` pending; chrXII rDNA `n_excluded` ≠ 0 is open)** — merged as `8b151c2`,
not pushed; cutover record in `POSTERIOR_SCALING.md` "Binary cutover". Phases A, D, C not started.

**Run order: B → A → D → C.** B lands the production fix; A builds log-space against it; D puts both in
the Run Browser so the agreement is visible; C refits once the arithmetic is settled.

## ⚠ Decided AFTER approval — read this before starting Phase B

Phase B step 4 ("rebuild `librobocop.so`") hits a blocker the plan body does not mention.

**Fix B adds an 8th C parameter**, `int silent_states_begin` — it is needed because the normalisation
divides by the sum over *emitting* states only (columns `[0, 3330)`), and C cannot derive that boundary
from `n_states` alone. But **67 of the 79 `robocop.py` copies still declare the 7-arg form**; only the
12 `_norm` / `_logspace` trees declare 8. Rebuilding the shared `.so` from fix B as-is would make those
67 pass 7 arguments to an 8-arg function — the output pointer lands in the `silent_states_begin` slot and
C writes to a wild address. No exception, no warning.

**A rejected idea, recorded so it is not retried:** "zero the silent columns, then sum over all
`n_states`, so the parameter is unnecessary." This is **circular** — zeroing them requires knowing where
they start, which is the parameter. Pre-zeroing from Python does not work either, since `p_table` is the
*output* and C overwrites every column.

**DECISION (user, 2026-10-06): edit only the trees that will actually be used, and add a range guard.**

```
edit (4 files):  pkg/robocop/robocop.py
                 pkgvar/{seq,fiber,seqonly}_maskoff_tune75/robocop/robocop.py
leave alone:     the other ~60 pkgvar trees
already correct: the 12 _norm / _logspace trees (they pass 8 args today)
```

Rationale: those ~60 trees are **records of finished runs** and will not be re-run. The 8th argument does
not invalidate what they record — "sequence layer on, 75 TFs live" stays true. Only *executing* one
against the new binary would break, and the guard below makes that loud instead of silent.

The edit is two lines per file and touches **only `robocop/robocop.py`**, never
`robocop/utils/robocopExtras.py` — the file that defines what a variant *is*. So no variant's layer/mask
meaning changes, and `CLAUDE.md` rule 3's purpose is preserved.

**Mandatory guard**, at the top of `posterior_decoding`, before any pointer dereference:

```c
if (silent_states_begin <= 0 || silent_states_begin > n_states) {
    fprintf(stderr, "posterior_decoding: silent_states_begin=%d out of range (n_states=%d)\n"
                    "  -- this caller passes 7 args to the 8-arg ABI. Add\n"
                    "     dshared['silent_states_begin'] to the posterior_decoding call\n"
                    "     in that tree's robocop/robocop.py\n", silent_states_begin, n_states);
    return;
}
```

A 7-arg caller puts a pointer value (~1e14) in that slot, so the guard fires immediately and names the
exact fix. Residual risk: stack garbage could in principle land inside `[1, 3485]`; unlikely, and the
alternative (editing all 67) carries its own one-missed-file risk.

**The call-site edit**, for reference — `dshared['silent_states_begin']` is already in scope:

```python
     robocopC.posterior_decoding(
         ftable, btable, bscaling_factors,
         scaling_factors, dshared['n_states'], n_obs,
+        dshared['silent_states_begin'],
         p_table)
```

**Root cause, not fixed here:** one C change multiplies into 79 copy-paste edits because layer/mask and
`.pkl` choices live in library *source* rather than `config.ini`. `things_to_revisit_before_shipping.md`
§3.1 already flags this **[P0]**: "Ship one package whose behaviour comes from `config.ini`, and delete
the variants." Doing that first would make the 8th argument a single-file edit. It is a real refactor and
deliberately out of scope here.

## Context

There are **two** bugs, not one.

**Bug 1 — posterior blow-up.** `posterior_decoding` (`algo.c:239-240`) multiplies α·β by `sr`, a
~5,000-term running product whose overflow guard is commented out (`algo.c:216`), with no per-position
normalisation. Loud and rare: 1 of 63 segments in a genome task; chrXII `segment_1044` reaches row sums
of 2.8e91. Already fixed three ways (A read-side, B in-C linear, C in-C log-sum-exp); all three agree.

**Bug 2 — emission floor.** `robocop.py:801` computes `binom.pmf` in linear double; at high coverage it
returns **exactly 0.0**, and `robocopExtras.py:104-106` replaces every zero with `1e-30`. **Silent** —
row sums stay 1, nothing detects it. It does not blur the answer, it **inverts** it. At n=4379, ratio of
most-protected to most-open state:

```
 k        true log10      floored log10
 0            +929.1            -16.8     <- SIGN FLIPPED
 4379       -5334.5              0.0      <- states indistinguishable
```

Extent: ~8,479 decoded positions, essentially all in chrXII's rDNA (451,435–468,729). `binom.logpmf` is
exact there, but a log emission matrix cannot feed a linear recursion (`algo.c:58, :85, :111, :159,
:175, :190` all *multiply*), so log-space is the **enabling change** for exact emissions.

**Outcome wanted:** fix B as the production decoder with the shared `.so` rebuilt, plus a validated
log-space implementation that fixes bug 2.

## Decisions (user, 2026-10-05)

1. Production decoder = **fix B, the full in-C fix with `librobocop.so` rebuilt** — not fix A.
2. Archive the old binary; **record the cutover** in a commit message *and* a permanent note.
3. **Fix A stays on `main` as the historical reader** for the ~1,600 existing decodes. Once decodes are
   normalised at source it is a divide-by-~1.0 no-op (idempotent; yields 0 not NaN on an all-zero row,
   `score_robocop.py:216-217`).
4. Log-space ships as a **validated reference**, not the default.
5. **Refit is Phase C**, a separate campaign, so causes stay attributable.
6. **chrXII rDNA is in scope** — it becomes the locus that proves the fix.
7. EM out of scope (`robocop_em_combined.py:107` has `iterations=10` but nothing references that file;
   `em_smoke_gate.py:3` records the re-estimation has never executed).

**Run B before A.** B is small and already agrees across 24 arms; landing it first means A validates
against a committed baseline with matching source. The one standing constraint this retires:
`librobocop.so` is invariant through Phase A and changes **exactly once**, deliberately, in Phase B.

---

## Phase B — fix B becomes production

Today `main` ships two binaries no committed source reproduces:

```
main       algo.c                 md5 d1ba123c…   PRISTINE, unfixed
branch B   algo.c                 md5 e800836f…   the fix, not on main
main       librobocop_norm.so                     binary, no source on main
PR #1      logspace-posterior -> main             points at fix C, not B
```

**Steps**

1. Merge `normalise-posterior-c` into `main` (commit `23a390d`; old code already commented, not deleted).
2. Retarget PR #1 off fix C — repoint at `normalise-posterior-c` or close and reopen.
3. **Archive** the current binary as `pkg/robocop/librobocop_prenorm.so`, md5
   `6a0724bf6ef7b8bc4313927b931ac685`, committed.
4. **Rebuild** `pkg/robocop/librobocop.so` with the existing `gccCompile` (unchanged flags, so fix B is
   the only delta). Record the new md5.
5. Add `cshared` + `so_md5` to `RUN_INFO.json` (`write_run_info.py:104-123`, schema 2) and an `so`
   column to `build_run_matrix.py`'s provenance dict, stamped from `dshared['robocopC']`. ~6 lines.
   Closes the gap where **no run currently records its binary** (only 10 single-fiber runs do).
6. Leave the 12 `*_norm` / `*_logspace` trees untouched (rule 3) so their finished runs stay readable.

**The cutover record** — three places, because a commit message is not findable in six months:

- **Commit message:** names this as the commit where `librobocop.so` changed, both md5s, the date, and
  what it means either side.
- **`POSTERIOR_SCALING.md`** (new, repo root) gains `## Binary cutover` with the rule that makes 1,600
  unstamped runs interpretable: *no `so_md5` in `RUN_INFO.json` ⇒ predates the cutover ⇒ produced by
  `librobocop_prenorm.so`.*
- **`CLAUDE.md`** gains one gotcha line pointing at it.

**Verify**

- `md5sum librobocop_prenorm.so` = `6a0724bf…` (the archive is the real old binary, not a rebuild).
- `git show main:pkg/robocop/algo.c | md5sum` = `e800836f…`.
- **Re-decode `w30k_bv75_49` with the stock driver and compare to `robocop_w30k_bv75_49_B_both`** — must
  agree to round-off, since it is the same code by a different route. This is the gate.
- `count_calls.py` `n_excluded` = 0 on a fresh chrXII decode; a fresh `RUN_INFO.json` carries `so_md5`.

**Rollback is not free.** `cp librobocop_prenorm.so librobocop.so` plus reverting the merge — but
decodes made in between are correct yet not comparable to pre-cutover runs. So: land B, verify on one
re-decoded window, *then* start new science on it.

---

## Phase A — log-space reference

**New file `pkg/robocop/algo_log.c`; `algo.c` untouched.** Three reasons: `librobocop.so` stays
reproducible from source; the new `.so` is a strict **superset** of old symbols, so both paths run in one
process (which makes the equivalence gate airtight); and "comment out, never delete" is satisfied
vacuously.

New symbols (prototypes in `bc.h`): `log_add`, `log_normalize`, `fward_log`, `bward_log`,
`posterior_decoding_log`, `posterior_decoding_normlin`.

- **`log_normalize` replaces `normalize()`, it is not removed.** Per-row max subtraction, offsets
  accumulated where `scaling_factors` lived. Reproduces the partial-sum (`[0, ssb)`) / full-row-divide
  asymmetry exactly; drops only the `DBL_MIN` addition. **Without it `|log_alpha|` reaches ~8e7 where
  double spacing is 3e-8 and the accuracy gates become unreachable.** What gets deleted is the running
  product `sr`, not per-position rescaling.
- `log_emis` is **2-D `(n_obs, n_states)`, pre-collapsed over channels**; `n_vars` dropped. Removes the
  7-term product `bward` recomputes inside the child loop (`algo.c:158-159, :174-175, :189-190`).
- Trailing `int abi = 20261005`, checked before any dereference, `restype = c_int`, Python asserts it —
  turns a silent ctypes/stack mismatch into a loud failure.
- **C initialises `log_ftable`/`log_btable` to `-INFINITY` itself.** `0.0` is `log(1)`, the worst wrong
  identity. Python also passes `np.full(-inf)`, redundantly.
- `posterior_decoding_log` **exponentiates internally and writes linear, row-normalised** values, so the
  h5 contract is unchanged and nothing downstream moves.
- Use `Ir(i,j,n_obs,n_states)` for `btable` — `bward` stores rows **reversed** (`algo.c:137-138, :240`).
- Preserve silent-state order literally: forward **ascending** (`:62, :90, :115`), backward **descending**
  (`:154, :185`). 154 silent→silent edges make this a correctness requirement.
- Write `0.0` to silent columns `[ssb, n_states)` and gate their `nnz` against the B arm — `B_seq` really
  has 44,957 non-zero silent cells.

**Build:** `pkg/robocop/gccCompile_logfull` → `gcc -O2 -ffp-contract=off -g -fPIC -Wall -Wextra
-pedantic bc.c algo.c algo_log.c -shared -o librobocop_logfull.so`. `-ffp-contract=off` because gcc
defaults to `fast` and would contract to FMA, breaking bit-identity. md5-check `librobocop.so` around
every compile — `gccCompile` writes that name and one mistyped `-o` destroys it.

**Python (new trees only):** `robocop.py:272` `np.ones`→`np.zeros`; `:801` `pmf`→`logpmf`; `:803`
`*=`→`+=`; `:917-918` `np.zeros`→`np.full(-inf)`; `:932/:966/:973` drop `np.longdouble`; `:943/:956`
argtypes; `:968` `calc_sr` call commented; `:1021-1046` diagnostic PNG block disabled (then
`del d['emission']` before `fward_log`). `bc.c:123` PWM→log **and `bc.c:126` silent columns `1.0`→`0.0`**.
Build the log matrix **in place** (976 MB; a second copy would double it).

**Trees:** `pkgvar/{seq,fiber,seqonly}_maskoff_tune75_logfull`, copies of the `_norm` trees with
`robocopC` overridden at `robocop_no_em.py:23`. Plus three `run_split_*_logfull.py` drivers — **not
optional**: `write_run_info.driver_tree` (`write_run_info.py:45-59`) resolves `pkg_tree` only from a
driver's literal `sys.path.insert`.

### ⚠ Four things that are silently catastrophic if done wrong

1. **`robocop.py:806`'s literal `0` means "no observation here" → `0.0` (log 1), NEVER `-inf`.** The
   floor lifts it to a uniform value that cancels in normalisation. Backwards, and ~60% of positions go
   all-`-inf` and save as all zeros — which `count_calls.py` will **not** flag, because 0 is inside
   [0,1]. Only the tune75/ABF1 masks and `pwm_emission` zeros mean forbidden → `-inf`.
2. **DELETE (comment out) the `1e-30` floor at `robocopExtras.py:104-106`. Do not convert it to `-inf`.**
   In a log matrix `0.0` is log 1 and is everywhere, so `== 0 → -inf` hits every one of them — the same
   catastrophe as #1. `logpmf` is finite for every `(k,n,p)` that occurs, so the floor is unnecessary.
   `+DBL_MIN` was also today's only `0/0 → NaN` guard, which is why the check below is mandatory.
3. **Three more copies of the same code exist**, and a missed one is silently wrong:
   `robocopExtras.py:101` (live in `fiber_` only) and `:110-111` (live in `seqonly_`) → `= 0.0`; plus
   `singlefiber/run_permol_window.py:137-139` (`*=`→`+=`), `singlefiber/pooled_emission_check.py:113-116`
   and `pkgvar/permol_seq_maskoff/.../robocopExtras.py:180-182` (two more floors).
4. **`build_run_matrix.py:343-345, :362` derive tree attributes by grepping for `= 1`.** A log tree writes
   `= 0`, so `seq_layer` / `fiber_layer` / `mask` would be **misreported in `runs.json` and the viewer**
   with nothing flagging it. Widen the regexes and assert each new tree's derived layers match its parent.

**Mandatory gate for #1–#3:** assert `np.nanmax(d['emission']) <= 0.0` before the C call (a leftover
linear layer is `> 0` and trips immediately), and assert no position has all emitting states `-inf`.

### Phase A verification

Four arms, so the recursion rewrite and the emission fix are never confounded: **B** (production), **D0**
(log recursion + today's floored emissions), **D1** (log recursion + exact `logpmf`), **S** (seqonly null
control — its fiber layers are off, so it can only ever be a Stage-1 control). D0 vs D1 is one env
switch, `ROBOCOP_LOG_EMISSION ∈ {floored, exact}`, printed and recorded in `RUN_INFO.json`.

**Stage 1 — equivalence, chrXIV:396,001–433,000.** Verified **provably underflow-free**: 1,877 distinct
`(n,k)` pairs × 249 fitted `p`, **zero** underflows, max n=123. So D0 **must** equal B.

- **A1** bit-identity: `fward`/`bward`/`posterior_decoding` from `librobocop_logfull.so` vs
  `librobocop.so` on identical inputs → bit-identical. Proves `-O2` + the extra TU changed nothing.
- **A2** in-process linear vs log, same emission matrix: **max abs and max rel < 1e-9** (not 1e-15 — a
  3,330-term log-sum-exp costs bits; 1e-9 is still five orders under the 1e-4 save floor).
- **A3** likelihood: D0 within **1e-6** of B's stored `-489955.03749446274` (segment_0). D0 writes
  `log(1e-30)` at the `:806` cells so this is a real gate. **D1's likelihood moves ~+470,000 per segment
  — record, never gate.** (That stored value is 96% `1e-30`-floor artefact: 6,815 floored
  channel-positions × `ln(1e-30)` = −470,763.)
- **A4** end-to-end via `compare_posterior_fixes.py` — needs `ARMS` (`:33`) and the dir template (`:73`)
  extended, **plus a raw `p_table` comparison**, since `region_optable` already re-normalises and sums
  over blocks so per-state errors can cancel.
- **A5** null control: `seqonly_logfull` vs `seqonly_norm` agree at 1e-9 **and on chrXII**.

**Stage 2 — divergence, chrXII:450,001–470,000** (5 overlapping 5 kb windows covering 451,435–468,729).
Predictions written down first:

1. count of `(k,p)` with `pmf == 0` and finite `logpmf` must be **> 0**, else the premise is void here;
2. **D1's posteriors become near-degenerate, D0's do not** — metric: mean of `max_j p[i,j]` over
   positions with n>1000; predict D1 ≫ D0;
3. **`logL(D1) < logL(D0)`** by order 1e6–1e8 over the window; wrong sign ⇒ something inverted;
4. neither arm emits a posterior outside [0,1]; `n_excluded` = 0 in both;
5. **D0 still equals B on chrXII**, which is what attributes the D1−D0 difference to the emission alone.

**Gate on likelihood-ratio direction at named `(k,n)` positions, not on row sums** — row sums → ~1 and
`n_excluded` → 0 follow from the posterior normalisation alone, and fix B already achieves both.
**Falsifier:** D1 ≡ D0 on chrXII while (1) holds ⇒ the log emission is not reaching the recursion.

**Rollback (Phase A):** delete `librobocop_logfull.so`, the three trees and three drivers. `main` and all
~78 existing trees untouched by construction; mid-flight abort is free.

---

## Phase D — the viewer deliverable (how you confirm it yourself)

**12 runs registered in the Run Browser**, two windows × two arms × three layer configs (the standing
instruction: tune on both layers, decode all three):

```
w30k_bv75_49_Bprod_{both,fib,seq}      chrXIV 396,001-433,000   post-cutover production B
w30k_bv75_49_logfull_{both,fib,seq}    chrXIV 396,001-433,000   full log-space
rdna_chrXII_Bprod_{both,fib,seq}       chrXII 450,001-470,000   post-cutover production B
rdna_chrXII_logfull_{both,fib,seq}     chrXII 450,001-470,000   full log-space
```

`_both` is the fib+seq arm — that is the pair to open side by side.

### ⚠ What "the same" should and should not mean

| window | expectation | why |
|---|---|---|
| **chrXIV** `_both` | **identical** (within the viewer's 0.01 resolution) | provably underflow-free, so log-space has nothing to repair. **This is the consistency check you asked for.** |
| **chrXII** `_both` | **must differ** | this is the rDNA array where `binom.pmf` underflows. If these two look identical, the log-space emission fix **did nothing** and Phase A has failed its own falsifier. |

So: same on chrXIV proves the implementations agree; **different on chrXII proves the second bug is
actually fixed.** Both results are required. Expect the chrXII difference as sharper, more confident
peaks in `logfull` where `Bprod` is flat and uncommitted.

### Honest limit of the viewer as evidence

The payload is uint8 at `q=100` (`extract_run.py:40, :141`), so viewer resolution is **0.01** — far
coarser than the 1e-9 numerical gate. The viewer is the visual confirmation; `compare_posterior_fixes.py`
on the raw `p_table` is the actual proof. Report both, never the viewer alone.

### Steps

1. `write_factor_table.py` per decode dir (no `--normalise` needed post-cutover — the posterior arrives
   normalised, and the flag would be a second divide-by-~1).
2. `viewer_site/extract_run.py <run_id> <chrom>` for each. **`chrXII 450001 470000` must be added to
   `viewer_site/windows.tsv`** — it currently holds only chrI, chrII, chrIV and chrXIV, so without this
   the rDNA runs extract zero blocks and appear empty. Note chrXIV's existing viewer window is
   400,001–430,000, narrower than the 396,001–433,000 decode, so the first and last segments are
   clipped from view; widen it or accept the crop.
3. `viewer_site/build_run_matrix.py`, then refresh `build_info.json` (`n_runs`, `built`, `git`).
4. Pin all 12 to one family in `viewer_site/run_overrides.tsv` — `family = posterior fix ablation` —
   otherwise the `_both` arms sort away from their own `_fib`/`_seq` siblings, exactly as happened with
   the first 24 ablation arms (`family_of` returns `"tuner"` whenever provenance carries a campaign,
   `build_run_matrix.py:483`).
5. Serve: `bash viewer_site/build_site.sh && bash viewer_site/serve.sh 8765`.

**Done means:** the site lists all 12, and opening `w30k_bv75_49_Bprod_both` against
`w30k_bv75_49_logfull_both` shows one curve, while the chrXII pair visibly differs.

---

## Phase C — refit, as its own campaign

Phase B makes chrXII's `n_excluded` (2,985,920 / 1,939,360 / 2,509,280 across campaigns) go to 0, so
those factor-positions re-enter `occ = post_sum / block_len` (`count_calls.py:177-178`) — the tuner's
objective. **Every tuned λ/w vector stops being a fixed point.**

New named campaign against corrected occupancies, not a continuation. Report `tf_prob` and entry cost,
not λ. Expect movement concentrated in factors with chrXII mass; chrXIV and chrII targets should barely
move (chrII has 19 rows above n=200, none that actually underflow). Out of scope: changing the tuner's
update rule, or masking rDNA from the objective (offered, declined in favour of refitting).

---

## Notes

- **Re-measure, don't hardcode:** row-sum bands (0.9968 min across bv75 arms, 0.9987 on one segment,
  0.9952 on a genome segment); min non-zero transition is **1.55e-39** in `robocop_train_tw_bv75_49`,
  not the 1.36e-12 of `robocop_train_fiberonly` — assert `> 1e-300` instead of quoting either.
- **Cost:** `-O0` today; measured baseline ~42 s/segment, dominated by the interpreted emission loop at
  `robocop.py:791-806`, not the C recursion. Budget +5–15% and measure.
- **A C-only change would fix nothing** — `binom.pmf` is scipy, in Python; there is zero binomial code in
  `algo.c`/`bc.c`. **Everything after decoding is unchanged**, because the posterior is exponentiated in
  C, so the h5 contract (linear, normalised, float64, CSR + `shape` attr, 1e-4 floor) is identical and
  ~44 analysis files plus 82 package copies keep working untouched.
- `POSTERIOR_SCALING.md` also carries the standing account of bug 1 (convention, symptoms, how to check
  row sums, what not to do) — draft written and verified, plus stitches into `CLAUDE.md`,
  `things_to_revisit_before_shipping.md` (beside `### 2.7`) and `HANDOFF.md` (beside `### 4.7`).
