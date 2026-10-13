# Posterior scaling — the convention, the bug, and the binary cutover

**Short version.** A stored posterior must sum to 1 at every position, over the **emitting states
only** (columns `[:silent_states_begin]`). Decodes made before 2026-10-06 do not guarantee this; decodes
made after the binary cutover do. To tell them apart, look for `so_md5` in the run's `RUN_INFO.json`.
The silent columns are **not** bounded by the fix; on chrXII rDNA a few still blow up (§5, open issue).

This note covers bug 1 (the posterior is not normalised). Bug 2 (the emission floor inverts evidence) is
**not** fixed by the cutover; it is fixed only in the log-space decoder (`_logfull` trees), which also
avoids bug 3 (`normalize()`'s `+DBL_MIN`, `algo.c:22,26`, wrong at ~1,200 rDNA positions). See HANDOFF §4.8
and `POSTERIOR_FIX_PLAN.md` "Results" and "Handoff, 2026-10-12".

Line numbers below were read on 2026-10-06. "Pre-cutover" anchors are in
`git show 8b151c2^1:pkg/robocop/algo.c`; "current" anchors are in the working-tree `pkg/robocop/algo.c`.

---

## 1. The convention

- The posterior is the probability of each state at each position. The paper says the output *is* that
  probability (NAR 49:7925, Fig 2E).
- Sum over **emitting states only**: `p[:, :silent_states_begin]`. This matches the decoder's own
  `normalize()`, which sums columns `0 … idx-1` (`algo.c:21-22`) and is called with
  `silent_states_begin` as `idx` (`algo.c:74`).
- Do **not** sum over all `n_states`. Silent states can hold mass (HANDOFF §4.8: 44,957 non-zero cells in
  one seq-only segment).
- Expect emitting row sums in about `[0.995, 1.0]`, not exactly 1. The shortfall is the save-time floor,
  `v[v < 1e-4] = 0` (`pkg/robocop/robocop.py:33`), not a defect.

## 2. The bug (pre-cutover binary)

**What the old code did.** `posterior_decoding` (now renamed `posterior_decoding_v2`) wrote `(sb[i]·b)·(sr[i]·f)` with no per-position
normalisation (pre-cutover `algo.c:239-240`). `sr` is a running product of ~5,000 `sb/sf` ratios
(pre-cutover `algo.c:214`), and this fork commented out its `isinf` guard (pre-cutover `algo.c:216`;
still commented at current `algo.c:218`). One bad ratio rescales every position upstream of it.

**Symptoms** (numbers from HANDOFF §4.8, not re-measured here):

- Emitting row sums far above 1 — up to **2.8e91** on chrXII `segment_1044` (4,707 of 5,000 rows).
- Values stay finite in float64, so `np.isfinite` passes them. A float32 cast turns them to `inf`.
- Rare: about 1 of 63 segments in a genome task. Not rDNA-only (see memory note on extreme weights).
- Overlap averaging of a scaled segment with a clean one gives spurious mid values (the viewer's 0.50 at
  chrXIV:412,651, per the comment at `analysis/score_robocop.py:213-214`).

**The fix (fix B, now in the binary).** Current `algo.c:287-311`: multiply `b·f` unscaled, sum the first
`silent_states_begin` columns, divide the whole row by that sum. Per-position scale factors cancel in that
ratio, so `sb` and `sr` are no longer used. A row with zero emitting mass is left all zeros
(`algo.c:304-310`) — treat an all-zero row as undecodable.

## 3. How to check any decode

Run from `analysis/` (tested 2026-10-06 on `robocop_w30k_bv75_49_A_both` and `_B_both`: min ≈ 0.998,
max = 1.0):

```python
import h5py, score_robocop as S
dec = S.load_decode("robocop_<label>/")
sb = int(dec["dshared"]["silent_states_begin"])
for fn in dec["infofiles"]:
    with h5py.File(fn, "r") as f:
        for k in f:
            if k.startswith("segment_"):
                rs = S._get_sparse_todense(f, k + "/posterior")[:, :sb].sum(axis=1)
                print(k, rs.min(), rs.max())     # expect ~[0.995, 1.0]
```

Also: `count_calls.py` reports `n_excluded`, the positions where any factor's posterior is outside
`[0, 1]` (`analysis/count_calls.py:147-155`). On a pre-cutover decode, non-zero means bug 1. On a
post-cutover decode, non-zero is the silent-column issue in §5, not bug 1.

**Reading pre-cutover decodes.** `score_robocop.region_optable` normalises on read
(`analysis/score_robocop.py:215-217`), over `[:silent_states_begin]`, before the overlap average. This is
fix A. It stays: it is the only correct reader for the ~1,600 pre-cutover decodes. On a post-cutover decode
it divides by ~1, so it is harmless.

## 4. What NOT to do

- **Do not clip to 1.** `analysis/write_factor_table.py:175` does `np.clip(..., 0, 1)`; that turns a 1e13
  artefact into a confident call.
- **Do not drop bad positions and call it fixed.** `count_calls.py` drops them and reports `n_excluded`;
  that is a tripwire, not a fix.
- **Do not check finiteness only.** 1e171 is finite. Check the `[0, 1]` bound on emitting row sums.
- **Do not normalise over all `n_states`.** Slice `[:silent_states_begin]`.
- **Do not rebuild absolute α/β from `sr`.** `sr` is the damaged quantity. For the likelihood use
  `calc_sr`'s return value (current `algo.c:226`, `return log_sf_sum`), which does not use `sr`.
- **Do not read a row sum of 1 as "the emissions are fine".** Bug 2 keeps row sums at 1 while inverting
  the evidence at high coverage (HANDOFF §4.8).

## 5. Open issue: silent columns are not bounded (chrXII rDNA)

**Status (2026-10-12): open, not fixed.** Not caused by `normalize()`'s `DBL_MIN` (bug 3): it is also present in the exact 80-bit replica (`analysis/posterior_fix/replica_longdouble.py`). Likely a
forward/backward time-index mismatch in the silent states — see `POSTERIOR_FIX_PLAN.md` "Handoff,
2026-10-12", open problem 2. The log-space decoder writes silent = 0, so it does not have this issue.

- **What:** after fix B, `count_calls.py` `n_excluded` on `coord_rdna_chrXII.tsv` is **not** 0:
  `both` 2, `fib` 7, `seq` 0 positions, of 21,000.
- **Why it is possible:** fix B divides the whole row by the **emitting** sum (`algo.c:300-303`). Silent
  columns (`≥ silent_states_begin`) are divided too but never summed, so nothing bounds them. At a few
  positions column 3330 reaches up to **1.7e17**.
- **How it reaches the factor table:** `sum_for_dbf_probs_*` set `nuc_padding_length = 0`
  (`pkg/robocop/robocop.py:1181`), so `actual_nuc_end = nuc_start + nuc_len` (`:1189-1191`). In
  `robocop_train_fiberonly/HMMconfig.pkl` that is 2799 + 531 = **3330 = `silent_states_begin`**, the first
  silent state. `posterior_table[i, actual_nuc_end]` (`:1225`) is then read as "nuc_end".
- **Not a fix-B regression:** fix A and fix B both keep the pre-fix silent:emitting ratio; they only rescale.
- **Link to bug 2 (the floor) — still UNVERIFIED.** All 7 positions lie within ±1 of a high-coverage underflowing base,
  but 91% of the stretch does, so that says nothing.
- **On chrXIV:** every silent cell is 0 after the 1e-4 save floor, so the issue does not show there.

**How to check it** on any decode, inside the §3 loop:

```python
p  = S._get_sparse_todense(f, k + "/posterior")
sc = p[:, sb:]                       # silent columns
print(k, sc.max(), (sc > 1).sum())  # expect max <= 1, count 0
```

---

## Binary cutover

**On 2026-10-06 `pkg/robocop/librobocop.so` was replaced by a fix-B build.** Every decode that loads the
shared library now gets a per-position normalised posterior (emitting states). Landed **locally only**
(merge `8b151c2`, not pushed; PR #1 untouched).

**Gate (manager-measured, 2026-10-06):** re-decode of `coord_w30k_chrXIV.tsv` with the stock tune75
drivers vs `robocop_w30k_bv75_49_B_{both,fib,seq}` (made with `librobocop_norm.so`): max abs diff **0** on
every cell, all 9 segments, all 3 layers; log-likelihood identical; emitting row sums 0.9968–1.0. This was
measured on the superseded `c6dfe41c` build. **Re-confirmed on `bf219a43`**: the
`robocop_w30k_bv75_49_Bprod_*` re-decodes carry `so_md5 = bf219a43…` in `RUN_INFO.json` and match the
`B` arm (`POSTERIOR_FIX_PLAN.md` "Results, 2026-10-06").

| | file | md5 | compiler (`strings`) |
|---|---|---|---|
| before | `pkg/robocop/librobocop_prenorm.so` (archived, the real old binary) | `6a0724bf6ef7b8bc4313927b931ac685` | GCC 7.5.0 |
| after | `pkg/robocop/librobocop.so` | `bf219a434d5125028e25f8798b0694c1` | GCC 15.2.0 |
| superseded | `analysis/_trash/20261006_012726/librobocop_c6dfe41c_unrenamed.so` (first build, before the rename; never production after the rename) | `c6dfe41c58427aa291e65acec263ccba` | GCC 15.2.0 |

All three md5s and the compiler strings were re-read on 2026-10-06 (`md5sum`, `strings | grep GCC:`).
A run stamped `so_md5 = c6dfe41c…` used the same arithmetic as `bf219a43…`; only the symbol name differs.

### Which binary made a run

**Rule: no `so_md5` in `RUN_INFO.json` (key absent or `null`) ⇒ the run predates the cutover ⇒ it was made
by `librobocop_prenorm.so`.** Exception: if the run's tree overrides the library in its
`robocop_no_em.py` (the 12 `*_norm` / `*_logspace` trees do, e.g.
`analysis/pkgvar/fiber_maskoff_fit9_bgopen_norm/robocop_no_em.py:22`; line 22 or 23 across the 12), that
tree's own `.so` made it.

`analysis/write_run_info.py` is now schema 2. It adds:

- `cshared` — the library path the decode loaded (config.ini `cshared`, or a tree's override; last
  literal assignment wins, as at runtime).
- `cshared_how` — which of those two it came from.
- `so_md5` — md5 of that file. **Written at decode time only**; a backfill leaves it `null`, because it
  would stamp whatever the path holds now, not what ran.
- `log_emission` — the `ROBOCOP_LOG_EMISSION` env var, for Phase A.

1,633 committed `config.ini` files name `cshared = ../pkg/robocop/librobocop.so` (`grep -rh '^cshared'`),
so the path alone says nothing about which binary ran. Only `so_md5` does.

### What changed besides the fix

- **The compiler.** The new binary was built with the unchanged `pkg/robocop/gccCompile`, but by system
  GCC 15.2.0, not the GCC 7.5.0 that built the old one. Command: `CPATH=$CONDA_PREFIX/include bash
  gccCompile` (system gcc lacks the GSL headers; `algo.c:6` includes GSL but never calls it). The fix-B
  validation binary `librobocop_norm.so` is also GCC 15.2.0, so the B arms already compared against
  pre-cutover runs carry the same compiler change.
- **The ABI and the name.** The function gained an 8th argument, `int silent_states_begin`, and was
  **renamed** `posterior_decoding_v2` (current `algo.c:229-243`). The old symbol is gone:
  `nm -D pkg/robocop/librobocop.so` lists only `posterior_decoding_v2`.

### Old 7-arg trees fail at symbol lookup

- **Ported to `posterior_decoding_v2` (5 call sites):** `pkg/robocop/robocop.py:973-978`, the same lines
  in `analysis/pkgvar/{seq,fiber,seqonly}_maskoff_tune75/robocop/robocop.py`, and the new tree
  `analysis/pkgvar/seq_maskoff_abi8/` (a copy of `seq_maskoff`; `diff -rq` shows only `robocop/robocop.py`
  differs). Its driver is `analysis/run_split_revfix_seq_maskoff_abi8.py`.
- **Not ported: 63 trees** still bind the old name and load the shared `.so` (no override). If one runs,
  Python fails before any call: `AttributeError: …/librobocop.so: undefined symbol: posterior_decoding`
  (reproduced 2026-10-06 with a bare `CDLL(...).posterior_decoding`). This is by design.
- **To port a tree** (as a new tree, rule 3): call `posterior_decoding_v2`, add one `c_int` to its
  `argtypes`, and pass `dshared['silent_states_begin']` before `p_table`.
- **Tuning campaigns.** `tune_concentrations.py:92`'s default `DRIVER` is now the `_abi8` driver (old line
  kept commented at `:91`). A campaign freezes its driver in `state.json` (`:140-146`), and `submit` reads
  it back (`:245`). So old campaigns still name 7-arg drivers and **will fail at lookup if resumed**.
  **Watch out:** `conc_tuning/u001/state.json` has no `driver` key, so `st.get("driver", DRIVER)` at `:245`
  would silently resume u001 on the new `_abi8` driver and the new binary instead of failing.
- **Second line of defence: the range guard** (`algo.c:251-259`). A caller that binds `_v2` with the wrong
  `argtypes` passes a junk int as `silent_states_begin`. Out of `(0, n_states]`, it prints which file to fix
  and calls `abort()` (exit 134). Catch probability is ~`1 - n_states/2^32`, not certainty
  (`algo.c:248-250`).
- **Why abort, not return:** `p_table` arrives as `np.zeros` (`pkg/robocop/robocop.py:919`). A plain
  return would save an all-zero posterior, which passes `count_calls.py`'s `[0, 1]` check.
- **Unaffected:** the 12 `*_norm` / `*_logspace` trees. They call the old name on their own `.so`
  (override at `robocop_no_em.py:22` or `:23`), which still exports it.

### Rollback

```bash
cd pkg/robocop && cp librobocop_prenorm.so librobocop.so     # restore the old binary
# then revert the merge (8b151c2) and the 5 ported call sites. Otherwise those trees fail at
# lookup (no posterior_decoding_v2 in the old .so), and tune_concentrations.py's default DRIVER
# must go back to the commented line at :91.
```

Rollback is cheap to do but **not free**. Decodes made between cutover and rollback are correct, but they
are not comparable to pre-cutover runs: different arithmetic, different compiler, and chrXII positions that
used to be excluded now count. Any comparison across the cutover must say which side each run is on
(check `so_md5`).
