# fiberCOP audit tab data: schema

Four JSON files, one per generator in `..` (`gen_f_*.py`). They mirror the audit page's data
(`robocop_c_audit/walkthrough/artifact_build/` and `variants/`), so the audit's tab JS can be reused with
the changes flagged **[changed]** / **[new]** below. All files are written with `allow_nan=False`:
any non-finite number is `null`.

Common conventions

- `n` = 6 positions (sequence `AATTAA`), `S` = 10 emitting states, `N` = 12 states (10, 11 are silent).
- State order (index → short label): 0 BG, 1 F1, 2 F2, 3 R1, 4 R2, 5 N1, 6 N2, 7 N3a, 8 N3b, 9 N4,
  10 CENTRAL, 11 TF-entry. Long names: `W.NAMES` (`names` key).
- "btable row" `r` holds position `t = n - 1 - r` (row 0 is the last base), as in the C code.
- Methods: `current` = fork's original algo.c (`pkg/robocop/librobocop_prenorm.so`); `fixB` = production fix B
  (`librobocop.so`); `fixed` = the audit's C1 fix (`variants/librobocop_fixed.so`); `logD0` = fork log-space
  decoder fed the log of the floored emissions; `logD1` = same decoder fed exact `binom.logpmf`
  (`ROBOCOP_LOG_EMISSION=exact`); `brute` = path enumeration on the floored emissions (truth for the model
  production actually builds); `brute_exact` = path enumeration on the exact emissions (truth for the real data).
  **[changed]** the audit had `current`, `fixed`, `log`, `brute`; there is no `log` key here.

---

## 1. `f_lbl.json` (the LB object, "Backward, line by line")

Scenario: fiber `c1` (220 reads at positions 1-4, all methylated; 25 reads / 6 methylated elsewhere).

| key | type | meaning |
|---|---|---|
| `steps` | list[step], 113 | "Now": every bward step from the traced upstream algo.c run with `n_vars = 7`. Same as the backward pass of `current` AND `fixB` (bit-identical btable/sb, asserted). |
| `steps_fix` | list[step], 113 | "Fix": Python replay of the C1-fixed order over 7 layers; its final table is bit-identical to `LIB_C1FIX`'s btable (asserted). |
| `code_b` | list[[int line, str text]], 56 | fork `algo.c` lines 149-204 (bward). **[changed]** fork numbers = upstream + 2. |
| `code_n` | list[[int line, str text]], 21 | fork `algo.c` lines 10-30 (normalize). **[changed]** +2. |
| `code_fix` | list[[str key, str display, str text]], 32 | the fixed bward listing. Keys are fork line numbers as strings (`"149"`, `"151"`, ...), plus `"NEW0"`, `"NEW"`, `"X168"`, `"X203"`, `""`. **[changed]** audit keys were upstream numbers (`"147"`, `"X166"`, `"X201"`). |
| `fix` | list[n][N] float | the correct ("should-be") final btable, row-indexed (= `LIB_C1FIX` btable). |
| `fixpre` | list[n][N] float | Fix side, per row: emitting cells before rescale, silent cells as computed. |
| `dbl_min` | float | 2.2250738585072014e-308 |
| `n`, `S` | int | 6, 10 |
| `seq` | str | `"AATTAA"` |
| `line_offset` | int | **[new]** 2 (fork = upstream + 2; verified: upstream lines 1-226 == fork lines 3-228). |
| `source` | str | **[new]** `"fork algo.c (identical bward in original and fix B)"` |
| `line_map` | {str upstream: int fork} | **[new]** every line number the audit JS hardcodes, mapped to the fork. |
| `scenario` | {key, n_w, k_w, n_c, k_c} | **[new]** the counts behind this run. |
| `c1_rows` | object | **[new]** where Now differs from Fix; see below. |
| `emit_log10` | list[n][S] float/null | **[new]** log10 of the 7-layer emission product (floored model, computed in log so it does not underflow); null = exact zero (N3b at A/T). |

`step` objects (identical field set to the audit; all `lines`/`nlines` are **fork** numbers):

| kind | fields |
|---|---|
| `init` | row, t, j, val, lines `[151,152]` |
| `silent0` (Now only, row 0 silent loop) | row, t, j, child, ev, epos, readrow, old, bchild, T, val, lines |
| `emit` | same fields; Now `lines` = fork 171-181 subset |
| `silent` | same fields; Now `lines` = fork 187-197 subset |
| `nsum` | row, t, terms[S], sum; Now also lines `[168]` (row 0) or `[203]`, nlines `[19,21,22]` |
| `ndiv` | row, t, before[], after[], sum; Now: before/after have N=12 entries (silent included), Fix: S=10; Now also lines, nlines `[25,26]` |
| `store` | row, t, sb; Now also lines `[168]`/`[203]`, nlines `[29]` |

`ev` = emissionValues (the product over all 7 layers), `old`/`val` = cell before/after the `+=`,
`bchild` = child's btable value read, `T` = transition probability. Fix-side steps carry no `lines`.

**JS hardcodes to update for the fork listing** (`lbl_tab.js` `hl()`): every line string +2, i.e.
`149,150→151,152`; `154,155,157,158,159,160→156,157,159,160,161,162`; `169-179→171-181`;
`185-195→187-197`; `166→168`; `201→203`; normalize `17,19,20→19,21,22`, `23,24→25,26`, `27→29`.
Fix side: same +2 (`"159","160"`→`"161","162"`, ...).

`c1_rows` object:

| key | value (this scenario) |
|---|---|
| `rule` | the `isWrong` test of `lbl_tab.js` (rel 1e-6 and abs 1e-250) applied to the final btables |
| `rows` | `[2, 3, 4, 5]`: btable rows where Now != Fix |
| `cells` | list[[row, j]], 21 cells: `[2,11]`, `[3,10]`, `[3,11]`; row 4 columns 0, 2, 4, 9, 10, 11; row 5 all 12 columns |
| `first_row`, `first_row_t`, `first_row_cols` | `2`, `3`, `[11]` |
| `subnormal_silent_cells` | `[[2,11],[3,10],[3,11]]`: silent cells whose Now value fell below DBL_MIN before normalize (the C1 cells proper) |
| `subnormal_silent_steps` | indices into `steps` of those silent steps: `[48,49,68,69,70,71]` |
| `now_vs_fix` | {"row,j": [now, fix]} for every cell in `cells` |

The audit hardcodes row 3 / column 10 as "the C1 row". Here C1 **starts one row earlier**: row 2 (t=3), column 11
(TF-entry), then row 3 (t=2) columns 10 (CENTRAL) and 11. Rows 4-5 are wrong by propagation.

---

## 2. `f_d.json` (the D object, plus the fiber data)

| key | type | meaning |
|---|---|---|
| `scenario` | str | `"c1"` **[new]** |
| `short` | list[12] str | short labels (same as audit `D.short`) |
| `names` | list[12] str | long labels (`W.NAMES`) |
| `seq`, `S`, `n`, `N` | | `"AATTAA"`, 10, 6, 12 |
| `n_vars` | int | 7 **[new]** |
| `emit` | list[n][S] float | same layout as audit `D.emit`: `emit[t][j]` = product over the 7 layers of the floored linear emission, as C receives it. Many entries are 0.0 (the product underflows below 5e-324) and N3b is exactly 0 (no C/G in the sequence). |
| `maxEmit` | list[n] float | max over j of `emit[t]` |
| `init`, `end` | list[12] float | start / end probabilities |
| `trans` | list[[i, j, p]] | nonzero transitions (same format as audit `B.trans`) |
| `dbl_min` | float | |
| `fiber` | object | **[new]**, below |

`fiber`:

| key | type | meaning |
|---|---|---|
| `seq` | str | `"AATTAA"` |
| `n_w`, `k_w`, `n_c`, `k_c` | list[n] int | Watson / Crick coverage and methylated count per position (only the scored strand is used) |
| `scored` | list[n] `"W"`/`"C"`/null | strand scored at each position: Watson at A, Crick at T (`W W C C W W`) |
| `p_watson`, `p_crick` | list[S] float | per-state methylation probability p (`fiber_model.P`) |
| `raw_log10` | list[2][n][S] float/null | exact `log10 binom.pmf(k, n, p_state)`; index 0 = Watson, 1 = Crick; null where that strand is not scored at that position |
| `floored` | list[n][S] bool | true where the scored strand's `binom.pmf` underflowed to 0.0 and was replaced by 1e-30. 16 cells in c1. |
| `n_floored` | int | count of true in `floored` |
| `layers` | list[n][S][3] float | the 3 non-trivial factors after the floor: [sequence PWM, Watson, Crick]. Layers 1-4 are exactly 1 (asserted). The unscored strand is always 1e-30 (production sets it to 0 and the floor lifts it). Product of the 3 == `emit` (rtol 1e-14). |
| `layer_names` | list[3] str | labels for the 3 factors |
| `floor` | float | 1e-30 |
| `emit_log10_floored` | list[n][S] float/null | log10 emission of the floored model, summed in log (no underflow) |
| `emit_log10_exact` | list[n][S] float/null | log10 emission of the exact model (unscored strand contributes 0) |

---

## 3. `f_emp.json` (EMP, "Empirical results")

Top level: `scenarios` (list[4]), `names` (list[12]), `S` (10), `n` (6), `methods` **[new]** (list of the 7 keys).

Per scenario (keys `ordinary`, `c1`, `c1x`, `rdna`, from `fiber_model.SCENARIOS`):

| key | type | meaning |
|---|---|---|
| `key`, `title`, `desc`, `seq` | str | |
| `n_w`, `k_w`, `n_c`, `k_c` | list[n] int | **[changed]** replaces audit `short`/`long`; audit `K` dropped |
| `scored` | list[n] str | **[new]** `"W"`/`"C"` per position |
| `floored_cells` | int | **[new]** (pos, state) cells whose scored-strand binom underflowed and got 1e-30 (0, 16, 36, 4) |
| `maxemit_log10` | list[n] float | max over emitting states of log10 emission, **exact** model |
| `maxemit_log10_floored` | list[n] float | **[new]** same, floored model |
| `zero_emission_rows` | list[int] | positions where the linear 7-layer product is 0 for every emitting state |
| `npaths`, `npaths_exact` | int | paths enumerated by `brute` / `brute_exact` (136 each) |
| `methods` | {method: M} | 7 methods: current, fixB, fixed, logD0, logD1, brute, brute_exact |

Per method `M`:

| key | type | meaning |
|---|---|---|
| `post` | list[n][N] float/null | posterior table as returned |
| `ll` | float/null | log-likelihood (natural log). Floored-model methods and `brute` agree; `logD1`/`brute_exact` are on the exact model, so differ by design. |
| `max_err` | float/null | max abs(post - brute.post) over emitting states (**floored truth**); null if any NaN |
| `max_err_exact` | float/null | **[new]** same against `brute_exact` (**exact truth**) |
| `rowsum` | list[n] float | sum of emitting posteriors per position |
| `dbf` | list[n][7] float | `robocop.sum_for_dbf_probs` (audit upstream package) on the posterior after the 1e-4 sparsification (`empirical.py:downstream`) |
| `em` | list[3] float | normalised [background, TF1, nucleosome] start-state sums, as in the audit |

---

## 4. `f_dep.json` (DEP, depth sweep)

| key | type | meaning |
|---|---|---|
| `sweep` | list[row], 31 | the `frac = 1.0` series (primary; same role as audit `depth.json` `sweep`) |
| `series` | {"1.0", "0.0", "0.6": list[row]} | **[new]** all three methylated fractions |
| `fracs` | [1.0, 0.0, 0.6] | **[new]** |
| `grid` | list[31] int | coverages: 24-point log grid 2..10000 plus 210, 220, 230, 250, 300, 1000, 4379 |
| `positions` | [1,2,3,4] | positions given coverage n (both strands); the rest keep 25 reads / 6 methylated |
| `methods` | list[5] | current, fixB, fixed, logD0, logD1 |
| `reads` | null | placeholder; to be filled from `build/realdata.json` |

`row`:

| key | meaning |
|---|---|
| `K`, `n` | coverage (same number; `K` kept for audit-JS compatibility) |
| `frac`, `k` | methylated fraction and count `round(frac*n)` |
| `maxemit_log10` / `maxemit_log10_floored` | max log10 emission over positions 1-4 and emitting states, exact / floored |
| `floored_cells` | floored binom cells in the model |
| `<m>` | max posterior error vs **floored** truth (`brute`), null if NaN. **[changed]** the audit's `<m>` was vs its only truth |
| `<m>_exact` | **[new]** max error vs **exact** truth (`brute_exact`) |
| `<m>_ll_ok` / `<m>_ll_ok_exact` | log-likelihood within 1e-6 relative of the floored / exact truth |
