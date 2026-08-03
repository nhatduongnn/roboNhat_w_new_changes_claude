# What To Do — Fiber-seq RoboCOP: Diagnosis & Recommendations

**Goal:** integrate Fiber-seq (single-molecule m6A methylation) into RoboCOP and tune
the model so decoding is consistent with well-known yeast biology (phased nucleosome
arrays, TF footprints at NDRs).

**Phased strategy (explicit user directive):** get the **Fiber-seq layer working alone and
biologically correct FIRST**, then incorporate the sequence and MNase layers. The sequence
layer (`data_emission_matrix[0][:]=1`) and the MNase layers (1–4) are turned off **on
purpose** to isolate Fiber-seq — this is not a bug and re-enabling them is a *later* phase,
not a near-term fix. Everything below is scoped to making **Fiber-only** decoding correct.

---

## MASK SEMANTICS + WORKING ABF1 FOCUS (2026-07-23) — READ FIRST

The Fiber emission matrix is a MULTIPLICATIVE likelihood factor. So in the mask:
- **`= 0` FORBIDS a state** (floored to 1e-30 in robocopExtras → effectively impossible). This
  is how you REMOVE states.
- **`= 1` does NOT remove — it FREES** a state (neutral, zero emission penalty). A freed state
  with a high transition prior (e.g. `unknown`, dbf_conc 0.1) then DOMINATES. The original
  hardcoded mask used `=1`, which is why "masking" never removed anything.

The original hardcode `data_emission_matrix[index][14:,:]=1` was also on the WRONG AXIS
(positions, not states) — emat is (7, n_obs, n_states); `[index][14:,:]` = positions 14+.

**Working ABF1-focus mask (single comment-in/out line, state axis, `=0`):**
```python
data_emission_matrix[index][:, 29:dshared['nuc_start']] = 0   # forbid states 29..2798
```
Keeps background(0), ABF1 fwd+rev(1..28), nucleosomes(2799+); forbids all other TFs + unknown.
Lives in `update_data_emission_matrix_using_binomial_fiber_seq` in robocop.py, commented out
by default (repo default = mask OFF). REMINDER: remove when sequence/MNase layers return.

**Verified result (ERV46 region, mask ON `=0`):** ABF1 max posterior 0.9998, nuc max 1.0,
bg max 1.0, unknown max 0.0000 (fully forbidden). Residual ROX1/SIG1 leak to ~0.2 at 1-2
isolated positions only (mean ~0.0002) — cosmetic, crosses the 0.1 legend threshold. Leak
cause: the 1e-30 forbid only bites at 'A' positions; at non-'A' positions the Binomial code
sets ALL states to 1e-30, so short forbidden TF motifs (5bp) can catch a sliver in A-poor
windows. Negligible.

**KEY POSITIVE:** Fiber-seq ALONE (no sequence, no MNase, no EM) yields a good ABF1-focused
decode: phased nucleosome array + clean ABF1 footprints. Mask-OFF also good (nuc + all TFs).

Mechanics for running mask ON: uncomment the line, submit, wait until RUNNING, re-comment
(env-var toggle was removed per user preference). trainDir=robocop_train_fiberonly,
driver=run_fiberonly_noem.py, sbatch=sbatch_noem.sh. Runs use run_robocop_without_em (keeps
tmpDir/info.h5; equivalent decode to with_em iter=0, which deletes tmpDir via robocop_em.py:162).

---

## EMPIRICAL RESULT (2026-07-22, clean re-run — earlier mask reasoning now corrected above)

Ran 4 clean fiber-only, no-EM jobs (run_robocop_without_em against an iter=0 HMMconfig),
mask ON vs OFF, on both the ERV46 single region (chrI 60001-65000) and the 9-region set.
Plots at chrI 60500-64500:

- **mask OFF (robocop_all_fiber, robocop_erv46_maskoff) → GOOD decode.** Nucleosomes (`nuc`)
  dominate in a phased array that aligns with the fragment-length dyads and the meth/A
  depletion; white NDR gaps; TF footprints (ABF1, REB1, NHP6A, UME6, ROX1...). **The
  Fiber-seq layer ALONE — no sequence, no MNase, no EM — already recovers sensible
  nucleosome + TF biology.**
- **mask ON (robocop_all_abf1, robocop_erv46_maskon) → EMPTY/diffuse decode.** The mask
  `data_emission_matrix[index][14:,:]=1` (+ the explicit nucleosome line) sets Fiber emission
  of nucleosomes AND all non-ABF1 states to uninformative 1, leaving only background + the
  first 13 ABF1-forward states with signal. Nothing drives the nucleosomes → diffuse
  posterior, everything below the 0.1 plot threshold.

**Conclusions that overturn the earlier speculation:**
1. This REVERSES the hypothesis drawn from the committed PNGs (whose provenance was flagged
   unreliable). mask-OFF is the good decode; mask-ON is empty. The earlier "restore ABF1
   focus" recommendation was wrong — the mask as written destroys the nucleosome signal.
2. The good Fiber-only baseline is mask-OFF. Do NOT re-enable the mask as written.
3. If an ABF1-focused view is still wanted, the mask must be REDEFINED to keep nucleosome +
   background + ABF1 states informative and only zero out the OTHER TF states (i.e. do not
   touch nuc_start:nuc_start+nuc_len, and don't blanket `[14:,:]`).

Next: quantify the mask-OFF nucleosome decode vs `Chereji_2018_+1_-1_nucs.bed`; assess TF
footprint calls; decide whether "unknown"/TF competition needs tuning. The big nucleosome
architecture is already there.

Artifacts: driver `run_fiberonly_noem.py`, generic `sbatch_noem.sh`, trainDir
`robocop_train_fiberonly/`, env-var mask toggle `ROBOCOP_ABF1_MASK` in `robocop.py`.

---

## Diagnosis (re-evaluated for the Fiber-only phase) — PARTLY SUPERSEDED, see above

Two committed fiber-only decodes tell different stories, and the difference is the key
insight:

- `analysis/robocop_all_abf1/figures/robocop_output_chrI_60001_64000.png` — **looks close to
  correct.** Nucleosomes (`nuc`, medium gray `0.7`) dominate the posterior track in a
  sensible phased pattern; `unknown` (light gray `#D3D3D3`) appears only at NDR-like gaps.
  This is roughly the ABF1-focused behavior we want.
- `analysis/robocop_all_fiber/figures/robocop_output_chrI_60500_64500.png` — **the problem
  case.** `nuc` is absent from the legend entirely, meaning nucleosome posterior never
  exceeds the 0.1 plotting threshold anywhere in the window (verified via
  `visualization.preprocess_occupancy_profile`), so the dominant gray is `unknown`. All ~150
  TFs are active here and fragment the decode.

**So the lever is the size of the active factor set, not the "unknown" prior per se.** When
the model is restricted toward ABF1 (`_abf1` run), nucleosomes and ABF1 compete cleanly and
the decode is good. When all ~150 TFs are live (`_fiber` run), nucleosomes lose and the
window collapses to `unknown`.

**Critical code-state finding:** the ABF1 restriction is currently **NOT applied** on the
active path. The `[14:,:]=1` ABF1 mask exists at `robocop.py:459` (inside the *dead*
duplicate NB function — never runs) and at `robocop.py:795` (inside the *active* binomial
function, but **commented out**). The live Fiber path is the Binomial function
(`..._counts_Bionomial` → `update_data_emission_matrix_using_binomial_fiber_seq`). So the
current committed code runs all TFs → `_fiber`-style unknown wash. The good-looking `_abf1`
figure was almost certainly produced when the mask was active (unverified — provenance of
committed PNGs is uncertain; re-running current code on one region would settle it).

Because sequence and MNase are intentionally off, the m6A signal **alone** carries all
discrimination — so the secondary issues below also matter more than they would otherwise:

**B. Weak emission discrimination.** Fitted p(m6A): background 0.138, nucleosome 0.024–0.14
(mean 0.052), TFs 0.08–0.20 — all clustered in a narrow low band. With highly variable
per-position coverage `n`, the binomial likelihood is near-flat when `n` is small.

**C. EM is disabled** (`pkg/robocop_em.py`: `iterations=0`), so the
background/nucleosome/TF/unknown balance is never calibrated to the Fiber-seq data.

**D. Emission-model correctness is now safety-critical.** With Fiber-seq the only signal, any
bug in the binomial emission directly breaks the biology. Needs an explicit correctness pass
(Crick reference base `nucleotide_ref=3`, coverage handling, pseudo-count shrinkage).

**Fundamental limitation to plan around:** without sequence, TF *identity* is largely
unidentifiable from an m6A footprint alone. So "biologically correct, Fiber-only" is judged
mainly on **nucleosome architecture, accessibility/NDRs, and ABF1 footprint occupancy**, not
TF identity. Identity accuracy is the payoff of re-adding sequence later.

**One-line summary:** the ABF1-focused decode already looks close to right; the current
committed code just isn't applying the ABF1 focus on the active binomial path, so it runs all
TFs and collapses to `unknown`. Restore the focus, verify the emission model, turn EM on.

---

## Recommendation (prioritized, Fiber-only phase)

### Tier 1 — make Fiber-only decoding biologically correct (highest leverage)
1. **Restore the ABF1 focus on the ACTIVE binomial path.** The `[14:,:]=1` mask is commented
   out at `robocop.py:795` (active fn) and only live at `:459` (dead NB fn), so the current
   code runs all ~150 TFs and collapses to `unknown`. Re-enable the restriction in the
   binomial path (or, cleaner, reduce the TF panel in `pwm.p` to ABF1 + nucleosome + bg +
   unknown). The `_abf1` figure shows this alone gets the decode close to correct.
   **REMINDER: remove this restriction when sequence/MNase layers come in.**
2. **Get nucleosomes right.** Once TFs are restricted, nucleosome positioning is the dominant
   verifiable feature. Tune `nucleosome_prob` (transition) and the nucleosome p-parameters
   until phased arrays hold, and validate against `Chereji_2018_+1_-1_nucs.bed`. Only revisit
   the `unknown` prior (`dbf_conc['unknown']=0.1`) if nucleosomes still lose to `unknown`
   after the TF restriction — in the `_abf1` run `unknown` behaves correctly (NDRs only), so
   it may not need changing.
3. **Re-enable EM** (`iterations>0`, still Fiber-only) so the bg/nuc/ABF1/unknown balance is
   learned from the m6A data. The constrained-EM machinery (`adjustEM`, threshold) exists.
4. **Verify the emission model** — since it's the only signal: confirm the Crick
   reference-base handling (`nucleotide_ref=3`) matches the 0/1/2/3 = A/C/G/T encoding;
   decide how to handle wildly varying coverage `n` (cap or down-weight low-`n` positions);
   and check whether the pseudo-count over-shrank the fitted p's toward uniform (would
   explain the narrow 0.05–0.14 band).

**First, though: re-run robocop on current code for one fiber region** to confirm what
today's code actually produces (the committed PNGs are of uncertain provenance — the good
`_abf1` one may predate the mask being commented out).

### Tier 2 — remove experiment-specific breakage (cheap; do first)
5. Delete the **dead** duplicate `update_data_emission_matrix_using_negative_binomial_fiber_seq`
   def (`robocop.py:402` and `:431` — Python keeps only `:431`; both are off the active path
   since the live Fiber path is the Binomial fn). Keep the ABF1 masking logic itself — see
   Tier 1 #1 — but consolidate it into the active binomial function so there's one clear,
   intentional switch instead of a commented line + a copy in dead code.
6. Delete the `transition_mat[3330, …]` prints/edits (crash risk + region-specific).
7. Make `inputs/*.pkl` / `.npy` paths **config-driven**, and gate the forced
   emission-plotting + `print` spam behind a debug flag (it writes 7 PNGs and dumps every
   TF's params per segment, per decode).

### Tier 3 — define "performs well" for the Fiber-only phase
8. **Establish a quantitative benchmark** appropriate to Fiber-only: nucleosome dyads vs
   `Chereji_2018_+1_-1_nucs.bed` (primary), accessibility vs known NDRs/promoters, and
   footprint **occupancy** at Abf1/Reb1 sites vs `MacIsaac_...Abf1_Reb1.bed` (presence, not
   necessarily identity). All ground truth is already in `analysis/inputs/`. Track these
   numbers across every tuning run.

### Suggested starting order
Tier 2 cleanup first (fast, removes the crash risk), then Tier 1 #1 → #2 (tame "unknown",
then dial in nucleosomes) since those most directly remove the gray wash — with Tier 3 #8
set up in parallel so nucleosome/accessibility accuracy is measurable at every step. Defer
sequence + MNase re-integration to the next phase, once Fiber-only is validated.

---

## Notes on repo vs FIBERSEQ_CHANGES.md
The doc is accurate on all major claims. Gaps worth noting:
- The emission-layer plotting block (`robocop.py:1005–1031`) and the
  `transition_mat[3330,…]` debug prints (`robocop.py:891–893`) are **active**, not
  commented; the latter will `IndexError` on any model whose transition matrix has fewer
  than ~3331 states.
- `getReads.getFiber_seq`'s `idx != None` branch is also broken (undefined `offset`,
  calls the buggy `readData` copy); only the non-idx branch is used.
- Only 12 of ~150 model TFs have fitted footprints; the rest silently use
  `combined_low_count`. Names do match `pwm.p` (e.g. `Abf1_murphy`), so matching works.
