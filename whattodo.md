# What To Do — Fiber-seq RoboCOP: Strategy & Open Work

**Goal:** integrate Fiber-seq (single-molecule m6A methylation) into RoboCOP and tune
the model so decoding is consistent with well-known yeast biology (phased nucleosome
arrays, TF footprints at NDRs).

**Phased strategy (explicit user directive):**
1. Get the **Fiber-seq layer working alone and biologically correct FIRST** — aggregate
   (modkit pileup) form. The sequence layer (`data_emission_matrix[0][:]=1`) and the MNase
   layers (1–4) are turned off **on purpose** to isolate Fiber-seq. This is not a bug.
2. Then incorporate the sequence layer, then MNase.
3. Then move from the aggregate pileup to **single-fiber** (per-read) decoding.

Companion docs: `HANDOFF.md` = current tooling (scorer, plots, layer/mask toggles, canonical
runs) — that is the up-to-date operational reference. `FIBERSEQ_CHANGES.md` = full code diff
vs upstream. This file = strategy + what is left to do.

---

## Where things stand (so nobody redoes finished work)

Done and validated:
- **ABF1-focus mask** works and is applied on the active binomial path (see mask semantics
  below). Fiber-only + mask ON recovers 3/5 chrI MacIsaac ABF1 sites; mask OFF recovers 0/5.
- **Nucleosome architecture is good on full chrI** with Fiber-seq alone — Chereji recall
  ≈ 0.79–0.81, median dyad error 7 bp, and it is essentially unchanged between mask ON and
  mask OFF. The earlier concern that a live ~150-TF panel collapses the decode to `unknown`
  does **not** hold at chromosome scale; it was a small-window artifact.
- **Quantitative benchmark exists**: `analysis/score_robocop.py` scores any decode against
  Chereji/Brogaard dyads, MacIsaac ABF1, phasing period, and accessibility. Run it on every
  tuning run and diff the JSON reports. Numbers for the 4 canonical chrI runs are in
  `analysis/chrI_5run_metrics.json`.
- **Sequence layer helps ABF1 a lot**: enabling layer 0 lifts chrI ABF1 enrichment from
  ~1.1× to ~29×, though site recall is still 2/5 (see the limitation below).
- **The shipped PWM collection has been audited** (2026-08-14 → 08-18, read-only —
  `inputs/motifs_meme.txt` is unchanged). All 153 shipped matrices were compared against
  JASPAR CORE fungi and Rossi ChExMix, three ways, one row per Rossi replicate. **7
  matrices come out as the odd one of the three** — `Abf1_murphy`,
  all four RAP1 matrices, `Pdr1_badis`, `Cad1_murphy`. See **`HANDOFF.md` §0** for the
  method, the tooling, and the published sheet; the new open work it created is Tier 0
  below. None of the items in this file were worked on.

---

## Mask semantics — read before touching the mask

The Fiber emission matrix is a MULTIPLICATIVE likelihood factor. So in the mask:
- **`= 0` FORBIDS a state.** This is how you REMOVE states.
- **`= 1` does NOT remove — it FREES** a state (neutral, zero emission penalty). A freed state
  with a high transition prior (e.g. `unknown`, `dbf_conc` 0.1) then DOMINATES.

The axis matters: `emat` is `(7, n_obs, n_states)`, so `[index][14:, :]` slices **positions**,
not states. The mask must slice the **state** axis: `[index][:, 29:nuc_start]`.

**The mask to use lives in `robocopExtras.py:113-114`** (commented out by default):
```python
data_emission_matrix[5][:, 29:dshared['nuc_start']] = 0   # Fiber Watson
data_emission_matrix[6][:, 29:dshared['nuc_start']] = 0   # Fiber Crick
```
It keeps background (0), ABF1 fwd+rev (1..28) and nucleosomes (`nuc_start..`); it forbids all
other TFs + `unknown`.

**It must run AFTER the 1e-30 floor** (`robocopExtras.py:105-106`), which is why it lives here
and not inside the binomial function. The floor turns every 0 into 1e-30; if the mask ran
first, forbidden states would be lifted back to 1e-30 — tiny but nonzero, so they can still
leak posterior. Running after the floor leaves the forbidden states **exactly 0**, and since
emission is a product over channels, a 0 in the Fiber channels ⇒ total emission 0 ⇒ posterior
exactly 0. Background and nucleosomes keep the floor, so no column goes all-zero (no NaN).

This **supersedes the older copy at `robocop.py:798`**, which sits before the floor and can
leak. Prefer the `robocopExtras.py` one; the `robocop.py:798` copy should be deleted (see
cleanup #4 below).

Mechanics: **do not comment these lines in and out by hand any more.** Every layer/mask
combination exists as a frozen copy of the package under `analysis/pkgvar/<variant>/`, which
the driver selects with `sys.path.insert(0, 'pkgvar/<variant>/')`. That removes the
flip-and-wait race and lets all variants run concurrently — see `HANDOFF.md` §0.5 for the
variant table. The older single-run path (trainDir `robocop_train_fiberonly`, driver
`run_fiberonly_noem.py`, sbatch `sbatch_noem.sh`) still works if you do edit by hand. Decodes
use `run_robocop_without_em` (keeps `tmpDir/info.h5`; equivalent to `with_em` iter=0, which
otherwise deletes tmpDir).

Also note: a decode **never re-reads the meme file** — `run_robocop_without_em` reads
`pwm_emission` / `tf_prob` / `transition_matrix` out of the trainDir's `HMMconfig.pkl`
(`robocop_no_em.py:51`). Changing `pwmFile` in a config does nothing without a retrain.

**REMINDER: this mask is experiment-specific — remove it when the sequence/MNase layers come
back in for good.**

---

## Fundamental limitation to plan around

Without the sequence layer, TF **identity** is largely unidentifiable from an m6A footprint
alone — a protected ~15 bp patch does not say which factor made it. So "biologically correct,
Fiber-only" is judged mainly on **nucleosome architecture, accessibility/NDRs, and ABF1
footprint occupancy**, not on TF identity. Identity accuracy is the payoff of re-adding
sequence, which is why the seq-layer runs improve ABF1 enrichment ~25× while nucleosome
metrics barely move.

---

## Open work

### Tier 0 — the PWM collection (NEW; gates Tier 1 #2)

0. **Decide what to do about the 7 matrices the audit flagged.** The sequence layer is the
   next phase of the plan, and turning it on with a wrong matrix makes decoding worse, not
   better. `Abf1_murphy` is already proven wrong end-to-end: its spacer (cols 5–9) is a
   GC-rich informative block that real ABF1 sites do not carry, it is anti-correlated with
   JASPAR's spacer at r = −0.411 while the two cores correlate at r = +0.998, and on a chrI
   FIMO scan it recovers **2 of 5** MacIsaac sites where the JASPAR matrix recovers
   **5 of 5**. In a full decode the swap is worth less — 3/5, site posterior 0.138 → 0.204
   — because the fiber layer is the binding constraint (Tier 1 #3 below). The
   other six (`Rap1_telomeric`, `Rap1_zhu`, `Rap1_motif1`, `Rap1_motif2`, `Pdr1_badis`,
   `Cad1_murphy`) have sheet evidence but **no FIMO or decode validation yet** — get that
   before swapping anything. **A confident TOMTOM q-value is not evidence a matrix works;
   score real sites.** Details, tooling and the shortlist: `HANDOFF.md` §0.

   Two constraints when swapping: keep the `jaspar_abf1_motifs_meme.txt` pattern (one
   alternate PWM file + one alternate config, never an in-place edit of
   `inputs/motifs_meme.txt`), and if the matrix is one of the 12 with a fitted Fiber-seq
   footprint (`Rap1_telomeric` and `Abf1_murphy` both are) keep its **motif ID, width and
   orientation** — `update_data_emission_matrix_using_binomial_fiber_seq` looks the p-vector
   up by ID and it is registered to the motif's column frame.

### Tier 1 — model quality
1. **Re-enable EM.** `pkg/robocop_em.py:116` is still `iterations = 0`, so the
   background/nucleosome/TF/`unknown` balance is never calibrated to the Fiber-seq data.
   Fiber-seq parameters come entirely from the pre-computed pickles. The constrained-EM
   machinery (`adjustEM`, threshold) already exists. Train once with EM, then decode with
   `run_robocop_without_em` so `tmpDir` survives.
2. **Retrain with the sequence layer ON.** The current `*_seq_*` decodes reuse
   Fiber-only-trained weights, so they are a lower bound on what sequence can do.
3. **Verify the emission model** — it is the only signal, so bugs here directly break the
   biology:
   - confirm the Crick reference-base handling (`nucleotide_ref = 3` in
     `update_data_emission_matrix_using_binomial_fiber_seq`) matches the 0/1/2/3 = A/C/G/T
     encoding;
   - decide how to handle wildly varying per-position coverage `n` (cap, or down-weight
     low-`n` positions) — the binomial likelihood is near-flat when `n` is small;
   - check whether the pseudo-count over-shrank the fitted `p`'s toward uniform. Fitted
     p(m6A): background 0.138, nucleosome 0.024–0.14 (mean 0.052), TFs 0.08–0.20 — a narrow
     low band, which is weak discrimination.
4. **Nucleosome/`unknown` tuning, only if metrics say so.** Nucleosomes already score well, so
   revisit `nucleosome_prob` and `dbf_conc['unknown'] = 0.1` only if a change regresses them.

### Tier 2 — cleanup (cheap, removes a real crash risk)
5. Delete the **dead duplicate** `update_data_emission_matrix_using_negative_binomial_fiber_seq`
   (defined twice, `robocop.py:402` and `:431` — Python keeps only `:431`; both are off the
   active path since the live Fiber path is the Binomial function). Delete the superseded mask
   copy at `robocop.py:798` at the same time so there is exactly one mask switch.
6. Delete the `transition_mat[3330, …]` **prints** at `robocop.py:892-894`. The assignments
   above them are already commented out, but the prints are live and will `IndexError` on any
   model whose transition matrix has fewer than ~3331 states.
7. Make the `inputs/*.pkl` / `.npy` paths **config-driven** (currently literal relative paths in
   `update_data_emission_matrix_using_fiber_seq_counts_Bionomial`, so the run only works from
   one working directory), and gate behind a debug flag: the forced
   `plot_all_factors_side_by_side` call (`robocop.py:624`), the emission-layer PNG dump
   (`robocop.py:1007-1032`, 7 PNGs per decode), and the per-TF `print` spam.

### Tier 3 — next phase
8. **Single-fiber decoding.** Everything today is aggregate: the modkit pileup gives
   per-position `(k methylated, n A trials)` summed over reads, and the binomial treats those
   reads as exchangeable. Per-read decoding needs a different ingestion path (read-level m6A
   calls, not pileup) and a per-molecule state chain. Not started; blocked on the aggregate
   phase being validated.

### Suggested order
Tier 2 first (fast, removes the `3330` crash risk and the path fragility), then Tier 1 #1,
then **Tier 0 before Tier 1 #2** — there is no point retraining with the sequence layer on
a matrix that is known to be wrong — then Tier 1 #3. Score with `score_robocop.py` after
every change so regressions are visible immediately. Defer Tier 3 until Fiber-only +
sequence is validated.

---

## Notes on repo vs FIBERSEQ_CHANGES.md
The doc is accurate on all major claims. Gaps worth noting (all re-verified against the current
tree):
- The emission-layer plotting block (`robocop.py:1007–1032`) and the `transition_mat[3330,…]`
  prints (`robocop.py:892–894`) are **active**, not commented.
- `getReads.getFiber_seq`'s `idx != None` branch is broken (undefined `offset`, and it calls the
  buggy `readData` copy of the extractor); only the non-idx branch is used.
- Only 12 of ~150 model TFs have fitted footprints in
  `inputs/all_TFs_1000pealVal_params_pseudo.pkl` (Abf1_murphy, Cin5_murphy, Fhl1_zhu, Fkh1_zhu,
  Mcm1_zhu, Nhp6a_zhu, Rap1_telomeric, Reb1_badis, Sko1_murphy, Spt15_zhu, Tbf1_zhu, Ume6_zhu);
  the rest silently fall back to `combined_low_count`. Names do match `pwm.p`, so the lookup
  itself works.
- **`combined_low_count` gets no pseudocounts (deferred — not a problem today).**
  In `abf1_reb1_dms_parameter_Fiber-seq_w_binom.py`, `add_pseudocounts_binomial(counts, 3, 58)`
  is called on the per-TF Fiber-seq binomial branch (line ~130) but **not** on the combined
  low-count Fiber-seq binomial branch (line ~206). The other `add_pseudocounts_binomial` call
  (line ~164) sits inside `if tech != "Fiber_seq"`, the DMS/nbinom branch, so it never runs here.
  Consequence in `inputs/all_TFs_1000pealVal_params_pseudo.pkl`:
    - 12 individual TFs: `['C'] = ['G'] = ['T'] = 0.051724` (= 3/58, pure pseudocount)
    - `combined_low_count`: `['C'] = ['G'] = ['T'] = 0.0` exactly
  **Harmless for now**: `robocop.py:662` only ever indexes `['A']`, and `['A']` *is* pooled over
  942 real sites, so the value RoboCOP actually consumes is fine. The C/G/T arrays are dead in
  both cases (Fiber-seq only calls m6A at A). Revisit if any layer ever reads a non-A base.
  Also note `combined_low_count` has `tf_len == 1` — intentional for now; the single value is
  broadcast across all `2 * tf_len` states of every fallback TF at `robocop.py:662`.
