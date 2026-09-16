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

Companion docs: `CLAUDE.md` = the fresh-session entry point (environment, rules, gotchas,
command cheat-sheet). `HANDOFF.md` = state of every thread and the tooling (scorer, plots,
layer/mask toggles, canonical runs) — the up-to-date operational reference. `FIBERSEQ_CHANGES.md` = full code diff
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
- **A widened-footprint series exists and is fully measured** (2026-08-27 → 09-04). Five
  live variants — `widememe` (ABF1 only, 7/2), `wide12` (the 12 fitted TFs, per-TF
  footprint), `wide10all` (all 153, ±10), `wide150` (12 TFs, ±150), `wide150all` (all 153,
  ±150) — each reached as **pure data plus one changed line**, since
  `get_transition_matrix_info` derives `tf_lens` from `pwm[tf].shape[1]`. All are scored on
  chrI and chrXIV except `wide150all`, which never ran. A sixth, the uniform ±10 padding of
  the 12 TFs, held the `wide12` name until 2026-09-04 and is retired to
  `analysis/retired/wide10_uniform/` (its scores are kept in the README there).
  **`wide150` scored empty: the ±150 pads crush the 12 widened TFs' concentration priors by
  1e-16…1e-30, so their posterior is zero. Fixable without retraining — see `HANDOFF.md` §7.** `wide150all` is permanently blocked by
  32-bit index overflow in `bc.c` at `n_states` 95285; ±70 is the largest uniform pad across
  all 153 motifs that fits under the 46340 ceiling. Full detail: `HANDOFF.md` §7 and
  `analysis/README_wide_implementations.md`.
- **The concentration prior was calibrated by a λ sweep, not by EM** (2026-08-24 → 09-01).
  EM collapses: after 10 iterations 66 of 154 TFs sit at exactly 0 and 28 are pinned to the
  constrained-EM ceiling (`mean + 2*sd` of the *initial* priors, computed once and never
  recomputed). A direct sweep of ABF1's λ does better — λ=0.01 gives F1 0.111 vs 0.045 at
  baseline, same true positives, false positives 81 → 29. The response is monotone in λ, so
  there are no local optima and sampling buys nothing. Runs `robocop_{chrI,chrXIV}_conclo_*`,
  scored in `conc_scores/`.
- **All 84 MacIsaac-targeted concentrations are now calibrated, and it settles what λ can
  do** (2026-09-10 → 09-12, `HANDOFF.md` §1). Two 8-round genome-wide campaigns —
  `u001` (all motifs live) and `m001` (69 non-MacIsaac motifs hard-masked) — drove per-factor
  call counts onto MacIsaac's site counts: 79 of 81 groups within 2×, typical factor 1.07×
  off, nucleosomes unmoved. **Site accuracy barely followed**: pooled F1 0.029 → 0.049 with
  all the gain in the first step, recall peaking at round 1, and every λ step adding or
  removing calls at only 1–2% MacIsaac precision. So **λ is a count knob, not an accuracy
  knob** — consistent with the flat AUROC in the ABF1 sweep. Masking the untargeted motifs
  made it slightly worse (F1 0.042). λ_unknown is pinned at 0.01 by user decision. The tuned
  models are `robocop_train_ct_{u001,m001}_07`; Rossi validation of them is the open question.
- **A model-free validation target now exists**: Rossi's genic/intergenic split for **378
  TFs** (`analysis/rossi_genic/rossi_genic_all_TFs.tsv`). For any TF the model emits, compare
  the fraction of its calls inside an ORF with that TF's row. 77 of the 153 RoboCOP motif TFs
  have a Rossi row, 47 with ≥100 peaks; their genic% spans 8.8% to 59.3% against a 73.0%
  random-genome null. `HANDOFF.md` §5.2.
- **The shipped PWM collection has been audited** (2026-08-14 → 08-18, read-only —
  `inputs/motifs_meme.txt` is unchanged). All 153 shipped matrices were compared against
  JASPAR CORE fungi and Rossi ChExMix, three ways, one row per Rossi replicate. **7
  matrices come out as the odd one of the three** — `Abf1_murphy`,
  all four RAP1 matrices, `Pdr1_badis`, `Cad1_murphy`. See **`HANDOFF.md` §6** for the
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
flip-and-wait race and lets all variants run concurrently — see `HANDOFF.md` §4.2 for the
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

### Tier −1 — DONE 2026-09-04

−1. ~~**Score the four `wide150` decodes.**~~ **DONE.** All four scored on chrI and chrXIV;
    the array bound was widened (and a guard added so a stale bound now aborts instead of
    silently skipping rows). **Result: empty.** ABF1's posterior is identically 0 at every
    MacIsaac site in all four. Not the block-width artifact and not a scoring bug — the
    `Abf1_murphy` column is present and non-widened factors in the same decode carry normal
    mass. The ±150 estimated pads crush the 12 widened TFs' `tf_prob` by 9.6e-19 (ABF1) to
    5.0e-30 (Fhl1) while every non-widened TF sits at ratio 1.00, which underflows the
    posterior to zero. Nucleosomes are unaffected (recall 0.81, dyad 7–8 bp, period 171), so
    the decodes themselves are healthy. `HANDOFF.md` §7.

−2. ~~**Add `widefp` and `wide150` to the viewer runs.**~~ **DONE.** Both added, label sets
    verified identical (18 labels after the `wide12` retirement), combined browser rebuilt
    and republished in place to `c6c7d1f3-…`.

−3. **Decide the all-motif wide experiment.** STILL OPEN. ±70 across all 153 runs today with
    no code change (`n_states` 46325, just under the 46340 ceiling; ~295 GB train, so a
    `compsci-cluster-fitz-*` node). The alternative is an `int64_t` rebuild of `bc.c` into a
    **separate** `librobocop.so` used by that variant only — a change to the numerical core
    that must be re-validated against an existing run first. `HANDOFF.md` §4.7.

−4. **NEW — decide whether to rescue `wide150`.** Its priors can be restored without
    retraining: `make_conc_trainDir.py --set <TF>=<lam>` takes all twelve at once, with
    lam = baseline `tf_prob` ÷ wide150 `tf_prob` (Abf1_murphy=1.04e18, Reb1_badis=1.99e20,
    Fhl1_zhu=1.99e29, …; recompute the full vector from the two trainDirs — `HANDOFF.md` §7). That would make `wide150` an
    actual test of the ±150 hypothesis rather than a test of switched-off factors. It needs
    a new trainDir plus four re-decodes; the training is a config build, not a fit, so the
    expensive part is only the decodes.

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
   score real sites.** Details, tooling and the shortlist: `HANDOFF.md` §6.

   Two constraints when swapping: keep the `jaspar_abf1_motifs_meme.txt` pattern (one
   alternate PWM file + one alternate config, never an in-place edit of
   `inputs/motifs_meme.txt`), and if the matrix is one of the 12 with a fitted Fiber-seq
   footprint (`Rap1_telomeric` and `Abf1_murphy` both are) keep its **motif ID, width and
   orientation** — `update_data_emission_matrix_using_binomial_fiber_seq` looks the p-vector
   up by ID and it is registered to the motif's column frame.

### Tier 1 — model quality
1. ~~**Re-enable EM.**~~ **Done, and the answer is no.** `iterations = 0` still stands in
   `pkg/robocop_em.py` and should stay: EM was run (`robocop_train_em10_chrII*`,
   `pkgvar/seq_maskoff_em10`) and it **degrades** ABF1 on the held-out chrXIV in both layer
   configurations. It drives 66 of 154 TFs to exactly 0 and pins 28 more to the
   constrained-EM ceiling — a ceiling set by `mean + 2*sd` of the *initial* priors, computed
   once and never recomputed, over a distribution skewed enough (mean 1.0e-4, median 1.2e-5,
   sd 2.9e-4) that 6 TFs are already above it before EM starts. Calibrate concentrations with
   the direct λ sweep instead (`make_conc_trainDir.py`, `sweep_conc.py`, `score_sweep.py`);
   λ=0.01 for ABF1 is the measured optimum. `HANDOFF.md` §2 has the diagnosis;
   `conc-calibration-wont-work` in the memory index has the numbers.
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
**Tier −1 first** — it only reads runs that already exist. Then Tier 2 (fast, removes the
`3330` crash risk and the path fragility), then **Tier 0 before Tier 1 #2** — there is no point retraining with the sequence layer on
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
