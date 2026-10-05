# `analysis/singlefiber/` — single-fiber RoboCOP prototype

Everything else in this repo decodes the **aggregate** Fiber-seq signal: `modkit pileup`
pools all molecules into (k methylated, n valid trials) per reference position and the
emission is a binomial (`pkg/robocop/robocop.py:775-813`). This directory decodes **one
molecule at a time** from that molecule's own m6A calls, with the same 3,485-state HMM,
the same transition matrix, the same PWM sequence layer and the same 531-state nucleosome
dinucleotide model.

Built to the plan of 2026-09-28. Nothing outside this directory and
`analysis/pkgvar/permol_seq_maskoff/` was touched.

## The one model change

`binom.pmf(k, n, p_j)` becomes a two-value Bernoulli lookup indexed by that molecule's
0/1 call:

| this molecule's call at position *i* | emission for non-silent state *j* |
|---|---|
| methylated | `ps[j]` |
| canonical | `1 - ps[j]` |
| no observation | `1.0` |

`1.0` for "no observation" is not a special case bolted on: the aggregate path already
behaves that way, because `binom.pmf(0, 0, p) == 1.0` at a zero-coverage adenine. And
exactly as in the aggregate (`robocop.py:804-806`), every position whose **reference**
base is not that layer's target base (A for Watson, T for Crick) is zeroed across the
non-silent states and then floored to `1e-30` in `robocopExtras.py` — a per-position
constant that cancels across states.

`ps` itself is untouched: `robocop.build_fiber_ps` in the frozen tree holds a **verbatim**
copy of the block that builds it in the aggregate path, so background, the 12 fitted TFs,
`combined_low_count` and the 531 nucleosome states keep their meaning.
`permol_params.build_ps_labelled` reproduces that block while also recording where each
state's value came from, and `permol_params.assert_verbatim` asserts the two vectors are
bit-identical at the start of every run.

## Files

| file | what |
|---|---|
| `read_calls.py` | BAM MM/ML → per-molecule 0/1/no-call, matched to `modkit pileup` |
| `validate_reader.py` | proves the reader: summed per-molecule calls vs the pileup's (k, n) |
| `permol_params.py` | the three footprint arms and the per-read efficiency correction |
| `collapse.py` | vectorised state→factor collapse, checked against `robocop.sum_for_dbf_probs` |
| `run_permol_window.py` | decode every molecule overlapping a window; one arm per invocation |
| `checks.py` | the four pre-registered checks |
| `sbatch_agg_baseline.sh` | the aggregate-binomial baseline for check 1 |
| `sbatch_permol_window.sh` | one array task per (arm × efficiency mode) |

Outputs go to `/usr/project/xtmp/nd141/permol_proto/`, **not** the repo.

## The frozen tree

`analysis/pkgvar/permol_seq_maskoff/` is a copy of `pkgvar/seq_maskoff` (the only tree
with the sequence layer ON, the fiber layers live and all 153 motifs). Exactly two files
differ from the source, both marked `PERMOL`:

* `robocop/robocop.py`
  * the forced `plot_all_factors_side_by_side` call is removed — it ran unconditionally,
    twice per segment, so 439 molecules would have emitted 878 throwaway figures;
  * the per-decode emission PNG dump after `posterior_decoding` is removed — it ran
    unconditionally and rewrote the same 7 PNGs after every forward-backward;
  * `build_fiber_ps` added (verbatim `ps` construction, callable on its own);
  * `update_data_emission_matrix_using_bernoulli_fiber_seq` added — the per-molecule
    emission, vectorised over the state axis (the aggregate's Python double loop is
    ~9.4e6 inner iterations for this window and would be ~3.0e9 over 439 molecules);
  * an **opt-in** parents/children cache in `posterior_forward_backward_loop`
    (`dshared['cache_parents_children']`), since those depend only on the fixed
    transition matrix. Off by default, because an EM run mutates it in place.
* `robocop/utils/robocopExtras.py`
  * `updatePerMoleculeEMMat` added: calls the Bernoulli emission for both strand layers
    and then applies the **same** `1e-30` floor as the aggregate path.

`updateMNaseEMMatNB` and `update_data_emission_matrix_using_binomial_fiber_seq` are
untouched, so the tree's aggregate path computes exactly what `pkgvar/seq_maskoff`
computes; the only behavioural difference is that it no longer writes the two sets of
diagnostic figures.
`getReads.py` and `robocop_no_em.py` are **not** modified: the driver takes molecules
straight from the BAM and runs its own per-molecule segment loop, so the package's pileup
ingestion never has to be bypassed.

Speed comes from two things that need no C change: the vectorised emission, and
pre-multiplying the 7 emission layers into 1 before the C call with `n_vars = 1` (the C
code multiplies the `n_vars` layers together — `algo.c:57-58`, `:84-85`, `:110-111` — so
this is algebraically identical).

## Reader semantics

`modkit pileup` keeps a base call only when the probability of the called state reaches
its confidence threshold, here **0.8105469** (from `pileup.log_whole_genome`). With
`p_mod = ML/255` that is:

* methylated `ML/255 >= 0.8105469`
* canonical `ML/255 <= 0.1894531`, **or** the adenine is absent from `MM` (the tag is
  mode `A+a.`, so unlisted adenines are implicitly canonical)
* dropped — "no observation" — anything in between; modkit excludes those from
  `Nvalid_cov` too.

`dx:i:0` on every read (no duplex), so a forward-aligned read carries Watson calls
(reference A) and a reverse-aligned read Crick calls (reference T) — the same contract as
the aggregate layers (`nucleotide_ref = 0` / `3`).

`pysam.AlignedSegment.modified_bases` reports positions in **stored**-query coordinates
(for a reverse read the key is `('A', 1, 'a')` and `query_sequence[pos]` is `T`), so they
line up with `get_aligned_pairs()` without any flipping.

**Ragged spans.** Each molecule is decoded over its own aligned span intersected with the
window; `n_obs` is a plain argument (`robocop.py:265-274`, `:282`), `end_probs` is all ones
with `end_at_any_state = 1`, and `initial_probs` is length-independent, so no C change is
needed. Clipping to the window is a memory decision, not a modelling one. A motif within
~70 bp of a read end is censored on that molecule, so `molecules.tsv` records every span
and `checks.py` reports per-site molecule counts.

## The three footprint arms

The shipped per-TF vectors were fitted as a population average over bound **and** unbound
molecules: ABF1's mean is 0.0843 against a background of 0.1383, only a 1.6× contrast.

* **A** — shipped vectors as-is. The single-variable control: only the emission *form*
  changes.
* **B** — deconvolved, same shapes: `p_bound = (p_agg - (1-pi) * p_bg) / pi`, floored at
  `1e-3`. `pi ∈ {0.3, 0.5, 0.7}`.
* **C** — flat protection floor `0.024286`, the minimum of the fitted nucleosome vector,
  i.e. a measured "fully protected" rate for this enzyme and basecaller.

**Decision (CLAUDE.md rule 7).** Arms B and C change only the state blocks of the 12 TFs
that have a fitted per-position footprint. Background, the 531 nucleosome states and
`combined_low_count` (the single scalar 0.249 shared by the 141 unfitted motifs, including
`unknown`) keep their shipped values. Reason: the mixture inversion needs "each TF's own
shape", which only those 12 have, and holding the 141 competitors fixed is what isolates
footprint depth as the variable under test.

**Per-read efficiency** (measured spread 12.9× p5→p95, against a ~2× footprint effect):
each molecule's whole `ps` vector is scaled by that molecule's own flanking m6A rate over
the ±400 bp **outside** the window (candidate excluded) divided by the pooled flanking
rate, clipped into (0, 1). Molecules with fewer than 50 informative flank calls fall back
to their whole-span rate, then to no correction; `molecules.tsv` records which. Every
emission entry stays a valid Bernoulli likelihood (`p` and `1-p`), so the scaled
forward-backward normalisation is unaffected; each arm still runs with the correction on
and off so the effect is measured rather than assumed.

## The four checks

See the table in `checks.py`'s docstring; thresholds were fixed in the plan before any
run. In short:

1. **Correctness** — molecule-average posterior vs an aggregate decode of the same window
   under the **same tree and trainDir**; pass at `r >= 0.90` on nucleosome occupancy.
2. **Nucleosomes** — pooled per-molecule dyad calls vs Chereji +1/-1, plus
   molecule-to-molecule dyad spread (if every molecule returns the same path, sequence and
   transitions are swamping the fiber layer).
3. **TF** — per-molecule ABF1 posterior at the two MacIsaac sites vs the same molecules at
   within-NDR shifted controls; `AUROC >= 0.70` would overturn the scoping verdict.
4. **Context survives the HMM** — fraction of molecules putting < 0.1 nucleosome posterior
   across the two motifs; the isolated LLR analysis predicts ~0.84.

Note on the check-1 baseline: `analysis/robocop_erv46_maskoff` decodes the same window but
was produced by `run_fiberonly_noem.py`, which inserts `../pkg/`, where
`robocopExtras.py:101` (`data_emission_matrix[0][:] = 1`) is **live** — i.e. it is
fiber-only, sequence layer **off**. Comparing against it would vary two things at once, so
`sbatch_agg_baseline.sh` runs the **unmodified** `run_split_revfix_seq_maskoff.py` on
`coord_erv46.tsv` instead and that is the baseline.

## How to run

```bash
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg R_HOME=$CONDA_PREFIX/lib/R

python singlefiber/validate_reader.py --json /usr/project/xtmp/nd141/permol_proto/reader_validation.json
sbatch singlefiber/sbatch_agg_baseline.sh            # check-1 baseline
sbatch --array=0      singlefiber/sbatch_permol_window.sh   # arm A, efficiency off
sbatch --array=1-9%5  singlefiber/sbatch_permol_window.sh   # the other nine
python singlefiber/checks.py --out /usr/project/xtmp/nd141/permol_proto/checks.json
```

Each per-molecule run writes, under its output directory:
`tmpDir/info_0_1.h5` (the molecule-**average** posterior, written as an ordinary
`segment_0` so `score_robocop.py` and `write_factor_table.py` read it untouched, plus a
`molecule_depth` array because `region_optable` divides by segment count, not molecule
count), `permol.npz` (per-molecule collapsed factor columns), `molecules.tsv`,
`run_stats.json`, `config.ini`, `coords.tsv` and `RUN_INFO.json`.

## Result of the first run (2026-09-28): arm A, efficiency off

439 molecules over `chrI:60001-65000`. 301 s of decode (**0.69 s/molecule**), peak RSS
**1.77 GiB**, against 77 s and **10.6 GiB** for the aggregate decode of the same window.
Numbers below are in `/usr/project/xtmp/nd141/permol_proto/checks.json`,
`reader_validation.json` and `pooled_check/pooled_check.json`.

**Reader — exact.** Summed over molecules, the per-molecule calls reproduce the modkit
pileup's (k, n) at **every** position the emission reads: 1,495 reference-A and 1,339
reference-T positions, `max|difference| = 0` for all four arrays. The 1.3 % of pileup
trials not recovered sit entirely at positions where the reference base is **not** A/T
(1,459 Watson + 900 Crick trials, from read-level A's at mismatches), which
`robocop.py:791-792` discards anyway. Ambiguity loss 9.22 % (22,291 of 241,839 calls with
`0.1895 < ML/255 < 0.8105`), matching the documented ~9.3 %.

**Check 1 — FAILS its threshold (r = 0.444 on nucleosome occupancy, pass was ≥ 0.90), but
the emission is correct.** The threshold assumed that averaging per-molecule posteriors
approximates the posterior of the pooled data. It does not, and
`pooled_emission_check.py` proves the code is right instead: multiplying all 439
molecules' Bernoulli factors into **one** chain — using the shipped emission function,
unchanged — reproduces the aggregate binomial decode at **r = 0.9999999+ on every factor
column**, `max|difference| ≤ 4.9e-4`. The identity behind it is
`∏ ps^m (1-ps)^(1-m) = ps^k (1-ps)^(n-k) = binom.pmf(k,n,ps) / C(n,k)`, and `C(n,k)` is
state-independent so it cancels.

What check 1 actually measured is **prior reversion**: one molecule carries ~1/50 of the
evidence, and the trained priors are `background_prob = 0.933` against
`nucleosome_prob = 0.00136`, so a single molecule's posterior falls back toward background.
Mean nucleosome occupancy is 0.560 per-molecule-average vs 0.858 aggregate (background
0.284 vs 0.080); where the aggregate says occupancy > 0.9 the molecule average says 0.599,
where it says < 0.1 the average says 0.308. The shape is partly recovered —
`r` rises 0.44 → 0.50 → 0.58 → 0.66 as the tracks are smoothed at 25/75/147 bp.
**Check 1 should be re-specified** as "pooled-emission decode vs aggregate decode", which
is the question it was trying to ask; the molecule-average correlation is a result, not a
correctness test.

**Check 2 — PASS.** Pooled per-molecule dyad calls recover **4/4** Chereji +1/-1 dyads in
the window at ≤ 20 bp (recall 1.00, median distance 0.5 bp), and the molecules do not
collapse onto one path: **434 of 439** posterior tracks are distinct and the
molecule-to-molecule dyad scatter is **27.2 bp** (per-site 19.5/28.3/26.1/28.6 bp), well
over the 10 bp floor. Precision is meaningless here — Chereji lists only +1/-1 nucleosomes.

**Check 3 — the ranking works, the calls do not.** Pooled AUROC **1.000** (281 vs 281)
against within-NDR controls 65 and 77 bp away with matched aggregate occupancy (0.0014 and
0.0008 vs 0.0 at the motifs), which formally clears the 0.70 bar. But no molecule is
anywhere near a call: the per-molecule ABF1 posterior at the motifs is 6.0e-6 (site 1) and
8.1e-4 (site 2) at the median, **0 of 281 molecules above 0.10**. The per-molecule
variation is nonetheless real and in the right direction — 15.6×/18.9× spread across
molecules at one site (CV 0.40/0.36), and the posterior is strongly *negatively*
correlated with that molecule's own methylated fraction over the motif (Spearman
**−0.68**, p = 1.5e-19, n = 133; **−0.74**, p = 1.2e-24, n = 137), i.e. protected molecules
score higher. So single-molecule ABF1 evidence exists and is ordered correctly; what is
missing is magnitude. That is exactly what arms B and C were designed to move.

**Check 4 — matches.** 0.900 of molecules put mean nucleosome posterior < 0.1 across the
two motifs (0.890 / 0.910 per site; 0.877 / 0.868 using the max over the motif), against
the ~0.84 the isolated LLR analysis predicted, and 0.35 at the shifted controls. Per-
molecule chromatin state survives the HMM's transition priors.

**Arms B and C were not run**, per the plan's stop rule on check 1. Everything is in place:
`sbatch --array=1-9%5 singlefiber/sbatch_permol_window.sh`.
