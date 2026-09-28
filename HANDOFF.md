# RoboCOP + Fiber-seq — Handoff

**Read `CLAUDE.md` first** for the environment, the rules, and the gotchas. This file carries
the state of each thread and where to pick it up. `whattodo.md` holds strategy and the
open-work tiers; `FIBERSEQ_CHANGES.md` the code diff against upstream;
`things_to_revisit_before_shipping.md` the debug leftovers and hardcoded knobs.

Sections: **§1 concentration calibration (current)** · §2 what a concentration is ·
§3 tools · §4 model mechanics · §5 ground truth · §6 motif audit · §7 widened footprints ·
§8 earlier side experiments · §9 published artifacts · §10 standing constraints ·
§11 repo state.

---

## 1. Concentration calibration — FINISHED 2026-09-12

### 1.1 What was run

The per-DBF prior (`tf_prob`, the "concentration") is never fitted in this fork (§2), so it
was calibrated by fixed-point iteration against **MacIsaac p005_c1 site counts**: decode the
genome, count each factor's calls, move its λ toward its target, repeat. Two campaigns, 8
rounds each, genome-wide, λ_unknown fixed at 0.01 and the nucleosome prior held:

| campaign | live motifs | driver | stopped because |
|---|---|---|---|
| `u001` | all 153 | `run_split_revfix_seq_maskoff.py` | hit the 8-round limit |
| `m001` | 84 MacIsaac motifs + `unknown`; **69 hard-masked** | `run_split_revfix_seq_maskoff_macisaac.py` | within-2× count stalled (77→77→77) |

Masking is emission-only (`pkgvar/seq_maskoff_macisaac/`, §4.3): the masked motifs keep their
prior mass, so the tuned factors' priors are identical to `u001`'s and the only difference is
that the competitors cannot be called. They hold 1.1% of the prior at every entry — more than
all 84 kept motifs together (0.96%).

### 1.2 Results

**Counts converge.** Groups within 2× of their MacIsaac count, over rounds 0→7:

| campaign | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | typical factor |
|---|---|---|---|---|---|---|---|---|---|
| `u001` | 16 | 27 | 40 | 60 | 66 | 71 | 78 | **79 / 81** | 1.07× off |
| `m001` | 15 | 31 | 42 | 64 | 74 | 77 | 77 | **77 / 81** | 1.05× off |

**Site accuracy does not follow.** Calls are posterior ≥ 0.10 matched to a MacIsaac site
within 30 bp; MacIsaac requires cross-species conservation, so these are lower bounds, good
for comparing rounds rather than as absolute accuracy:

| round | u001 precision | u001 recall | u001 F1 | m001 F1 |
|---|---|---|---|---|
| 0 | 1.7% | 8.4% | 0.029 | 0.025 |
| 1 | 2.7% | 10.5% | 0.043 | 0.040 |
| 3 | 3.3% | 8.7% | 0.048 | 0.041 |
| 7 | 3.5% | 7.8% | **0.049** | 0.042 |

Almost all of the gain is step 0→1, which cut the gross over-callers; rounds 4–7 each add
≤ 0.0002, and m001's last round is negative. Recall *peaks at round 1* and then falls as the
counts are pushed onto target.

**Why: marginal precision is 0.8–2.4% for every step, in both directions** (Δmatched/Δcalls,
from `tuning_trajectory.py`). Calls a raise adds are about as wrong as the ones already
there; calls a lowering removes were 1–2% real. **λ moves the model along the count axis, not
the accuracy axis** — the same conclusion as the ABF1 λ sweep, where AUROC was flat at every
λ (0.56–0.59) while F1 moved 2.5×.

**Masking does not help; it slightly hurts** (final F1 0.042 vs 0.049). With competitors
gone the kept factors absorb their calls, so more of them bottom out: at the λ = 1e-6 floor
and still over target, `m001` has ABF1 (1,638 copies vs 300 sites), REB1, FKH1 and MCM1;
`u001` has only REB1 (1,177 vs 279). **The can't-fix list only covers the λ cap**, so these
floor-stuck over-callers are never flagged — a small, worthwhile addition to `cmd_update`.

**Masking does fix HAP1**: within 2× at λ = 1e3, 22 of 184 sites found. In `u001` it stays on
the can't-fix list (54.6 copies vs 184, would need λ ≈ 8e3).

**Per-factor extremes (u001, round 7 vs 0):** gains ZAP1 +0.297, REB1 +0.076, SUT1 +0.052,
MBP1 +0.041; losses CBF1 −0.046 (it started near target and ~10% of the calls its lowering
removed were real), TYE7 −0.025, RPH1 −0.020, GCN4 −0.020.

**Nucleosomes never moved**: 66,545 (u001) and 66,799 (m001) copies against 66,521 untuned,
i.e. +0.4% at the widest, with the prior held every round.

### 1.3 The machinery

| file | what |
|---|---|
| `tune_concentrations.py` | the loop: `status` / `build` / `submit [--chain]` / `update` / `next`, namespaced by `--run` |
| `make_conc_targets.py` | per-motif targets from MacIsaac c1 (or Rossi); `--verify` pins the merge rule; `load_macisaac_c1_sites()` gives site centers |
| `make_conc_trainDir.py` | rewrites one trainDir row from a λ vector, with fidelity + blast-radius gates; `--hold-nucleosome` solves the compensating λ_nuc |
| `count_calls.py` | per-factor `occ` / calls per chromosome, plus the MacIsaac match sidecar under `macisaac/` |
| `tuning_trajectory.py` | per-step precision/recall/F1 and marginal precision → `conc_tuning/<run>/trajectory.tsv` |
| `sbatch_tune_next.sh` | one link of the auto-chain |
| `conc_tuning/make_conc_sheet.py` | rebuilds the published concentration sheet |

A campaign lives in `conc_tuning/<run>/` (`state.json`, `report_NN.tsv`, `trajectory.tsv`,
`chain.log`, `STOPPED`) with trainDirs `robocop_train_ct_<run>_NN`, decodes
`robocop_genome_ct_<run>_NN` and counts `conc_tuning/counts_ct_<run>_NN.tsv`.

**Step size.** `λ *= (T/E)^(α/β)` with α = 0.7 damping and β seeded at 0.23 — the measured
elasticity `d log count / d log λ`, re-estimated per group by secant after round 1. A naive
`(T/E)^0.7` closes only ~16% of the log gap per round (~25 rounds instead of 4–8). λ is
clamped to [1e-6, 1e3]; above ~1e4 the shared unbound root collapses and `p^147` destroys the
nucleosome model.

**Cross-TF coupling is real but sparse** — dropping one factor's λ 100× moved the other 147
by a median of 1.0000 (95th pct 1.0029), with Rsc3 +24.7%, Nhp6a +9.3%, Spt15 +5.5% — which
is why all factors are tuned at once with damping rather than one at a time.

**The auto-chain.** `submit --chain` appends a `ctNext_<run>_NN` job (`afterany` on the count
job) that runs `update`, evaluates the stop rules, then builds and submits the next round.
Stops write `conc_tuning/<run>/STOPPED` with a reason: nucleosome copies >5% from 66,521;
within-2× not risen for 2 rounds; 8 rounds; or any exception. The count job carries
`--kill-on-invalid-dep=yes` so a dead decode cancels it and the chain stops **visibly**
instead of pending forever.

**One failure mode seen.** Every decode task copies the trainDir's `config.ini` into the
output directory and immediately parses it, and `cp` truncates before writing, so a task that
reads mid-copy dies with `NoSectionError: 'main'` (u001 round 3, task 32). The decode script
now retries once after 60 s.

### 1.4 Where to pick this up

1. **Validate against Rossi** (§5.2), which was deliberately held back the whole time. That
   is the outstanding question: did count-matching against a conserved-site catalogue make
   the model better or just narrower?
2. The λ vectors worth carrying forward are in `conc_tuning/{u001,m001}/state.json`; the
   trainDirs `robocop_train_ct_u001_07` and `..._m001_07` are the tuned models.
3. Do not expect more λ rounds to improve accuracy (§1.2). The next real lever is the fiber
   emission (§4.6) and then the PWMs (§6).
4. Optional: flag λ-floor-stuck over-callers, mirroring the can't-fix rule.

**A standing two-site test case (ERV46).** Two MacIsaac ABF1 sites sit in
`chrI:60,001-65,000` and **no run has ever got both**: the baseline nails `62,657-62,671`
(posterior 0.99) and misses `61,163-61,177` (0.002), while the low-ABF1 variant recovers the
upstream one (0.55) and collapses the downstream one (0.001). A correctly retuned prior should
get both — a quick, cheap check on any new model. Compare with
`python make_posterior_viewer.py --region chrI:60001-65000 --run a=<dir> --run b=<dir>`.

**Other open threads, not started.** Retrain with the sequence layer ON — every current
`fib_seq` decode reuses Fiber-only-trained weights, so they are a lower bound on what the
sequence layer can do. And the scorer re-collapses the whole ~230 kb chrI segment on every
`score()` call (~9 min/run); caching the collapsed factor track per region would pay for
itself in any multi-run sweep.

---

## 2. What a concentration is

Published sheet: <https://claude.ai/artifact/XdB264uQyfr9b8F23B5sgy> (old id `f7ff1c1b-…`)
(rebuild with `python conc_tuning/make_conc_sheet.py u001=7 m001=7`).

The paper assigns weights and turns them into transition probabilities:

> To initialize the probabilities, we assign weight 1 to the "empty" DBF (representing an
> unbound DNA nucleotide) and 35 to the nucleosome. To each TF, we assign a weight which is
> that TF's dissociation constant K_D … α_k = w_k · α₀^{L_k}

1. **Per motif** (`parameterize.calculateKD`): `K_d = Π_i bg(b*_i) / p_i(b*_i)` over columns,
   where `b*` is the column's most likely base. A sharp 14 bp motif scores ~1 for its best
   base everywhere, so each column contributes roughly the background frequency; a vague 7 bp
   motif contributes ~1 per column.
2. **By hand**: empty DNA 1.0, nucleosome 35, `unknown` (a flat 10 bp matrix) 0.1.
3. **To probabilities** (`concentration_probability_conversion.convert_to_prob`): solve the
   single unbound root α₀ of `Σ_k w_k α₀^{L_k} = 1` by `np.roots`, scale each weight by
   α₀^{L_k}, normalise.

So concentrations differ **because the motifs differ**: Nhp6a (7 bp) starts at 1.17e-3 and
Abf1 (14 bp) at 4.73e-7, a 2,478× gap, with λ = 1 for both. λ multiplies the weight.

**The cancellation to keep in mind.** At its own consensus a motif's sequence likelihood beats
background by exactly `1/K_d`. Setting `w_k = K_d` cancels that, so every factor is about
equally callable at its own best word: the starting concentration encodes **motif sharpness,
not protein abundance**. Physically the statistical weight should be `[TF]_free / K_d`, and
the missing per-factor scalar is exactly what λ stands in for.

**Nothing here is fitted.** `robocop_em.py` hardcodes `iterations = 0`, so the Baum-Welch
update the paper describes never runs: all 154 states in
`robocop_train_fiberonly/HMMconfig.pkl` are bit-identical to values recomputed from `pwm.p`,
and `robocop_train/likelihood.txt` has one line. When EM *was* forced on (`pkgvar/
seq_maskoff_em10/`, `ROBOCOP_EM_ITERS=10`) it moved ABF1 3,723× **up** while the data wants it
~100× down, pinned 28 of 154 factors at the published cap (`mean + 2·std` of the *initial*
priors = 6.69e-4, computed once and never recomputed) and drove 66 more to exactly 0. EM is a
dead end here; `unknown` is also exempt from that cap (`adjustEM`'s `range(1, n_tfs)` skips
the last TF and `unknown` sorts last).

**`unknown` is the wildcard.** Flat emission, so it matches anywhere, and it held 3.3× the
prior of all 153 real motifs combined. A sweep of λ_unknown ∈ {1, 0.3, 0.1, 0.03, 0.01} on
chrIV+VII+XV (nucleosome prior held) cut its occupancy 2.4× and gave real TFs +47% mass, but
the gain is *global*, not discriminating: the median gap improved 8.3× → 4.5× while the number
of factors within 2× stayed flat, and REB1 got worse. It is now pinned at **0.01** by user
decision. HAP1's posterior maximum is 3.4e-4 genome-wide at λ=1 — extinct, not out-competed.

---

## 3. Tools

### `score_robocop.py` — the scorer, single source of truth
Scores a decode against Chereji ±1 dyads, MacIsaac ABF1 sites, phasing period and MNase
accessibility, reading the sparse posterior from `tmpDir/info.h5`.
- `score(outDir, regions=None, tol_nuc=20, tol_abf1=20, abf1_global_max=None,
  return_abf1_tracks=False)` → metrics + `_per_region`.
- **One peak-caller**: `_above_threshold_runs` → `call_abf1(track, pos, threshold)` returns
  the **center of each above-threshold run** (not an argmax — the sequence layer produces flat
  saturated plateaus where an argmax jitters with the window). Threshold
  `abf1_call_threshold(gmax) = max(0.10, 0.30*gmax)`; pass `abf1_global_max` when scoring a
  window so it uses the run's real global threshold (cached in `abf1_thresholds.json`).
- Nucleosome dyads use `call_peaks` (find_peaks), unchanged.

### The rest
| tool | what |
|---|---|
| `score_factors.py` | the same caller over **every** factor with ground truth, for many runs; `--runs-from <runs.tsv>` |
| `count_calls.py` | reference-free per-factor `occ` (expected copies) + calls + MacIsaac matches; what the tuning loop consumes |
| `tuning_trajectory.py` | what each λ step did to precision/recall/F1 |
| `plot_abf1_locus.py` | per-locus ABF1 panels; **invokes** the scorer's tracks and calls, so it moves with any change to the caller |
| `make_posterior_viewer.py` | the published occupancy browser; label is the join key across regions |
| `make_factor_chart.py` | per-factor detection chart from `layer_scores/` reports |
| `nhp6a_diag.py` | why Nhp6a over-calls in a given decode |

`occ = Σ posterior / block_len` is the tuning signal: threshold-free and the conjugate of the
prior. `n_pred_adaptive` is a poor tuning signal because its threshold is 30% of the factor's
own max, so it moves with the concentration; `n_pred_fixed` (≥0.10) is stable.

---

## 4. Model mechanics

### 4.1 Emission layers
`np.ones((7, n_obs, n_states))`, each active layer multiplied in (a layer left at 1.0 is OFF):
0 sequence/PWM · 1–2 MNase short/long · 3–4 ATAC · 5–6 Fiber-seq Watson/Crick. Fiber layers
are filled by `update_data_emission_matrix_using_binomial_fiber_seq`, with zeros floored to
1e-30. MNase/ATAC are off by design: the phased plan is to get Fiber-seq right, then add
sequence, then MNase.

### 4.2 `analysis/pkgvar/` — layer/mask state is frozen, not hand-commented
Each variant is a full copy of the `robocop` package with its toggles applied, selected by the
driver (`sys.path.insert(0, 'pkgvar/seq_maskoff/')`). 28 copies exist; `robocop.py` is
byte-identical across them and only `utils/robocopExtras.py` differs. Hand-commenting was a
race — with a Slurm array you cannot know when a task imports the file.

| variant | sequence layer | mask |
|---|---|---|
| `fiber_maskon` / `fiber_maskoff` | OFF (`[0][:] = 1`) | ABF1-only / none |
| `seq_maskon` / `seq_maskoff` | ON | ABF1-only / none |
| `seqonly_maskon` | ON | fiber layers set to 1, **mask moved onto layer 0** |
| `seq_maskoff_{bgtss,lowabf1}` | ON, no mask | differ only in which `inputs/*.pkl` they load (`bg_params_tss.pkl` at `robocop.py:606`; `all_TFs_1000pealVal_params_pseudo_lowabf1.pkl` at `robocop.py:598`) |
| `seq_maskoff_12tfs` | ON | **mask**: keeps the 12 fitted-footprint TFs (`FITTED_12`, `robocopExtras.py:123`), masks all other TFs and `unknown` |
| `seq_maskoff_macisaac` | ON | keep-list: 84 MacIsaac motifs + `unknown` (§1.1) |
| `seq_maskoff_em10` | ON, no mask | EM on via `ROBOCOP_EM_ITERS` |

### 4.3 The hard mask
In `robocopExtras.py`, **after** the 1e-30 floor, zero the Fiber channels over the states to
forbid: `data_emission_matrix[5|6][:, s:e] = 0`. Emission is a product over channels, so those
states get posterior exactly 0 while background and nucleosomes keep the floor (no NaN).
Masked states **keep their transition prior**, which is simply lost — the model is not
renormalised. Derive the slice from `dshared['tfs']` + `tf_starts`/`tf_lens` and assert the
names exist (`seq_maskoff_12tfs` and `_macisaac` do); the old ABF1-only form hardcoded
`29:nuc_start`, which only worked because Abf1 sorts first. State layout: `0` background,
`1..28` ABF1 fwd+rev, `29..nuc_start-1` other TFs + `unknown`, `nuc_start..` nucleosomes.

### 4.4 trainDir vs decode
A decode reads `pwm_emission` / `tf_prob` / `transition_matrix` from `HMMconfig.pkl`
(`robocop_no_em.py:51`) — so a new PWM file needs a **retrain**, while a concentration change
does not: `make_conc_trainDir.py` rewrites only row `silent_states_begin` of the transition
matrix (`set_transition` writes background, nucleosome and the TF entries there) in ~1 min,
and its two gates prove it (fidelity: λ=1 reproduces the source bit-exactly; blast radius:
only that row changed). Decode with `run_robocop_without_em`, which keeps `tmpDir/info.h5`.

### 4.5 Run naming
Directory == label, punctuation aside: `fib+seq+lam0.01` → `robocop_<chrom>_fib_seq_lam0p01`.
`fib_seq_` = both layers, `fib_` = Fiber only. Not renamed, because they are not runs:
`pkgvar/*` (frozen packages), `sbatch_*_wide10.sh` and `*_widememe*.py` (generic drivers where
the name is the method, not the run).

**Runs on disk.** The current chrI/chrXIV series is `robocop_<chrom>_fib{,_seq}[_<variant>]`;
prefer anything with `_revfix` or a post-`revfix` name, because the original four
(`robocop_chrI_{maskon,maskoff,seq_maskon,seq_maskoff}`) predate the reverse-strand
fiber-parameter fix in `90b05c3` and are kept only because older numbers refer to them. Also
present: the three `*_JASPAR` runs (trainDir `robocop_train_jaspar/`, built by
`sbatch_train_jaspar.sh` from `config_jaspar.ini`, which differs from `config_fiberonly.ini`
on the `pwmFile` line alone), the fiber-parameter variants
`robocop_chrI_seq_maskoff_{12tfs,bgtss,lowabf1}`, and the tuning campaigns'
`robocop_genome_ct_{u001,m001}_NN`.

**TF colours are deterministic**, keyed on the factor's name by `_color_for_name` in
`plotRoboCOP{,ax}.py`, and cached per run as `<outDir>/dbf_color_map.pkl`. If a factor ever
changes colour between decodes, delete that stale pickle and re-plot. `nucleosome` = grey 0.7,
`unknown` = `#D3D3D3`.

### 4.6 The fiber emission is the real limit
The fitted ABF1 p-vector runs 0.029–0.137 against a background of 0.138 — every position below
background — so it is a generic protection detector with almost no positional specificity, and
with raw coverage `n` untempered it produces likelihood ratios of 1e9–1e10 at spots 8–91 bp
off-motif that no sequence term can overturn. This is why a PWM swap that goes 2/5 → 5/5 on a
pure FIMO scan only reaches 3/5 inside the decode. The emission layer also consumes only the
14-column vector, discarding the elevated flanks and not even spanning the ~21 bp real
footprint. **Fixing this outranks any further PWM change** (`whattodo.md` Tier 1 #3).

### 4.7 Numeric limits
`bc.c`'s index arithmetic is 32-bit: `n_states` ≤ 46,340. `wide150all` (all 153 motifs at
±150, `n_states` 95,285) overflows it — verified by calling the shipped `.so` directly — and
its driver refuses to submit. ±70 across all 153 motifs (`n_states` 46,325) is the largest
uniform pad that fits and needs no code change; it was measured at ~295 GB to train and ~65 GB
to decode, so only the 1,150,000 MB `compsci-cluster-fitz-*` nodes hold the training job.
Widening the index type to `int64_t` in `bc.c`/`bc.h`/`algo.c` for a separate `.so` was scoped
but not done; it changes the numerical core and must be re-validated against an existing run.

---

## 5. Ground truth and validation sets

### 5.1 MacIsaac (the tuning target)
`/usr/project/xtmp/nd141/projects/replicate_prob_dyad_plot/data/ref-data/MacIsaac_p005_c1_V64_SGD.gff3`
— 27,870 rows, 119 TFs, sacCer3, md5 `898b4e62cc5317da9167c568fb6fdd06`. **Outside the repo**;
`inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed` is only a hand-filtered 2-TF slice of it (a
repo-only search wrongly concludes MacIsaac covers just ABF1 and REB1). Rows are redundant:
merge by interval union with 20 bp slop, anchored on the motif-interval midpoint, which
reproduces ABF1 300 / REB1 279 / RAP1 233 (raw 315 / 293 / 282) and 27,870 → 25,104 sites.
81 of the 150 motif prefixes join it.

### 5.2 Rossi ChExMix — held back as validation

**Where the data lives** (outside the repo; scripts spell it `/usr/project/xtmp/…` or
`/usr/xtmp/…`, the same directory):
`/usr/project/xtmp/nd141/projects/data/rossi_strand/`

| what | files | contents |
|---|---|---|
| **Merged ChExMix calls** | `<TF>_CX.bed`, 381 files (`Abf1_CX.bed`, …) | one file per TF, merged across replicates; 1 bp summits + score, **no motif, no strand**. This is the `_cx` set below. |
| **Per-sample zips** | `<sample_id>_YEP.zip`, 778, each extracted to `<sample_id>/<sample_id>_YEP/` | one ChIP-exo replicate each, Rossi's per-sample output: bound-feature beds (promoter/TSS/TES/gene-middle), heatmaps/composites, that sample's MEME motifs (`<id>_MEME_Motifs.txt`) and FIMO instances (`<id>_FIMO_Motifs`, `<id>_Motif_<n>_FourColor.bed`) |
| **Per-TF motif-anchored beds** | `output/<tf>_rossi_peak_w_strand.bed` | `rossi_strand.py` (same dir) downloads each zip and keeps `_CX` summits with a sample FIMO motif nearby, taking the strand from the motif |

Sample id → TF / replicate / assay / condition / on-disk: `analysis/inputs/rossi_sample_conditions.tsv`
(built by `fetch_rossi_conditions.py` from GEO GSE147927 + the Rossi GitHub sample key).

Derived files in `analysis/inputs/`:
- `rossi_peak_w_strand_all_TFs.bed` — the per-TF motif-anchored beds concatenated: header + 29,328 rows,
  358 TFs, columns `chr start end peakVal strand motif sample_id replicate TF`. **This is the `_motif`
  set.** HANDOFF's 29,105 is after dropping the 223 rows from heat-shock samples (`--normal-only`,
  joined on `rossi_sample_conditions.tsv`).
- `rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed` — 2,839 rows, 74 TFs, every
  `peakVal` = 1000: each `_motif` peak re-anchored to the best-scoring site of RoboCOP's own PWM
  (`TF` in model naming, `score`, `best_seq`, `agree` = strand matches Rossi's). Built by
  `analysis/conform_TFs_to_PWM.py`, which hardcodes paths into the older `roboNhat_w_new_changes`
  repo and writes an unfiltered `…_peakVal.bed`; where the `peakVal == 1000` filter was applied was
  not found. This is what `score_factors.py` scores against **and what the fitted fiber footprints
  were trained on** — see the circularity caveat below.

The motif audit (§6) reads the zips' `_MEME_Motifs.txt` (`build_motif_dbs.py`, "Rossi ChExMix 856",
one motif per replicate); `rossi_genic_all.py` reads the `_CX.bed` files directly.

Genic/intergenic is the distributional target. **One rule**: a position is `genic` if it lies
between some gene's ATG and its stop codon. An earlier four-class scheme (promoter / gene-end
/ gene-body / intergenic) was built and **abandoned** — its windows and priority order moved
the answer several points without adding a fact, and labelled peaks inside gene A's ORF as
"promoter of gene B". Do not reintroduce windows. Coordinates come from `inputs/sacCer3.gtf`
(the `gene` span *is* the ORF, audited over 6,516 genes), not `Park_2014_TSS.csv`, which
misses 26% of ORFs. **The null is 73.0% genic**, so every table carries
`genic_vs_null = genic% / 73.0%`.

Two peak sets, different targets: `_cx` = Rossi's merged ChExMix calls (378 TFs, 182,582
peaks); `_motif` = the subset with a YEP FIMO motif within 30 bp (358 TFs, 29,105). The motif
filter removes genic peaks preferentially (Fhl1 53.2 → 17.9%), so score against whichever set
the decode resembles. Usable scope: 77 of 153 motif TFs have a Rossi row, 47 with ≥100 peaks,
genic% spanning 8.8% (Spt15) to 59.3% (Cad1), median 27.8%. Do not score against one global
expectation — pooled genic% is 42.2% but 82 of 378 TFs sit at or above the null.

The classifier validates itself at both ends: the intergenic extreme is Pol II preinitiation
(Spt15/TBP 8.8%, Sua7 9.8%) plus the whole Pol III machinery and Orc1 (2.0%); the genic
extreme is Paf1C (89–93%), COMPASS (Bre2 94.6%) and Set2 (94.3%) — elongation factors that
ride the ORF. Files in `analysis/rossi_genic/`; scripts `rossi_genic.py` (the 12 fitted TFs),
`rossi_genic_all.py` (all 378), `make_genic_report.py`.

ABF1's in-ORF peaks are **real, not filter leakage**: per-peak replicate support 2.29 in gene
bodies vs 2.24 in promoters, 0 of 55 resting on the pooled analysis alone
(`rossi_abf1_support.py`). Without the motif filter, gene body *is* the weak tail — the motif
requirement, not the location, separates strong from weak.

**Circularity caveat:** the fitted Fiber-seq footprints in
`inputs/all_TFs_1000pealVal_params*.pkl` were trained on Rossi peak locations, and MacIsaac is
77.5% inside Rossi. Any template-based result must refit leave-one-chromosome-out or it is
measuring memorisation.

### 5.3 The 12 TFs with a fitted footprint
Abf1_murphy, Cin5_murphy, Fhl1_zhu, Fkh1_zhu, Mcm1_zhu, Nhp6a_zhu, Rap1_telomeric,
Reb1_badis, Sko1_murphy, Spt15_zhu, Tbf1_zhu, Ume6_zhu. The other 141 fall back to
`combined_low_count`, whose p (0.2489/0.2647) is **above** the background p (0.1383) — so
those states outscore background exactly where DNA is most accessible, which is why TFs get
decoded into linkers. That is the artifact `pkgvar/seq_maskoff_12tfs` was built to remove.

---

## 6. Motif audit (2026-08-14 → 08-18) — read-only, nothing swapped

Published sheet ("Odd One Out"):
<https://claude.ai/code/artifact/7c63a39b-8e04-4cd1-9660-7797d0dec154>. Rebuild with
`python motif_distance_sheet.py` then `python make_motif_sheet_page.py`.

**ABF1's shipped matrix is wrong in its middle.** `Abf1_murphy` (w=14) and JASPAR MA0265.3-rc
agree on both half-sites (column r = +0.998 over 9 columns) and are **anti-correlated in the
5-column spacer** (r = −0.411), where Murphy carries a GC-rich informative block
(P(GC) 0.678 vs JASPAR 0.300 vs genome 0.381). Real sites do not carry it, so Murphy charges
them for it: as a pure FIMO scan Murphy recovers **2 of 5** MacIsaac chrI sites and JASPAR
**5 of 5**, and every miss is a spacer penalty, not a core mismatch. Inside the decode the
gain is smaller — 3/5, with mean posterior at sites 0.204 vs 0.138 — because the fiber layer
limits it (§4.6). Nucleosome architecture is untouched either way.

**Why TOMTOM misses it:** its q-value asks "closer than two random matrices", which saturates
for any pair sharing a strong core. FIMO asks whether a 14-mer clears a threshold **summed
over all 14 columns**, and the spacer is 5 of them. **Never validate a PWM swap on
similarity scores alone — score real sites.**

**The audit, blind, over all 153:** three databases (`shipped` 153 / JASPAR fungi 193 / Rossi
ChExMix 856) assembled by `build_motif_dbs.py` into `motifdb/` (~60 MB, regenerate, do not
commit). No privileged reference: with three matrices there is always a closest pair, and the
**odd one out** is the member opposite it. Distances are raw mean-per-aligned-column ED/KLD,
not p-values. Widths differ, so a two-stage frame is used: search offsets/orientations against
a fixed W-column window (W = shortest motif) minimising the sum of the three pairwise KLs,
then report all three pairs over the **intersection** of those placements. Alignment is not
transitive, so the three pairwise optima are jointly realisable in only 59% of rows; on those
the two schemes agree on 78 of 82 verdicts and all 4 disagreements favour the intersection.

Verdicts: 82 rows have no Rossi motif, `rossi_is_odd` 29, `no_consensus` 22,
`all_three_agree` 9, **`native_is_odd` 7**, `ambiguous` 2, `two_datasets_only` 2. Median
distance to the other two: native 0.167, JASPAR 0.150, Rossi 0.175 — the shipped collection is
not systematically the outlier; the problem is motif-by-motif.

**The shortlist:** `Abf1_murphy` (unanimous, 3/3 replicates), `Rap1_motif2`, `Rap1_telomeric`,
`Rap1_zhu`, `Pdr1_badis` (all unanimous), `Rap1_motif1` (2/3), `Cad1_murphy` (1/2). ABF1 is
the only one with FIMO + decode validation; **the RAP1 matrices and Pdr1 have sheet evidence
only.** `Rap1_telomeric` matters most after ABF1 because it is one of the 12 with a fitted
footprint.

**If a matrix with a fitted footprint is swapped**, the p-vector is keyed on the motif ID and
registered to its column frame — keep the ID, width and orientation, which is why
`inputs/jaspar_abf1_motifs_meme.txt` keeps the name `Abf1_murphy` reverse-complemented into
Murphy's orientation. Honest limitation: 120 of 213 rows are scored on fewer than 8 columns
(many native matrices are 5–8 wide); do not act on a short-window row without a FIMO check.
Power too: the median surviving Rossi motif rests on 28 sites, 10% on fewer than 10.

---

## 7. Widened footprints

**`wide150` scored EMPTY, and the cause is the prior, not the pads.** The four decodes
(`robocop_{chrI,chrXIV}_fib{,_seq}_wide150`, Slurm 12492826–12492829, `n_states` 10,685 against
the 3,485 baseline, trainDir `robocop_train_wide150/`) are complete and were scored: ABF1's
posterior is **identically 0** at every MacIsaac site in all four. Not a block-width artifact
and not a scoring bug — the `Abf1_murphy` column is present and non-widened factors in the same
decode carry normal mass. The ±150 **estimated pads crush the 12 widened TFs' `tf_prob`** by
9.6e-19 (ABF1) to 5.0e-30 (Fhl1) while every non-widened TF sits at ratio 1.00, which
underflows the posterior to zero. Nucleosomes are unaffected (recall 0.81, dyad 7–8 bp, period
171), so the decodes themselves are healthy.

**It is rescuable without retraining**: `make_conc_trainDir.py --set <TF>=<lam>` takes all
twelve at once with `lam = baseline tf_prob ÷ wide150 tf_prob` (Abf1_murphy 1.04e18,
Reb1_badis 1.99e20, Fhl1_zhu 1.99e29, …). Recompute the full vector by reading `tf_prob` out of
both `robocop_train_fiberonly/HMMconfig.pkl` and `robocop_train_wide150/HMMconfig.pkl` rather
than copying stale numbers. That turns `wide150` into an actual test of the ±150 hypothesis
instead of a test of switched-off factors; it needs one new trainDir (a config build, not a
fit) plus four re-decodes, which is where the cost sits.

When adding any run to the scoring tables, widen the `#SBATCH --array` bound in
`sbatch_score_layers*.sh` as well — the bound is not derived from `layer_runs_*.tsv`, so a
stale one silently skips the new rows.

**Read widened enrichment with the block-width correction.** `sum_for_dbf_probs` is unmodified,
so the posterior collapses over the whole padded block: at ±150 an ABF1 call renders as a
314 bp plateau, not a 14 bp peak, deflating enrichment by roughly 314/14 ≈ 22×. A raw
enrichment drop is **expected, not evidence the model is worse** — compare recall and site
posteriors. Estimated-pad prior suppression is undone without retraining via
`make_conc_trainDir.py --tf <name> --lam <1/ratio>`.

Earlier results: pads 7/2 (`wideABF1`) made ABF1 **worse**; the discriminating signal is the
hyperaccessible flank further out, not more protection. ABF1's real motif is ~18 bp — exactly
4 flank columns survive Bonferroni — and beyond ±2 the pads should be background, not
estimated. There is **no ±75 run**; do not look for one. `wide150all` is blocked by §4.7.

**A gate that was wrong, and the fix.** `check_widememe_traindir.py` failed `wide150` on a
6.1e-6 residual spread against a 1e-8 tolerance; the model was fine and the gate was wrong —
it trusted the `background_prob` stored by `convert_to_prob`, which converges less tightly than
the `tf_prob`s built from it (`corr(residual, tf_len) = 1.0000` gave it away). It now refits
the root by least squares from the priors themselves (spread 1.1e-15). The gate needs
`--params inputs/all_TFs_1000pealVal_params_pseudo_<variant>.pkl`.

Full rationale for the family: `analysis/README_wide_implementations.md`.

---

## 8. Earlier side experiments

**Sliding the fitted ABF1 footprint across the genome** (chrI, read-only). A matched filter:
mean-centre the template (`w_j = p_j − mean(p)`, shape only), form variance-stabilised
residuals from the pileup (`y_j = (k_j − n_j p̂)/sqrt(n_j p̂(1−p̂))`, so evidence scales like
`sqrt(n)`), score `S = Σ w_j y_j / sqrt(Σ w_j²)` both orientations. Mean-centring is what makes
it an ABF1 detector rather than a nucleosome detector; `p̂` must be **local** (±500 bp) — with
`bg_params.pkl`'s 0.1383 it ranks true sites at 56.8%, worse than chance.

Result: at ±25 all 5 chrI sites land in the top 0.375% (permutation p = 0.025) but precision
is 0.58% at 5/5 recall — ~10× worse than the Murphy PWM alone. Controls are the finding: Reb1
scores 1.86% (22× worse) but **Rap1 scores 0.056%, better than ABF1** — both are
notch-in-NDR factors and the filter cannot separate them. So it detects "protected notch
inside an accessible region", not ABF1 identity. A retracted early claim ("a flat template
does as well") came from a ±100 window and un-whitened LLRs; at ±7 that correlation is 0.233,
not 0.986 — always compare whitened shape against whitened level. **The scan code was never
saved**; only `abf1_profile_pm100_agentA.{npz,png}` survive.

**Open issue this surfaced:** `inputs/bg_params.pkl` has p = 0.1383/0.1384, but the
genome-wide pooled rate is **0.0790** (Σn 689,036,863; Σk 54,428,639). 0.1383 looks like an
accessible-region fit. If it is mis-calibrated it biases **every** fiber-layer likelihood
ratio, not just this scan. Needs the user to confirm which segments went into that fit.

Also on disk from this era: the `sacCer3.fai` bug — 12 of 17 chromosomes were misindexed by
87,501 bytes (inherited from upstream). RoboCOP is unaffected because it reads with SeqIO;
only faidx users were.

---

## 9. Published artifacts

This is the one list of the user's published artifacts; keep it current when one is published.
Edit the source file and re-publish **to the same URL** (pass it as `url`), or a second artifact
is created instead of updating the first. Updated 2026-09-16.

**Run Browser (local site, 2026-09-22) replaces the per-window browsers for everyday use.** One static
site lists every decode run (run matrix: layers, mask, phi, EM, lambda, ABF1 width/pads, decoy,
trainDir, pkgvar, coverage, Fiber-seq check) and opens 1-N runs side by side on a chromosome (one
lane per run over shared m6A / depth / genes / reference sites). Code `analysis/viewer_site/`
(`build_site.sh` rebuilds everything, `serve.sh [PORT]` serves it); output
`/usr/project/xtmp/nd141/viewer_site/` (not in git, no size limit, opened through VS Code port
forwarding). Only the 4 windows in `viewer_site/windows.tsv` are extracted for now; data are stored at
whole-chromosome coordinates so more blocks slot in later. Every decode now also writes
`RUN_INFO.json` (provenance: driver, pkgvar tree, real trainDir, campaign/round/role) and
`factor_tables/part_*.npz` (per-factor optable) from `sbatch_genome_decode.sh` (non-fatal hooks,
backups `*.pre_viewer`). The Occupancy Browser artifacts below are frozen and will go stale.

**URL format changed.** Artifacts now live at `https://claude.ai/artifact/<short id>`. The old
`claude.ai/code/artifact/<uuid>` ids still identify the same pages (reading an artifact reports its
old uuid); the "old id" column maps between them. Paths below are relative to `analysis/` unless
they start with `presentation/`.

| artifact | URL | old id | built from |
|---|---|---|---|
| **RoboCOP Meets Fiber-seq** (the talk deck) | https://claude.ai/artifact/1vT1DuTfTbuYvFzTyjwMmy | — | `presentation/talk.html` (offline copy `talk_offline.html`); plan and change log `presentation/PLAN.md`; locus slides from `presentation/templates/locus_slide/` |
| Tuner Continuation Board (status + held-out F1 of the six campaigns after the 1.25× → 1.1× continuation) | https://claude.ai/artifact/MNuieUxLJJNMZMKXvaPQGV | — | scratchpad `tuning_status/tuning_status.html` (numbers from `conc_tuning/<run>/validation.tsv`, `presentation/continue_results.md`); rebuild by hand |
| Tuned Occupancy Browser (fw01 / sw01 / bw01 r0 + final, bt02/05/10 finals, u001 r7; 42 windows on chrXIV, chrII, chrIV) | https://claude.ai/artifact/55vxej7MWpqoQN7hfdfB6h | — | `presentation/tuned_occupancy_viewer_tempered.html` ← `presentation/build_layer_viewers.py tempered` (wraps `build_tuned_viewer.py`) |
| Same Weights, Three Layers (overnight job A: sw01 weights decoded seq / seq+fiber / fiber) | https://claude.ai/artifact/JigVragrGjB7MzDin1ydkM | — | `presentation/layer_same_weights_viewer.html` ← `presentation/build_layer_viewers.py sameweights` |
| ABF1 Layer Viewer (24 MacIsaac ABF1 sites, seq / fib / both, untuned + tuned) | https://claude.ai/artifact/4U6MnRVNN9idFMPEuF8grX | — | `presentation/abf1_layer_viewer.html` ← `make_posterior_viewer.py` with regions/runs in `/usr/project/xtmp/nd141/scratch_fiber_vs_seq2/` |
| RoboCOP Concentration Sheet (u001/m001 tuning, MacIsaac + Rossi validation) | https://claude.ai/artifact/XdB264uQyfr9b8F23B5sgy | `f7ff1c1b-5987-4ef8-9276-51c39f97acd6` | `conc_tuning/conc_sheet.html` ← `conc_tuning/make_conc_sheet.py u001=7 m001=7` |
| RoboCOP Tuning Ledger | https://claude.ai/artifact/MDihRrKrtKzaq7qwBhaKpP | — | scratch build, campaigns 01–06 |
| RoboCOP Occupancy Browser (chrI + chrXIV, untuned and EM runs) | https://claude.ai/artifact/RYgeitfcR21v7zwY8XpTmN | `c6c7d1f3-d62a-4858-a38a-4ee1c7891e0d` | `posterior_viewer_all.html` ← `make_posterior_viewer.py` |
| chrI Occupancy Browser (ERV46) | https://claude.ai/artifact/5XtKJ7bVN4ueAV2fFf29pV | `24b47df3-9f5f-4372-b6ec-d4b1976c6f2a` | `posterior_viewer_erv46.html` |
| chrXIV Occupancy Browser (186–191 kb) | https://claude.ai/artifact/LjeWVUHdWVaHWu9CD3GR1X | `9fd1fd00-ad6d-4d85-9e92-177d30888836` | `posterior_viewer_chrXIV_187k.html` |
| chrXIV Occupancy Browser (55.5–60.5 kb) — same title as the one above | https://claude.ai/artifact/54ic75fzu4rx7uXiFxNKdR | `20e96d14-e171-4170-a11b-f32eb7711680` | `posterior_viewer_chrXIV_58k.html` |
| Where the Twelve Bind | https://claude.ai/artifact/GE7SMg3VaPRrdivjZC4yxy | `7b4db749-5827-40b4-80a3-854fbb56a6b6` | `rossi_genic/where_the_twelve_bind.html` |
| ORF Versus Gene Body | https://claude.ai/artifact/61h2XiNhJC7qvYseNd2BSr | `28965c44-fe42-4433-b918-c42a5fe5550b` | `orf_vs_gene_body.html` |
| Factor Detection on chrXIV | https://claude.ai/artifact/PRygmQbvc3r26RkWRQrvZk | `b5a5d5b2-3ba0-4260-b63c-ce74115e33b7` | `chrXIV_factor_chart.html` ← `make_factor_chart.py` |
| Odd One Out (motif audit) | https://claude.ai/artifact/GMtQu62hERnEcaX8R3RL5Z | `7c63a39b-8e04-4cd1-9660-7797d0dec154` | `motif_distance_sheet.html` |
| Fiber-seq HSMM — model lineage | https://claude.ai/artifact/2z2SQpL7psukePJyw15xC9 | — | not recorded |

The chrXIV-browser mapping was checked by reading each page's region. The other old→new pairs are
matched by title.

**Known stale content.** `ORF Versus Gene Body`'s decision chain describes the abandoned four-class
scheme (§5.2); its Figure 1 (ORF/gene-body/TSS anatomy) is still correct and is the reason to keep
it. `presentation/tuned_occupancy_viewer.html` is the superseded 9.4 MB local build; the live Tuned
Occupancy Browser matches `tuned_occupancy_viewer_tempered.html`. When adding runs to the untuned
Occupancy Browser, keep `viewer_runs_chrI.tsv` and `viewer_runs_chrXIV.tsv` label sets
**identical**, since the label is the join key across regions.

---

## 10. Standing constraints

Also listed in `CLAUDE.md`:

- **Never overwrite** `inputs/all_TFs_1000pealVal_params_pseudo.pkl`, `inputs/bg_params.pkl`,
  `inputs/motifs_meme.txt`. New variants get new filenames.
- **Do not modify** `robocop_em.py`'s line-162 tmpDir cleanup or its `iterations = 0`.
- **Do not modify any existing `pkgvar/*` tree** — create a new one.
- MacIsaac bed used **exactly as shipped**: no offset correction, no strand flip. The 1 bp
  ABF1 phase difference against Murphy is a motif-definition difference, not an error.
- **Commit and push only when explicitly asked.**
- State every parameter a run changes besides the one under test, as a decision, before
  running it.

---

## 11. Repo state

Committed through `918effb` ("Add widened-footprint runs, concentration sweep, and the Rossi
genic/intergenic target"), which included `analysis/pkgvar/` (28 frozen variants; the 40 KB
`librobocop.so` copies are tracked and all byte-identical, md5
`6a0724bf6ef7b8bc4313927b931ac685`), the widened meme files and params pkls, decode-directory
metadata (`config.ini`, `coords.tsv`, `pwm.p`, with `tmpDir/` and `HMMconfig*.pkl` gitignored)
and all of `analysis/rossi_genic/`.

**Uncommitted, from the concentration work (2026-09-10 → 09-12):** everything in §1.3 —
`tune_concentrations.py` and its `--run`/`--chain`/`next` machinery,
`make_conc_targets.py`'s site loader, `count_calls.py`'s MacIsaac sidecar,
`make_conc_trainDir.py --hold-nucleosome`, `tuning_trajectory.py`, `sbatch_tune_next.sh`,
`pkgvar/seq_maskoff_macisaac/`, `run_split_revfix_seq_maskoff_macisaac.py`, the
`conc_tuning/{u001,m001}/` state and reports, `conc_tuning/conc_sheet*.html` +
`make_conc_sheet.py`, and this rewrite of `HANDOFF.md` + the new `CLAUDE.md`. Nothing under
`pkg/` has been modified. `git status` also carries a large tail of older untracked analysis
scripts and generated figures; `motifdb/` (~60 MB) and the `rossi_locus_class*/` outputs of the
abandoned four-class scheme are safe to delete rather than commit.
