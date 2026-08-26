# RoboCOP + Fiber-seq — Session Handoff

Pick-up notes for the next agent. Read `whattodo.md` (strategy + open work) and
`FIBERSEQ_CHANGES.md` (full code diff vs upstream) first — this file adds the **scoring +
plotting tooling** and the **layer / mask toggles** that are easy to get wrong.

**Start with §0** — the most recent work (the shipped PWM collection was audited against
JASPAR and Rossi; ABF1's matrix is demonstrably wrong and six others are suspect). **§8 is
the repo state: none of it is committed.** §6 covers a self-contained side experiment
(sliding the fitted ABF1 fiber footprint across the genome), finished as an exploration but
whose code was never saved.

Environment:
```bash
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
```
All decode outputs (`robocop_chrI_*`, `robocop_seqlayer_*`, `robocop_erv46_*`,
`robocop_train_fiberonly/`) live on THIS filesystem, not in git (they are 20–210 MB each and
git-ignored). A new session on this machine already has them.

---

## 0. Most recent work (2026-08-14 → 08-18) — the motif-source audit

**Nothing in `whattodo.md`'s open-work list was touched.** These four days went into a
question that list did not contain: *is the PWM collection RoboCOP decodes with actually
right?* The answer is "not everywhere", and that matters because the sequence layer
(§2 layer 0) is the next phase of the plan — turning it on with a wrong matrix makes
things worse, not better. Everything below is **read-only diagnosis**;
`inputs/motifs_meme.txt` has NOT been changed.

### 0.1 The finding that started it — ABF1 Murphy vs JASPAR

`inputs/motifs_meme.txt`'s `Abf1_murphy` (w=14) and JASPAR MA0265.3-rc (w=14) are the
same motif in their core and **anti-correlated in their middle**:

| columns | what they are | column r (Murphy vs JASPAR) | P(GC) Murphy / JASPAR / genome |
|---|---|---|---|
| 0–4, 10–13 | the two half-sites | **+0.998** (9 cols, all ≥ +0.98) | — |
| 5–9 | the spacer | **−0.411** | **0.678** / 0.300 / 0.381 |

Murphy's spacer is a **GC-rich, informative** block (per-column KL 0.16–0.75 against the
yeast background). JASPAR's is nearly flat, as a spacer should be. Real ABF1 sites do not
carry that GC block, so Murphy charges them for it. Per-column numbers:
`analysis/abf1_column_distances.{py,tsv,png}` and `analysis/abf1_murphy_vs_jaspar_pwm.txt`.

**Why it is invisible to TOMTOM.** TOMTOM's q-value asks "closer than two random
matrices", which saturates for any pair sharing a strong core — Murphy-vs-JASPAR ABF1 is
a confident match by q. FIMO asks a different question: does this 14-mer clear a
log-odds threshold, **summed over all 14 columns**. The spacer is 5 of those 14.
`analysis/explain_abf1_score_split.py` → `abf1_score_split.tsv` decomposes both scores on
the 5 MacIsaac chrI ABF1 sites. Both matrices score the core within 1.3 bits of each other (max gap at site 2);
the spacer is where they part:

| site | spacer contribution Murphy | spacer JASPAR | FIMO p Murphy | FIMO p JASPAR | passes p<1e-4 |
|---|---|---|---|---|---|
| 1 | +3.50 | +0.24 | 3.6e-06 | 2.9e-05 | both |
| 2 | −2.15 | +1.36 | 3.1e-04 | 2.1e-05 | **JASPAR only** |
| 3 | −7.87 | +2.97 | 3.9e-04 | 6.5e-09 | **JASPAR only** |
| 4 | −0.74 | −0.06 | 1.6e-05 | 1.6e-05 | both |
| 5 | −4.87 | +2.67 | 3.1e-04 | 1.8e-06 | **JASPAR only** |

**As a pure sequence scan, Murphy recovers 2 of 5 and JASPAR 5 of 5.** Every miss is a
spacer penalty, not a core mismatch. (Inside the full RoboCOP decode the gain is smaller —
3 of 5, not 5 of 5. See §0.2 for why.) This is the general lesson: **a confident TOMTOM match can still lose
sites, so never validate a PWM swap on q-values alone — score real sites.**

Supporting figures (all regenerate-able): `abf1_murphy_vs_jaspar_logos.png`,
`abf1_motif_anatomy*.png`, `abf1_score_split.png`, `abf1_column_distances.png`,
`macisaac_vs_user_fimo.png`, `chrI190559_decoded_fimo.png`. FIMO scans in
`analysis/chrI_fimo/` (whole chrI) and `analysis/chrI190k_fimo/`, driven by
`compare_fimo_macisaac_chrI.py` — note it scores against `motifdb/`-style backgrounds
(`robocop_bg.txt` = the same yeast background RoboCOP computes, not MEME's uniform).

### 0.2 The RoboCOP runs that tested the swap

`inputs/jaspar_abf1_motifs_meme.txt` is a **copy of `inputs/motifs_meme.txt` with exactly
one motif replaced**: JASPAR MA0265.3, reverse-complemented into Murphy's orientation.
The other 152 are byte-identical, and the **motif ID is deliberately still
`Abf1_murphy`** — `update_data_emission_matrix_using_binomial_fiber_seq` in `pkg/robocop/robocop.py`
looks the Rossi-fitted Fiber-seq p-vector up by that exact string (`loaded_params['p'][tf_name]`), and the 14-column register has to keep applying column-for-column. Do not
"tidy" that ID.

`analysis/config_jaspar.ini` differs from `config_fiberonly.ini` on **one line**:
```
pwmFile = inputs/jaspar_abf1_motifs_meme.txt      # vs inputs/motifs_meme.txt
```
trainDir `robocop_train_jaspar/`. Drivers: `sbatch_train_jaspar.sh`,
`sbatch_jaspar_seq_maskon.sh`, `sbatch_jaspar_seq_maskoff.sh`,
`sbatch_jaspar_seqonly_maskon.sh`.

The matched pairs on disk (same coords, same layers, same mask — only the ABF1 matrix
differs):

| layers / mask | Murphy | JASPAR |
|---|---|---|
| Fiber+seq, ABF1-only | `robocop_chrI_seq_maskon_revfix` | `robocop_chrI_seq_maskon_JASPAR` |
| Fiber+seq, all TFs | `robocop_chrI_seq_maskoff_revfix` | `robocop_chrI_seq_maskoff_JASPAR` |
| seq only, ABF1-only | `robocop_chrI_seqonly_maskon_revfix` | `robocop_chrI_seqonly_maskon_JASPAR` |

**What the swap actually buys in a decode — re-measured 2026-08-18 with
`score_robocop.py` on whole chrI; raw JSON in `analysis/jaspar_vs_murphy_chrI_metrics.json`:**

| run | ABF1 recall | predicted ABF1 peaks | mean posterior at the 5 sites | ABF1 enrichment | Chereji nuc recall | median dyad err |
|---|---|---|---|---|---|---|
| Murphy, Fiber+seq, ABF1-only | 2/5 | 190 | 0.138 | 15.2× | 0.785 | 7.0 bp |
| **JASPAR**, Fiber+seq, ABF1-only | **3/5** | 256 | **0.204** | 16.3× | 0.793 | 7.0 bp |
| Murphy, Fiber+seq, all TFs | 2/5 | 83 | 0.135 | 33.6× | 0.800 | 6.5 bp |
| **JASPAR**, Fiber+seq, all TFs | **3/5** | 152 | **0.202** | 29.0× | 0.807 | 7.0 bp |

**Read this carefully — the decode gains far less than the FIMO scan does.** On sequence
alone JASPAR goes 2/5 → **5/5** (§0.1). Inside the full model it goes 2/5 → **3/5**, and
`mean_post_at_sites` only rises 0.138 → 0.204. Nucleosome architecture is untouched, as
expected (the ABF1 matrix is one of 153).

The gap between 5/5 and 3/5 is the **Fiber layer, not the PWM**. The fitted ABF1 p-vector
runs 0.029–0.137 against a background of 0.138 — every position below background — so it is
a generic protection detector with almost no positional specificity, and with raw coverage
`n` untempered it produces likelihood ratios of 1e9–1e10 at spots 8–91 bp off-motif that no
sequence term can overturn. **Fixing the fiber emission outranks any further PWM change**;
see `whattodo.md` Tier 1 #3. Also note the JASPAR runs put *more* mass on ABF1 overall
(83 → 152 predicted peaks in the all-TF run), so enrichment dips slightly even as recall and
site posterior improve — judge a swap on posterior-at-sites and recall together, not on
enrichment alone.

### 0.3 Is ABF1 the only one? — the three-way sheet

One motif found by hand says nothing about the other 152. `inputs/motifs_meme.txt` is
**not "Murphy"** — it is 153 matrices across mixed sources (64 `_zhu`, 60 `_badis`,
26 `_murphy`, plus `Rap1_telomeric` / `_motif1` / `_motif2`), so any answer has to break
down by source.

Three databases, assembled by `analysis/build_motif_dbs.py` into `analysis/motifdb/`:

| file | what | n |
|---|---|---|
| `shipped.meme` | copy of `inputs/motifs_meme.txt` — what RoboCOP decodes with | 153 |
| `jaspar.meme` | JASPAR CORE, `tax_group=fungi`, `version=latest`, via the API | 193 |
| `rossi.meme` | Rossi ChExMix pre-computed MEME motifs, E-filtered, renamed | 856 |
| `yeast_bg.txt` | genome background via `getDBFconc.computeBackground` — **the background the model assumes**, not MEME's uniform 0.25 | — |
| `coverage.tsv` | per shipped **motif**: which sources exist, how many matrices each | 153 |

Rossi sample→TF mapping needs no external sheet: `inputs/rossi_peak_w_strand_all_TFs.bed`
already carries `sample_id` + `TF`. Coverage came out **3-way 69 / 2-way-JASPAR 70 /
2-way-Rossi 2 / nothing-to-compare 12** = 153. NHP6A is a real "no JASPAR at all" case —
an HMG-box architectural protein, so RoboCOP is scanning for a motif JASPAR does not
think exists.

**Independence caveat.** Rossi and JASPAR are independent of the *shipped PWMs* (which are
Murphy 2011 / Zhu / Badis lineages), which is what makes them usable as a check here. They are
NOT independent of everything else in this repo: the fitted Fiber-seq footprints in
`inputs/all_TFs_1000pealVal_params*.pkl` were trained on Rossi peak locations, and MacIsaac is
77.5% inside Rossi (§6). So "Rossi agrees with JASPAR against native" is evidence about the
matrix; it is not independent evidence about the fiber layer.

`analysis/compare_motif_sources.py` runs TOMTOM six ways (three pairs × `-dist ed` and
`-dist kullback`, `-bfile yeast_bg.txt`, `-thresh 1 -evalue` so weak same-name pairs are
still reported) and writes `motif_comparison.tsv` + `motif_comparison_report.txt`. It also
correlates each pair's **log-odds track along real chrI sequence** (`r_track`) — the only
metric tied to what the model actually does. Two matrices can be formally similar and
still place mass differently; `Gal4_zhu` is q_ed 3e-07 with r_track 0.56.

### 0.4 The published sheet — native vs JASPAR vs Rossi

**Artifact: <https://claude.ai/code/artifact/7c63a39b-8e04-4cd1-9660-7797d0dec154>
("Odd One Out").** Rebuild with:
```bash
python motif_distance_sheet.py      # -> .tsv + .json (225 rows, incl. aligned logo PWMs)
python make_motif_sheet_page.py     # -> motif_distance_sheet.html from motif_sheet_template.html
```
then re-publish **to that same URL** (pass it as `url` to the Artifact tool) — do not
create a second one.

Design decisions worth not re-litigating:

- **No privileged reference.** Rossi and JASPAR are not ground truth, they are two more
  measurements. With three points there is always a closest pair; the **odd one out** is
  the member opposite it. That is the whole verdict logic.
- **Raw distances, no p-values.** TOMTOM's own column functions
  (`ED = sqrt(Σ(X−Y)²)`, `KLD = ½(Σ X ln(X/Y) + Σ Y ln(Y/X))`) reported as a **mean per
  aligned column** after `-motif-pseudo 0.1` spread by the yeast background. The p-value
  saturates and is exactly what hid the ABF1 spacer.
- **One row per Rossi replicate.** Rossi runs each factor as several independent ChExMix
  samples. Collapsing them to a "best" match threw away the replication, which is the
  most valuable thing in the data. Within a sample the top motif is taken when it leads
  the runner-up by ≥ 2 orders of magnitude in E (87% of samples clear that —
  `rossi_evalue_structure.py`); otherwise the runner-up gets its own row.
  `rossi_pick_sensitivity.py` shows why "pick the Rossi motif nearest native" is a biased
  rule: it shrinks N↔R by construction *and* lets native choose the matrix for the J↔R
  leg, the one leg native is supposed to have no say in.
- **E-values are not comparable between factors.** log10 E correlates with log10 nsites at
  r = −0.54, so any absolute cutoff is meaningless across factors; the tie-break is
  relative, within one run.
- **Heat shock is labelled and excluded from consensus.** The `<id>_YEP` directory names
  are a Yeast Epigenome Project tag, not a growth medium. Real conditions come from GEO
  GSE147927 via `fetch_rossi_conditions.py` → `inputs/rossi_sample_conditions.tsv`;
  21 samples on this disk are 37 °C heat shock (3 or 6 min).
- **Power is on every row.** The median surviving Rossi motif rests on **28 sites** and
  10% on fewer than 10. A large distance from a 9-site motif is noise.

**The width problem, and how it is solved (`common_window`).** The three matrices are
rarely the same width — RAP1 is native 20 / JASPAR 12 / Rossi 16 — so aligning each pair
independently leaves the three numbers in a row resting on different column counts. They
are then not commensurable, and "which pair is closest" can be decided by how many columns
each pair happened to share. **TOMTOM cannot fix this: it is strictly one query motif vs
target motifs, no three-motif mode, and no multiple-motif alignment exists anywhere in
MEME 5.5.9.** Its complete-scoring is also *asymmetric* — the distance depends on which
motif you call the query — so it cannot be used symmetrically. Two stages instead:

1. **Frame.** Search every offset and orientation against a **fixed** W-column window
   (W = width of the shortest motif), columns a motif cannot reach priced against the
   genome background, minimising the **sum of the three pairwise mean KLs**
   (sum-of-pairs multiple alignment). Fixing the width is what stops the window being
   minimised by shrinking onto whichever columns happen to match.
2. **Numbers.** Report all three pairs over the **intersection** of those placements —
   the columns every matrix actually reaches. One column set, all three pairs.

"Align each pair the way TOMTOM would, then intersect" is the obvious approach and this
**reduces to it wherever it is well defined** — but alignment is not transitive, so the
three pairwise optima are mutually realisable in one frame in only **82 of 140 rows
(59%)**; `pairwise_frame_consistency.py` measures this. On those 82 rows the two schemes
agree on **78** verdicts and all 4 disagreements favour the intersection. The grey band on
each logo is exactly the scored columns; a `−n` tag marks columns of the shortest matrix
dropped because another matrix does not reach them.

Current window statistics (213 rows with a window): scored columns median **7**, min 4,
max 15, **under 8 in 120 rows** — the honest limitation, and it is flagged amber on every
row. Dropped columns: 0 in 169 rows, 1 in 36, 2 in 8. `window_cost_kl` (how much worse the
worst pair is under the joint frame than under its own free alignment): median 0.0000,
p75 0.0019, p90 0.0634, max 0.5041, above 0.05 in 25 rows.

**Thresholds** `CLOSE_KL = 0.060` and `MARGIN_KL = 0.030` were re-checked, not assumed:
the closest-of-three distance is median 0.032 / p75 0.074, putting 0.060 at the 69th
percentile. `margin_kl` = second-smallest minus smallest of the three pairwise KLs; below
`MARGIN_KL` the row is `ambiguous`.

**Results, per native motif (153 total, gene-level consensus over replicates):**

| consensus | n |
|---|---|
| no Rossi motif at all (blank) | 82 |
| `rossi_is_odd` | 29 |
| `no_consensus` | 22 |
| `all_three_agree` | 9 |
| **`native_is_odd`** | **7** |
| `ambiguous` | 2 |
| `two_datasets_only` | 2 |

Median distance to the other two: native **0.167**, JASPAR **0.150**, Rossi **0.175** —
i.e. the shipped collection is not systematically the outlier; the problem is
motif-by-motif.

**The 7 `native_is_odd` motifs — the shortlist of matrices to consider replacing:**

| motif | Rossi replicates | reached a verdict | called native odd | unanimous |
|---|---|---|---|---|
| `Abf1_murphy` | 3 | 3 | 3 | **yes** |
| `Rap1_motif2` | 3 | 3 | 3 | **yes** |
| `Rap1_telomeric` | 3 | 3 | 3 | **yes** |
| `Rap1_zhu` | 3 | 3 | 3 | **yes** |
| `Pdr1_badis` | 2 | 2 | 2 | **yes** |
| `Rap1_motif1` | 3 | 2 | 2 | no |
| `Cad1_murphy` | 2 | 1 | 1 | no |

("reached a verdict" excludes replicates that came out `no_consensus` or `ambiguous`;
none of these seven had a heat-shock replicate.)

**ABF1 came out of the audit unanimously wrong in all three replicates** — the hand
finding of §0.1 reproduced blind, with JASPAR and Rossi agreeing with each other
(KL 0.016–0.019) and native 0.17–0.22 away from both. **`Rap1_telomeric` is on this list
and is one of the 12 TFs with a fitted Fiber-seq footprint** (list at the end of
`whattodo.md`), so it
is the next most consequential matrix after ABF1.

### 0.5 `analysis/pkgvar/` — layer and mask state is no longer hand-commented

**This supersedes the "comment the lines IN, submit the sbatch, wait until RUNNING,
re-comment" mechanics described in §2 and §3.** That dance was a race: with a slurm array
you cannot know when a task actually imports the file. Instead `analysis/pkgvar/<variant>/`
holds a **frozen copy of the whole `robocop` package** with the toggles already applied,
and each driver does

```python
sys.path.insert(0, 'pkgvar/seq_maskon/')     # run_split_revfix_seq_maskon.py
from run_robocop import run_robocop_without_em
```

so every task of every array imports exactly the state it is meant to, and **all variants
can run concurrently**. `robocop.py` is byte-identical across variants; the only differences
are in `utils/robocopExtras.py`:

| variant | line 101 (sequence layer) | lines 113+ (ABF1 hard mask) |
|---|---|---|
| `fiber_maskon` | `[0][:] = 1` — sequence OFF | mask on layers 5/6 |
| `fiber_maskoff` | `[0][:] = 1` — sequence OFF | commented out |
| `seq_maskon` | commented out — sequence **ON** | mask on layers 5/6 |
| `seq_maskoff` | commented out — sequence **ON** | commented out |
| `seqonly_maskon` | sequence **ON** | layers 5/6 set to **1** (fiber off) and the **mask moved onto layer 0** |
| `seq_maskoff_{12tfs,bgtss,lowabf1}` | sequence ON, mask off | differ only in which `inputs/*.pkl` they load |

`seqonly_maskon` is the one to read carefully: because layers 5 and 6 are neutralised there,
a mask placed on them would be silently wiped and the run would quietly become a full
154-TF decode — so the mask rides on layer 0, the only live layer. Same semantics (a 0 in
any channel zeroes the product).

**The gotcha that made the JASPAR retrain necessary:** a decode **never re-reads the meme
file**. `run_robocop_without_em` takes `pwm_emission` / `tf_prob` / `transition_matrix`
straight out of the trainDir's `HMMconfig.pkl` (`robocop_no_em.py:51`). Changing `pwmFile`
in a config does nothing unless you **retrain** — which is why there is a separate
`robocop_train_jaspar/` built by `sbatch_train_jaspar.sh`.

### 0.6 Where to pick this up

1. **Nothing is committed** — see §8. Commit before building on it.
2. Decide whether to act on the shortlist. The ABF1 evidence is complete enough to swap
   (FIMO 5/5 vs 2/5 on chrI, decode 3/5 vs 2/5 with site posterior 0.204 vs 0.138,
   unanimous across all three Rossi replicates, mechanism understood) — but note §0.2:
   the decode is fiber-limited, so a PWM swap alone will not reach 5/5;
   the four RAP1 matrices and `Pdr1_badis` have sheet evidence but **no FIMO / decode
   validation yet**. Do that first — §0.1 is the cautionary tale about trusting a
   similarity score without scoring real sites.
3. A swap means editing `inputs/motifs_meme.txt`, which changes **every** decode. Keep the
   `jaspar_abf1_motifs_meme.txt` pattern: one alternate PWM file + one alternate config,
   never an in-place edit, so the two are comparable.
4. If a matrix with a **fitted Fiber-seq footprint** is swapped (`Rap1_telomeric`,
   `Abf1_murphy`), the footprint is keyed on the motif ID **and registered to the motif's
   column frame**. Keep the ID and the width/orientation, or the p-vector silently
   mis-registers. This is exactly why the JASPAR ABF1 file keeps the name `Abf1_murphy`
   and is reverse-complemented into Murphy's orientation.
5. The sheet's real limitation is the 120 rows scored on fewer than 8 columns — many
   native matrices are only 5–8 wide. Those verdicts are weak; do not act on a
   short-window row without a FIMO check.

---

## 1. The two tools to keep using

### `analysis/score_robocop.py` — the scorer (single source of truth)
Quantitatively scores a decode against yeast ground truth: Chereji +1/−1 nucleosome dyads,
MacIsaac ABF1 sites, phasing period, MNase accessibility. Loads the sparse per-segment
posterior out of the decode's `tmpDir/info.h5`, collapses HMM states → factor tracks, and
matches predicted peaks to reference within a tolerance.

Public API you will actually call:
- `score(outDir, regions=None, tol_nuc=20, tol_abf1=20, abf1_global_max=None, return_abf1_tracks=False)`
  → dict of metrics + `_per_region` list. `regions=[(chrm,lo,hi)]` scores just a window.
- **ABF1 peak-caller = one code path**: `_above_threshold_runs(track, height)` finds contiguous
  above-threshold runs; `footprint_centers()` and `call_abf1(track, pos, threshold)` both derive
  from it. `call_abf1` returns `[{center,start,end}]` in genomic coords. **A predicted ABF1
  anchor = the CENTER (midpoint) of an above-threshold run**, NOT an argmax — this is robust to
  the flat, saturated posterior plateaus the sequence layer produces (argmax there jitters with
  the window; the midpoint does not).
- Threshold: `abf1_call_threshold(gmax) = max(0.10, 0.30 * gmax)`. When scoring a small window
  pass `abf1_global_max=<whole-chrI max>` so the window uses the run's REAL global threshold
  (~0.30 for all 4 chrI runs, since each run's chrI ABF1 max ≈ 1.0). Cache in
  `analysis/site4_thresholds.json` / `abf1_thresholds.json`.
- Nucleosome dyads still use `call_peaks` (find_peaks) — unchanged.

### `analysis/plot_abf1_locus.py` — scorer-driven per-locus plot (reusable)
Renders one panel per model showing the ABF1 posterior around a locus across several decodes.
**It does NOT re-implement peak calling — it INVOKES `score(..., return_abf1_tracks=True)` and
draws exactly what the scorer returns** (track, threshold line, and a ▼ arrow at each of the
scorer's `call_abf1` centers). So if you change the peak-caller or threshold inside
`score_robocop.py`, this plot moves with it automatically — single source of truth.
```bash
python plot_abf1_locus.py                                     # MacIsaac site #4, 4 chrI runs
python plot_abf1_locus.py --motif chrI:45318-45332 --out siteN.png   # any locus / models
```
`DEFAULT_MODELS` = the 4 chrI runs (fiber_abf1, fiber_all, seq_abf1, seq_all). For a new run not
in the threshold cache, it scores that run whole-chrI once to record its global ABF1 max, then
every later locus renders instantly.

Companion deliverables (regenerate-able, kept): `plot_abf1_grid.py` +
`abf1_sites_per_run.py` → `abf1_recovery_grid.png` (5 MacIsaac sites × 4 models);
`make_5run_chart.py` + `score_5runs.py` → `chrI_5run_recall.png`.

---

## 2. Layer toggles — sequence / MNase / Fiber-seq

The emission tensor is **7 layers**, built in `robocop.py:_build_data_emission_matrix` as
`np.ones((7, n_obs, n_states))` (all neutral), then each ACTIVE layer is multiplied in
(emission = product over layers, so a layer left at 1.0 contributes nothing = OFF):

| Layer | Channel | State (default) | How to toggle |
|-------|---------|-----------------|----------------|
| 0 | Sequence (PWM) | **OFF** | `robocopExtras.py:101` `data_emission_matrix[0][:] = 1` forces it neutral. **Comment out line 101 to turn sequence ON.** (This is the only difference between the `*_maskon`/`*_maskoff` fiber runs and the `*_seq_*` runs — the config.ini files are identical.) |
| 1–2 | MNase short/long | **OFF** | the `update_..._negative_binomial` calls are commented out in the fiber path (`robocop.py:565-568`); layers stay 1.0. |
| 3–4 | ATAC short/long | **OFF** | never filled. |
| 5–6 | Fiber-seq Watson/Crick | **ON** | filled by `update_data_emission_matrix_using_binomial_fiber_seq` (`robocop.py:676-680`); zeros floored to 1e-30 in `robocopExtras.py:105-106` to avoid NaN. |

**Phased strategy (user directive):** get Fiber-seq alone biologically correct FIRST, then add
sequence, then MNase. Sequence/MNase being off is intentional, not a bug.

**Do not edit these lines by hand any more — see §0.5.** Every combination already exists as
a frozen package copy under `analysis/pkgvar/`, selected by the driver's `sys.path.insert`.

---

## 3. ABF1-only decoding — the hard-forbid mask (apply AFTER the 1e-30 floor)

To force a TRUE ABF1-only decode (absolutely nothing but ABF1 + background + nucleosomes gets
any posterior), use the **hard mask in `robocopExtras.py:113-114`** (commented out by default):
```python
data_emission_matrix[5][:, 29:dshared['nuc_start']] = 0   # Fiber Watson
data_emission_matrix[6][:, 29:dshared['nuc_start']] = 0   # Fiber Crick
```
**Why it must run AFTER the 1e-30 floor (lines 105-106), not before:** the floor turns every 0
into 1e-30 so no position is all-zero. If the mask ran before the floor, the forbidden states
would be lifted back to 1e-30 (tiny but nonzero → they can still leak posterior). Running the
mask AFTER the floor leaves the non-ABF1 TF states (indices `29 .. nuc_start`, incl. `unknown`)
**EXACTLY 0**. Emission is a product over channels, so a 0 in the Fiber channels ⇒ total
emission 0 ⇒ posterior exactly 0 for every TF except ABF1 (states 1..28). Background (0) and
nucleosomes (`nuc_start..`) keep the 1e-30 floor, so no column goes all-zero (no NaN).

This `robocopExtras.py` mask **supersedes the older `robocop.py:798` mask** (which set =0 inside
the binomial function, before the floor, and could leak). Prefer the robocopExtras one.

State layout for the slice: `0` = background; `1..28` = ABF1 fwd+rev; `29 .. nuc_start-1` = all
other TFs + `unknown`; `nuc_start..` = nucleosomes.

Mechanics: **the comment-IN / submit / re-comment dance is obsolete — use `analysis/pkgvar/`
(§0.5).** Pick the variant that already has the mask baked in (`*_maskon`) and run its driver;
nothing is edited and variants can run concurrently. The older single-run path
(`run_fiberonly_noem.py` + `sbatch_noem.sh`, trainDir `robocop_train_fiberonly`) still works if
you edit lines 113-114 by hand. Decodes use `run_robocop_without_em` (keeps `tmpDir/info.h5`;
equivalent to `with_em` iter=0, which otherwise DELETES tmpDir).

---

## 4. TF colors are consistent across decodings

`colorMap()` in both `plotRoboCOP.py` and `plotRoboCOPax.py` assigns each DBF a **deterministic
color keyed on its NAME** via `_color_for_name(name)` (not on `set()` ordering or the process
hash seed). So ABF1 is the same color in every plot and every output folder. The map is cached
per run as `<outDir>/dbf_color_map.pkl` and reused if present. **If you ever see a TF change
color between two decodes, delete that folder's `dbf_color_map.pkl` and re-plot** — a stale
pickle from before this fix is the only way colors drift. `nucleosome` = grey `0.7`,
`unknown` = light grey `#D3D3D3`.

---

## 5. The chrI runs on disk

The original four (scored and plotted throughout §1–§4):

| outDir | layers | mask | plot color |
|--------|--------|------|------------|
| `robocop_chrI_maskon`      | Fiber only | ABF1-only | blue |
| `robocop_chrI_maskoff`     | Fiber only | all TFs   | blue |
| `robocop_chrI_seq_maskon`  | Fiber+seq  | ABF1-only | orange |
| `robocop_chrI_seq_maskoff` | Fiber+seq  | all TFs   | orange |

**These four predate the reverse-strand fiber-parameter fix (commit `90b05c3`).** The
post-fix reruns carry a `_revfix` suffix — `robocop_chrI_maskon_revfix`,
`robocop_chrI_maskoff_revfix`, `robocop_chrI_seq_maskon_revfix`,
`robocop_chrI_seq_maskoff_revfix`, plus `robocop_chrI_seqonly_maskon_revfix` — and are the
ones to compare against. Prefer `_revfix` for anything new; the un-suffixed four are kept only
because the numbers below and in `abf1_thresholds.json` refer to them.

Also on disk: the three `*_JASPAR` runs (§0.2), and three fiber-parameter variants
`robocop_chrI_seq_maskoff_{12tfs,bgtss,lowabf1}` driven by `run_split_variant_*.py`.

Latest ABF1 site-#4 result (from `plot_abf1_locus.py`, midpoint 62664): fiber_abf1 → 62663
(1 bp ✓), fiber_all → MISSED (local max 0.147 < 0.30), seq_abf1 → 62665 (1 bp ✓), seq_all →
62665 (1 bp ✓). Matches `abf1_sites_per_run_centered.txt`.

## 6. Side experiment — sliding the fitted ABF1 fiber footprint across the genome

**Question asked:** is the Fiber-seq-estimated ABF1 methylation profile, on its own, enough to
*find* ABF1 in the genome — independent of the HMM and of the sequence layer? Run as an
exploratory side branch (agent "agentA"), read-only, chrI only. **Answer: no, but it finds
something real.** Details below; the scan code itself was NOT saved (see "What is on disk").

### 6.1 What it computes

Not autocorrelation — a **cross-correlation / matched filter**. (In this repo "autocorrelation"
already means the nucleosome-phasing-period metric in `score_robocop.py`; don't overload it.)

Two vectors of equal length are formed per candidate window and dot-producted.

*Template side* (from `inputs/all_TFs_1000pealVal_params_pseudo.pkl`), computed once:
```
w_j = p_j - mean(p)          # mean-centered => SHAPE only, level discarded
```
For the shipped 14-column `p['Abf1_murphy']['watson_signal']['A']`, `mean(p) = 0.0843` and
`sqrt(sum w_j^2) = 0.1314`. Columns 3 and 13 (deep notch) and 14 (high rim) carry ~half of
`sum w_j^2`; column 7 (`p=0.0872`, sits on the mean) contributes essentially nothing.

*Genome side*, per position, from the modkit pileup (`k` = methylated calls, `n` = A trials):
```
y_j = (k_j - n_j*p_hat) / sqrt(n_j * p_hat * (1 - p_hat))
```
i.e. a variance-stabilised residual — observed minus expected, in units of its own standard
error. This is where **coverage weighting** comes from for free: evidence scales like `sqrt(n)`,
so 0/100 is a loud protection signal while 0/3 is barely anything.

*Score:*
```
S = sum_j (w_j * y_j) / sqrt(sum_j w_j^2)
```
Each column casts a signed vote (template says protected + genome is protected => positive).
Slide 1 bp, repeat, score **both orientations** (mirror `[::-1]` AND Watson<->Crick channel
swap — see the reverse-strand fix in commit 90b05c3) and keep the better one.

Why each normalisation matters:
- **mean-centering `w`** is what makes this an ABF1 detector rather than a nucleosome detector.
  Without it every protected patch scores high; with it, a window must also have the *elevated
  rim*, not just the notch.
- **dividing by `sqrt(sum w_j^2)`** cancels the template's arbitrary scale, puts `S` on a unit-
  variance z-scale under the null, and makes different widths / different TFs comparable.
- the data side is deliberately **not** normalised (so this is a matched filter, not Pearson
  `r`) — amplitude should count, a 300-read window tracing the shape must beat a 3-read one.
- `p_hat` is a **local** background from +/-500 bp, NOT `inputs/bg_params.pkl`. See 6.4.

### 6.2 Results (chrI, ~230k windows, template refit leave-one-chromosome-out)

| width | median true-site percentile | worst true site |
|-------|-----------------------------|-----------------|
| +/-7 (14 bp — the width the emission layer actually uses) | 0.401% | 2.139% |
| +/-25 (51 bp) | **0.086%** | **0.375%** |

+/-10 (21 bp) and intermediate widths were **never run**. Widths were judged on *worst-case*
site rank, not median. At +/-25: all 5 MacIsaac chrI ABF1 sites in the top 0.375%, permutation
null `p = 0.025`, top-50 hits 13x enriched for annotated Rossi TF sites.

**But precision is poor: 0.58% at 5/5 recall — 863 chrI windows outscore the worst true site.**
The Murphy PWM alone narrows chrI to ~92 positions, so as a standalone detector this is ~10x
worse than sequence. Consistent with the main-line result that fiber-only TF *identity* is at
its ceiling.

**Controls are the real finding:**
- Reb1 template: 1.86% (22x worse than ABF1) — so it does discriminate.
- **Rap1 template: 0.056% — better than ABF1.** Both are notch-in-NDR factors and the filter
  cannot separate them. Rap1, not Reb1, is the meaningful wrong-TF control.

Conclusion: this is a strong **"protected notch inside an accessible region"** detector, not an
ABF1-identity detector.

### 6.3 A retracted claim — don't repeat this mistake

An early pass concluded "a flat template scores as well as the ABF1 template" (r = 0.986).
**That was wrong**, on two counts: (a) it used a +/-100 window, where 14 notch columns are
diluted by 187 flank columns, and (b) it correlated *un-whitened raw likelihood scores*, which
mostly compares two total-protection measures. At +/-7 the same correlation is 0.233. Always
compare **whitened shape vs whitened level**, never raw LLR vs raw LLR.

The profile extractor was also rebuilt: reuse `make_params_pm50.py`'s own `Pileup` /
`window_for` / `combine_motif_counts_binom` / `add_pseudocounts_binomial(3,58)` /
`fit_binomial_parameters` rather than re-deriving. The rebuilt +/-100 profile is bit-identical
(max|diff| = 0.000) to both the pm50 pkl and the shipped 14-column pkl. An ad-hoc extractor
drifted (max|diff| 0.053 W / 0.170 C) by filtering on reference A/T instead of bucketing on the
modkit base column, and by omitting pseudocounts.

### 6.4 Open issue this surfaced — `bg_params.pkl` may be mis-calibrated

`inputs/bg_params.pkl` has `p = 0.1383 / 0.1384`. The **genome-wide pooled rate is 0.0790**
(sum n = 689,036,863; sum k = 54,428,639 over 11,414,910 pileup rows). 0.1383 looks like an
*accessible-region* fit. Scoring with 0.1383 turns the ABF1 template into a nucleosome detector
that ranks true sites at 56.8% — worse than chance, which is why the scan used a local +/-500 bp
background instead. **If this is a real mis-calibration it biases every fiber-layer likelihood
ratio in the model, not just this scan.** Needs the user to confirm what segments were passed to
that background fit.

### 6.5 What is on disk

Only the profile artifacts — **the scan itself was never written out and must be rebuilt**:
- `analysis/abf1_profile_pm100_agentA.npz` — 201-length float64 arrays, keys `half, n_sites,
  n_plus, n_minus, motif_len, p_all_W, k_all_W, n_all_W, p_all_C, k_all_C, n_all_C, p_refA_W,
  k_refA_W, n_refA_W, p_refA_C, k_refA_C, n_refA_C`
- `analysis/abf1_profile_pm100_agentA.png`

### 6.6 Where to pick it up

1. Rebuild the scan as a committed script (it currently exists nowhere) and write results to
   TSV instead of leaving them in conversation context.
2. Fill the width sweep: +/-7, +/-10 (21 bp — the user specifically wants this), +/-12, +/-25,
   +/-50, judged on worst-case rank.
3. Add the plain-Pearson-`r` baseline at 14 bp and 21 bp alongside the weighted score, to show
   what the coverage weighting and whitening actually buy.
4. Settle 6.4 before trusting any genome-wide fiber likelihood ratio.
5. Note the structural limit: the emission layer consumes only the **14-column** vector, which
   discards the elevated flanks and does not even span the ~21 bp real footprint. A windowed /
   contextual fiber emission is the obvious follow-on but has not been scoped.

**Circularity caveat:** Rossi IS the training set for these parameters and MacIsaac is 77.5%
inside it. The +/-25 numbers above used a leave-one-chromosome-out template refit; any new
result must do the same or it is measuring memorisation.

---

## 7. Open threads (not started)
- Retrain EM with the sequence layer ON — current `*_seq_*` decodes reuse Fiber-only-trained
  weights, so they are a lower bound on what seq can do.
- Optional scorer speedup: cache the collapsed factor track per region (currently re-collapses
  the whole ~230 kb chrI segment on every score call, ~9 min/run).

---

## 8. Repo state — read this before you write anything

`git log` stops at **`90b05c3` "Fix reverse-strand fiber params; stop generator clobbering
the shipped pkl"**. Everything in §0 is **uncommitted and untracked**, and the user has
**not approved a commit or a push** — ask before running either.

Tracked files modified: `HANDOFF.md`, `whattodo.md` (this update). Nothing under `pkg/` or
`robocop.py` has changed since `90b05c3` — the motif audit touched no model code.

New, untracked, worth keeping (all under `analysis/`):

| file | what |
|---|---|
| `build_motif_dbs.py` | assembles `motifdb/` — shipped / JASPAR / Rossi / yeast background / coverage |
| `compare_motif_sources.py` | six TOMTOM runs + chrI log-odds track correlation → `motif_comparison.tsv`, `motif_comparison_report.txt` |
| `plot_motif_comparison.py` | the three overview figures |
| `motif_distance_sheet.py` | the sheet: common window, verdicts, replicate consensus → `.tsv` + `.json` |
| `make_motif_sheet_page.py` + `motif_sheet_template.html` | JSON → `motif_distance_sheet.html` (the artifact) |
| `pairwise_frame_consistency.py` | proves alignment is not transitive (59%); justifies the two-stage window |
| `rossi_evalue_structure.py` | why E-values are not comparable across factors; sets the 2-orders rule |
| `rossi_pick_sensitivity.py` | why "pick the Rossi motif nearest native" is biased |
| `fetch_rossi_conditions.py` | GEO GSE147927 → `inputs/rossi_sample_conditions.tsv` (heat-shock labels) |
| `abf1_column_distances.py` + `plot_abf1_column_distances.py` | per-column ED/KLD, Murphy vs JASPAR |
| `explain_abf1_score_split.py` + `plot_abf1_score_split.py` | TOMTOM-vs-FIMO gap on the 5 chrI sites |
| `compare_fimo_macisaac_chrI.py` | whole-chrI FIMO scan vs MacIsaac |
| `config_jaspar.ini`, `sbatch_jaspar_*.sh`, `sbatch_train_jaspar.sh` | the JASPAR-ABF1 run series |
| `inputs/jaspar_abf1_motifs_meme.txt` | the one-motif-swapped PWM file |
| `inputs/rossi_sample_conditions.tsv` | Rossi sample → condition / replicate |

**`analysis/pkgvar/` is untracked too** (§0.5) — 8 frozen copies of the `robocop` package,
each with its layer/mask toggles baked in. It is the mechanism every current run driver
depends on, so it is not disposable; but it carries compiled `librobocop.so` and
`__pycache__`, so if it is committed, commit the sources and gitignore the binaries. The
alternative is to record the per-variant `robocopExtras.py` diff (it is ~4 lines) and
regenerate.

Generated outputs also untracked: `jaspar_vs_murphy_chrI_metrics.json` (the §0.2 table),
`motif_distance_sheet.{tsv,json,html}`,
`motif_comparison.tsv`, `abf1_*.png`/`.tsv`, `motifdb/` (~60 MB, mostly TOMTOM tables —
**do not commit `motifdb/`**, regenerate it with `build_motif_dbs.py`), `chrI_fimo/`,
`chrI190k_fimo/`, and the decode directories (already git-ignored).

Also untracked and from **earlier** sessions, not this one — they belong to the
reverse-strand-fix and fiber-parameter-variant work described in §5–§6:
`fiber_params_lib.py`, `make_params_pm50.py`, `make_bg_tss.py`, `make_low_abf1.py`,
`compare_variants.py`, `run_split_*.py`, `plot_abf1_5sites_*.py`,
`plot_abf1_base_overlap*.py`, `plot_abf1_motif_anatomy*.py`,
`plot_macisaac_*.py`, `plot_site5_decoded_fimo.py`, `plot_native_region.py`,
`plot_factor_p_values.py`, and the `inputs/*.pkl` variants
(`all_TFs_1000pealVal_params.pkl`, `..._pseudo_lowabf1.pkl`, `..._pseudo_pm50bp.pkl`,
`bg_params_tss.pkl`).

## 9. EM training of the concentrations — QUEUED 2026-08-24, running unattended

**The finding that prompted it.** The per-DBF concentration prior is never fitted.
`parameterize.getDBFconc` sets `tf_prob` from `calculateKD` (Kd of the motif consensus,
a pure function of motif length and information content), and `robocop_em.py` hardcodes
`iterations = 0`, so the Baum-Welch loop under it never runs. Verified: all 154 states in
`robocop_train_fiberonly/HMMconfig.pkl` are bit-identical to values recomputed from
`pwm.p`, and `robocop_train/likelihood.txt` has exactly one line. Result: Nhp6a (7 bp)
= 7.2239e-4, 6th of 154; Abf1_murphy (14 bp) = 1.7974e-07, 148th — a **4,019x** gap,
while the chrI posterior implies only ~6.9x.

**Confirmed clean:** the ABF1 lambda multipliers are NOT in any source. `parameterize.py`
is byte-identical across `pkg/` and all ten `pkgvar/` copies; the sweep lives only in
`robocop_train_conc{3,10,30,100,300,1000}/`, each stamped with `conc_patch.json`; and all
three live decodes read the unpatched `robocop_train_fiberonly`.

**What is queued** (job ids in `analysis/.em10_jobids`, chained with `--dependency=afterok`
so a bad fit is never decoded and mistaken for a result):

| job | script | what |
|---|---|---|
| 12420006 | `sbatch_train_em10_smoke.sh` | 2 windows x 2 iters, then `em_smoke_gate.py` |
| 12420008 | `sbatch_train_em10.sh` | 20 windows x 10 iters -> `robocop_train_em10_chrII/`, then `em_trace_report.py` + gate |
| 12420009 | `sbatch_chrI_em10.sh` | chrI 6-way decode -> `robocop_chrI_seq_maskoff_em10/` |

- Training coords: `coord_train_chrII_20.tsv`, 20 x 5 kb on **chrII** (median A-trials
  66-89, comparable to chrI's 66), built by `make_train_coords.py --seed 0`. chrI is a
  fully held-out test set.
- Training variant: `pkgvar/seq_maskoff_em10/` — exactly two files differ from
  `pkgvar/seq_maskoff/`: EM on via `ROBOCOP_EM_ITERS` (default 10) with a compact
  `em_trace/iter{i}.npz` replacing the 97 MB-per-iteration `HMMconfig{i}.pkl` dumps, and
  the live emission-plot block (7 PNGs per segment per iteration) commented out.
- **Decoding uses `pkgvar/seq_maskoff`, NOT `_em10`** (`run_split_em10_decode.py`), so the
  only variable between `robocop_chrI_seq_maskoff_em10` and `..._revfix` is the trainDir.

**When it finishes, read in this order:** `logs/train_em10_12420008.out` (the gate verdict
and `em_trace_report`'s tables), `em_trace_robocop_train_em10_chrII.png/.tsv`, then
`python nhp6a_diag.py robocop_chrI_seq_maskoff_em10` and
`python score_robocop.py robocop_chrI_seq_maskoff_em10`.

**What to watch for.** The constrained-EM cap is `mean + 2*std` of the INITIAL priors =
6.69e-4, computed once and never recomputed; Nhp6a already sits at 108% of it, ABF1 at
0.027%. `unknown` is exempt (`adjustEM`'s `range(1, n_tfs)` skips the last TF, and
'unknown' sorts last) and has ~1042 implied binding events — the named failure mode is
`unknown` absorbing the freed mass instead of ABF1.

**The ERV46 test.** Two MacIsaac ABF1 sites sit in `chrI:60,001-65,000`. Today no single
run gets both: revfix/capA nail `62,657-62,671` (0.99/1.00) and miss `61,163-61,177`
(0.002/0.017); capB recovers the upstream one (0.55) but collapses the downstream one to
0.001. A correctly retuned prior should get both. Compare in the viewer:
`python make_posterior_viewer.py --region chrI:60001-65000 --run revfix=... --run em10=...`

**Still not committed** — see section 8. Nothing under `pkg/` was modified.
