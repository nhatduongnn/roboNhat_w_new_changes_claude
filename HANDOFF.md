# RoboCOP + Fiber-seq — Session Handoff

Pick-up notes for the next agent. Read `whattodo.md` (strategy + open work) and
`FIBERSEQ_CHANGES.md` (full code diff vs upstream) first — this file adds the **scoring +
plotting tooling** and the **layer / mask toggles** that are easy to get wrong.

**Start with §10** — the newest work (2026-09-01 → 09-04): the widened-footprint runs,
including four `wide150` decodes that are **finished on disk and not yet scored**; the
published artifact URLs; and the Rossi genic/intergenic validation table. §0 is the older
motif-source audit (the shipped PWM collection vs JASPAR and Rossi; ABF1's matrix is
demonstrably wrong and six others are suspect) and still stands. §6 covers a self-contained
side experiment (sliding the fitted ABF1 fiber footprint across the genome), finished as an
exploration but whose code was never saved.

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

> **Superseded in part.** This section was written when `git log` stopped at `90b05c3`.
> The §0 motif-audit work described below was committed in `3384105`; §10.8 has the current
> repo state. The rules at the end of this section — how to treat `pkgvar/` and `motifdb/` —
> still apply.

`git log` stopped at **`90b05c3` "Fix reverse-strand fiber params; stop generator clobbering
the shipped pkl"** when this was written. Everything in §0 was then uncommitted and
untracked, and the user had **not approved a commit or a push**.

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

---

## 10. Most recent work (2026-09-01 → 09-04) — widened footprints, the viewer, and the Rossi genic/intergenic target

**Read this section first.** It supersedes the "start with §0" pointer at the top of this
file: §0 is the motif audit from August and still stands, but everything below is newer.

### 10.1 What is DONE AND UNSCORED — pick this up first

The `wide150` decodes **all four finished** (Slurm 12492826–12492829, COMPLETED
2026-09-03 19:24 → 23:20, 1.5–5.5 h each). Their `tmpDir/info_*_6.h5` files are complete on
disk. **Nothing has scored them, nothing has charted them, and they are in no `*_runs.tsv`.**

| decode dir | layers | chrom | h5 splits |
|---|---|---|---|
| `robocop_chrI_wide150` | fib+seq | chrI | 6 |
| `robocop_chrI_fib_wide150` | fib only | chrI | 6 |
| `robocop_chrXIV_wide150` | fib+seq | chrXIV | 12 |
| `robocop_chrXIV_fib_wide150` | fib only | chrXIV | 12 |

Same trainDir `robocop_train_wide150/` for both layer variants; the sequence layer is
switched at decode time by which `pkgvar` the driver imports. `n_states` 10685 (baseline
3485). Built by `run_wide150_all.sh`; full rationale in `README_wide_implementations.md`.

**To score them**, add four rows to `layer_runs_chrI.tsv` / `layer_runs_chrXIV.tsv` —

    fib+wide150       robocop_chrI_fib_wide150
    fibseq+wide150    robocop_chrI_wide150
    fib+wide150       robocop_chrXIV_fib_wide150
    fibseq+wide150    robocop_chrXIV_wide150

— then widen the `#SBATCH --array` range in `sbatch_score_layers.sh` /
`sbatch_score_layers_chrXIV.sh` to match the new row count (they index
`layer_runs_*.tsv` by array task id; the array bound is **not** derived from the file, so a
stale bound silently skips the new rows). Reports land in `layer_scores/` and
`layerXIV_scores/`; `make_factor_chart.py` builds the comparison chart from them.

**Read the enrichment numbers with the block-width correction.** Like every `widememe`-method
run, the posterior collapses over the whole padded block (`sum_for_dbf_probs` unmodified), so
at ±150 an ABF1 call renders as a **314 bp plateau, not a 14 bp peak**. That inflates the
genome-wide mean and deflates enrichment by roughly the width ratio (314/14 ≈ 22×) — far
larger than `widememe`'s 1.6×. A raw enrichment drop is therefore **expected and not
evidence the model is worse**; compare recall and site posteriors, or correct for the block
width, before concluding anything. Same for the estimated-pad prior suppression
(`README_wide_implementations.md`, "wide10all's pooled path"), which at ±150 will be larger
still and is undone without retraining via
`make_conc_trainDir.py --tf <name> --lam <1/ratio>`.

### 10.2 There is NO ±75 run

Nothing at ±75 was ever built or submitted; do not go looking for it. What exists at the
wide end is `wide150` (§10.1, the 12 fitted TFs at ±150, running and done) and
`wide150all` (all 153 motifs at ±150) which is **built and gated but permanently blocked**:
`n_states` 95285 overflows the 32-bit index arithmetic in `pkg/robocop/bc.c`, verified by
calling the shipped `.so` directly (`sbatch_wide150all_memcheck.sh` —
`I(n-1,n-1,n)` returned 489,296,632 instead of 9,079,231,224, and `I3` returned negative).
`run_wide150all_all.sh` refuses to submit.

**±70 is the largest uniform pad across all 153 motifs that fits** (`n_states` 46325 against
the ceiling of 46340 = floor(sqrt(INT_MAX))). That needs no code change and is the obvious
next experiment if an all-motif wide run is wanted; memory was measured at ~48 GB for the
three dense square matrices, ~295 GB train, ~65 GB decode, so only the 1,150,000 MB
`compsci-cluster-fitz-*` nodes will hold the training job. The alternative — widening the
index type in `bc.c`/`bc.h`/`algo.c` to `int64_t` and building a **separate**
`librobocop.so` for that variant only — was scoped but not done; it changes the numerical
core and must be re-validated against an existing run before it is trusted.

### 10.3 Published artifacts

| artifact | URL | built from |
|---|---|---|
| RoboCOP Occupancy Browser (combined, chrI + chrXIV) | `https://claude.ai/code/artifact/c6c7d1f3-d62a-4858-a38a-4ee1c7891e0d` | `posterior_viewer_all.html` ← `make_posterior_viewer.py --regions viewer_regions.tsv` |
| chrI Occupancy Browser | `https://claude.ai/code/artifact/24b47df3-9f5f-4372-b6ec-d4b1976c6f2a` | `posterior_viewer_erv46.html` |
| chrXIV Occupancy Browser | `https://claude.ai/code/artifact/9fd1fd00-ad6d-4d85-9e92-177d30888836` | `posterior_viewer_chrXIV_187k.html` |
| chrXIV Occupancy Browser (55–60 kb) | `https://claude.ai/code/artifact/20e96d14-e171-4170-a11b-f32eb7711680` | `posterior_viewer_chrXIV_58k.html` |
| Factor Detection on chrXIV | `https://claude.ai/code/artifact/b5a5d5b2-3ba0-4260-b63c-ce74115e33b7` | `chrXIV_factor_chart.html` ← `make_factor_chart.py` |
| Where the Twelve Bind (genic/intergenic) | `https://claude.ai/code/artifact/7b4db749-5827-40b4-80a3-854fbb56a6b6` | `rossi_genic/where_the_twelve_bind.html` ← `make_genic_report.py` |
| ORF Versus Gene Body (teaching diagram) | `https://claude.ai/code/artifact/28965c44-fe42-4433-b918-c42a5fe5550b` | `orf_vs_gene_body.html` |
| Odd One Out (motif audit sheet) | `https://claude.ai/code/artifact/7c63a39b-8e04-4cd1-9660-7797d0dec154` | `motif_distance_sheet.html` |

To update one, edit the file and re-publish **to the same URL** — a publish without the URL
creates a second artifact instead of updating the first.

**The combined viewer c6c7d1f3 is stale.** `viewer_runs_chrI.tsv` and
`viewer_runs_chrXIV.tsv` are missing `widefp` and `wide150`; add both label pairs and rebuild
with `make_posterior_viewer.py --regions viewer_regions.tsv --out posterior_viewer_all.html`.
The label sets in the two files must stay **identical** — the label is the join key that
keeps the selected run when you switch region.

**28965c44 (ORF Versus Gene Body) is partly obsolete.** Its decision-chain section and
cross-tab describe the four-class promoter / gene-end / gene-body / intergenic scheme that
was abandoned on 2026-09-03 (§10.4). Its Figure 1 — the ORF/gene-body/TSS anatomy diagram —
is still correct and is the reason to keep it. Either retire it or strip the decision chain.

### 10.4 The Rossi genic/intergenic target — a validation set for any decode

**One rule, and only one:** a position is `genic` if it lies between some gene's ATG and its
stop codon, `intergenic` otherwise. No promoter window, no terminator window, no priority
order. An earlier four-class scheme (promoter / gene-end / gene-body / intergenic) was built
and then **abandoned** — its windows and the priority order needed to arbitrate overlaps moved
the answer by several points without adding any fact, and it produced the absurdity of a peak
inside gene A's ORF being labelled "promoter of gene B". Do not reintroduce windows.

The coordinate source is `inputs/sacCer3.gtf` (Ensembl R64-1-1 = SGD R64), **not**
`inputs/Park_2014_TSS.csv` — Park covers only actively transcribed genes and misses 1,707 of
the 6,692 ORFs (26%). The GTF `gene` span for a protein-coding gene **is** the ORF: audited
over all 6,516 genes carrying both a CDS and a stop codon,
`gene.start − CDS.start ∈ {−3, 0}` and `gene.end − CDS.end ∈ {0, 3}` (that 3 bp is only
whether the stop codon counts inside the CDS), `+` strand `gene.end == stop_codon.end` and
`−` strand `gene.start == stop_codon.start` for every gene. No UTRs are annotated. The audit
re-runs and re-prints on every invocation rather than being trusted from this note.

**The null: 73.0% of this genome is inside an ORF** (union 8,901,290 of 12,157,105 bp,
confirmed against 200,000 uniform random positions). A raw genic% means nothing without it;
every table carries `genic_vs_null = genic% / 73.0%`.

| file | rows | what |
|---|---|---|
| `analysis/rossi_genic.py` | — | the 12 fitted TFs. `--normal-only`, `--outdir` |
| `analysis/rossi_genic_all.py` | — | **all 378 TFs** — the validation table. imports `Genome`/`read_orfs` from `rossi_genic.py` |
| `analysis/make_genic_report.py` | — | renders the artifact from the TSVs |
| `analysis/rossi_genic/rossi_genic.tsv` | 13 | the 12 + the random-genome null |
| `analysis/rossi_genic/rossi_genic_all_TFs.tsv` | **378** | per-TF counts, both peak sets, `in_robocop` flag |
| `analysis/rossi_genic/rossi_peaks_genic_all_cx.tsv` | 182,582 | every merged summit with its genic flag |
| `analysis/rossi_genic/rossi_peaks_genic_all_motif.tsv` | 29,105 | the motif-filtered subset, same flag |
| `analysis/rossi_genic/rossi_peaks_genic.tsv` | 3,455 | the 12 TFs, per peak |
| `analysis/rossi_genic/genic_bars.png`, `set_comparison.tsv` | — | figure + zip/merged/+motif comparison |

**Two peak sets, both in the table, and they are different targets.** `_cx` columns are
Rossi's merged ChExMix calls read straight from
`/usr/project/xtmp/nd141/projects/data/rossi_strand/{TF}_CX.bed` (381 files, 3 of them empty
— Kti12, Rpa190, Rsc2 — hence 378 TFs, 182,582 peaks). `_motif` columns are the same calls
kept only where a YEP FIMO motif sits within 30 bp
(`inputs/rossi_peak_w_strand_all_TFs.bed`, 358 TFs, 29,105 peaks after `--normal-only`).
**The motif filter is not neutral — it removes genic peaks preferentially** and always moves
genic% down, sometimes hard (Fhl1 53.2 → 17.9%, Tbf1 20.4 → 5.2%, Fkh2 33.7 → 10.4%). Score a
decode against whichever set the decode resembles; mixing them reads as model error. The 12-TF
`_motif` columns reproduce `rossi_genic.tsv` exactly (all deltas zero) — that is the
cross-check between the two scripts.

**How to use it as validation.** For each TF the model emits, count its calls that land genic
and intergenic and compare the fraction with that TF's row. This is a *distributional* claim,
weaker than site-level agreement but far broader: a model can be wrong site-by-site and still
be asked whether it puts the right share of each factor inside genes. The per-peak files keep
the stricter site-level comparison open. Scope: **77 of the 153 RoboCOP motif TFs have a Rossi
row, 47 with ≥100 merged peaks** — that is the usable set, and their genic% spans 8.8%
(Spt15) to 59.3% (Cad1), median 27.8%, so it is a real target and not a constant. The
remaining 73 PWMs (Pho4, Tec1, Gat1, Msn4, most of the YBR/YDR orphans) have no Rossi target
at all; the `in_robocop` column marks the join both ways.

**Do not score against a single global expectation.** Pooled over all 378 TFs only 42.2% of
peaks are genic against the 73.0% null — but 82 of the 378 TFs sit *at or above* the null.

**The table validates itself at both ends**, which is the reason to trust the classifier.
Sorted by genic%, with no input about what these proteins do, the intergenic extreme is the
Pol II preinitiation complex (Sua7/TFIIB 9.8%, Spt15/TBP 8.8%, Tfb1, Tfb2, Rad3) plus the
whole Pol III machinery (TFIIIC Tfc1/3/6/8, Brf1, Bdp1 — tRNA genes are not protein-coding
ORFs so they read intergenic by construction) and Orc1 at replication origins (2.0%); the
genic extreme is Paf1C (Paf1, Leo1, Rtf1, Ctk2 89–93%), COMPASS (Bre2 94.6%, Sdc1, Swd3,
Spp1, Shg1), Set2 94.3%, Chd1, and Rad6/Bre1 — every one a co-transcriptional elongation
factor that rides the ORF with Pol II. Promoter machinery lands at 0.03–0.13× the null,
elongation machinery at 1.22–1.30×.

The 12 fitted TFs, merged + motif set, normal condition only:

    TF                   n    genic  interg    genic% interg%   vs null
    Tbf1               155        8     147       5.2    94.8     0.07x
    Reb1               583       54     529       9.3    90.7     0.13x
    Spt15/TBP          259       26     233      10.0    90.0     0.14x
    Rap1               356       36     320      10.1    89.9     0.14x
    Fkh1               297       35     262      11.8    88.2     0.16x
    Mcm1               179       25     154      14.0    86.0     0.19x
    Abf1               502       88     414      17.5    82.5     0.24x
    Ume6               236       42     194      17.8    82.2     0.24x
    Fhl1                84       15      69      17.9    82.1     0.24x
    Nhp6a              138       26     112      18.8    81.2     0.26x
    Cin5               264       80     184      30.3    69.7     0.42x
    Sko1               182       82     100      45.1    54.9     0.62x
    random genome   200000   145953   54047      73.0    27.0     1.00x

`--normal-only` drops the 223 motif-set rows annotated from a heat-shock sample (13 TFs,
mostly Spt15/RSC/SAGA). It cannot apply to the merged set: those calls are pooled across
replicates and carry no sample attribution to filter on.

### 10.5 Abf1's genic peaks are real — the question that started §10.4

Asked whether Abf1's in-ORF calls are replicate-supported or filter leakage. They are real.
Reconstructed per-peak replicate support and significance from the per-sample
`{id}_chexmix_allevents.tabular` files (which carry `YPD_Sig`, `YPD_Ctrl`, `YPD_log2Fold`,
`YPD_log2P` — the only route to per-peak significance, since every score in `{TF}_CX.bed` is
the constant 1000), matched at 30 bp across 3 replicates by `analysis/rossi_abf1_support.py`:

- promoter+motif mean replicate support **2.24**, gene-body+motif **2.29** — gene-body peaks
  are *better* supported, not worse;
- 45.5% vs 46.2% called by all three replicates;
- **0 of 55** rest on the pooled analysis alone.

Without the motif filter, gene body *is* the weak tail (31.4% zero-support, median log2fold
2.83 vs 3.45) — so the motif requirement, not the location, is what separates strong from
weak. Outputs in `analysis/rossi_abf1_support/`.

### 10.6 A gate that was wrong, and the fix

`wide150` training "failed" `check_widememe_traindir.py` after 2h20m: residual spread 6.1e-6
against a 1e-8 tolerance. **The model was fine; the gate was wrong.** It trusted the
`background_prob` stored by `convert_to_prob`, which solves its unbound root numerically —
and the value it *stores* converges less tightly than the `tf_prob`s built with it (4.4e-7
relative at 320-column motifs vs 3e-15 at 40). `corr(residual, tf_len) = 1.0000` gave it
away. The gate now **refits the root** from the priors themselves instead of trusting the
stored one:

```python
coef = np.linalg.lstsq(np.vstack([np.ones_like(L), L]).T, np.log(raw), rcond=None)[0]
r_fit = float(np.exp(coef[1]))
spread = float((raw / r_fit ** L).max() - (raw / r_fit ** L).min())   # 1.1e-15
drift  = r_fit / (pn / pb) - 1.0
```

`wide10` and `widefp` were re-verified and still pass. **The gate needs
`--params inputs/all_TFs_1000pealVal_params_pseudo_<variant>.pkl`** — omitting it fails every
variant with "every params entry matches its block length", which is the gate correctly
complaining that the baseline pkl does not describe a widened model.

### 10.7 Standing constraints — carried forward, do not violate

- **Never overwrite** `inputs/all_TFs_1000pealVal_params_pseudo.pkl`, `inputs/bg_params.pkl`,
  or `inputs/motifs_meme.txt`. New variants get new filenames.
- **Do not modify** the `robocop_em.py` line-162 tmpDir cleanup.
- **Do not modify any existing `pkgvar/*` tree** — create a new one.
- The MacIsaac bed is used **exactly as shipped**: no offset correction, no strand flip. The
  1 bp ABF1 phase difference against Murphy is a motif-definition difference, not an error.
- **Commit and push only when explicitly asked.**

### 10.8 Repo state as of this commit

Everything in §10 **is committed**, including `analysis/pkgvar/` (which now carries 28
frozen package variants; the 40 KB `librobocop.so` binaries are tracked too, matching what
`3384105` already did — all 28 are byte-identical copies of the
shipped library, md5 `6a0724bf6ef7b8bc4313927b931ac685`, verified), the widened meme files and params pkls, the
decode-directory metadata (`config.ini`, `coords.tsv`, `pwm.p` — `tmpDir/` and
`HMMconfig*.pkl` stay gitignored, so a run is reproducible from what is committed without
the 60–630 MB of posteriors), and all of `analysis/rossi_genic/`.

**Deliberately NOT committed, still on disk:** `analysis/rossi_locus_class{,_v2,_v3,_normalonly,_setcmp,_setcmp_v2}/`
— the outputs of the abandoned four-class scheme (§10.4). `rossi_locus_class.py` and
`rossi_locus_report.py` ARE committed as the record of what was tried, but their output
directories are noise and are safe to delete.

Nothing under `pkg/` has been modified. `robocop_em.py`'s `iterations = 0` and its line-162
tmpDir cleanup are both untouched.
