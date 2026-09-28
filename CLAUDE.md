# CLAUDE.md — RoboCOP + Fiber-seq

RoboCOP (Mitra/Hartemink, NAR 49:7925) is an HMM that decodes nucleosome and TF occupancy
from sequence + chromatin accessibility. This fork replaces MNase-seq with **Fiber-seq m6A**
data in yeast (sacCer3). `pkg/` is the upstream package; **all work happens in `analysis/`**.

## Environment

```bash
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg R_HOME=$CONDA_PREFIX/lib/R
```

Slurm, partition `compsci`. A genome decode is 48 array tasks: ~45 min each on
`compsci-cluster-fitz-*`, **~1 h 55 on `linux3x` nodes**, so a round takes ~2 h wall clock.
Decode outputs (20–630 MB each) live on this filesystem and are gitignored; the metadata
(`config.ini`, `coords.tsv`, `pwm.p`, `conc_patch.json`) is committed.

## Where things stand (2026-09-12)

Concentration calibration is **finished**; see HANDOFF.md §1 for the numbers.

- Two campaigns of 8 rounds each tuned all 84 MacIsaac-targeted motifs' λ toward MacIsaac
  site counts: `u001` (all 153 motifs live) and `m001` (69 non-MacIsaac motifs hard-masked).
- Counts converge (u001 79/81 groups within 2× of target) but **site accuracy does not**:
  pooled F1 0.029 → 0.049, with all the gain in the first step. Every λ step adds or removes
  calls at only 1–2% MacIsaac precision, so λ sets *how many* calls a factor makes, not
  *which* ones are right.
- Conclusion carried forward: **λ is a count/operating-point knob**. Better site accuracy has
  to come from discrimination — the PWMs and the Fiber-seq emission model.
- Next, in order: (1) validate the tuned λ against the held-out **Rossi** set (HANDOFF §5);
  (2) fix the fiber emission (HANDOFF §4.6, `whattodo.md` Tier 1 #3); (3) the PWM shortlist
  (HANDOFF §6). Optional loose end: flag factors stuck at the λ **floor** (only the cap is).

## Rules

1. **Commit and push only when explicitly asked.**
2. Never overwrite `inputs/all_TFs_1000pealVal_params_pseudo.pkl`, `inputs/bg_params.pkl`,
   or `inputs/motifs_meme.txt`. A variant gets a new filename.
3. Never modify an existing `analysis/pkgvar/*` tree — copy it to a new one.
4. Do not touch `robocop_em.py`'s `iterations = 0` or its line-162 tmpDir cleanup.
5. Use the MacIsaac bed exactly as shipped: no offset correction, no strand flip.
6. A decode directory is named for its run **label** (`fib+seq+lam0.01` →
   `robocop_<chrom>_fib_seq_lam0p01`); keep the two in step.
7. When a run changes any parameter besides the one under test, say so as a decision before
   running, and restate it next to the result.
8. **Never hand-wave a claim.** Every statement about results, code behaviour, data or cause
   must be backed by evidence actually run or read: a `file:line` quote, a command and its
   output, or a number read from a named file. If something has not been verified, say
   "unverified" and how to check it. Check agents' reports against the files before relaying
   them.

## Gotchas that have each cost a session

- **A decode never re-reads the MEME file.** `run_robocop_without_em` reads `pwm_emission`,
  `tf_prob` and `transition_matrix` out of the trainDir's `HMMconfig.pkl`. Changing `pwmFile`
  in a config does nothing without a retrain.
- **λ does not compose.** `make_conc_trainDir.py`'s fidelity gate recomputes every
  concentration from `calculateKD` and demands a bit-exact match to its source, so patching a
  patched trainDir aborts. Always apply the cumulative λ vector in one `--set` call against
  pristine `robocop_train_fiberonly`.
- **Layer/mask state lives in frozen package copies** under `analysis/pkgvar/<variant>/`,
  selected by each driver's `sys.path.insert`. Never hand-comment toggles.
- **A hard mask must be applied AFTER the 1e-30 floor** in `robocopExtras.py`, or masked
  states get lifted to 1e-30 and still leak posterior.
- **Lowering a small prior inflates the nucleosome.** All priors share one normalisation and
  the 147 bp nucleosome amplifies it as `p^147`: λ_unknown = 0.01 alone raises the nucleosome
  prior 21×. Use `make_conc_trainDir.py --hold-nucleosome` to pin it.
- **chrXII's rDNA array emits invalid posteriors** (1e0–1e171, finite in float64). Validate
  the `[0,1]` bound, not finiteness; `count_calls.py` drops those positions loudly.
- **MacIsaac gff rows are redundant** — merge by interval union with 20 bp slop before
  counting (ABF1 315 rows → 300 sites). `make_conc_targets.py --verify` pins this.
- **Decode with `run_robocop_without_em`** (keeps `tmpDir/info.h5`); `with_em` deletes tmpDir
  for ≤500 coords.
- `plt.close('all')` in long decode loops — a figure leak OOM'd whole-chrI runs.
- **`bc.c` index arithmetic is 32-bit**: `n_states` must stay ≤ 46,340. ±70 pads fit across
  all 153 motifs; ±150 across all motifs does not.

## Command cheat-sheet

```bash
# score a decode against Chereji nucleosomes / MacIsaac ABF1 / phasing
python score_robocop.py <outDir>
python score_factors.py --chrom chrXIV --runs-from chrXIV_runs.tsv   # per-factor, many runs

# concentration tuning: one campaign = state + trainDirs + decodes under one name
python tune_concentrations.py status --run u001
python tune_concentrations.py build  --iter N --run R [--driver <run_split_*.py>]
python tune_concentrations.py submit --iter N --run R [--chain]   # --chain self-drives rounds
python tune_concentrations.py update --iter N --run R             # counts -> report + next λ
python tuning_trajectory.py --run R                               # what each λ step bought

# per-factor counts for any decode (occ, calls, MacIsaac matches within 30 bp)
python count_calls.py <outDir> --chrom chrIV --out counts_chrIV.tsv
python count_calls.py --merge 'conc_tuning/counts_<tag>/*.tsv' --out counts.tsv

# rebuild published pages
python conc_tuning/make_conc_sheet.py u001=7 m001=7   # concentration sheet
python make_posterior_viewer.py --regions viewer_regions.tsv --out posterior_viewer_all.html
bash viewer_site/build_site.sh && bash viewer_site/serve.sh 8765   # local Run Browser (all runs), HANDOFF §9
```

Republish an artifact **to its existing URL** (pass it as `url`), or a second one is created.
URLs are in HANDOFF.md §9.

## Figures: the locus slide is the preferred style

The user likes the **locus slide** (talk slide 5, "One ABF1 locus up close"): a ~2 kb window drawn
the way the RoboCOP Occupancy Browser draws it. It has one row per run with the highlighted factor's
posterior over grey nucleosome occupancy, one shared m6A Watson/Crick panel, genes and an axis, with
reference sites outlined and labelled "hit/miss · coords · posterior". Default to this style when
showing a decode at a locus. It is saved as a reusable template in
`presentation/templates/locus_slide/`: `build_locus_slide.py` builds one slide's data from decodes
(it imports `make_posterior_viewer.py` unchanged), `locus_slide_snippet.html` holds the component,
and `README.md` covers inputs, sizing (3 rows fit 1600×900) and the hit/miss rule. The talk deck
(`presentation/talk.html`, artifact `1vT1DuTfTbuYvFzTyjwMmy`) uses it on slides 5, 5b–5d, 7b, 10b and 10c.

## Docs

| file | what |
|---|---|
| `HANDOFF.md` | state, results, machinery, and where to pick each thread up |
| `whattodo.md` | strategy and the open-work tiers |
| `FIBERSEQ_CHANGES.md` | the full code diff against upstream |
| `things_to_revisit_before_shipping.md` | debug leftovers and hardcoded knobs, by priority |
| `analysis/README_wide_implementations.md` | the widened-footprint run family |
