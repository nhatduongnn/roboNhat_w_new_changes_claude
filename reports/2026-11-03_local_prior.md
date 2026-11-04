# Local prior at ERV46: before / local prior / raised global prior (2026-11-03)

## Answers

- **Does the local prior work?** Yes, at both ERV46 ABF1 sites. Median per-read P(ABF1) goes from ≤ 0.0009 to 0.81 (site 1) and 0.79 (site 2), with no ABF1 calls anywhere else in the window.
- **Is it better than raising the genome-wide prior?** Yes. ABF1 ×1500 lifts only site 2 (median 0.57). Site 1 stays at 0.0097, and a non-site spot (61,660) reaches 0.50.
- **Do reads still differ from each other?** Somewhat. Per-read SD is 0.13 / 0.12, against 0.33 for site_mixture at the same sites. P(ABF1) falls as methylated motif As rise (Spearman −0.58 / −0.72).
- **Part 2b (decision point).** The spread is above the plan's 0.076 trigger. That 0.076 was the median over all 86 sites, not these two. At these two sites the spread is about 2.5× smaller than site_mixture's, so RoboCOP's per-read m6A evidence is still weaker. I did **not** build a bound-only ABF1 rate. That is the user's call.
- **Nucleosomes:** essentially unchanged. The mean |Δ| of the averaged occupancy is 0.0009, and the Chereji dyad results match "before".

## What was built

| file | what |
|---|---|
| `analysis/singlefiber/site_mixture.py` | current model: nucbrogsym, filtnordna nucleosome/background pkls, background floor plumbed into `em()`, driver read filters + keep list + thresh; new `heldout_run` (backup `.site_mixture.py.bak_localprior`) |
| `/usr/project/xtmp/nd141/tf_compare/scripts/run_site_mixture_filtnordna.py` (+ `.sh`) | copy of the 86-site runner → `tf_compare/mixture_filtnordna_pb0p02/` (the old `mixture_pb0p02` is untouched) |
| `analysis/singlefiber/local_prior.py` | table loader, row/state mapping (ref and flipped frames), weight in layer 1 as −log w on the other emitting states |
| `analysis/singlefiber/run_permol_window_logfull_5to3.py` | `--local-prior <sites_w.tsv>`, default off. Each read uses the w row of its own held-out run. Records `local_prior` in RUN_INFO / run_stats; molecules.tsv gains `local_prior_run` and `local_prior_sites` only when the option is on (backup `.bak_localprior`) |
| `analysis/singlefiber/calibrate_local_prior.py` | bisects log w so that prior-only P(ABF1) = f (m6A all no-obs, sequence on) |
| `analysis/singlefiber/test_local_prior.py` | toy test and real-decoder test |
| `analysis/singlefiber/fig_permol_fibers.py` | `--order-from <json>`, `--abf1-vmax`; the json now stores the read order and labels (backup `.bak_compare`) |
| `analysis/singlefiber/compare_localprior_erv46.py` | numbers + 3-way figure |

The driver was replaced atomically (temp file, then `ast.parse`, then `mv`) while job 13256087 was running. Its later tasks run the new file. With the option off it is bit-identical (see Checks), but `perread_sites_cur.sh` prints `md5sum` of the driver. Tasks that started after the swap (about 13:08) will show md5 `136e4fe9…` instead of `53c696b5…`.

## Checks

| check | result | source |
|---|---|---|
| Toy: weight on the emission = weight on the entry transition | max \|Δ\| = 0; other-state form 3.8e-16; odds ratio = w | `compare_localprior/test_local_prior.log` |
| Real decoder, w = 1 | bit-identical to no hook | same |
| Real decoder: odds of "ABF1 starts at 62659" change by w | rel. error ≤ 5e-13 (w = 1e3, 1e5, 1e-2); flipped frame 1.7e-13 | same |
| Every TF first state has one parent | pass (`check_single_parent`, all 154 TFs) | same |
| Regression: option off vs `filt/erv46_5to3_nordna` | `per_mol`, `avg_optable` bit-identical; names, cov and loglik identical; same label (fitz-39 vs fitz-36) | `compare_localprior/regression_check.txt` |
| Calibration: prior-only P = f | ref frame max \|P−f\| 2.4e-4. **Flipped frame 1.5e-3**, just above the 1e-3 target, because of the small nucleosome-model asymmetry | `compare_localprior/sites_w.json` |
| Same read order in all three composites | clustered order, shuffled order and labels identical | the three `fig_permol_fibers_*.json` |

## f per site (Part 1, p_bound 0.02) — `tf_compare/mixture_filtnordna_pb0p02/sites_f.tsv`

| site | f [90% CI] | n | f without Jun25 | f without Mar20 | f without May07 |
|---|---|---|---|---|---|
| site 1 chrI:61165-61178 | 0.715 [0.626, 0.793] | 140 | 0.841 | 0.706 | 0.677 |
| site 2 chrI:62659-62672 | 0.615 [0.503, 0.719] | 132 | 0.709 | 0.586 | 0.673 |

The two controls have f ≈ 0 (≤ 1e-9). Calibrated w (`compare_localprior/sites_w.tsv`):

- site 1: pooled 7.2e5, range 6.0e5–1.5e6. Prior-only P without a weight is 3.5e-6.
- site 2: pooled 3.1e3, range 2.7e3–4.7e3. Prior-only P without a weight is 5.2e-4.

## 3-way numbers — `/usr/project/xtmp/nd141/permol_proto_logfull/compare_localprior/compare_summary.json`

Per-read P(ABF1) = max over the motif interval, for reads that cover the whole motif.

| site | condition | n | median | max | frac > 0.5 | SD | ρ vs meth. motif As | ρ vs site_mixture | call agreement |
|---|---|---|---|---|---|---|---|---|---|
| 1 | before | 142 | 6.5e-6 | 7.3e-6 | 0 | 2e-6 | −0.74 | 0.69 | 0.26 |
| 1 | **local** | 142 | **0.808** | 0.916 | **0.94** | 0.132 | −0.58 | 0.64 | **0.80** |
| 1 | global ×1500 | 142 | 0.0097 | 0.011 | 0 | 0.003 | −0.74 | 0.69 | 0.26 |
| 1 | site_mixture | 140 | 0.917 | 0.967 | 0.74 | 0.326 | −0.69 | – | – |
| 2 | before | 138 | 8.8e-4 | 1.1e-3 | 0 | 3e-4 | −0.73 | 0.45 | 0.40 |
| 2 | **local** | 138 | **0.786** | 0.840 | **0.93** | 0.121 | −0.72 | 0.49 | 0.67 |
| 2 | global ×1500 | 138 | 0.572 | 0.631 | 0.72 | 0.126 | −0.73 | 0.43 | **0.79** |
| 2 | site_mixture | 132 | 0.766 | 0.931 | 0.60 | 0.322 | −0.58 | – | – |

Call agreement = the fraction of reads where both calls fall on the same side of 0.5, against site_mixture's pooled post_bound. Against site_mixture's run-held-out post_bound, the local prior scores ρ 0.81 / 0.54 and agreement 0.94 / 0.77.

**Local-prior median by methylated motif As (0 / 1 / 2 / 3):**

| site | 0 | 1 | 2 | 3 | reads (0 / 1 / 2 / 3) |
|---|---|---|---|---|---|
| site 1 | 0.82 | 0.74 | 0.46 | 0.20 | 106 / 24 / 9 / 1 |
| site 2 | 0.80 | 0.65 | 0.44 | 0.19 | 101 / 23 / 7 / 1 |

**Over-calling.** The local prior's mean posterior sits above f: 94% of reads exceed 0.5, against f = 0.72. Reads with no methylated motif As push toward "bound" under RoboCOP's m6A rates. This is one more sign that RoboCOP's emission is not the site_mixture model (Part 2b).

**Window-wide ABF1 calls > 0.5 outside the two sites (±10 bp):**

| condition | calls | highest off-site per-read P(ABF1) |
|---|---|---|
| before | 0 | 6.6e-4 |
| local prior | 0 | 6.6e-4 |
| global ×1500 | 0 | 0.498 at 61,660 (median 0.44 over covering reads) |

**Nucleosomes vs before:**

| condition | mean \|Δ\| averaged | max \|Δ\| averaged | mean \|Δ\| per read | max \|Δ\| per read |
|---|---|---|---|---|
| local prior | 0.0009 | 0.022 | 0.0010 | 0.48 |
| global ×1500 | 0.0022 | 0.044 | 0.0023 | 0.17 |

The Chereji dyad check passes in all three conditions (recall 0.75). The median distance at each dyad is the same (0 / 34 / 14 / 11 bp), and the fraction within 20 bp changes by ≤ 0.008 (`compare_localprior/nucq/nuc_quality_before_localprior_globalprior.json`).

Posterior validity for all three runs: no NaN, row-sum deviation ≤ 1.3e-14, silent columns 0.

## Figures — `analysis/singlefiber/figures/erv46_localprior_2026-11-03/`

- `compare_abf1_3way_erv46.png` (.pdf). Panels, in the same row order:
  - raw m6A, clustered;
  - P(ABF1) before / local / global, each on a 0–1 scale, drawn over chrI:60,901–62,900;
  - per-site scatter of per-read P vs methylated motif As, with site_mixture in grey.
- `{before,localprior,globalprior}/permol_composite_full_*.png`: full window, P(ABF1) on 0–1. The order comes from the "before" run's clustering (`before/fig_permol_fibers_erv46_5to3_nordna.json`; its cluster sizes 45/39/24/14 noise match the 11-02 figure).

## Caveats

- **Global prior run.** It is the sweep's existing `abf1sweep/x1500`, reused and not re-run. Its trainDir is the only difference from "before".
- **Held-out f is per run, not per read.** Each read's w comes from f fitted without its own run. With f pooled, the local prior's agreement with pooled post_bound would differ. That case is unverified; to check it, decode with a table that has only `pooled` rows.
- **Site 1 needs an extreme weight.** Its w is ~7e5 because the PWM score there is weak (prior-only P 3.5e-6). The weight applies only to reads that cover the whole motif.
