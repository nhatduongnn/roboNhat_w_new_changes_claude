# Single-fiber FiberCOP — handoff and plan (updated 2026-11-04)

**Start here for the single-fiber thread.** Read `CLAUDE.md` first (environment, rules). This file says
what exists, what was decided, what the model is, what failed and why, and what to do next.
Detail: `analysis/singlefiber_review/REPORT.md` (review of everything done), `analysis/singlefiber/README.md`
and `analysis/singlefiber/LOGFULL_PORT.md` (the code and the log-space port).

## Goal

Call TF binding (and nucleosomes) on **individual Fiber-seq reads**, using each read's own m6A plus the
DNA sequence (PWM layer) to make up for how little one read carries. The end question is how many fibers
are bound at a site, and which ones.

## Next session (2026-11-04): start here

**State in one paragraph.** The single-fiber defaults are the filtered, rDNA-free 7-barcode model:
- tree `pkgvar/permol_seq_maskoff_logfull_nucbrogsym_filtnordna`;
- `inputs/*_filtnordna.pkl` (background, nucleosome and TF m6A, all refit on one read set);
- keep list `/usr/project/xtmp/nd141/filtered_refit_nordna/keep_reads.txt`;
- read filters MAPQ ≥ 20, aligned ≥ 500, qs ≥ 10, one pooled MAD cap;
- fixed call threshold 0.8105469; start anywhere; 5'→3'.

Nucleosomes per read look good. Per-read TF calls are still the problem. Details are in the "Done 2026-11-02",
"Local prior, 2026-11-03" and "Revisit later" sections below, and in `reports/`.

**Open, in order (the user decides):**
1. **No-call handling — the PI is not happy with it.**
   - Today a no-call A (0.19 < p < 0.81) scores 1 in every state, the same as C/G/T, so it cancels.
   - Measured no-call fraction: 14.6% of linker A's vs 7.3% in Brogaard nucleosome cores
     (`/usr/project/xtmp/nd141/nocall_rates/nocall_rates.json`), so the "no evidence" assumption is false.
   - Options:
     - (a) 3-outcome categorical emission, P(meth) = (1−ν)θ, P(unmeth) = (1−ν)(1−θ), P(no-call) = ν, with ν per state
       fitted from the pileup's N_fail;
     - (b) soft calls θp + (1−θ)(1−p), which need calibrated dorado p;
     - (c) keep as is.
   - The user wants to see what is best before choosing.
   - Nov 3 deck slide 4b shows today's math.
   - Not measured yet: ν at bound TF sites.
2. **Per-read TF-call method (open #1 in the decisions sheet).** Three options compared at ERV46
   (`reports/2026-11-03_local_prior.md`, `reports/2026-11-03_abf1_prior_sweep.md`):
   - **Before:** ≤ 0.001.
   - **Raised genome-wide ABF1 prior:** lifts all reads together. ×1500 calls site 2 only; 84-site Rossi AUROC is 0.56,
     against 0.94 for the per-site model.
   - **Local prior:** per-site f from `site_mixture.py` fed in with `--local-prior`. Calls both sites but over-calls:
     94% of reads > 0.5 against f = 0.72.
   - Next, if the local prior is chosen: **Part 2b**, a bound-only ABF1 m6A rate (RoboCOP's 0.084 is pooled over
     bound and unbound fibers). Needs a new TF pkl and tree; user OK required.
3. **Parameters averaged over bound and unbound fibers.** This is the general version of 2b: nucleosome and TF p's
   are pooled fits. Not yet a row in the decisions sheet; the user was asked whether to add it.
4. The rest of the ranked list: `reports/2026-11-03_singlefiber_decisions.md` (14 items), mirrored on Nov 3
   deck slide 3b and in the sheet.

**Figures** (user preference: in the repo):
- `analysis/singlefiber/figures/erv46_2026-11-02/filtnordna/permol_composite_full_*.png` — current decode, with a
  raw-m6A panel in the clustered order;
- `analysis/singlefiber/figures/erv46_localprior_2026-11-03/compare_abf1_3way_erv46.png`;
- `analysis/singlefiber/figures/abf1_prior_sweep_2026-11-03/`.

Clustering (`fig_permol_fibers.py`) uses only the ±125 bp window around ABF1 **site 2** (chrI:62,540–62,790),
and only reads covering 100% of it (122 of 387 at ERV46).

**Artifacts:**
- Nov 3 deck: https://claude.ai/artifact/QJgpxNtLXFFfCdYWLPvYio, source `presentation/lab_nov03/`.
  Slides come from `make_fragment.py` + `update_slides.py` → `make_template.py`, then republish to the same URL.
  The images in `img/` are published files.
- Decisions sheet "FiberCOP model choices": https://claude.ai/artifact/8KkLfcKovncXqmNwFZh5TT. Its rows live in
  the page's database, collection `choices`. `presentation/lab_nov03/decisions_rows.json` mirrors them; keep the two
  in sync.

**Waiting on the user:** the slide 4b "P(open) from 50:50" row is a two-state toy, not decoder output — remove,
relabel, or replace it with a real flip-one-call re-decode.

## 2026-10-31 → 11-03 (DONE): is the background m6A rise real?

**Answer:** a mixture of sequencing runs; Jun25 is about 1.65× May07. Not hypermethylated reads, not the threshold.
See "Decisions 2026-11-02" and `reports/2026-11-02_report.md`. The original plan follows.

**Why.** Refitting background m6A on all 7 merged barcodes raised it from 0.138 (barcode01) to **0.155**:
- Watson 0.1548 / Crick 0.1527;
- `inputs/bg_params_all7.pkl`, fit by `analysis/nucbrogsym/bg_fit.py`.

Barcodes and experiments differ, so first make sure the rise is not caused by:
- a minority of hypermethylated reads (for example fully methylated, failed-chromatin or naked-DNA reads);
- one outlier barcode;
- the merged pileup's lower modkit threshold (0.8105 merged vs 0.8223 barcode01; the threshold is the 10th percentile of call confidence, chosen per pileup, unseeded).

**Facts already established:**
- per-barcode m6A rates are 0.063–0.084 (data-pipeline catalogue, choices sheet);
- there are no read filters at all: no MAPQ, Q-score or length filter (`singlefiber/read_calls.py`, `min_mapq=0`; the modkit pileup uses defaults);
- read-group → barcode mapping is solved in `singlefiber/site_mixture.py` (bc22/23/24 and bc04 under hash-suffixed RG IDs);
- only Mar20 barcode01 has recorded sample conditions (`XL_hia5`).

**To do:**
1. **Per-read m6A rate distribution in linker/background windows**, per barcode:
   - look for a heavy tail or a second mode of near-fully-methylated reads;
   - compute the background rate with and without the top 1/5/10% of reads, and from the median of per-read rates.
2. **Background rate per barcode**, each at the SAME call threshold (0.8105 and 0.8223, from the BAM via `read_calls`). This splits threshold effect vs barcode effect.
3. **Read-quality covariates per read:** MAPQ, read length, mean Q-score, fraction of no-calls (ambiguous). Do hypermethylated reads share them?
4. **Propose read filters**, e.g. MAPQ ≥ 20, min length, a per-read m6A outlier cap. Record the choice in the choices sheet.
5. **Refit everything from one consistent, filtered read set:** background + nucleosome (`nucbrogsym/bg_fit.py`, `nucbrogsym/m6a_fit.py`), and TF footprints (`abf1_reb1_dms_parameter_Fiber-seq_w_binom.py` `compute_Fiber_seq_TFPhisMus`). New filenames only (CLAUDE.md rule 2).
6. **Re-run the checks:**
   - 100-read direction test (`/usr/project/xtmp/nd141/permol_proto_logfull/dirtest100/` or `.../all7/`);
   - ERV46 Chereji dyads (`nucbrogsym/nuc_quality.py`);
   - chrXIV whole-chromosome score (`analysis/nucbrogsym/chrXIV_score/`, runs `analysis/robocop_chrXIV_logfull_*`);
   - per-site model (`/usr/project/xtmp/nd141/tf_compare/`).

   Watch ABF1 aggregate enrichment: it fell 33× → 24× when background moved to all 7 barcodes.

**REVISIT — move everything to 7 barcodes (user decision 2026-10-31).** Every model and script must use
all 7 barcodes; code that works on one barcode gets rewritten. Do this AFTER the read-filter and
background-rate questions above are answered, then refit once and switch all models together. Known
single-barcode spots so far:
- `singlefiber/site_mixture.py:77-79` — `bg_params.pkl`, `nucleosome_params.pkl`, `robocop_train_fiberonly`;
- `singlefiber/sbatch_permol_window_logfull.sh:42-44` — runs the old driver and old trainDir (no sbatch for `_5to3`);
- `nucbrogsym/m6a_fit.py:30-31,105-106` — defaults to the barcode01 pileup and the barcode01 output name;
- `singlefiber/permol_params.py:53-54` — defaults `nucleosome_params.pkl` / `bg_params.pkl`;
- `singlefiber/test_5to3.py:46` — passes the barcode01 pkls;
- tuner default driver `run_split_revfix_seq_maskoff_abi8.py` (`tune_concentrations.py:92`) — its tree loads barcode01
  nuc + bg; the tune75 / logfull trees load barcode01 nuc + `bg_params_open.pkl` (0.249, copied from
  `combined_low_count`, `make_bg_open.py`);
- `pkg/robocop/robocop.py:602,606` — shipped package hardcodes `nucleosome_params.pkl` / `bg_params.pkl`;
- `posterior_fix/fibercop_audit/fiber_model.py:57-58`, `bg_likelihood_arithmetic.py:64`; plotting scripts with
  `BG_METH = 0.1383` (`plot_abf1_*`, `plot_factor_*`); param builders `make_bg_open.py`, `make_bg_tss*.py`,
  `make_low_abf1.py`, `make_params_wide.py`.
- Frozen pkgvar trees (rule 3, never edit — make new ones): 87 of 89 load a barcode01 nucleosome pkl; only
  `permol_seq_maskoff_logfull_nucbrogsym_all7` and `seq_maskoff_logfull_nucbrogsym_all7` are fully all-7.
TF footprints were fit on all 7 barcodes from the shipped pkl's first commit (`1480f85`, 2025-12-14;
`abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:1121-1129` reads the merged pileup). All 1,673 decode
`config.ini` files read the merged pileup. Inventory 2026-10-31.

**Decisions 2026-11-02 (user).**
- Background answer: the 0.138 → 0.155 rise is a mixture of runs (Jun25 bc21-24 ~1.65× May07, in linkers,
  nucleosome cores and genome-wide alike); not threshold, not hypermethylated reads
  (`/usr/project/xtmp/nd141/bg_investigation/{perread,nov03}/`).
- Single-fiber background = ONE pooled value from all 7 barcodes (0.1548 / 0.1527). The user expects to raise it
  by hand later, as in the aggregate runs.
- Aggregate decodes keep the hand-set `bg_params_open.pkl` (0.2489 / 0.2647) for now.
- Read filters: primary + mapped; MAPQ ≥ 20; aligned length ≥ 500 bp (not 1 kb); qs ≥ 10; per-read m6A outlier
  cap on the methylation rate (box-plot rule vs MAD still being weighed; the lower fence is < 0 in every run, so
  neither removes hypo reads).
- **One background file:** `inputs/bg_params_all7.pkl` was byte-identical to `inputs/bg_params_merged.pkl`. The
  original name `bg_params_merged.pkl` is kept as the real file; `bg_params_all7.pkl` is now a symlink to it
  (the copy moved to `inputs/_trash/20261102_bg_all7_dedupe/`). The link stays because the frozen trees
  `pkgvar/{seq,permol_seq}_maskoff_logfull_nucbrogsym_all7` and the `_5to3` driver default read that name.
  New code should use `bg_params_merged.pkl`.

**Done 2026-11-02 — filtered refit, new single-fiber defaults.**
- **Read set:** `/usr/project/xtmp/nd141/filtered_refit/`, 719,979 of 1,189,672 mapped primary reads kept.
  - Filters: nuclear chromosomes only, MAPQ ≥ 20, aligned length ≥ 500, qs ≥ 10.
  - Outlier cap: one pooled median ± 4·1.4826·MAD on the per-read m6A rate, giving bounds [−0.075, 0.204]. It removed 13,656 hyper reads and 0 hypo reads.
  - Threshold: modkit `sample-probs` on the filtered BAM gives 0.8105469 (3 runs, identical).
- **New fits** (each checked first: on the unfiltered pileup it reproduces the old file bit-exactly):

  | file | Watson | Crick | change from old |
  |---|---|---|---|
  | `inputs/bg_params_filt.pkl` | 0.1565 | 0.1545 | +1.1% |
  | `inputs/nucleosome_params_brogaard_sym_filt.pkl` | | | ×0.973, correlation 0.9998 |
  | `inputs/all_TFs_1000pealVal_params_pseudo_filt.pkl` | | | every position within 0.017 |

  - The TF file was made with the new `nucbrogsym/tf_fit.py`.
- **Driver** `_5to3` defaults to:
  - tree `pkgvar/permol_seq_maskoff_logfull_nucbrogsym_filt`, the three `_filt` pkls, the read filters and the keep list;
  - the docstring gives the command that reproduces the 10-30 runs.
- **Sbatch:** `singlefiber/sbatch_permol_window_logfull_5to3.sh`.
- **ERV46 decode:** `/usr/project/xtmp/nd141/permol_proto_logfull/filt/erv46_5to3` (387 reads; its figures moved 2026-11-03 to `analysis/singlefiber/figures/_trash/20261103/erv46_2026-11-02/filt_with_rdna/`; rDNA-free run figures: `analysis/singlefiber/figures/erv46_2026-11-02/filtnordna/permol_composite{,_full}_erv46_5to3_nordna.png`).
  - With the same 387 reads, new params vs old: nucleosome occupancy correlation 0.99993; Chereji check recall 0.75 (PASS) in both.
  - Per-read ABF1 maximum at the two sites is ≤ 0.0011 in every run, i.e. still no per-read TF calls.

**Done 2026-11-02 (later) — rDNA excluded; these are the current defaults.**
- **Read set:** reads overlapping chrXII:451,418-468,929 are removed as their own filter step, before the MAD fence
  (`/usr/project/xtmp/nd141/filtered_refit_nordna/`).
  - Kept: 667,502 of 1,189,672 reads (56.1%).
  - Fence: [−0.070, 0.194], removing 9,270 hypermethylated reads.
  - Threshold unchanged at 0.8105469.
- **Refit:** `inputs/*_filtnordna.pkl`; everything moved by less than 1.5% from `_filt`.
  - background 0.15635 / 0.15437
- **Tree** `pkgvar/permol_seq_maskoff_logfull_nucbrogsym_filtnordna` is now the `_5to3` driver default, with keep list
  `filtered_refit_nordna/keep_reads.txt`.
- **ERV46 decode:** `/usr/project/xtmp/nd141/permol_proto_logfull/filt/erv46_5to3_nordna`.
  - Same 387 reads.
  - Read-averaged tracks within 8e-4 of the `_filt` run.
- **Why exclude rDNA:**
  - 17% of reads, about 19× genome depth;
  - MAPQ ≥ 20 there does not mean a unique copy;
  - a higher m6A rate (0.107 vs 0.065);
  - 43.5% of the old fence removals were rDNA reads;
  - no fitting window touches it.
  - Evidence: `/usr/project/xtmp/nd141/rdna_check/`.

**Revisit later (noted 2026-11-02):**
- Aggregate log-space driver (`run_split_seq_maskoff_logfull_nucbrogsym_all7.py`) still starts motif-first: it uses the
  trainDir's `initial_probs` (310 non-zero states, `init[0]` 0.933). End-anywhere is on (end_probs all 1;
  `algo_log.c:197-198`). Single-fiber starts anywhere (`stationary_initial`, `_5to3.py:222-255`). User: port later.
- `site_mixture.py`: move to the filtered params and plumb a bg file into `em()` (:284). Rerun into a new output dir,
  because the runner would otherwise overwrite `tf_compare/mixture`.
- `test_5to3.py:45-46` pins the old barcode01 model.
- Latent bug in the never-run branch at `robocop.py:932` (`motif_starts` undefined there).
- Latent (upstream): `end_probs` is applied in BOTH the forward pass (`algo_log.c:148-149`, `algo.c:111`) and the
  backward init (`algo_log.c:197-198`). With end-anywhere it is all ones, so this has no effect today. A
  non-uniform end vector would be counted twice; drop it from one of the two passes. Verified 2026-11-02
  (`/usr/project/xtmp/nd141/startend_check/end/`).
- Other `read_calls` callers (plots, `site_mixture`, `checks`, `pooled_emission_check*`) still read unfiltered reads.
- **(user, 2026-11-02) Per-read TF calls: per-site bound fraction (`site_mixture.py`) vs raising the TF entry
  transition.**
  - The ×138 ABF1 prior test called 2 of 84 sites. Revisit both approaches side by side on the filtered model.
- **(user, 2026-11-02) Methylated / unmethylated / no-call.**
  - Today each A on the read's strand is one of three codes (`read_calls.py`: 1, 0, −1).
  - The emission is a Bernoulli on calls 1 and 0 only: +log p for 1, +log(1−p) for 0 (`robocop.py:1581-1582` in the
    `_filt` tree). A no-call, a non-target base or an uncovered base adds 0, i.e. it is skipped
    (`:1583`, exact mode).
  - No-calls are 2–4× more common in open DNA (`/usr/project/xtmp/nd141/emission_audit/`), so skipping them throws
    away signal.
  - Revisit a 3-outcome (categorical) emission and its effect on single-fiber reads.
  - **Measured 2026-11-03** (filtered rDNA-free pileup; `/usr/project/xtmp/nd141/nocall_rates/nocall_rates.json`).
    The no-call fraction is N_fail / (N_valid + N_fail) on A calls:

    | region | no-call fraction |
    |---|---|
    | linkers | 14.6% |
    | Brogaard nucleosome cores | 7.3% |
    | Chereji cores | 8.2% |
    | genome | 9.0% |

    - So no-calls depend on the state, and treating them as "no evidence" is not justified.
    - One no-call would favour open about 2× (larger than an unmethylated A's 1.16–1.26× toward nucleosome).
    - The PI flagged this. Slide 4b of the Nov 3 deck shows the math.

**Then (open user decisions):**
- rerun the per-site model with the new defaults and extend to REB1;
- the lost Chereji dyad 61,336 (not investigated);
- `unknown` / nucleosome prior shares (29% / 11.5%; revisit later);
- optional 3-outcome emission.

## Update 2026-10-26 → 10-30 (read this first)

**New defaults** in `analysis/singlefiber/run_permol_window_logfull_5to3.py` (the old driver `run_permol_window_logfull.py` is unchanged). The old setup is reproducible with explicit flags; the command is in the docstring at :57-59.

| choice | default now | evidence |
|---|---|---|
| direction | each read decoded 5'→3' on its own strand (`--direction 5to3`) | user decision |
| start rule | start anywhere = long-run vector (`--start any`); end anywhere is upstream since `8debe63` | 100-read left↔right test: 0.75 → 0.21 (old model) → 0.067 (new) |
| nucleosome sequence model | Brogaard top-2000, Watson + RC pooled, centre = BED start (`robocop_train_fiberonly_nucbrogsym`) | asymmetry 0.40 → 0.058 log10; chrXIV Chereji recall unchanged 0.808, dyad error 8 bp |
| m6A rates | background + nucleosome refit on all 7 barcodes (`bg_params_all7.pkl`, `nucleosome_params_brogaard_sym_all7.pkl`; pkgvar `permol_seq_maskoff_logfull_nucbrogsym_all7`) | bg p 0.138 → 0.155; TF footprints unchanged |
| sequence | reference always; positions with ≥10 reads and >50% one other base are FLAGGED only (`consensus_seq.py`; genome scan 11,433 flags, `/usr/project/xtmp/nd141/consensus_scan/`); flags drawn in the posterior viewer | the strain differs from S288C at ~1/kb |
| trimming | not implemented (post-decode only, if ever) | user note |

**TF calls per read: use the per-site fraction-bound model** (`analysis/singlefiber/site_mixture.py`; results `/usr/project/xtmp/nd141/tf_compare/`).
- It fits f (fraction of reads bound) per site from all reads, then uses f as each read's prior.
- Across 42 MacIsaac ABF1 sites + 42 PWM look-alikes on chrI+chrXIV: Rossi AUROC 0.96; barcode hold-out Spearman 0.80; controls ≈ 0.
- Raising ABF1's prior ×138 (`robocop_train_fiberonly_abf1x138`) calls almost no reads (2 of 84 sites, one false). Don't use it for per-read work.

**Why one read can't call a TF** (worked out in full in the artifact *single-fiberCOP math*, https://claude.ai/artifact/NszXsMkgqaL8N1hMncbMbN):
- the aggregate emission equals the sum of the reads' Bernoulli emissions + log C(n,k);
- so a single read has ~1/n of the evidence against the same −16.2-nat ABF1 entry cost.

**Open / noted:**
- Chereji dyad 61,336 is lost with the Brogaard model at ERV46 (cause not investigated).
- The new all-7 m6A raises chrXIV ABF1 aggregate calls 327 → 463 (enrichment 33× → 24×); the sequence model alone leaves ABF1 unchanged. The TF footprints were already fit on the merged 7 barcodes (generator `analysis/abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:1123`), so the old setup mixed barcode01 background with merged-barcode footprints.
- `unknown` takes 29% and nucleosome only 11.5% of bases in the prior (revisit later).
- 3-outcome emission (no-calls are 2–4× more common in open DNA) is an optional upgrade (`/usr/project/xtmp/nd141/emission_audit/`).
- All model choices are tracked in the shared sheet https://claude.ai/artifact/8KkLfcKovncXqmNwFZh5TT.

## User decisions (2026-10-25)

| decision | value |
|---|---|
| decoder | **log space** (`librobocop_logfull.so`, `ROBOCOP_LOG_EMISSION=exact`); the linear path is kept only as a record |
| layers | **fiber + sequence first**; MNase later (not dropped) |
| test locus | **ERV46**, chrI:60,001-65,000, MacIsaac ABF1 sites chrI:61,163-61,177 and 62,657-62,671 |
| footprint fitting data | all 7 merged barcodes are fine (for accuracy numbers, cross-fit: fit on 6, test on the 7th) |
| still open | how to estimate per-site bound fraction (see "Next steps"); which factors after ABF1; FiberHMM as official baseline? |

## What exists and works

| piece | path |
|---|---|
| per-read driver (log space) | `analysis/singlefiber/run_permol_window_logfull.py` (+ `sbatch_permol_window_logfull.sh`) |
| package copy | `analysis/pkgvar/permol_seq_maskoff_logfull/` (copy of `permol_seq_maskoff` + the three logfull edits) |
| aggregate reference | `analysis/pkgvar/seq_maskoff_logfull/`, driver `analysis/run_split_seq_maskoff_logfull.py` |
| read-level m6A reader | `analysis/singlefiber/read_calls.py` (BAM MM/ML -> per-read 0/1/no-call; threshold 0.8105469) |
| checks | `singlefiber/checks.py`, `pooled_emission_check_logfull.py`, `test_logfull_permol.py`, `compare_permol_runs.py` |
| evidence-ceiling measurement | `analysis/singlefiber_review/site_evidence_ceiling.py` |
| outputs | `/usr/project/xtmp/nd141/permol_proto_logfull/` (log space), `/usr/project/xtmp/nd141/permol_proto/` (old linear, 10 arms) |
| old linear prototype (record only) | `analysis/singlefiber/run_permol_window.py`, `pkgvar/permol_seq_maskoff/` (pre-fix-B .so, 1e-30 floor) |

Validation of the log-space port (all on the ERV46 window, 439 reads, arm A, efficiency off):
- per-read posteriors match the old linear prototype to 3e-8 (0 cells off by > 0.01);
- all reads pooled into one chain reproduce the log-space aggregate decode to 9e-14;
- prototype checks 1-4 identical; 7/7 unit tests pass (`permol_proto_logfull/unit_tests_{exact,floored}.json`).

How missing data is handled (no DBL_MIN anywhere):

| case | value | where |
|---|---|---|
| no information (uncovered base, non-A on the read's strand, no-call, silent state) | 0.0 = log 1 | `pkgvar/permol_seq_maskoff_logfull/robocop/robocop.py:300, 1569-1571` |
| m6A call | +log(p) methylated, +log(1-p) unmethylated; p = 0 or 1 refused | `robocop.py:1563-1565` |
| forbidden (PWM zero, mask) | -inf in that cell | `robocop.py:335` |
| impossible (every state -inf at a position) | run stops: Python assert + C rc -2 | `robocop.py:975-978, 1038-1061`; `pkg/robocop/algo_log.c:138, 209` |

## The model as it stands

| part | value (source) |
|---|---|
| data | merged 7-barcode BAM `/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/combine_sequencing_runs/merged_Mar20_barcode01_Jun25_barcode21-24_May07_barcode03-04.bam`; one strand per read (forward -> Watson A, reverse -> Crick T) |
| trainDir | `robocop_train_fiberonly` (pristine; lambda untuned) |
| states | 3,485: 1 background, 2,798 TF states, 531-state nucleosome block, 155 silent junction states |
| TFs | all 154 live (153 motifs + `unknown`), forward and reverse; lengths 5-20 bp, median 9; ABF1 14, REB1 8, unknown 10 |
| chance per base of starting a feature | background 0.933, unknown 0.050, nucleosome 0.00136, REB1 1.7e-5, **ABF1 1.8e-7** (all TFs 0.065) |
| layers | 0 = sequence PWM (on); 1-4 MNase (off); 5/6 = per-read Bernoulli on Watson A / Crick T |
| m6A p per state | background 0.138 (`inputs/bg_params.pkl`); nucleosome 147-position profile 0.024-0.14; ABF1 mean 0.084/0.080; REB1 0.22/0.24; all other motifs and `unknown` one flat fallback 0.249/0.265 (`inputs/all_TFs_1000pealVal_params_pseudo.pkl`, 13 entries) |

Known issue: the flat fallback (0.249) is above background (0.138), so most TF states claim *more*
methylation than open DNA.

## What failed, and why

- Nucleosomes per read work: Chereji +1/-1 dyads 4/4 within 20 bp; 434 of 439 reads get distinct paths.
- TF calls per read do not: 0 reads reach ABF1 >= 0.10 at either site in any of the 10 arms (best 0.033).
- Cause is the data, not the code (`singlefiber_review/site_evidence_ceiling.py`):

| site | reads | informative calls in motif (median) | reads with 0 m6A in motif | P(no m6A in a 14 bp open window) |
|---|---|---|---|---|
| 61,163-61,177 | 146 | 5 | 0.801 | 0.404 |
| 62,657-62,671 | 144 | 5 | 0.75 | 0.508 |

  So one read's m6A favours "bound" by at most ~2-4x. The HMM's posterior is roughly
  start chance (1.8e-7) x sequence fit x m6A evidence, so the m6A can never move it far.
- Other problems: the 1.0 site-vs-control AUROC comes from sequence, not m6A; poly-dA:dT tracts are
  hypomethylated even when open (unmodelled); the ABF1 footprint is too narrow (signal sits in the
  hyperaccessible flanks; motif is 18 bp).

## The idea for next: estimate each site's bound fraction from its own reads

The HMM has one start chance per TF for the whole genome; it cannot learn that a particular site is bound
on most fibers. So split the job in two:
1. **Per site, from all reads together:** what fraction f of fibers is bound. Example, site 1: 80% of reads
   have no m6A in the motif, an unbound open window has none 40% of the time by chance, so
   0.80 = f + (1-f) x 0.40 gives f ~ 0.67 (rough; ignores A/T content). This is the "how many fibers are
   occupied" answer and is well determined with ~146 reads.
2. **Per read:** weigh the read against f instead of 1.8e-7. Computed with ABF1's mean m6A rate (0.084) vs
   the open-window rate at site 1 (0.272), 5 informative calls, f = 0.67:

   | methylated calls of 5 | likelihood ratio bound:free | posterior bound |
   |---|---|---|
   | 0 | 3.15 | 0.86 |
   | 1 | 0.77 | 0.61 |
   | 2 | 0.19 | 0.28 |
   | 3 | 0.05 | 0.09 |

   Per read the answer stays probabilistic; per site the fraction is solid.

Do this as one mixture model fitted per site (EM over the reads), with bound / free / nucleosome
alternatives, rather than plugging in an outside number.

## Next steps (in order)

| # | do | start from | validate by |
|---|---|---|---|
| 1 | Per-site mixture model at ERV46 (bound / free / nucleosome; f per site by EM; per-read posterior) | `singlefiber/read_calls.py`, `site_evidence_ceiling.py` | f stable across barcodes; reads called bound are nucleosome-free with phased +1 |
| 2 | Extend the evidence measurement to all MacIsaac ABF1/REB1 sites on chrI and chrXIV, split by A/T run length; simulate per-read ROC at real call density | `site_evidence_ceiling.py` | per-site calls in motif, open-window rate |
| 3 | Fit per-read bound/unbound footprint shapes (motif +/- 50 bp, per strand) by EM at ChIP-supported sites | `singlefiber/permol_params.py` (arm B logic) | cross-fit over barcodes; bound profile low in motif, high at flanks |
| 4 | Add an A/T-context term so unmethylated poly-A/T stops looking bound | memory note polyAT-hypomethylation-artifact; external `kmer_hsmm` | A/T-matched unbound sites stop scoring as bound |
| 5 | Validate: motif-matched unbound controls (matched on A/T run and call count); per-site fraction vs Rossi ChIP-exo (held out); FiberHMM per-fiber calls as baseline | FiberHMM outputs `/usr/project/xtmp/nd141/projects/Fiber_seq/Fiber_tools_claude/fiberhmm_run/outputs/fire/` | |
| 6 | Only if 1-5 work: full per-read HMM with a per-position TF start chance taken from step 1 | `pkgvar/permol_seq_maskoff_logfull` | matches step 1 at sites |

Not yet re-run in log space: arms B, C and efficiency-on (`--array=1-9` on `sbatch_permol_window_logfull.sh`).

## Related, outside the repo
- FiberHMM + FIRE run on 13 BAMs: `/usr/project/xtmp/nd141/projects/Fiber_seq/Fiber_tools_claude/fiberhmm_run/`
- per-fiber HSMM with a k-mer (sequence-context) emission: `/usr/project/xtmp/nd141/projects/Fiber_seq/hmm_for_clustering_claude_code/`

## Local prior, 2026-11-03

Built (plan `let-s-go-with-mad-snazzy-lollipop`; full report `reports/2026-11-03_local_prior.md`):
- **Part 1.** `singlefiber/site_mixture.py` moved to the current model (nucbrogsym trainDir, filtnordna nuc/bg
  pkls, bg floor from `bg_params_filtnordna.pkl`, driver read filters + keep list + thresh 0.8105469) and gained
  `heldout_run`. Runner `/usr/project/xtmp/nd141/tf_compare/scripts/run_site_mixture_filtnordna.py` →
  `tf_compare/mixture_filtnordna_pb0p02/sites_f.tsv` (86 sites). ERV46: f = 0.715 (site 1), 0.615 (site 2).
- **Part 2.** `singlefiber/local_prior.py` (weight w on both ABF1 first states at the motif-start row, layer 1,
  as −log w on the other states), `--local-prior <sites_w.tsv>` in `run_permol_window_logfull_5to3.py`
  (default off; off = bit-identical to `filt/erv46_5to3_nordna`), `singlefiber/calibrate_local_prior.py`
  (bisects w so prior-only P(ABF1) = f; one table per held-out run), `singlefiber/test_local_prior.py`.
- **Result at ERV46** (`/usr/project/xtmp/nd141/permol_proto_logfull/compare_localprior/compare_summary.json`):
  the local prior lifts per-read P(ABF1) at both MacIsaac sites to median ~0.8 (site 1 0.81, site 2 0.79) with no
  off-site calls. The raised global prior (×1500) lifts site 2 only (0.57); site 1 stays 0.0097, and a non-site spot
  at 61,660 reaches 0.50. Per-read spread with the local prior is SD 0.13 / 0.12, against site_mixture's 0.33 at
  the same sites. RoboCOP's per-read m6A evidence is still weaker than the per-site model's (Part 2b). Building a
  bound-only ABF1 rate needs the user's decision; it is not built.
- Figures: `analysis/singlefiber/figures/erv46_localprior_2026-11-03/`.
