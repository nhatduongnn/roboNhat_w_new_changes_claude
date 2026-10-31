# Single-fiber FiberCOP — handoff and plan (2026-10-26)

**Start here for the single-fiber thread.** Read `CLAUDE.md` first (environment, rules). This file says
what exists, what was decided, what the model is, what failed and why, and what to do next.
Detail: `analysis/singlefiber_review/REPORT.md` (review of everything done), `analysis/singlefiber/README.md`
and `analysis/singlefiber/LOGFULL_PORT.md` (the code and the log-space port).

## Goal

Call TF binding (and nucleosomes) on **individual Fiber-seq reads**, using each read's own m6A plus the
DNA sequence (PWM layer) to make up for how little one read carries. The end question is how many fibers
are bound at a site, and which ones.

## Next session (2026-10-31): is the background m6A rise real?

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
