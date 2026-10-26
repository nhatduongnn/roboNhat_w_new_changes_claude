# Single-fiber FiberCOP — handoff and plan (2026-10-26)

**Start here for the single-fiber thread.** Read `CLAUDE.md` first (environment, rules). This file says
what exists, what was decided, what the model is, what failed and why, and what to do next.
Detail: `analysis/singlefiber_review/REPORT.md` (review of everything done), `analysis/singlefiber/README.md`
and `analysis/singlefiber/LOGFULL_PORT.md` (the code and the log-space port).

## Goal

Call TF binding (and nucleosomes) on **individual Fiber-seq reads**, using each read's own m6A plus the
DNA sequence (PWM layer) to make up for how little one read carries. The end question is how many fibers
are bound at a site, and which ones.

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
