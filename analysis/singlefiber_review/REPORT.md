# Single-fiber FiberCOP: what exists and what to do next (2026-10-25)

Review by a subagent; arm results (`/usr/project/xtmp/nd141/permol_proto/checks_*.json`), the stale README line and
the per-read evidence ceiling (`site_evidence_ceiling.py`, re-run 2026-10-25) re-checked against the files.

## Summary
- A working per-read prototype exists: `analysis/singlefiber/` + frozen tree `analysis/pkgvar/permol_seq_maskoff/`.
  It decodes each read alone with the full 3,485-state HMM, sequence (PWM) layer on, Bernoulli m6A term per A.
- Ran 2026-09-28: 10 arms over chrI:60,001-65,000, 439 molecules. Outputs in `/usr/project/xtmp/nd141/permol_proto/`.
- Plumbing is right: the read reader matches the modkit pileup exactly (`reader_validation.json`); pooling all reads'
  Bernoulli terms into one chain reproduces the aggregate decode (r = 0.99999996).
- Nucleosomes per read work (Chereji +1/-1 dyads 4/4 within 20 bp; 434 of 439 paths distinct).
- TF calls per read do not: 0 reads reach ABF1 >= 0.10 at either MacIsaac site in any arm; best read 0.033.
- Cause is the data: one read covers one strand, ~5 informative bases inside a 14 bp motif, and 40-51% of open 14 bp
  windows have no m6A anyway. So one read's motif evidence is at most a ~2-4x likelihood ratio, against a global ABF1
  entry prior of ~5e-7.
- It still runs on the old numerics (pre-fix-B .so md5 6a0724bf with DBL_MIN padding, 1e-30 floor, linear layer product).

## Results across arms (`checks_armC_effoff.json` per_arm; site 1 / site 2)
| arm | efficiency | r nucleosome vs aggregate | ABF1 max at sites | reads > 0.10 |
|---|---|---|---|---|
| A (shipped p) | off / on | 0.444 / 0.472 | 1.1e-3 / 6.1e-3 (site 2) | 0 / 0 |
| B pi 0.3 / 0.5 / 0.7 | on | 0.472 | 0.033 / 0.018 / 0.010 (site 2) | 0 / 0 |
| C flat 0.0243 | off / on | 0.443 / 0.471 | 1.5e-3 / 0.022 (site 2) | 0 / 0 |

Per-read evidence ceiling (`site_evidence_ceiling.py`, chrI:60,001-65,000):

| site | reads | informative calls in motif (median) | reads with 0 m6A in motif | open-region m6A rate per call | P(no m6A in a 14 bp open window) |
|---|---|---|---|---|---|
| site 1 (61,163-61,177) | 146 | 5 | 0.801 | 0.272 | 0.404 |
| site 2 (62,657-62,671) | 144 | 5 | 0.75 | 0.2205 | 0.508 |

## Problems in the existing work
| # | problem | consequence |
|---|---|---|
| P1 | Information ceiling: one strand per read (no duplex), ~5 calls in a 14 bp motif | best-case likelihood ratio from the motif alone ~2-4x |
| P2 | Global prior (TF entry ~5e-7; runs used untuned `robocop_train_fiberonly`) | posterior reverts to the prior; per-read average nucleosome 0.56 vs aggregate 0.86 |
| P3 | Check 3's AUROC 1.0 comes from sequence, not fiber (controls have no ABF1 motif; gap ~1e6) | not evidence of per-read TF detection |
| P4 | Check 1 mis-specified (mean of per-read posteriors != pooled posterior) | use the pooled check instead |
| P5 | Old numerics (DBL_MIN padding, floors) | row sums fine (0.9949-0.9999); hidden DBL_MIN damage unverified |
| P6 | Poly-dA:dT hypomethylation not modelled | an unmethylated A/T run on one read looks like a footprint |
| P7 | Footprint too narrow; evidence sits in the hyperaccessible flanks; ABF1 motif is 18 bp | the 14 bp state ignores most of a read's evidence |
| P8 | 9.2% of calls ambiguous and dropped | minor |
| P9 | One 5 kb window, two sites, one TF | accuracy at scale unknown |

Stale docs: `singlefiber/README.md:245` says "Arms B and C were not run" (they were); `whattodo.md` Tier 3 #8 says
"Not started".

## Recommended approach
A ~2-4x per-read likelihood ratio can update a good prior but cannot discover a site. Split the job:
1. Where and how often a factor binds: the population (log-space aggregate decode, PWM, ChIP).
2. Which reads are bound there: a per-read test with prior = the site's population occupancy, evidence = the read's m6A
   over motif + flanks, against both "free" and "nucleosome" alternatives.

Example: site 60% bound, read with 5 unmethylated motif calls (LR ~3.5) -> posterior 0.84; with the global 5e-7 prior
the same read gives ~2e-6, which is what the prototype saw. Rough occupancy from the ceiling table (unverified, ignores
A/T content): ~0.67 and ~0.49 at the two sites.

| option | verdict |
|---|---|
| O1 prototype ported to log space | correctness reference only; still capped by P1/P2 |
| O2 full per-read HMM with per-position TF entry prior from the population posterior | long-term target |
| O3 site-local per-read test (motif +/- ~50 bp, prior = site occupancy) | do first: fast, transparent, tests whether the signal exists |
| O4 FiberHMM / FIRE footprints (already run on 13 BAMs outside the repo) | baseline; no TF identity |
| O5 external per-read HSMM with k-mer emission | borrow its sequence-context term for P6 |

Validation (no per-read ground truth): simulation at real call density; motif-matched unbound controls (matched on A/T
run length and call count); per-site fraction bound vs Rossi ChIP-exo (held out); co-occupancy (bound read
nucleosome-free with phased +1); held-out barcodes; FiberHMM baseline.

## Ordered next steps
| # | build / run | start from | decision needed |
|---|---|---|---|
| 1 | log-space per-read tree `pkgvar/permol_seq_maskoff_logfull` (log emission, no floor, layers added) | `pkgvar/seq_maskoff_tune75_logfull` + permol code | open decision 4 |
| 2 | evidence ceiling for all MacIsaac ABF1/REB1 sites on chrI/chrXIV, split by A/T run length; per-read ROC simulation | `site_evidence_ceiling.py`, `singlefiber/read_calls.py` | none (read-only) |
| 3 | fit per-read bound/unbound footprints (motif +/- 50 bp, per strand) by EM at ChIP-supported sites | `singlefiber/permol_params.py` (arm B) | training sites; factors |
| 4 | site-conditioned per-read caller (O3) with a nucleosome alternative | new script in `singlefiber/` | source of site occupancy |
| 5 | A/T-context term in the per-read emission | memory polyAT-hypomethylation-artifact; external kmer_hsmm | run length vs k-mer |
| 6 | only if O3 works: full per-read HMM with per-position TF prior (O2) | step-1 tree | whether to change C |
| 7 | fix stale docs | — | OK to edit |

## Open questions for the user
1. Accept per-read calling at known/candidate sites (de novo per-read TF calls at 14 bp look out of reach)?
2. Port single-fiber to log space now (open decision 4)?
3. Site occupancy prior from which source: an aggregate decode (which tuned model?) or Rossi (then Rossi is no longer held out)?
4. Which factors first (ABF1, REB1)?
5. FiberHMM as the official baseline?
6. Is the MNase step of the phased plan dropped?
7. Fit footprints on all 7 merged barcodes or hold some out?

## Paths
- Prototype `analysis/singlefiber/`; tree `analysis/pkgvar/permol_seq_maskoff/`; outputs `/usr/project/xtmp/nd141/permol_proto/`
- External: `/usr/project/xtmp/nd141/projects/Fiber_seq/Fiber_tools_claude/fiberhmm_run/` (FiberHMM + FIRE),
  `/usr/project/xtmp/nd141/projects/Fiber_seq/hmm_for_clustering_claude_code/` (per-fiber HSMM, k-mer emission)
