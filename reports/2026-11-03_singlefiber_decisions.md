# Single-fiber decisions still open, 2026-11-03

This is a read-only review of the single-fiber (per-read) code and docs as they stand on 2026-11-03. It lists every choice that still needs the user, ranked by how much it blocks useful per-read results. Nothing was edited, run on Slurm, or committed. Every number is from a named file or a command run during this review. "Unverified" marks the rest.

---

## Summary (ranked)

| # | decision | recommendation | blocks |
|---|---|---|---|
| 1 | **Local occupancy: how each read gets a site-specific TF prior** (estimator, where it enters, circularity, sites, factors, nucleosomes) | Estimate f per site with the per-site EM mixture. Use it two ways: (a) now, as the per-read answer at known sites; (b) later, as a per-position weight inside the per-read HMM, added through the existing emission hook (no C change). Cross-fit f by **run** for validation. Candidate sites from a genome-wide PWM scan; MacIsaac and Rossi used only as labels. Do not do this for nucleosomes. | Every per-read TF call. Today per-read ABF1 is ≤ 0.0011 everywhere. |
| 2 | **Approve moving `site_mixture.py` (and other `read_calls` users) to the `_filtnordna` defaults**, with a new output dir | Yes. It is mechanical, but it changes the reads and params under the 0.96 AUROC result, so it needs your OK (rule 7). | Rerunning #1 on the current model. |
| 3 | **Validation design**: controls and held-out labels | Add accessibility-matched controls: PWM hits in open DNA with no ChIP. Today's controls are mostly nucleosomal (median f_nuc 0.886), so AUROC 0.96 partly measures "open vs closed". Keep Rossi held out. | Trusting any result from #1. |
| 4 | **Run labelling difference** (Jun25 ≈ 1.65× May07) | Try one scale factor per run (3–7 numbers) before per-read `--eff 1`. Never fit f across runs without it. | Per-read posteriors are biased by run (mean post_bound 0.36 May07 vs 0.45 Jun25). |
| 5 | **Background value + TF fallback** (0.156 pooled; hand-raise like 0.249?; fallback 0.25 for 141 motifs + `unknown`) | Decide these together. Keep 0.156 for now. Either set the fallback to background, or treat `unknown` openly as the "open DNA" state. | What "free" means in the HMM; the size of `unknown`. |
| 6 | **Prior shares and concentrations** (`unknown` 29%, nucleosome 11.5%, pristine untuned λ) | Fix the nucleosome share first. It drives per-read prior reversion (read-average nucleosome 0.58). | Calibrated per-read nucleosome occupancy, and the HMM route in #1(b). |
| 7 | **ABF1 footprint for per-read work** (motif 14 vs 18 bp, flank width, fixed p_bound 0.02) | In the per-site model, give the bound class its own elevated flank rate (offsets about −30..−14 and +16..+43). Keep p_bound fixed and report how sensitive results are to it. | How sharp per-read posteriors get (today 0 of 42 MacIsaac sites have a median read > 0.9). |
| 8 | **Poly-dA:dT hypomethylation** | Add a run-length term to the "free" class in the per-site model first, then the HMM. | False "bound" calls at A/T-rich sites. |
| 9 | **No-call emission and dorado p calibration** | Do a 3-outcome emission after #1–#4. No control samples were found for calibration, so ask whether any exist. | Small gain in open DNA. Not blocking. |
| 10 | **Which factor after ABF1** | REB1, but check its footprint sign first: the fitted REB1 m6A (0.22/0.24) is *above* background (0.156). | Extending #1. |
| 11 | **Scale, output and viewer** | Next: all MacIsaac ABF1/REB1 sites genome-wide with the per-site model (cheap). Genome-scale HMM only after #1(b) works. Estimated ~56 CPU-h (unverified). | Genome-scale per-read results. |
| 12 | **Trimming and consensus flags** | Keep no trim. Decode the whole read span, not window-clipped pieces. Mark TF calls on flagged positions. | Minor. |
| 13 | **Aggregate model**: move to `_filtnordna` pileup + params and start anywhere? | Yes, so the two decoders share one model. This needs a new aggregate tree (rule 3). | Comparing per-read with aggregate on equal terms. |
| 14 | **Code inconsistencies** found in review | Fix list in §14. `test_5to3.py` will fail on current defaults. | Regression testing. |

---

## 1. Local occupancy: a site-specific TF prior for each read

**The problem, in one line.** One read carries about a 2–4× likelihood ratio at an ABF1 motif. The HMM's genome-wide ABF1 prior is about 1e-6 per base, so the posterior can never move. Measured: per-read ABF1 max at the two ERV46 sites ≤ 0.0011 (`/usr/project/xtmp/nd141/permol_proto_logfull/filt/erv46_5to3_nordna/permol.npz`, `per_mol[..., Abf1_murphy].max()` = 0.0011244). Stationary ABF1 share is 1.45e-6 of bases. `unknown` is 0.289 and nucleosome is 0.115 (recomputed from `robocop_train_fiberonly_nucbrogsym/HMMconfig.pkl` in this review).

**Why raising the global prior is not the answer.** The ×138 ABF1 trainDir lifts every read at a site by the same amount; it does not separate reads.
- Per-site SD of per-read posterior: ×138 = 0.00084; per-site mixture = 0.076 (`/usr/project/xtmp/nd141/tf_compare/compare_summary_mixture_pb0p02.json`, `median_per_site_sd_*`).
- At ERV46 site 2, ×138 puts every read near 0.10 (median 0.102, SD 0.031, max 0.127) (`tf_compare/perread/erv46_perread_summary.json`).
- Reads > 0.5, Rossi AUROC: ×138 0.503; mixture 0.938 (same compare file).
- Note: the ×138 test used the old `robocop_train_fiberonly` model, not the current one (`erv46_perread_summary.json` `traindir`).

**Sub-decisions:**

### 1a. Which estimator of local occupancy f

| option | what it is | evidence | verdict |
|---|---|---|---|
| **Per-site EM mixture** (`site_mixture.py`) | f = fraction of reads in the "bound" class, fitted from that site's ~160 reads (bound / free / nucleosome) | MacIsaac median f 0.449, PWM-only 1e-10; Rossi Spearman 0.82, AUROC 0.96 (`tf_compare/mixture_pb0p02/summary.json`) | **Recommend.** It is a real fraction of fibers, and it comes from the same data type. |
| Aggregate RoboCOP posterior at the site | ABF1 posterior from the pooled decode | Aggregate ABF1 site accuracy is poor (F1 0.03–0.05, λ is a count knob; CLAUDE.md 2026-09-12). Pooling ~150 reads into one chain gives one consensus path, not a fraction of fibers. | Not an estimate of f. Use it only as a candidate filter, if at all. That it is not a fraction is reasoning; unverified numerically. |
| ChIP (Rossi ChIP-exo, MacIsaac) | external occupancy score | Rossi is the held-out label. Not a fraction. Only for factors that were profiled. | Use for validation only. Using it as the prior spends the held-out set. |

### 1b. Where f enters

| option | how | cost | verdict |
|---|---|---|---|
| (i) Per-site Bayes outside the HMM | per-read posterior = EM responsibility with f as the prior (`site_mixture.fit_site`, `per_read.post_bound`) | exists and runs | **Use now** at known sites. Caveat: its nucleosome class is crude (whole-window only; no partial nucleosome, `site_mixture.py:39`). |
| (ii) Per-position TF weight inside the per-read HMM (plan "Next steps" #6) | multiply the emission of the TF's **first** state at the motif start by w/w_global. That state is entered only from the junction, so this equals a position-specific entry transition. In log space it is one addition, through the existing `emission_hook` (`run_permol_window_logfull_5to3.py:388-391, 469-471`). | No C change. Flipped (Crick 5'→3') reads need the mirrored state (`state_mirror`). The equivalence is my derivation; **unverified**. Check it on a toy against a modified transition matrix. | **Later**, after (i) is validated. It gives consistent nucleosome / other-factor competition. |
| (iii) Raise the genome-wide entry (λ) | concentration | see ×138 above | **Reject** for per-read work (already the 10-30 decision). |

Extra decision for (ii): **calibration.** A weight w is not an occupancy, because nucleosome and `unknown` compete. Pick w so that the HMM's prior occupancy at the site, with all calls set to no-call (sequence only), equals f. One forward-backward per bisection step. Unit test T2 shows that a read with no calls equals "sequence only" (`analysis/singlefiber/LOGFULL_PORT.md`).

### 1c. Circularity (f and the read scored on the same reads)
- Today, f is fitted on all reads and each read is then scored with that f (`site_mixture.fit_site`). This is standard empirical Bayes. Each read is ~1/160 of f, so the bias is small, but per-read accuracy numbers are then in-sample.
- `heldout_barcode` cross-fits by **barcode** (`site_mixture.py:348-373`). But 4 of 7 barcodes are one run (Jun25), so holding out bc21 still trains on bc22-24 from the same run. That is not a run-level hold-out.
- The barcode CV Spearman of 0.80 over 602 pairs is inflated by the between-class split (f ≈ 0.45 at MacIsaac vs ≈ 0 at controls). Within-class agreement is unverified.
- **Recommend:** production f from all reads. For validation, cross-fit by **run** (Mar20 / May07 / Jun25). Also report within-class (MacIsaac only) f agreement.

### 1d. Which sites
- **Today:** 42 MacIsaac ABF1 sites + 42 PWM-only hits on chrI+chrXIV + 2 ERV46 controls (`tf_compare/sites/sites.tsv`). The PWM-only hits are matched on PWM score and read depth only (`tf_compare/scripts/make_sites.py:73-87`), not on accessibility.
- **Options:** MacIsaac (ChIP-chip; biased to strong sites); Rossi (held out); genome-wide PWM scan (`site_mixture.scan_pwm`).
- **Recommend:** candidates = a genome-wide PWM scan above a fixed score. MacIsaac and Rossi are used only as labels. This estimates f without picking sites by the answer.

### 1e. Which factors after ABF1
See §10. REB1 is next, but its footprint sign needs checking first.

### 1f. How to validate
See §3.

### 1g. Nucleosomes: local occupancy too?
**No.**
- Per-read nucleosomes already work: Chereji recall 0.75 (`SINGLEFIBER_PLAN.md` "Done 2026-11-02"). Earlier: 434 of 439 paths distinct (`analysis/singlefiber/README.md:222-226`).
- A positional prior taken from the population would pull each read toward the average and erase the read-to-read variation that is the point.
- The real nucleosome problem is the **global** share (11.5%, §6). Fix that instead.

---

## 2. Move `site_mixture.py` and other readers to the current defaults

Where they stand:
- `site_mixture.py:77-79` reads `robocop_train_fiberonly`, `nucleosome_params.pkl` (barcode01, Chereji-centred) and `bg_params.pkl` (0.138).
- `em()` floors p_open at `bg_rate()` with that default file (`:284`); the floor was hit at 1 of 86 sites (p_open min 0.1383, `mixture_pb0p02/site_fits.tsv`).
- `load_site` calls `read_molecules(..., min_mapq=0)` with no keep list (`:209-215`), so it reads unfiltered reads.
- Other unfiltered callers: `checks.py:302`, `plot_permol_window.py:85`, `fig_permol_fibers.py:160`, `pooled_emission_check*.py`, `test_logfull_permol.py:94`, `singlefiber_review/site_evidence_ceiling.py:10` (grep for `read_molecules(` in this review).
- The runner writes to `tf_compare/mixture` or `mixture_pb<pb>` (`run_site_mixture.py:12`), so a rerun would overwrite.

**Recommend:** pass nuc/bg files and the keep list through `load_site` / `em`, then rerun into a new directory (for example `mixture_pb0p02_filtnordna`). Rule 7: this changes reads, bg floor and nucleosome profile at once, so report it as one "model update" against the old result, not as a test of any one of them.

## 3. Validation design
- **Controls are not matched on accessibility.** PWM-only sites have median f_nuc 0.886; MacIsaac sites 0.264 (`mixture_pb0p02/site_fits.tsv`). Rossi-positive sites are almost all MacIsaac (36 of 42 MacIsaac vs 1 of 42 PWM-only). So "AUROC 0.96" mostly says "MacIsaac sites are open and bound, look-alikes are nucleosomal."
- **Missing control:** PWM hits in open DNA (NDRs) with no ChIP signal, also matched on A/T run length and informative-call count (plan "Next steps" #5).
- **Per-read confidence is low even at good sites.** Median fraction of reads with post > 0.9 is 0 at MacIsaac sites (max 0.61 at one site); median fraction > 0.5 is 0.49.
- **Other checks already planned** (`singlefiber_review/REPORT.md` "Validation"): simulation at real call density; co-occupancy (bound read nucleosome-free with phased +1); FiberHMM per-fiber footprints as baseline (outputs exist: `/usr/project/xtmp/nd141/projects/Fiber_seq/Fiber_tools_claude/fiberhmm_run/outputs/{fire,tagged}`); Chereji dyads for nucleosomes.
- **Decisions:** (a) add NDR-matched no-ChIP controls; (b) is FiberHMM the official baseline? (open since 10-25); (c) run-level hold-out (§1c).

## 4. Run labelling difference
- **Rates** (`/usr/project/xtmp/nd141/bg_investigation/nov03_final/rates_final.tsv`, via `reports/2026-11-02_report.md:187-195`): linker m6A May07 0.114–0.117, Mar20 0.143, Jun25 0.183–0.193. Nucleosome core and genome-wide rates show the same gap (Jun25/May07 ≈ 1.65–1.70×).
- **It shows up per read.** Mean per-read post_bound at the 42 MacIsaac sites, by barcode (computed from `mixture_pb0p02/per_read.json`):

  | barcode | mean post_bound |
  |---|---|
  | May07 bc03 / bc04 | 0.377 / 0.364 |
  | Mar20 bc01 | 0.411 |
  | Jun25 bc21–24 | 0.424–0.452 |

  The cause is unverified (likely low-labelling reads look more nucleosome-like).
- **Options:**
  - nothing (one pooled model, the current default);
  - one scale factor per run (or barcode) on every p;
  - per-read scaling (`--eff 1`, `permol_params.read_scale`: flank rate ÷ pooled, ±400 bp outside the window; falls back to span rate);
  - separate param sets per run.
- `--eff 1` has not been re-run in log space (`SINGLEFIBER_PLAN.md:291`). The old linear arms moved nucleosome r only 0.444 → 0.472 (`singlefiber_review/REPORT.md:22`).
- **Recommend:** per-run scale factors first. They are stable, there are few of them, and the cause is a run property. Compare against `--eff 1` on the per-site model, where it is cheap.

## 5. Background value and the TF fallback
- **Values now** (`inputs/*_filtnordna.pkl`, read in this review): background 0.1564 / 0.1544; `combined_low_count` 0.2504 / 0.2657. The fallback is used by the 141 unfitted motifs **and** `unknown`.
- **What this means:** in open DNA (site p_open median 0.32 at MacIsaac sites), any fallback TF state explains m6A better than background does. So `unknown` (29% prior, §6) acts as a second "open DNA" state. Read-averaged `unknown` at ERV46 is 0.13 (`erv46_5to3_nordna/permol.npz` `avg_optable`).
- **The user's plan to hand-raise background toward 0.249** (as aggregate `bg_params_open.pkl`, which was copied from `combined_low_count`) would remove that gap. But background would then no longer be a fitted linker rate.
- **Options:**
  - keep 0.156 + fallback 0.25 (now);
  - raise background to the fallback (aggregate style);
  - set the fallback to background, so unfitted motifs compete by sequence only;
  - fit a separate "NDR / hyperaccessible" rate.
- **Recommend:** decide jointly with §6. Do not raise background before #1 is re-validated, because the per-site model floors p_open at background (§2).

## 6. Prior shares and concentrations
- **Shares (recomputed in this review):** background 0.538, `unknown` 0.289, all other TFs 0.057, nucleosome 0.115. Real yeast nucleosome occupancy is far higher, so per-read results revert toward non-nucleosome: read-averaged nucleosome is 0.581 at ERV46 (`avg_optable`). The old model gave 0.56 per read vs 0.86 aggregate (`singlefiber/README.md:213-216`).
- **The driver's trainDir is pristine** `robocop_train_fiberonly_nucbrogsym` (`run_permol_window_logfull_5to3.py:101`), so λ is untuned (ABF1 tf_prob 1.8e-7). The tuned λ sets (u001, tuner-v2 bo09/bu01/bv75) are not used for single-fiber.
- **Decisions:**
  - (a) target nucleosome share (for example set from the aggregate genome-wide occupancy), with `make_conc_trainDir.py --hold-nucleosome`-style control;
  - (b) shrink `unknown`. Past sweep: freed mass goes to background (memory note unknown-sweep-result);
  - (c) which λ set for single-fiber.
- **Recommend:** (a) first. It changes per-read nucleosome calibration directly, and it sets the competition that #1(b) must calibrate against.

## 7. ABF1 footprint for per-read work
- **Motif:** 18 bp (Murphy 14 + 2 + 2 significant columns; memory abf1-extended-motif-is-18bp).
- **Footprint:** protected core about −7..+15; hyperaccessible flanks −30..−14 and +16..+43 (memory wide-footprint-abf1-result). Widening the aggregate state with protected pads made ABF1 worse (33.6× → 8.6×).
- **In `site_mixture`:** window is motif ± 20 bp (`fl=20`). The bound and free classes share one p_open in the flanks (`_estep`, `site_mixture.py:270-273`), so a bound read gets no credit for elevated flanks, and the flank is only partly inside the window. p_bound is fixed at 0.02; fitted freely it hits the 0.001 floor at 30 of 42 sites (`site_mixture.py:32-39`).
- **Options:**
  - keep it;
  - add a bound-class flank rate p_flank > p_open with a wider window (±45);
  - fit a per-read bound/unbound profile by EM at ChIP sites (plan "Next steps" #3).
- **Recommend:** the bound-class flank rate. It is the one known discriminating signal, and it is cheap to add.

## 8. Poly-dA:dT hypomethylation
- **Measured:** A/T runs ≥ 12 are 4.6× hypomethylated even in the most accessible stratum (memory polyAT-hypomethylation-artifact, chrXIV pileup). Neither the HMM nor `site_mixture` has a sequence-context term. So a read that is unmethylated over an A/T run looks bound or protected.
- **Options:**
  - a run-length multiplier on p in the free and background classes;
  - a k-mer emission (external `kmer_hsmm`);
  - ignore it, and match controls on run length only (§3).
- **Recommend:** a run-length multiplier in the per-site model first (the free class at each site), fitted on isolated-A/T accessibility strata as in the memory note. Then decide on the HMM.

## 9. No-call emission and dorado p calibration
- **Now:** methylated p ≥ 0.81, unmethylated p ≤ 0.19 or unlisted, no-call in between. A no-call adds 0 (`pkgvar/permol_seq_maskoff_logfull_nucbrogsym_filtnordna/robocop/robocop.py:1581-1583`).
- **No-calls carry signal.** Ambiguous rate is 0.187 in own-read nucleosome < 0.1 vs 0.041 in nucleosome > 0.9, a 4.6× difference (`/usr/project/xtmp/nd141/emission_audit/mar_and_evidence.json`).
- **Side observation:** A mismatches are also higher in open DNA (0.027 vs 0.003, same file). They are dropped as no observation too. The cause is unverified (m6A-induced basecall errors?).
- **Options:**
  - current;
  - 3-outcome categorical emission per state (meth / canon / no-call);
  - soft calls (use p directly, which needs calibration);
  - split at 0.5 (linker rate 0.192 vs 0.156, `reports/2026-11-02_report.md:111-117`).
- **Calibration:** no unmethylated or fully methylated control sample was found in the docs or `bg_investigation/provenance` (only Mar20 has recorded conditions). Ask whether one exists. Without one, soft calls are not calibrated.
- **Recommend:** 3-outcome emission, after #1–#4. It needs no calibration, because the no-call rate per state is fitted like the m6A rate.

## 10. Which factor after ABF1
- **REB1's fitted m6A is above background:** mean 0.2206 / 0.2409 vs 0.156 (`inputs/all_TFs_1000pealVal_params_pseudo_filtnordna.pkl`, read in this review). Cin5, Fhl1, Mcm1 and Tbf1 are also above background.
- So in the HMM, a REB1 "footprint" means *more* m6A than linker. It is an aggregate average over bound and unbound reads in hyperaccessible NDRs, not a protection signature.
- The REB1 motif is 8 bp, so it has fewer informative calls per read than ABF1 (unverified; measure with `site_evidence_ceiling.py`).
- **Recommend:** REB1 next, in the per-site model with its own fitted p_bound (not 0.02), after the §7 flank change. Check first, on a few MacIsaac REB1 sites, that the bound class comes out protected.

## 11. Scale, runtime, output, viewer
- **Runtime:** 0.477 s per kb of read span + 0.07 s per read, from a fit to `erv46_5to3_nordna/molecules.tsv` (387 reads, 0.96 s per read mean, median in-window span 1,406 bp). Peak RSS 1.7 GiB (`run_stats.json`).
- **Genome estimate (unverified):** about 35× non-rDNA depth after filters (52× × 67% kept), × 12 Mb ≈ 4e8 bp ≈ 56 CPU-hours.
- **Clipping:** reads are clipped to the 5 kb window (`run_permol_window_logfull_5to3.py:443-444`). A full-read decode needs ~28 kB per bp for the dense posterior (3,485 states × 8 B), about 280 MB for a 10 kb read. That is estimated, not measured.
- **Output:** `permol.npz` keeps 7 columns (background, nucleosome, nuc_center, nuc_padding, Abf1, Reb1, unknown); other TFs are dropped.
- **Decisions:**
  - (a) next scale: recommend all MacIsaac ABF1 (~300) and REB1 sites genome-wide with the per-site model. It is cheap, and it answers "how many fibers are bound";
  - (b) when to do the genome HMM (after #1(b));
  - (c) per-read output: columns, float16 vs float32, one file per chromosome;
  - (d) viewer: extend the Run Browser / locus slide with a per-read panel (fiber stack, one row per read, coloured by post_bound).

## 12. Trimming and consensus flags
- **Trimming** is not implemented; the user said post-decode only (`lab_meeting/oct_27_notes.md:31-33`). With start and end anywhere, the left↔right test fell to 0.067 (`SINGLEFIBER_PLAN.md:166`). Window clipping (§11) makes artificial ends at window edges, which trimming would not fix.
- **Recommend:** decode whole read spans at scale. Trim only if a direction test on real read ends still shows an edge effect.
- **Consensus flags:** 0 at ERV46 (`run_stats.json consensus_summary`), 11,433 genome-wide. Decision: drop, or just annotate, per-read TF calls overlapping a flagged position? Recommend annotate.

## 13. Aggregate model
- **Still starts motif-first:** 310 start states, background 0.933 (`SINGLEFIBER_PLAN.md:129-131`).
- **Still on old data and params:** it reads the unfiltered merged pileup (for example `robocop_tw3_bo09_48_logfull/config.ini:10`) and barcode01 nucleosome + `bg_params_open` (`SINGLEFIBER_PLAN.md:56-58`).
- A filtered, rDNA-free pileup exists (`/usr/project/xtmp/nd141/filtered_refit_nordna/filtered_pileup_all_chr`). No aggregate `_filtnordna` pkgvar tree exists (`ls pkgvar/*filtnordna*` shows only the per-read one).
- **Decision:** move the aggregate to the filtered pileup + `_filtnordna` pkls + start anywhere (new tree, rule 3)? **Recommend yes.** Then the per-read vs pooled identity check holds again on the same reads, and the user's "all 7 barcodes everywhere" decision (2026-10-31) is met. The hand-set background (§5) stays a separate choice.

## 14. Code inconsistencies found
| where | issue |
|---|---|
| `singlefiber/test_5to3.py:45-46` | `OLD_MODEL` passes the old tree / nuc / bg but not `--tf-params` or the read filters. The driver now defaults to `_filtnordna` TF params and the keep list. The old tree reads `all_TFs_1000pealVal_params_pseudo.pkl` (`pkgvar/permol_seq_maskoff_logfull/robocop/robocop.py:1441`), so `assert_verbatim` should fail. Not run; derived from code. |
| `site_mixture.py:77-79, 209-215, 284` | old params, unfiltered reads, barcode01 bg floor (§2) |
| `permol_params.py:53-55` | module defaults still the barcode01 nuc/bg. The driver overrides them, but other importers get the old ones. |
| `run_permol_window_logfull_5to3.py:43-49` | docstring item 4 still says TF params are "unchanged (pseudo.pkl)". Item 5 supersedes it. |
| `SINGLEFIBER_PLAN.md:131`, report §4 | cite `stationary_initial` at `:222-255` / `:237-268`; it is now `:242-275`. |
| `sbatch_permol_window_logfull.sh` | runs the old driver and old trainDir. The current one is `sbatch_permol_window_logfull_5to3.sh`. Fine, but easy to pick the wrong one. |
| latent | `robocop.py:932` `motif_starts` undefined in a never-run branch; `end_probs` applied in both passes (harmless while all ones) (`SINGLEFIBER_PLAN.md:135-139`) |
| `site_mixture.heldout_barcode` | "barcode" hold-out is not a run hold-out (§1c) |
