# ABF1 genome-wide prior (λ) sweep on the current single-fiber model — 2026-11-03

## Answer

- **Does raising ABF1's λ give per-read calls that separate bound from unbound reads?** No. The prior multiplies every read's odds by the same factor. Reads keep the same order at every λ, so calls appear on most reads of a site at once, or on none.
- **What does it cost?** At λ=1500, 7 of 42 MacIsaac sites get calls, but so do 6 of 42 PWM-only sites (5 of them Rossi-negative). Rossi AUROC of the fraction of reads > 0.5 goes only from 0.503 to 0.560; the per-site mixture model gets 0.938. At λ=5000, a spot with no site in ERV46 is called on 133/135 reads. Nucleosomes barely move.

The only parameter under test is ABF1's λ. Everything else is the current driver default. The side effects through normalisation are listed below; `--hold-nucleosome` was not used.

## ERV46 (chrI:60,001-65,000), 387 reads

Per-read ABF1 posterior is the max over the motif interval. The site intervals come from `tf_compare/sites/sites.tsv`: site1 61,165-61,178 (PWM 1.95), site2 62,659-62,672 (PWM 3.99), the same as the old comparison.

| λ | ABF1 tf_prob | site2 median / IQR / max | site2 SD | site2 >0.5 | site1 median / max | spot 61,656-61,669 median (>0.5 reads) | reads with a call outside the sites | nuc_prob | read-averaged nuc mean / max abs change vs default |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.797e-7 | 0.00088 / 0.0004 / 0.0011 | 0.0003 | 0 | 7e-6 / 7e-6 | 0.0005 (0) | 0 | 1.357e-3 | 0 / 0 (reproduces 11-02 run) |
| 138 | 2.480e-5 | 0.109 / 0.046 / 0.135 | 0.033 | 0 | 0.0009 / 0.0010 | 0.067 (0) | 0 | 1.354e-3 | 0.0003 / 0.006 |
| 400 | 7.185e-5 | 0.261 / 0.094 / 0.311 | 0.074 | 0 | 0.0026 / 0.0029 | 0.173 (0) | 0 | 1.349e-3 | 0.0008 / 0.017 |
| 1500 | 2.690e-4 | 0.572 / 0.129 / 0.631 | 0.126 | 0.725 | 0.0097 / 0.011 | 0.440 (0) | 0 | 1.327e-3 | 0.0022 / 0.044 |
| 5000 | 8.922e-4 | 0.819 / 0.086 / 0.854 | 0.118 | 0.949 | 0.032 / 0.035 | 0.724 (133/135) | 133 | 1.258e-3 | 0.0045 / 0.077 |

**Does the evidence separate reads?** Only by ordering them, and the order never changes.
- Per-read posterior vs methylated As in the motif gives Spearman −0.734 (site2) and −0.737 (site1) at every λ.
- At site2, median posterior by m6A count (0/1/2/3 m6A): λ=1500 0.595/0.386/0.199/0.070; λ=5000 0.833/0.686/0.453/0.204. The 0-vs-3 odds ratio is about 20× at both λ.
- 77% of reads have 0 m6A in the motif (site2: 101/132; median 5 observed A/T; motif m6A rate 0.069 vs flank 0.237). Those reads all get nearly the same posterior, so the 0.5 threshold is crossed by most of them together.
- The spot at 61,656 (motif m6A rate 0.039, flank 0.038, so inside protected chromatin) rises in step with site2 and is called on every read at λ=5000.
- Site1 (weak PWM) stays at or below 0.035. The old mixture model put it at 71% bound (`tf_compare/mixture_pb0p02/site_fits.tsv`, old model; shown for scale only).

**Nucleosomes** (`nuc_quality.py`): Chereji recall 0.75 PASS at every λ. Per-dyad distances are unchanged (0/34/14/11 bp). The share of reads with a call within 20 bp of each dyad changes by ≤0.008. Read-averaged occupancy goes 0.5806 → 0.5770.

## 84-site set (current model: untuned / ×138 / ×1500)

These numbers are directly comparable to `compare_summary_mixture_pb0p02.json`, which used the old model.

| | old ×138 | current untuned | current ×138 | current ×1500 | mixture (a) |
|---|---|---|---|---|---|
| Rossi AUROC, fraction of reads >0.5 | 0.503 | 0.500 | 0.503 | 0.560 | 0.938 |
| Rossi AUROC, median posterior | 0.703 | 0.700 | 0.700 | 0.699 | — |
| sites with any read >0.5 | 2 (1 false) | 0 | 2 (1 false) | 13 (7 MacIsaac, 6 PWM-only) | — |
| ...of which Rossi-negative | 1 | 0 | 1 | 5 | — |
| median per-site SD across reads | 0.00084 | 7.4e-6 | 0.0010 | 0.0088 | 0.076 |

- At ×138 the current model calls the same two sites as the old one: mac_chrXIV_559493 and pwm_chrI_190561 (Rossi 0).
- At ×1500 the six PWM-only sites with calls are chrXIV_708367 (Rossi 142, so true), chrXIV_483210, chrI_57546, chrXIV_76709, chrXIV_361813 and chrI_190561.
- The median-posterior AUROC (~0.70) does not change with λ. That is the same point as ERV46: λ shifts every site and read together and adds no ranking information.

## Side effects of the λ change on other priors

Source: `HMMconfig.pkl` of each trainDir, tabulated in `prior_table.json`.

| λ | background | nucleosome (ratio) | unknown (ratio) | other 152 TFs (ratio range) | stationary start: bg / nuc / TFs |
|---|---|---|---|---|---|
| 1 | 0.933234 | 1.357e-3 | 0.050108 | — | 0.5385 / 0.1151 / 0.3464 |
| 138 | 0.933221 | ×0.998 | ×0.99986 | 0.99972–0.99993 | 0.5385 / 0.1149 / 0.3466 |
| 400 | 0.933195 | ×0.994 | ×0.99959 | 0.99917–0.99979 | 0.5386 / 0.1145 / 0.3470 |
| 1500 | 0.933089 | ×0.977 | ×0.9984 | 0.9969–0.9992 | 0.5389 / 0.1126 / 0.3484 |
| 5000 | 0.932751 | ×0.927 | ×0.9948 | 0.9897–0.9974 | 0.5399 / 0.1070 / 0.3531 |

- In every trainDir only transition row 3330 (`silent_states_begin`) changed.
- `make_conc_trainDir.py` passed both gates on `robocop_train_fiberonly_nucbrogsym` exactly: fidelity rel diff 0, and blast radius = row 3330 only.
- The `config.ini` copied into each new trainDir still names the pristine trainDir. The driver does not read it; it reads only `HMMconfig.pkl`.

## Evidence

- **trainDirs:** `analysis/robocop_train_fiberonly_nucbrogsym_abf1x{138,400,1500,5000}/` (`conc_patch.json`). Build logs: `/usr/project/xtmp/nd141/permol_proto_logfull/abf1sweep/build/`.
- **ERV46 decodes:** `/usr/project/xtmp/nd141/permol_proto_logfull/abf1sweep/x{1,138,400,1500,5000}/`.
  - Slurm 13256058, sbatch `abf1sweep/sbatch_abf1sweep.sh`.
  - Driver md5 `53c696b5`. The driver has since been replaced by another agent (now `136e4fe9`, adds a default-off `--local-prior`), so these runs predate it.
- **ERV46 analysis:** `abf1sweep/analyze_sweep.py` → `sweep_summary.json`; `abf1sweep/prior_table.py` → `prior_table.json`.
- **Nucleosome check:** `abf1sweep/nucq/nuc_quality_x1_x138_x400_x1500_x5000.json`.
- **84 sites:**
  - decodes: `/usr/project/xtmp/nd141/tf_compare/perread_sites_cur/<site>/{untuned,abf1x138,abf1x1500}/`, from Slurm 13256087 (`tf_compare/scripts/perread_sites_cur.sh`, driver md5 `53c696b5`, all 258 decodes, no tracebacks);
  - comparison: `tf_compare/scripts/compare_ab_cur.py` (a copy of `compare_ab.py` with new tags and output names) → `tf_compare/compare_summary_cur_mixture_pb0p02.json` and `compare_sites_cur_mixture_pb0p02.tsv`.
- **Figures:**
  - `analysis/singlefiber/figures/abf1_prior_sweep_2026-11-03/abf1_prior_sweep_summary.png` (script `abf1sweep/fig_sweep.py`);
  - composite for λ=1500: `.../x1500/permol_composite_x1500.png`.
- **Caveat (unverified for the current model):** mixture-model numbers (a) are from the old run (`tf_compare/mixture_pb0p02`). They were not refit on the current filtered read set.
