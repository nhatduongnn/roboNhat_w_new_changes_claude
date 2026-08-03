# Fiber-seq implementation changes in RoboCOP

This document catalogs every code change that was made to integrate **Fiber-seq
(single-molecule m6A methylation) data** into RoboCOP, relative to the original
RoboCOP that had no Fiber-seq support.

## Baseline / how this was computed

- **Original RoboCOP** = the `HarteminkLab/RoboCOP` upstream repo at commit
  `79320ab` ("updates to robocop_em steps", Oct 2025). This is the fork point of
  this repo and the last commit *before* any Fiber-seq work. It is the same
  lineage as the local `RoboCOP/` sibling folder (that folder is an older 2022
  snapshot of the same upstream and, aside from unrelated 2022→2025 "dynacop"
  drift, contains none of the changes below).
- **This repo** (`roboNhat_w_new_changes_claude`, branch `main`) = that baseline
  plus 9 commits of Fiber-seq work (Oct 2025 → Apr 2026), plus one `.gitignore`
  commit.
- Everything below is the diff `79320ab..HEAD` restricted to source files, i.e.
  the actual Fiber-seq implementation (upstream drift excluded).

To regenerate the raw diff:
```bash
git diff 79320ab..HEAD -- pkg/
```

## New config keys

Fiber-seq is driven by three new keys in the `[main]` section of `config.ini`
(see `analysis/robocop_train/config.ini`):

| Key          | Example value                    | Meaning                                              |
|--------------|----------------------------------|------------------------------------------------------|
| `tech2`      | `Fiber`                          | Second assay type. When `== 'Fiber'`, Fiber-seq path runs. |
| `pileupFile` | `/usr/xtmp/.../..._pileup_all_chr`| modkit pileup BED file of per-position m6A counts.   |
| `nucleotide` | `A`                              | Base whose methylation is modeled (adenine for m6A). |

`tech` (e.g. `MNase`) is unchanged and still selects the MNase/ATAC path. The two
run side by side.

---

## File-by-file changes

### 1. `pkg/robocop/utils/getReads.py`  (+88, data ingestion)
New functions:
- **`getFiber_seq(modkit_df, tmpDir, info_file, coords, nucleotide, idx=None, tech="Fiber")`**
  — mirror of `getMNase`. Iterates over coordinate segments, extracts per-base
  methylation counts, and writes four sparse arrays per segment into the info
  HDF5 file (`save_sparse`), keyed as
  `segment_<i>/Fiber_count_meth_watson`, `..._meth_crick`, `..._A_watson`,
  `..._A_crick`.
- **`getValuesFiber_seqOneFileNucleotide(modified_bases_df, chrm, minStart, maxEnd, nucleotide)`**
  — core extractor. Filters the modkit dataframe to the region, then for each row
  reads modified-base count (col 11) and trial count (col 9), bucketed by strand
  (`+` → Watson, `-` → Crick). Positions are 0-based relative to `minStart`.
  Returns `(count_meth_watson, count_meth_crick, count_A_watson, count_A_crick)`.

### 2. `pkg/robocop/utils/readData.py`  (+69)
- **`getValuesFiber_seqOneFileNucleotide(modkit_df, chrm, minStart, maxEnd, nucleotide, offset=0)`**
  — an earlier/alternate extractor (well-documented docstring) that re-reads the
  modkit file from disk and returns only Watson/Crick **meth** counts (no A/trial
  counts). ⚠️ Superseded by the `getReads.py` version; see Known issues.

### 3. `pkg/robocop/utils/parameterize.py`  (+1)
- Added an `'unknown'` DNA-binding-factor concentration:
  `dbf_conc['unknown'] = 0.1`. Supports an "unknown TF" state used in the
  Fiber-seq transition experiments.

### 4. `pkg/robocop/robocop.py`  (+498, model core — the biggest change)
- **Emission tensor grown from 5 to 7 layers.** `d['n_vars']` and the
  `data_emat` shape changed `5 → 7`. Layers 0–4 are the original
  (sequence + MNase long/short + ATAC long/short); **layer 5 = Fiber Watson**,
  **layer 6 = Fiber Crick**.
- **`int` → `longlong` across all C-extension calls.** Every `np.long` was
  changed to `np.longlong` in `argtypes` and `.astype(...)` (transition matrix,
  initial probs, fward/bward, find_parents_and_children, PWM emission). Required
  for the rebuilt `librobocop.so` (also updated in this repo) and 64-bit
  correctness.
- **New Fiber-seq emission functions:**
  - `update_data_emission_matrix_using_binomial_fiber_seq(...)` — the **active**
    model. For each observation where the reference base is `A`, multiplies the
    emission by a **Binomial** pmf `binom.pmf(k, n, p_j)` where `k` = methylated
    count, `n` = total A trials, `p_j` = state-specific methylation probability.
    Non-`A` positions get emission 0.
  - `update_data_emission_matrix_using_fiber_seq_counts_Bionomial(...)` — driver
    that loads per-state methylation-probability parameters from pickles
    (`inputs/all_TFs_1000pealVal_params_pseudo.pkl`, `inputs/nucleosome_params.pkl`,
    `inputs/bg_params.pkl`), assigns forward/reverse `p` vectors per TF motif,
    fills nucleosome and background `p`, then calls the binomial updater for
    Watson (layer 5) and Crick (layer 6).
  - `update_data_emission_matrix_using_fiber_seq_counts_onePhi(...)` and
    `update_data_emission_matrix_using_negative_binomial_fiber_seq(...)` — an
    earlier **Negative-Binomial** approach (loads `phi`/`mu` params). Superseded
    by the Binomial version but left in place.
- **Diagnostic plotting** of emission-matrix layers (histograms + boxplots
  written to `<tmpDir>/emission_plots/emission_layer_*.png`), plus debug prints
  of specific transition-matrix entries in `posterior_forward_backward_loop`.
- `import matplotlib.pyplot as plt` and `binom` added; `tf_prob == []` changed to
  `len(tf_prob) == 0`.

### 5. `pkg/robocop/utils/robocopExtras.py`  (+35, emission driver)
- `updateMNaseEMMatNB` was **repurposed**: the two MNase negative-binomial calls
  are commented out and replaced by two
  `update_data_emission_matrix_using_fiber_seq_counts_Bionomial` calls
  (`FiberType='watson'` and `'crick'`, `tech='Fiber'`).
- After building emissions it **zeroes the sequence layer** (`data_emission_matrix[0][:] = 1`)
  so sequence/PWM doesn't dominate, and floors zeros in the Fiber layers
  (5 and 6) to `epsilon = 1e-30` to avoid log(0).

### 6. `pkg/robocop_em.py`  (+17) and `pkg/robocop_no_em.py`  (+13)
- Both read `tech2`; when `tech2 == 'Fiber'` they load the modkit `pileupFile`
  once (splitting column 9 into sub-columns), then call
  `getReads.getFiber_seq(...)` right after `getReads.getMNase(...)` to populate
  the Fiber-seq arrays in the info file.
- In `robocop_em.py`, EM `iterations` was changed **`10 → 0`** (Fiber-seq runs
  currently do inference only, no EM re-estimation).

### 7. `pkg/robocop/utils/plotRoboCOP.py`  (+202, visualization)
- **`plotOutput(...)`** gained a `fiber_info=None` argument and dynamically adds
  2 extra rows (Watson + Crick strand panels) to the figure when Fiber-seq data
  is present.
- **`plot_fiberseq(allinfofiles, coords, chrm, start, end, tech)`** — new. Loads
  the four Fiber-seq count arrays from the info HDF5 files across overlapping
  segments and averages them over the region.
- **`plotFiberseqAx(ax_w, ax_c, fiber_info, start, end)`** — new. Scatter-plots
  the per-base **meth/A ratio** (0–1) for Watson (blue) and Crick (orange).
- `plot_output` now calls `plot_fiberseq` and passes the result into `plotOutput`.
- Misc: `exit(0)` → `sys.exit(0)`; several debug prints added.

---

## New files introduced by the Fiber-seq work

| File | Purpose |
|------|---------|
| `pkg/robocop_train/config.ini` | Minimal train-dir config stub. |
| `pkg/debug.py` | Ad-hoc debugging scratch script. |
| `robocop-2024_env.txt`, `robocop-2024_env_no_munkres.txt`, `robocop-2024_env_with_pyranges_12142025.txt` | Conda/pip environment snapshots for the Fiber-seq work. |
| `adapt_from_sneha.yaml` | Environment YAML. |
| `query_names.txt` | List of query/read names (Fiber-seq data prep). |
| `robocop_train/config.ini` | Top-level train config used by the analysis runs. |
| `analysis/robocop_train/`, `analysis/robocop_all_subset/`, `analysis/robocop_all_fiber/` | Example run inputs/outputs (configs, `coords.tsv`, `pwm.p`, `nuc_emission.npy`, dinucleotide model, emission-layer PNGs). |
| `analysis/test.ipynb` | Development/testing notebook. |

Rebuilt binary: `pkg/robocop/librobocop.so` (recompiled for the `longlong`
change; `pkg/robocop/gccCompile` tweaked).

---

## Known issues / things to be aware of before extending (for the agent)

These are rough edges in the current Fiber-seq code — worth knowing before making
new changes:

1. **Hard-coded input paths.** `update_data_emission_matrix_using_fiber_seq_counts_Bionomial`
   and `..._onePhi` load parameter pickles from a literal relative `inputs/`
   directory (`inputs/all_TFs_1000pealVal_params_pseudo.pkl`,
   `inputs/nucleosome_params.pkl`, `inputs/bg_params.pkl`,
   `inputs/abf1_reb1_params.pkl`, `inputs/fiber_seq_data_count_meth_*.npy`).
   These are not config-driven and assume a specific working directory.
2. **Duplicate function definition.** `update_data_emission_matrix_using_negative_binomial_fiber_seq`
   is defined **twice** in `robocop.py`; the second definition (which hard-codes
   an ABF1-only mask and zeroes nucleosome emissions) silently overrides the first.
3. **Hard-coded ABF1-only / magic indices.** Several places hard-code state
   indices (e.g. `data_emission_matrix[index][14:, :] = 1`, transition entries at
   `3330`, `2779+20`) that are specific to one experiment and marked
   "delete later".
4. **`readData.getValuesFiber_seqOneFileNucleotide` has a latent bug** — it
   references an undefined `r1['start']` and re-reads the file from disk. The
   `getReads.py` version is the one actually used.
5. **EM disabled.** `iterations = 0` in `robocop_em.py` means no parameter
   re-estimation; Fiber-seq parameters come entirely from the pre-computed
   pickles.
6. **Debug artifacts.** Numerous `print(...)` statements, forced emission-layer
   plotting on every run, and commented-out `sys.exit` calls remain in
   `robocop.py`, `robocopExtras.py`, and `plotRoboCOP.py`.
7. **MNase path is currently short-circuited** in `updateMNaseEMMatNB` (the MNase
   NB calls are commented out and the sequence layer is flattened to 1), so this
   branch is effectively Fiber-seq-only right now.
