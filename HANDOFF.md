# RoboCOP + Fiber-seq — Session Handoff

Pick-up notes for the next agent. Read `whattodo.md` (diagnosis + phased strategy) and
`FIBERSEQ_CHANGES.md` (full code diff vs upstream) first — this file adds the **scoring +
plotting tooling** and the **layer / mask toggles** that are easy to get wrong.

Environment:
```bash
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
```
All decode outputs (`robocop_chrI_*`, `robocop_seqlayer_*`, `robocop_erv46_*`,
`robocop_train_fiberonly/`) live on THIS filesystem, not in git (they are 20–210 MB each and
git-ignored). A new session on this machine already has them.

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

Mechanics to run mask ON: comment IN lines 113-114 (and optionally comment OUT line 101 to also
enable sequence), submit the sbatch, wait until RUNNING, then re-comment (there is no env-var
toggle — that was removed per user preference). Drivers: `run_fiberonly_noem.py` +
`sbatch_noem.sh`, trainDir `robocop_train_fiberonly`. Decodes use `run_robocop_without_em`
(keeps `tmpDir/info.h5`; equivalent to `with_em` iter=0, which otherwise DELETES tmpDir).

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

## 5. The 4 canonical chrI runs (scored/plotted throughout)

| outDir | layers | mask | plot color |
|--------|--------|------|------------|
| `robocop_chrI_maskon`      | Fiber only | ABF1-only | blue |
| `robocop_chrI_maskoff`     | Fiber only | all TFs   | blue |
| `robocop_chrI_seq_maskon`  | Fiber+seq  | ABF1-only | orange |
| `robocop_chrI_seq_maskoff` | Fiber+seq  | all TFs   | orange |

Latest ABF1 site-#4 result (from `plot_abf1_locus.py`, midpoint 62664): fiber_abf1 → 62663
(1 bp ✓), fiber_all → MISSED (local max 0.147 < 0.30), seq_abf1 → 62665 (1 bp ✓), seq_all →
62665 (1 bp ✓). Matches `abf1_sites_per_run_centered.txt`.

## 6. Open threads (not started)
- Retrain EM with the sequence layer ON — current `*_seq_*` decodes reuse Fiber-only-trained
  weights, so they are a lower bound on what seq can do.
- Optional scorer speedup: cache the collapsed factor track per region (currently re-collapses
  the whole ~230 kb chrI segment on every score call, ~9 min/run).
