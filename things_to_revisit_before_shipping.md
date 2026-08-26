# Things to revisit before shipping

Everything below is a way this tree diverges from published RoboCOP. Some are deliberate
science (the Fiber-seq layer), some are development scaffolding that must not ship (twelve
frozen copies of the package), and some are debug leftovers that are actively costing
runtime today.

Baseline for every "upstream" claim is commit **`5e6fa4e` — "Initial commit of RoboCOP
current working state"**, the last state before this fork's changes. Reproduce any diff with:

```bash
git diff 5e6fa4e HEAD -- pkg/<file>
```

Scope of the drift, for orientation:

```
pkg/robocop/robocop.py                +427 lines
pkg/robocop/utils/plotRoboCOP.py      +230
pkg/robocop/utils/getReads.py         +128
pkg/robocop/utils/robocopExtras.py     +65
pkg/robocop_no_em.py                   +48
pkg/robocop_em.py                      +34
                                  23 files, +1496 / -282
```

Priority key: **[P0]** must not ship · **[P1]** decide and hardcode · **[P2]** document and
validate · **[P3]** nice to clean up.

---

## 1. Debug leftovers that are live right now

### 1.1 [P0] `print('bob')` on every emission update
`pkg/robocop/utils/robocopExtras.py:97` (and all twelve `pkgvar/*` copies). Fires once per
segment per iteration. Delete.

### 1.2 [P0] Per-TF debug block in the fiber emission builder
`pkg/robocop/robocop.py:655` and the ~12 lines after it — `---- TF {i} ----`, `p_forward
shape`, `p_forward[:5]`, etc., printed for **every one of 154 TFs, on every segment**. This
is the bulk of the multi-MB job logs. Delete, or gate behind a verbosity flag.

### 1.3 [P0] Emission-layer plotting runs unconditionally
`pkg/robocop/robocop.py:1021-1046` writes **7 PNGs per segment per call** into
`tmpDir/emission_plots/`. A near-identical block directly above at `:983-1017` is already
commented out, which suggests `:1021` was meant to be too. Currently disabled **only** in
`pkgvar/seq_maskoff_em10/` and `pkgvar/fiber_maskoff_em10/`; still live in `pkg/` and the
other ten variants. Delete or put behind a flag.

### 1.4 [P3] Dead `sys.exit` and commented duplicates
`robocop.py:1017` (`sys.exit("Stopping after plotting emission layers.")`) and the large
commented-out mirrors of `updateMNaseEMMatNB`'s body. Harmless but they make the file hard
to read.

---

## 2. Knobs that should become fixed, considered values

### 2.1 [P1] EM iterations — currently 0
`pkg/robocop_em.py:116` and ten of twelve `pkgvar/*` copies: `iterations = 0`.
**Upstream RoboCOP ships `iterations = 10.`** Changed to 0 in commit `ad6c2be`
(2025-12-13), deliberately, during development.

Consequence: training was a pure config build. `tf_prob` stayed exactly `calculateKD`'s
output and was never fitted, which is why Nhp6a (7 bp) sat at `7.2239e-4` and `Abf1_murphy`
(14 bp) at `1.7974e-07` — a 4,019x gap set by motif length alone.

`pkgvar/seq_maskoff_em10/` and `pkgvar/fiber_maskoff_em10/` now read
`ROBOCOP_EM_ITERS` (default 10). **Decide the shipping value and hardcode it.** Open
question: was 0 chosen because EM was misbehaving, or purely to save time during
development? That answer changes how much the em10 results should be trusted.

### 2.2 [P1] `ROBOCOP_EM_CAP` / `ROBOCOP_EM_CAP_SD` — added 2026-08-24
`pkgvar/{seq,fiber}_maskoff_em10/robocop_em.py`, replacing upstream's
`threshold = np.mean(thresholds) + 2*np.std(thresholds)`.

Env unset reproduces the upstream formula **bit-identically**
(`0.00066917569069160432`) — verified. `ROBOCOP_EM_CAP=off` sets the threshold to 1.0
(unreachable); `ROBOCOP_EM_CAP_SD=k` changes the multiplier.

This exists to answer one question: in the first 10-iteration run, **28 of 154 states ended
up pinned at the cap**, ABF1 and Nhp6a among them, so their tidy `1.0x` prior ratio was
imposed rather than fitted. **Collapse this to a single hardcoded constant once the right
ceiling is known.** An env-var switch is not a model parameter.

### 2.3 [P1] Sequence layer is toggled by hand-commenting
`robocopExtras.py:101` — `data_emission_matrix[0][:] = 1` sets the sequence emission to
uniform, i.e. turns the sequence layer off. Active or commented depending on the variant.
This is the single line separating `seq_maskoff` from `fiber_maskoff`. **Promote to a
config option** (`useSequence = true/false` in `config.ini`).

### 2.4 [P1] ABF1 hard-forbid mask is toggled by hand-commenting
`robocopExtras.py:113-114`. Zeroes the Fiber channels for all non-ABF1 TF states, forcing
an ABF1-only decode. Same treatment: a config option, not a comment.

### 2.5 [P1] `dbf_conc['unknown'] = 0.1`
`pkg/robocop/utils/parameterize.py:83` — **added by this fork**; upstream sets only
`background = 1.0` and `nucleosome = 35`. It is an unexplained magic number that lands
`unknown` at prior `0.0501`, ~70x Nhp6a and ~280,000x ABF1, making it the largest single TF
state. It is also **exempt from the constrained-EM cap** (`adjustEM` loops
`range(1, n_tfs)`, skipping the last TF, and `'unknown'` sorts last). Justify or remove.

### 2.6 [P2] `epsilon = 1e-30` floor on the Fiber channels
`robocopExtras.py:104-106`, layers 5 and 6. Prevents all-zero emissions producing NaN. Fine
in principle, but the value is arbitrary and interacts with the ABF1 mask (which
deliberately re-zeroes after the floor). Document the interaction.

### 2.7 [P2] Posterior storage silently truncates at 1e-4
`pkg/robocop/robocop.py:33`, `save_sparse_posterior`: `v[v < 1e-4] = 0`. **New in this
fork.** This is a sparsity step on the **posterior** table on its way into `info.h5`, and it
is unconditional — no flag, no state range, no relation to the ABF1 mask. It runs on every
write, in every variant, mask on or off.

Not to be confused with the `1e-30` floor and the ABF1 hard-forbid mask at
`robocopExtras.py:104-114`, which act on the **emission** matrix and are opt-in (and are
paired: the floor lifts genuine zeros so nothing goes NaN, then the mask re-zeroes the
non-ABF1 states so they drop out of the product). Turning the mask off does not avoid the
1e-4.

Two consequences:

- Every stored zero means "< 1e-4", not zero. Any published posterior figure or metric
  inherits the floor and should say so.
- **It now feeds back into training.** `update_transition_probs` sums `p_table` columns read
  back through `get_sparse_todense`, so the reestimated priors are computed from the
  *thresholded* posterior. A state with a tiny prior — ABF1 starts at `1.797e-07` — has much
  of its diffuse low-level posterior discarded before EM ever counts it, which biases the
  first update against precisely the states the fit is meant to rescue. This coupling was
  inert while `iterations = 0` and is live now.

Measured on `robocop_chrI_seq_maskoff_revfix/tmpDir/info_0_6.h5`, `segment_0` (5000 x 3485;
99.76% of the stored table is exactly zero). "Max lost" is the worst case, `n_zeros x 1e-4`:

| state | start col | kept mass | zeros | max lost | as % of kept |
|---|---|---|---|---|---|
| `Abf1_murphy` | 1 | 0.1989 | 4995 | 0.4995 | **251%** |
| `Nhp6a_zhu` | 1137 | 1.9869 | 4933 | 0.4933 | 25% |
| `Reb1_badis` | 1519 | 3.1786 | 4990 | 0.4990 | 16% |
| `unknown` | 2779 | 13.7038 | 4751 | 0.4751 | 3.5% |

That bound is loose -- it assumes every stored zero sat just under the threshold, when most
are genuinely ~0 -- so the real loss is much smaller. The asymmetry is what matters: the
bound is a flat `n_positions x 1e-4` per column while retained mass varies ~70x across
states, so the threshold bites hardest on exactly the low-prior states the fit is meant to
rescue, and is negligible for `unknown`. Cheap to settle: re-run one fit with the threshold
lowered (say 1e-6) and compare the resulting priors.

---

## 3. Architecture that is scaffolding, not a design

### 3.1 [P0] `pkgvar/` — twelve frozen copies of the package
```
fiber_maskoff  fiber_maskoff_em10  fiber_maskon  seq_maskoff  seq_maskoff_12tfs
seq_maskoff_bgtss  seq_maskoff_capA  seq_maskoff_capB  seq_maskoff_em10
seq_maskoff_lowabf1  seq_maskon  seqonly_maskon
```
Each is a full copy of `pkg/` with layer/mask state and input-file choices **baked in**,
selected at run time by `sys.path.insert`. This solved a real problem — concurrent slurm
arrays racing on hand-commented toggles — but it means twelve places to fix any bug, and
they have already drifted (`iterations` is 0 in ten of them and 10 in two).

**Ship one package** whose behaviour comes from `config.ini`, and delete the variants. Every
distinction they encode is a boolean or a filename:

| variant | what it actually is |
|---|---|
| `seq_maskoff` | sequence on, no mask — the workhorse (revfix) |
| `fiber_maskoff` | sequence off |
| `*_maskon` | ABF1 hard-forbid mask on |
| `seqonly_maskon` | fiber off |
| `seq_maskoff_capA/capB` | different `bg_params` + `caplow` pkl |
| `seq_maskoff_bgtss` | `bg_params_tss.pkl` |
| `seq_maskoff_lowabf1` | `..._pseudo_lowabf1.pkl` |
| `seq_maskoff_12tfs` | reduced motif set |
| `*_em10` | EM on |

### 3.2 [P1] Hardcoded relative paths to parameter files
`pkg/robocop/robocop.py:522, 598, 602, 606`:
```python
open('inputs/abf1_reb1_params.pkl')
open('inputs/all_TFs_1000pealVal_params_pseudo.pkl')
open('inputs/nucleosome_params.pkl')
open('inputs/bg_params.pkl')
```
Relative, so **every run must have cwd = `analysis/`**, and swapping a parameter set means
editing library source (which is why `pkgvar/` exists). Move to `config.ini`.

### 3.3 [P2] EM holds every segment's emission tensor in memory at once
`robocop_em.py:createInstances` builds and retains `d_segments` for all segments, whereas
`robocop_no_em` streams one at a time. At `n_states = 3485`, 7 layers, 5 kb, that is **~1 GB
per window** — 20 windows measured at 28.7 GB peak. This is a fork refactor (upstream passed
`(t, dshared)` and reloaded per segment). It caps training-set size; worth revisiting if
training on more than ~100 kb.

---

## 4. Real model changes — keep, but document and validate

These are the actual science and should be described in any writeup.

### 4.1 [P2] Emission tensor grew from 5 to 7 layers
`robocop.py:203`, `d['n_vars'] = 7` (upstream `5`). The two new layers are the Fiber-seq
Watson and Crick m6A channels. Emission is a product over layers, so a layer at 1.0 is off.

### 4.2 [P2] Fiber-seq data path
New throughout: `config.ini` gains `tech2 = Fiber`, `pileupFile`, `nucleotide`;
`getReads.getFiber_seq` / `getValuesFiber_seqOneFileNucleotide` parse the modkit pileup;
`update_data_emission_matrix_using_fiber_seq_counts_Bionomial` builds the binomial emission.
Note the pileup is read **in full into pandas at every run** (823 MB, ~7 GB resident).

### 4.3 [P2] Reverse-strand fiber parameter fix ("revfix")
`robocop.py:~640-660`. The reverse block needs both a mirror **and** a Watson/Crick channel
cross, because `reverse_complement()` applies both to the PWM. Validated: crossed minus
group correlates **+0.998** with the plus group, vs **-0.508** uncrossed. This is a genuine
bug fix — make sure it is described, not just present.

### 4.4 [P2] Sparse posterior storage
`save_sparse_posterior` / `update_sparse_posterior` / `get_sparse_todense` replace dense h5
datasets. Note `update_sparse_posterior` does `del f[k]` then recreate; HDF5 does not
reclaim that space, so `info.h5` grows across EM iterations.

### 4.5 [P3] Nucleosome/TF caller rewrites
`getNucleosomesRoboCOP_new.py` (+229) and `gettfsRoboCOP_new.py` (+360) exist alongside the
originals. Decide which ship.

---

## 5. Input data divergences

Eleven parameter pkls in `analysis/inputs/`, of which the code only ever reads four by
hardcoded name. The rest are alternates swapped in via `pkgvar/`:

```
all_TFs_1000pealVal_params.pkl              baseline
all_TFs_1000pealVal_params_pseudo.pkl       + pseudocounts  <- the shipped default
..._pseudo_caplow.pkl                       capped combined_low_count (capA/capB)
..._pseudo_lowabf1.pkl                      ABF1 down-weighted
..._pseudo_pm50bp.pkl                       +/-50 bp windows
bg_params.pkl                               shipped bg (m6A 0.1383)
bg_params_merged.pkl                        capA (0.1548)
bg_params_tss.pkl / _tss_top10.pkl          TSS-derived (capB 0.4402)
```

**[P1]** Decide which single parameter set ships, and record how it was fit. Standing rule
already in force: never overwrite `all_TFs_1000pealVal_params_pseudo.pkl` or
`bg_params.pkl`; reach the generator only through `fiber_params_lib.load()`.

**[P2]** The motif set itself: 7 of 153 shipped PWMs disagree with JASPAR/Rossi — see the
motif-source audit. Murphy's ABF1 spacer is a training artifact.

---

## 6. Open questions to settle

1. **Why was `iterations` set to 0?** Convenience, or did EM misbehave? Gates the whole
   concentration-training direction.
2. **What is the right cap?** 28 of 154 states pinned at `mean + 2*sd` means the ceiling,
   not the data, is setting the priors. The `nocap` run (job 12423037) is testing this.
3. **Does the prior help without a localizing signal?** The fiber-only EM run (job 12423035)
   answers it. If not, the lever is the emission model, not the concentrations.
4. **ABF1 evaluation rests on 5 chrI sites.** `n_ref = 5`, so TP 2 -> 3 is one locus. Any
   shipping claim about ABF1 accuracy needs more chromosomes.
5. **Is `unknown` at 0.1 defensible?** It is the largest TF state and cap-exempt.

---

## 7. Analysis tooling added by this fork

Not part of the model, but new relative to the paper, and worth deciding whether it ships:
`score_robocop.py` (quantitative scorer vs Chereji/Brogaard/MacIsaac), `make_conc_trainDir.py`
(lambda-patch a single concentration without retraining), `make_train_coords.py`,
`train_em10.py` / `train_em10_fiber.py`, `em_trace_report.py`, `em_smoke_gate.py`,
`make_posterior_viewer.py` + `posterior_viewer_template.html` (interactive posterior browser),
`nhp6a_diag.py`, `compare_caplow_runs.py`, and the `sbatch_*.sh` fleet.

---

*Last updated 2026-08-24. Line numbers are against `pkg/` at HEAD; the `pkgvar/*` copies
carry the same code at the same lines unless noted.*
