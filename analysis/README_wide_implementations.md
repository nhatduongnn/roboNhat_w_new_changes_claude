# Two implementations of the same widened-TF model

There are two ways to give a TF a state block longer than its motif in this tree. They
produce **bit-identical geometry** (verified: `n_states` 3503, `nuc_start` 2817,
`silent_states_begin` 3348, and all 154 entries of `tf_lens` / `tf_starts` match). Use the
second one for new work.

## 1. `wide` / `wideAnull` -- superseded, kept as evidence

Pads implemented by editing `robocop.py` inside frozen package copies: **124 changed lines
across 6 sites**, with per-TF `(left, right)` read from `tf_pads.tsv` at train time and
frozen into `HMMconfig.pkl` as `dshared['tf_pads']`.

    pkgvar/seq_maskoff_wide/  pkgvar/seq_maskoff_widenull/
    robocop_train_wideA/      robocop_train_wideAnull/
    robocop_chrI_wideA/       robocop_chrI_wideAnull/
    train_wide.py  run_split_variant_wide.py  verify_wide.py  check_wide_traindir.py
    sbatch_train_wide.sh  sbatch_chrI_wide.sh  run_wide_all.sh

**Its scoring outputs were deleted** on request (2026-09-01): `wide_scores/`,
`wide_factor_scores.{tsv,json}`, `wide_runs.tsv`, `sbatch_score_wide.sh`. The numbers below
are the record; nothing re-reads those files. `make_factor_chart.py` still carries `wideA` /
`wideAnull` rows in its FAMILY table -- harmless, they simply match no run now.

The decodes and trainDirs above are still on disk. Re-scoring them would need
`--pkg pkgvar/seq_maskoff_wide`, since their posterior collapse is motif-core-only; without
that flag the numbers would not mean what the originals meant.

Its result stands: widening ABF1 to 7/2 made detection **worse** (enrichment 33.6x -> 8.6x,
FPs 81 -> 164, background posterior tripled). Pad columns say "protected", and so does
nucleosomal DNA. In that build the pads were sequence-neutral (`pwm['background']`) and
`tf_prob` was held bit-identical to baseline, so `wideAnull` existed as the length-prior
control.

## 2. `widememe` -- current

The same model, reached as pure data plus **one changed line**:

| piece | file |
|---|---|
| widened motif | `inputs/motifs_meme_wide.txt`, built by `make_meme_wide.py` |
| widened Fiber vector | `inputs/all_TFs_1000pealVal_params_pseudo_wide.pkl` (reused) |
| the one line | `pkgvar/seq_maskoff_widememe/robocop/robocop.py:598`, the pkl filename |
| config | `config_fiberonly_widememe.ini`, `pwmFile =` the widened meme |
| drivers | `train_widememe.py`, `run_split_widememe.py` |
| gates | `check_widememe_traindir.py` |
| launcher | `run_widememe_all.sh` |

`diff -rq pkgvar/seq_maskoff_widememe pkgvar/seq_maskoff` returns exactly one file, and the
diff on it is exactly one line. `get_transition_matrix_info` already derives `tf_lens` from
`pwm[tf].shape[1]`, `stack_pwms` vstacks any width, `reverse_complement` mirrors the pads
onto the reverse block, and `bc.c` always took `motif_lens` as a per-TF array -- so the
shipped code widens the model unaided.

**Two differences from `wideA` that change how results read:**

1. The pad columns carry the **real estimated flank composition** (from the 341 Rossi ABF1
   sites), not background. `calculateKD` scores each column's argmax, so RoboCOP reads the
   wider motif as tighter-binding and -- because that Kd is fed in as a concentration --
   assigns it a *lower* prior. ABF1's `tf_prob` lands at **0.105x baseline** (0.537x from
   the `0.9332^9` length cancellation, 5.12x further from the pads). Deliberately left
   uncompensated; undo without retraining via
   `make_conc_trainDir.py --tf Abf1_murphy --lam 5.1185`.
2. The posterior collapses over the **whole 23 bp block**, since the shipped
   `sum_for_dbf_probs` is used unmodified. ABF1 renders as a 23 bp plateau rather than a
   14 bp peak -- which is the point, it is visible in the posterior viewer -- but it inflates
   the genome-wide mean ~1.6x and deflates enrichment by the same factor. `score_robocop.py`
   numbers need that correction before being compared to baseline or `wideA`.

A `tf_pads*.tsv` remains the single source of truth for pad widths: both `make_meme_wide.py`
and `make_params_wide.py` read one, so the sequence and Fiber layers cannot disagree.

## The widened runs built with method 2

| run | widened | pads | `n_states` | pad config | Fiber source |
|---|---|---|---|---|---|
| `widememe` | Abf1_murphy only | 7 left / 2 right | 3503 | `tf_pads.tsv` | pm50 |
| `wide10` | the 12 fitted TFs | 10 / 10 | 3965 | `tf_pads_wide10.tsv` | pm50 |
| `widefp` | the 12 fitted TFs | per-TF footprint | 3765 | `tf_pads_widefp.tsv` | pm50 |
| `wide10all` | all 153 motifs | 10 / 10 | 9605 | `tf_pads_wide10all.tsv` | pm50 |
| `wide150` | the 12 fitted TFs | 150 / 150 | 10685 | `tf_pads_wide150.tsv` | pm200 |
| `wide150all` | all 153 motifs | 150 / 150 | 95285 | `tf_pads_wide150all.tsv` | pm200 -- **BLOCKED, see below** |

Each has its own meme file, params pkl, pkgvar (one line), config, trainDir and two decodes
(chrI + chrXIV), gated by `check_widememe_traindir.py`.

## Pads wider than +/-50: the pm200 refit

`make_params_wide.py` slices the widened Fiber vector out of a fixed-window refit whose
centre columns are bit-identical to the shipped motif-length fit (its GATE 1). The original
refit is only 101 columns (`..._pm50bp.pkl`, centre +/-50), so it cannot supply a +/-150
pad. `make_params_wide.py` therefore takes `--src` and `--half`, defaulting to the pm50
behaviour so every earlier variant rebuilds bit-for-bit, and `wide150`/`wide150all` pass

    --src inputs/all_TFs_1000pealVal_params_pseudo_pm200bp.pkl --half 200

the 401-column refit built by the same generator (`make_params_pm50.py --half 200`). GATE 1
passes on it unchanged for all 12 fitted TFs on both strands, and it carries
`combined_low_count` at 401 columns, which is what makes the all-153 variant possible.
That refit bounds the pad: the widest motif is 20 bp at [190,210), so +/-190 is the
maximum it can supply.

`make_meme_wide.py` now checks each site's flank window against the contig length before
fetching and drops -- loudly -- any site that would be clipped, instead of letting
`pysam.fetch` silently truncate and put motif bases in a flank column. At +/-150 no Rossi
site in the bed is within reach of a chromosome end, so nothing is dropped in practice; the
check exists because +/-10 never came close and +/-150 does.

## The state-space ceiling: n_states <= 46340

`wide150all` is built and gated but **cannot be run against the shipped
`pkg/robocop/librobocop.so`**. Every 2-D and 3-D index in the C core is linearized in
32-bit `int`:

    int I(int row, int col, int ncol) { return row * ncol + col; }   // pkg/robocop/bc.c:14

The last element of an `n_states x n_states` matrix sits at `(n-1)*n + (n-1)`, which passes
`INT_MAX` at **n_states 46341**. At 95285 the true index 9,079,231,224 comes back as
489,296,632 -- verified by calling the shipped `.so` directly in
`sbatch_wide150all_memcheck.sh`. `construct_transition_matrix`,
`find_parents_and_children`, `set_initial_probs` and the forward/backward passes would all
write outside their buffers, and the emission index
`I3(6, n_obs-1, n_obs, n-1, n_states)` = 3,334,974,999 overflows too. A segfault is the
good outcome; silently wrong posteriors is the other one.

So with the library as shipped the ceiling is n_states 46340, i.e. a uniform pad of at most
~70 columns across all 153 motifs. `wide150` (10685) is far inside it; `wide10all` (9605)
always was. Lifting the ceiling means widening the index type in `bc.c` / `bc.h` / `algo.c`
to `long`/`size_t`, rebuilding a **separate** `librobocop.so`, and pointing only that
variant's `cshared` at it -- a change to the numerical core that has to be re-validated
against an existing run before it is trusted. `run_wide150all_all.sh` refuses to submit
until then (override with `I_KNOW_INDEX_OVERFLOW=1`).

Memory would also be severe. Measured on a compute node (`sbatch_wide150all_memcheck.sh`,
`--mem=400G`): the three dense square matrices -- `t_mat` float64 plus `parents_mat` and
`children_mat` int64 -- allocate and fully touch at **203 GB resident**, in 16 s. On top of
that sits the `(7, n_obs, n_states)` emission tensor at 24.8 GB per segment, and
`robocop_em` holds all 20 training windows live -- 497 GB. That is the term that actually
dominates (it explains `wide10all`'s measured 75 GB train / 20 GB decode, being linear in
n_states), giving a train peak near 711 GB and a decode peak near 238 GB, plus a ~68 GB
`HMMconfig.pkl` on disk because the transition matrix is pickled into it. Only the
1,150,000 MB `compsci-cluster-fitz-*` nodes could hold the training job.

**`wide10all`'s pooled path.** 141 of the 153 motifs have no individual Fiber-seq fit
(RoboCOP's own >=50-site rule, `abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:54,147`). They
get the pooled treatment on both layers: sequence flanks stacked once from the 942 sites of
the 62 sub-threshold TFs and shared by all 141 (79 of which have no sites at all), and a
`combined_low_count` +/-50bp slice cut **to each TF's own motif length**. That last point
matters -- the 141 have different motif widths, and numpy would silently broadcast a
length-1 scalar to any width, so a mis-sized slice would degrade to a flat scalar with no
error. `check_widememe_traindir.py` gates against exactly that.

**Measured, 20 kb of chrXIV, posterior runs above 0.10:**

    run        Abf1   Reb1  Nhp6a   nucleosomes
    baseline   14 bp   8 bp   7 bp  153 calls, 112 bp spacing
    widememe   23 bp   8 bp   7 bp  151 calls, 112 bp
    wide10all  38 bp  28 bp  27 bp  154 calls, 109 bp

Every plateau matches its designed block width (Reb1 8+20, Nhp6a 7+20), and nucleosomes are
untouched even at 2.76x the state space. But TF call counts collapse in `wide10all`
(Nhp6a 61 -> 7, Abf1 10 -> 1): that is the estimated-pad prior suppression, measured at
0.0003x-0.075x of baseline per TF (median 0.069x). Recoverable per-TF via
`make_conc_trainDir.py --tf <name> --lam <1/ratio>` without retraining.

**A gate lesson worth keeping.** The first `wide10` train "failed" on a check asserting that
non-widened TFs keep their baseline prior. They do not, and cannot: `convert_to_prob` solves
ONE unbound root `p` across all motif lengths and then renormalises, so
`prob_new/prob_base = (p_new/p_base)^len * (S_base/S_new)` for every TF. Widening 12 TFs by
20 bp moves untouched TFs by ~1%. The check now asserts the two things that really are
invariant: `tf_prob` is bit-identical to a first-principles `getDBFconc` recomputation, and
once the shared root shift is divided out every non-widened TF lands on one constant
(residual spread 4.4e-14). No tolerance guessing.
