# Retired runs

Moved here rather than deleted, so nothing is lost that cost compute. Safe to `rm -rf` any
subdirectory once you are satisfied with what replaced it.

## `wide10_uniform/` — the uniform ±10 widening of the 12 fitted TFs

Retired 2026-09-04 at the user's request. It padded all 12 individually-fitted TFs by a
**uniform 10 columns** on each side, which is not the per-TF geometry wanted: a footprint is
not symmetric about its motif, and the widths that matter differ per factor. The per-TF
footprint variant (formerly `widefp`) now carries the `wide12` name and occupies this slot.

Contains the whole run: four decodes, the trainDir, `tf_pads_wide10.tsv`,
`config_fiberonly_wide10.ini`, `run_wide10_all.sh`, its meme file and params pkl under
`inputs/`, its two frozen pkgvars under `pkgvar/`, and under `reports/` the four
`score_robocop.py` reports it produced (prefixed by chromosome, since the two chromosomes
use the same report basename).

**It is NOT reproducible from what remains at the top level alone** — restore by moving the
pieces back to `analysis/` and `analysis/inputs/` and `analysis/pkgvar/`. The sbatch drivers
it used (`sbatch_train_wide10.sh`, `sbatch_{chrI,chrXIV}_wide10.sh`) were **not** moved: despite
the name they are the generic widened-run drivers that `wide12`, `wide150`, `wide10all` and
`wide150all` all call.

**Its measured result, kept because it is real evidence.** The uniform ±10 padding scored
*better* on ABF1 than the per-TF footprint padding that replaced it:

    chrXIV (held out, 19 MacIsaac sites)   F1      enr(block-corrected)   AUROC
    uniform ±10        (this, retired)     0.192          393x            0.702
    per-TF footprint   (now `wide12`)      0.038           39x            0.645

The per-TF variant was chosen on geometric grounds — it encodes the specific left/right
widths intended for each factor — not because it scored higher. Anyone comparing the two
later should know the uniform run existed and won on this metric.

## `widefp_pkgvars_superseded/`

The two frozen package copies (`seq_maskoff_widefp`, `fiber_maskoff_widefp`) that the run
now called `wide12` was decoded with. Superseded by `pkgvar/{seq,fiber}_maskoff_wide12`,
which are byte-identical except for the params-pkl filename on line 598. Kept because a
decode's provenance is the pkgvar it ran under, and these are what actually produced the
posteriors now sitting in `robocop_*_wide12/`.
