# Locus slide template

One slide = one ~2 kb genomic window drawn the way the **RoboCOP Occupancy Browser** draws it
(`analysis/posterior_viewer_template.html`), sized for the deck's 1600 × 900 stage. Built from it in
`presentation/talk.html`: slides 5 ("One ABF1 locus up close") and 5b–5d, 6 (EM), 10b (tuned models at ERV46) and hidden 10c
(same concentrations, job A).

## What the slide shows

| row | content |
|---|---|
| sites strip (step 3) | one chip per reference site: `hit/miss · start–end · <call run> <TF> <posterior>` |
| one row per run (step 1: first run, step 2: the rest) | the highlighted factor's posterior (factor colour from the decode's `dbf_color_map.pkl`) + nucleosome occupancy (grey area), 0–1; reference sites as tinted bands with a top bar; optional Rossi summits as small dark triangles on the top edge |
| m6A per A | Watson (blue) and Crick (orange) methylated fraction per A, one dot per A with coverage, dashed background rate; counts from one run (`fiberRun`, default the first), identical in every run |
| genes | sacCer3 GTF transcripts, + strand above / − strand below |
| axis | bp ticks at 1/2/5 × 10^k |
| outlines (step 3) | a box around each reference site spanning all run rows and the m6A panel |

Captions stay neutral: what is plotted, not why.

## Files

- `build_locus_slide.py` builds the data JSON and fills the section markup.
- `locus_slide_snippet.html` holds the component: `<style id="locus-slide-css">`, `<template id="locus-section">` (placeholders `{{...}}`, filled by the builder) and `<script id="locus-slide-js">` (`drawLocus(root, data)` + `drawAllLoci()`).

## Inputs

- `--region chrom:start-end` is the window, 1-based inclusive. 2,001 bp (for example `chrXIV:186900-188900`) gives about 0.6 px per bp at stage width, so the posteriors draw as polylines rather than envelopes.
- `--factor` is an optable column, e.g. `Abf1_murphy` or `Reb1_badis`. `--tf-label` / `--ref-tf` default to the prefix in upper case (`ABF1`, `REB1`).
- `--runs analysis/viewer_runs_<chrom>.tsv --labels fib,fib+seq,seq` is the run table (label → decode dir) and the rows, in order. `--run LABEL=DIR` adds or overrides a run.
- `--sites macisaac_bed` (the default) reads `inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1_match_PWM.bed` through `make_posterior_viewer.load_ref_sites`. These are the browser's bands, ABF1 and REB1 only. `--sites macisaac_c1` uses the merged 119-TF gff intervals (interval union, 20 bp slop, as in `make_conc_targets.py`).
- `--rossi /usr/project/xtmp/nd141/projects/data/rossi_strand/<Tf>_CX.bed` is optional; it draws the Rossi ChExMix summits in the window.
- There are two data sources, and both call `analysis/make_posterior_viewer.py`, which is imported, never edited:
  - **Decode mode** (the default) runs `build_region(<window>, None, runs)`, which reads `tmpDir/info*.h5`. A 2 kb window takes about a minute per run, so use sbatch on the cluster.
  - **Viewer mode** (`--from-viewer analysis/posterior_viewer_all.html --viewer-region erv46`) slices the payload of an already built browser page. It gives the same numbers in seconds, but only for windows inside the browser's regions (`analysis/viewer_regions.tsv`).
- The slide chrome options are `--slide-id --data-id --kind main|hidden|backup --sec --tag --title --eyebrow --msg --foot`.
- Layout options:
  - `--labels` sets the rows, with `--descs "a|b|c"` and `--swatches fiber,both,seq` for their text and colour chip.
  - `--row-steps 1,2,1,2` sets the build step of each row.
  - `--post-h` sets the row height (default 100) and `--meth-h` the m6A panel height (default 104; `0` drops the panel and its legend entries).
  - `--call-run` and `--call-also run1,run2` choose which run decides hit/miss and which other runs' maxima the chip also lists.
  - `--extra-html file` inserts markup between the tracks and the footnote, for example a numbers strip.
  - `--tf-color` overrides the factor colour.

`--from-json` reuses a built payload so the section text can be re-filled without touching the decodes.

Output: `--out-json` (the payload) and `--out-section` (the filled `<section>` followed by its `<script type="application/json" id="trk<id>-data">`).

### Data schema

```
{chrom, s, e, region:[start,end] (source/browser region, for the footnote),
 factor, tfLabel, refLabel, runOrder:[labels],
 runs:{label:{tf:[round(p*1000)]*(e-s+1), nuc:[...]}},
 fiberRun, fiber:{mw, mc, aw, ac}, sites:[[start,end,strand]], rossi?:[summit],
 genes:[{name,start,end,strand}], bg, colors:{tf, nuc, ref},
 call:{run, thr, tol}}
```

## Build a new locus slide and insert it

```bash
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
cd presentation/templates/locus_slide
python build_locus_slide.py --region chrXIV:465700-467700 --factor Abf1_murphy \
    --runs ../../../analysis/viewer_runs_chrXIV.tsv --labels fib,fib+seq,seq \
    --rossi /usr/project/xtmp/nd141/projects/data/rossi_strand/Abf1_CX.bed \
    --slide-id 5c --title "One ABF1 locus up close (3)" --kind hidden \
    --msg "..." --foot "<span>...</span>" --out-json 5c.json --out-section 5c.section.html
```

Then, in `talk.html`:

1. Paste the `<section>` part of `5c.section.html` where the slide belongs in `#stage`. Order in the DOM is the navigation order.
2. Paste its `<script type="application/json">` next to the other locus data scripts, before the main `<script>`.
3. Add a `NOTES["5c"]` entry.
4. The CSS and the `drawLocus` JS are already in the deck, once. Every `.trk[data-locus]` is drawn on load, resize, font load and theme change. For a new deck, paste the snippet's `<style>` rules into the main stylesheet (the `@media` lines go into the deck's narrow-screen block) and the `<script>` body into the main script before the navigation code.
5. Regenerate `talk_offline.html`, run `node --check` on the extracted scripts, and republish.

Hidden slides use `data-kind="hidden"` and are skipped by arrow navigation. `h` includes them and `o` (the overview) shows them with a dashed border.

## Hit/miss rule (step-3 chips)

`hit` if the **call run's** (default `fib+seq`) posterior for the factor reaches **≥ 0.10 anywhere within ±20 bp of the site interval** (`[start−20, end+20]`), else `miss`. The chip prints that maximum. The same rule is implemented in `site_calls()` (Python, printed at build time) and in `drawLocus` (JS).

This is **not** the slide-4 scorer rule. `score_robocop.py` calls runs of posterior ≥ `max(0.10, 0.30 × whole-chromosome max)`, which is ≈ 0.30 for these runs. It takes the centre of each run and matches it greedily to MacIsaac site centres within ±20 bp. It is also not the tuning rule: `count_calls.py` uses a fixed 0.10, run centres, and a 30 bp match. At slide 5's two sites all three rules agree (0.00 → miss, 0.99 → hit). A site whose maximum falls between 0.10 and 0.30 would be a chip "hit" but a slide-4 "not found".

## Sizing

The stage is 1600 × 900 with 80 px side padding, so the track grid is 1440 px wide: a 150 px label column, a 14 px gap, and the canvas. Inside the canvas, 56 px is the y-label gutter and 12 px is right padding. Row heights are 3 runs × 100, m6A 104, genes 52 and axis 28, plus the 32 px chip strip and 5 px gaps. With the header, legend and footnote this fits 900 px with room to spare. **More than 3 runs will not fit** at the default heights. Slide 6 (the EM slide, 7b until v4.5) has 2 runs at 130 px with `--meth-h 0` plus a one-line numbers strip; 4 runs plus a table also fit at `--post-h 64 --meth-h 0`. Chips are clamped inside the plot. Two sites closer than about 300 px (about 450 bp in a 2 kb window) will have overlapping chips, so pick windows accordingly or shorten `--msg`. Below 760 px wide, the deck's narrow-screen rules shrink the label column and hide the descriptions.

## Theme

All chrome colours are deck tokens (`--rule`, `--muted`, `--ink-2`, `--wat`, `--cri`, `--bgline`, `--gene-*`), read with `getComputedStyle` at draw time. A `MutationObserver` on `data-theme` and a `prefers-color-scheme` listener redraw the canvases, so light, dark and system all resolve. Data colours are fixed so they match the browser and the PNGs:

- factor line: from `dbf_color_map.pkl`. Some entries are too pale to read on the light plot, for example REB1's `#d9f3e2`; override those with `--tf-color` (slide 5d uses `#1fa81c`) and say so in the footnote
- nucleosome: `#b3b3b3`
- reference: MacIsaac ABF1 `#d4145a`, REB1 `#0d7a8c`

The reference colour reaches the chips, bands and legend through `--locus-ref`, which is set inline on the `<section>` only when it differs from ABF1's.
