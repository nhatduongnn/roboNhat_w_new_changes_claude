# Talk plan — RoboCOP + Fiber-seq (lab meeting, 2026-09-15) — FINAL v4

Maintained by agent_ppt. **v4, 2026-09-14 ~22:10 EDT.** This version folds in the user's
review of v3 and the finished tuner-v2 campaigns (fw01 / sw01 / bw01).

**v4.1 edits (2026-09-15, user requests):** title slide now visible (first slide, name Nhat
Duong); outline sentence removed; RoboCOP recap hidden (H1a, after slide 1; slide numbers
unchanged, so visible numbering skips 2); roadmap box 1 has two sub-steps; slide 5 now
embeds occupancy-browser tracks instead of PNG crops.

**v4.2 edits (2026-09-15, user requests, agent_ppt):**
- Slide 5 is now drawn by the reusable locus-slide template (`presentation/templates/locus_slide/`). Its data are value-identical and its markup unchanged apart from `data-locus`.
- New hidden locus slides 5b, 5c and 5d.
- New **7b**, a visible alternative to slide 7 tagged "pick one": EM at one chrXIV ABF1 locus.
- Slide 8 loses its "Removed" and "Guards" columns; the guards move to hidden **8b**.
- Slide 9 has wider table spacing and a P/R/F1 definition; slides 10, H9a, H10a, B1 and B2 reference it.
- Slide 10's title says each model was tuned separately.
- New **10b**, visible companion to 10: the three tuned models at ERV46.
- New hidden **10c**: same concentrations (job A) at a chrXIV locus.

**v4.3 edits (2026-09-15, user decision, Rossi scope):**
- Rossi precision / recall / F1 on the slides now use **all 58 tuning groups** (`fitted=all`, `ref=rossi_cx`), including the 9 fitted-footprint groups (ABF1, CIN5, FHL1, FKH1, MCM1, RAP1, REB1, SKO1, UME6). The circularity is stated, not excluded.
- Slide 6: the circularity line is replaced (all 58 groups; 9 fitted footprints estimated at Rossi peaks, so fiber-layer runs are optimistic; sequence-only unaffected). Notes updated.
- Slide 10: table is now MacIsaac all + Rossi all on both tuning and held-out chromosomes (the two MacIsaac non-fitted columns are dropped so the columns are comparable); chart uses all-groups held-out values, y-axis 0–0.10; sequence-only bullet now "roughly tripled or better (2.9–4.6×)"; caveat chip replaced by the footnote "Rossi: all 58 groups; see slide 6 on circularity"; non-fitted values kept in the speaker notes only.
- B2: Rossi values unchanged (they were already pooled over all 58 groups); caveat reworded, footnote added, non-fitted values in notes.
- Full old → new table and sources: §9, "v4.3".
- Slide 5 (coordinator request): Rossi ABF1 ChExMix summits added at 61,171 and 62,665 (data `rossi`, legend "Rossi ABF1 summit", aria-label, speaker notes). 7b / 10b / 10c already carried theirs; unchanged.

**v4.5 edits (2026-09-15, user request): EM before ground truth.**
- The EM slide ("EM makes the calls worse", was **7b**) is now **slide 6**, right after 5d.
- "Ground truth, and the tuning set" (was **6**) is now **slide 7**, right after the new 6.
- The hidden "Why the priors need tuning" (was **7**) is now **H6a**, placed after slide 6 and tagged "Hidden · replaced by slide 6"; its audit line now reads "(slide 8)" instead of "(next slide)", since the slide after it is now ground truth.
- References updated: the circularity pointer on slide 10's caveat chip and B2's footnote now says slide 7; 10c notes "same site as slide 7b" → "slide 6"; notes prefixes "[Replaces hidden slide 7.]" → "[Replaces hidden slide H6a.]" and "[Hidden; slide 7b is shown in its place.]" → "[Hidden; slide 6 is shown in its place.]"; notes keys, URL-hash ids (`#6.n` = EM, `#7.n` = ground truth, `#H6a.n`; old `#7b.n` falls back to the first slide), the locus payload id `trk7b-data` → `trk6-data`, and a CSS comment.
- Outline ranges unchanged: Part 2 "slides 4–5d", Part 3 "Ground truth and tuning" "slides 6–8" (the EM slide opens Part 3 as the motivation for tuning; `data-sec="3"` kept).
- Visible order: T, 1, 3, 4, 5, 5b, 5c, 5d, 6 (EM), 7 (ground truth), 8, 9, 10, 10b, 11.
- The v4.2–v4.4 change notes below keep the ids used at the time: their 6 is now 7, their 7 is now H6a, their 7b is now 6.

**v4.7 edits (2026-09-15, user request): fiber-only rows back on slide 6.**
- Slide 6 ("EM makes the calls worse") has 4 locus rows again: fib+seq, fib+seq+em10, fib, fib+em10 (decodes `robocop_chrXIV_fib`, `_fib_em10`). Title, message, legend, Rossi summit, prior line and fib+seq numbers unchanged.
- `trk6-data` restored from the v4 page's `trk7b-data` (scratchpad `talk_v4_before_v44.html`): `fib+seq` / `fib+seq+em10` arrays and all other fields are identical to v4.6; `fib` / `fib+em10` added, `runOrder` and `call.also` list all four runs (chip shows all four site posteriors); `fiberRun` kept at fib+seq (fiber payload identical either way, no m6A panel). Checked: site max fib 1.00 → fib+em10 0.375; window runs ≥ 0.30 fib 3 → 8.
- Rows 72 px each (was 130 × 2; v4 used 64 × 4). Build steps 4: 1 fib+seq row → 2 fib+seq+em10 row → 3 fib and fib+em10 rows → 4 site outline/chip + numbers + prior line.
- Numbers block is now a two-row aligned grid (`.emline.two`): the unchanged fib+seq line plus **fiber only: found 7 → 5, calls 1,429 → 1,640, F1 0.0097 → 0.0060**, re-read from `layerXIV_scores/report_fib.json` (tp 7, n_pred 1429, f1 0.00967) and `chrXIV_scores/report_fibem10.json` (tp 5, n_pred 1640, f1 0.00603).
- Estimated body height 631 px of 661 available (slide content ≈ 870 of 900). Speaker notes rewritten: fib rows are on the slide; sequence-only and prior details stay in notes. Footnote lists the four decodes.

**v4.6 edits (2026-09-15, user request): slide 11 rewritten.**
- Slide 11 body replaced by the user's five bullets (see §11 entry); title kept; message now "Where these runs point so far, and what to try on the fiber layer."
- **Qualifier added to bullet 2** so it does not contradict bullet 3 (sw01 tripled F1, slides 9–10): the user's "doesn't quite help with accuracy" became "with the fiber layer on, accuracy barely improves". Bullet 3 "Sequence-layer tuning" became "Tuning the sequence layer alone" for the same reason. No new results added.
- Removed from the slide and moved to its speaker notes: the two ✔ lines' roadmap context, "examine the runs", "fix the tuner's bisection bracket", "re-tune on more chromosomes → single-fiber decoding → cluster fibers", and the compact roadmap (box 1 status). The ✔ "tuner that matches counts without distorting the model (three campaigns, identical rules)" line also moved to the notes; the ✔ "sequence-only was most accurate" line is covered by bullet 3.
- Build steps: 5 (one bullet each). Offline copy regenerated with `scratchpad/mk_offline.py`.

**Format:** a **web artifact** presentation, not PowerPoint.
- One slide, or one build step, per scroll-wheel notch or arrow key.
- **Hidden** slides are present in the artifact but skipped during navigation; toggle them
  with a key (suggest `H`).
- Figures are PNGs embedded in the artifact. Tables and charts are drawn inline from the data
  in §5.

**Path conventions:** paths are relative to `analysis/` unless they start with `/`. `CT/` means
`analysis/conc_tuning/`. Artifact IDs take the prefix `https://claude.ai/code/artifact/`.

**Status tags used below:**
- **VERIFIED** — re-read from the named file, by me or the coordinator.
- **HANDOFF** — taken from HANDOFF.md, not re-derived.
- **PENDING** — waiting on an investigation.
- **CAVEAT** — the user examines the runs before any claim goes on the slide.

---

## 0. Bottom line

- **The talk, in one sentence.** We tuned three RoboCOP configurations (fiber-only,
  sequence-only, both layers) with identical rules. **In these runs**, the sequence-only model
  improved most, and its gain held on a held-out chromosome and against Rossi. Adding the
  fiber layer lowered accuracy relative to sequence-only. Fiber-only matched call counts but
  not accuracy.
- **Length:** 11 visible slides (Title, Outline, then slides 3–11), about 30–35 min. There are
  3 hidden slides (H1a, H9a, H10a), 3 backups, and 5 deferred slides.
- **Two slides are gated tonight or tomorrow morning:**
  - Visible slide 4 (ABF1 table): **CAVEAT**; the user examines the runs first.
  - Visible slide 5 (one locus): **PENDING**; the mechanism is under investigation.
- **Old campaigns.** u001 / m001 (genome-wide, old tuner) are **legacy context only**. fm001 /
  sm001 / fm002 / sm002 are gone from the main slides.

---

## 1. Final slide order

### Visible
| # | slide (legacy ID) | one-line message |
|---|---|---|
| T | **Title** (S1) | RoboCOP meets Fiber-seq: aggregate decoding and concentration tuning. |
| 1 | **Outline** (new) | Section names only (no message line). |
| 3 | **Goal and roadmap** (S3) | Goal: single-fiber decoding with better TF locations; we are in the aggregate Fiber-seq phase. |
| 4 | **ABF1 calls: fiber only vs fiber + sequence** (S9) — CAVEAT | Shows ABF1 calls against MacIsaac sites on chrI and chrXIV for the two layer configurations (descriptive, no conclusion). |
| 5 | **One ABF1 locus up close** (S10) — PENDING | Occupancy-browser tracks for fib / fib+seq / seq at one MacIsaac ABF1 hit and one miss (the mechanism claim is withheld until confirmed). |
| 5b | **One ABF1 locus up close (2)** | chrXIV:186,900–188,900; the Rossi-supported site 187,879 is called by fiber alone (fib 1.00, fib+seq 1.00, seq 0.00), next to MacIsaac site 187,697, which no run calls. (Unhidden v4.4.) |
| 5c | **One ABF1 locus up close (3)** | chrXIV:465,700–467,700; the Rossi-supported site 466,725 is missed by every run, with site m6A 0.21 against a background of 0.138. (Unhidden v4.4.) |
| 5d | **One REB1 locus up close** | chrXIV:416,650–418,650; the Rossi-supported REB1 site 417,645 is a hit in fib+seq (fib 0.72, fib+seq 1.00, seq 0.06). (Unhidden v4.4.) |
| 6 | **EM makes the calls worse** (was 7b) | EM training adds many calls but few correct ones. Held-out chrXIV near GIS2 (site 167,958–167,971): fib+seq / fib+seq+em10 rows, a found/calls/F1 line and a priors-after-EM line (§9, "6"). |
| 7 | **Ground truth, and the tuning set** (S11; was 6) | Tune against MacIsaac, validate against held-out Rossi, restricted to the 58 factor groups the two share. |
| 8 | **Tuning method (v2)** (S13) | Count-matching per group on chrXIV + chrII, chrIV held out, three campaigns with identical rules. (Removed/Guards columns dropped 2026-09-15; guards on H 8b.) |
| 9 | **Results per campaign** (S14) | All three reach their count targets to different degrees, and accuracy moves very differently between them. |
| 10 | **Layer comparison, each model tuned separately** (S15) | In these runs, sequence-only is the most accurate on tuned and held-out data; adding fiber lowered accuracy; fiber-only matched counts, not accuracy. |
| 10b | **Tuned models at the ERV46 locus** — ALT, "companion to slide 10" | chrI:60,900–62,900 (slide 5's window), rows fw01 / sw01 / bw01, each at its own tuned concentrations (job C genome decodes). |
| 11 | **Lessons and next steps** (S16) | Count-matching sets operating points; the fiber emission is the next thing to fix before single fibers. |

### Hidden (in the artifact, skipped unless toggled)
| # | slide | why hidden |
|---|---|---|
| H1a | **RoboCOP recap** (S2) | User decision (2026-09-15): hidden. Placed after slide 1. |
| H6a | **Why the priors need tuning** (S12; was 7) | User decision (v4.4): replaced by the EM slide, now slide 6. Placed after slide 6, tagged "Hidden · replaced by slide 6". |
| 8b | **Tuning search rules** | After slide 8. The guard list plus a plain-language sentence. |
| H9a | **Fiber-only detail: why fw01 stalls** | Placed after slide 9. Explains fw01's 13 unconverged groups if asked. |
| 10c | **Same concentrations, three layer setups** | After 10b. Job A (sw01 round-5 weights) at chrXIV 167,958. Site posterior: seq 0.00, fib+seq 1.00, fib 1.00. Calls in the window (≥ 0.10): 0 / 5 / 14. |
| H10a | **Legacy comparison: old tuner on the same scope** | Placed after slide 10. For the question "did the old campaigns do better?" |

### Backup (after slide 11)
| # | slide |
|---|---|
| B1 | **What the tuning audit found** (decoder weight w, the hold artefact, held vs unheld evidence) |
| B2 | **Legacy genome-wide campaigns u001 / m001** (old S14: counts converge, F1 flat after step 1) |
| B3 | **Motif audit, "Odd One Out"** (7 shipped matrices are the outlier) |

### Deferred (not built into the artifact; content kept in §3)
The deferred slides are old S4 (what Fiber-seq measures), S5 (how Fiber-seq enters the model),
S6 (code changes), S7 (fitted footprints) and S8 (nucleosomes).

---

## 2. Slides in full

Every slide entry below has title, message, on-slide content, figure/table and its source,
**build steps** (one per scroll or arrow step), **figure source path**, speaker notes, and
priority.

---

### T. Title (S1) — VISIBLE (first slide)
- **Message:** RoboCOP reading Fiber-seq, tuned so layers can be compared fairly.
- **On slide:**
  - Title: "RoboCOP meets Fiber-seq: aggregate decoding and concentration tuning"
  - Nhat Duong · lab meeting · 15 Sep 2026 (name from the repo's git user.name)
- **Build steps:** 1 (static).
- **Figure source path:** none.
- **Speaker notes:** none.
- **Source:** n/a.
- **Priority:** core; made visible by user decision (2026-09-15, see D1).

---

### 1. Outline (new)
- **Message:** where the talk goes.
- **On slide:** four numbered sections, each listing its slide numbers; no message sentence.
  1. **RoboCOP and the goal**: slide 3
  2. **A first look at TF calls with Fiber-seq**: slides 4–5d
  3. **Ground truth and tuning**: slides 6–8 (6 = EM, 7 = ground truth, 8 = tuning method)
  4. **Results and next steps**: slides 9–11
- **Build steps:** 1 (static). Optionally each section highlights as the talk reaches it, if
  the builder adds a progress rail.
- **Figure source path:** none (text).
- **Speaker notes:** "Quick orientation. I'll remind you what RoboCOP is and where we're taking
  it, show a first look at TF calls with Fiber-seq, then spend most of the time on how we tuned
  the model and what the three tuned configurations tell us."
- **Source:** this plan.
- **Priority:** core.

---

### H1a. RoboCOP recap (S2) — HIDDEN (after slide 1)
- **Message:** RoboCOP is an HMM that turns sequence plus accessibility into a probabilistic
  occupancy map of nucleosomes and TFs, and each factor has a prior weight.
- **On slide:**
  - Hidden states: unbound DNA · one block per TF motif (fwd and rev) · a 147 bp nucleosome
  - Emissions: sequence (PWM) × accessibility layers (upstream: MNase-seq)
  - Output: a posterior for each position and factor; a factor is called where its
    posterior ≥ 0.10
  - Each factor has a prior weight, its "concentration"
- **Figure:** a state-architecture diagram. Background → TF blocks / nucleosome block →
  background, with emission layers stacked under it.
- **Build steps:**
  1. States diagram and the first bullet
  2. Emissions bullet, with the layer stack added under the diagram
  3. Output and prior bullets
  4. *Optional (see D2):* one extra line, "Here: + two Fiber-seq layers (Watson, Crick), a
     binomial m6A count per state"
- **Figure source path:** **NEEDS MAKING**, as inline SVG in the artifact. Content comes from
  `HANDOFF.md` §2 and §4.1. The paper PDF is not in the repo; only `SupplementaryMaterials.pdf`
  at the repo root.
- **Speaker notes:** "Most of you know RoboCOP: an HMM whose path walks through unbound DNA, TF
  motif blocks and nucleosomes, each emitting sequence and accessibility. The thing to hold on
  to is that every factor has a prior weight, and those weights decide how often a factor gets
  called; that's what the second half of the talk is about."
- **Source:** `HANDOFF.md` §2 (weights and priors), §4.1 (layers), §1.2 (0.10 call threshold).
- **Priority:** core.

---

### 3. Goal and roadmap (S3)
- **Message:** the goal is single-fiber decoding with better TF locations; we are in the
  aggregate phase.
- **On slide:** three boxes joined by arrows.
  1. **Aggregate Fiber-seq** — "we are here". Two sub-steps: "Incorporate fiber layer" →
     "Tune concentrations".
  2. **Single-fiber decoding**
  3. **Cluster fibers** into chromatin states
- **Build steps:**
  1. The three boxes, greyed
  2. Box 1 lights up with "we are here"
  3. Box 1's two sub-steps appear, with "Tune concentrations" highlighted as today's topic
- **Figure source path:** **NEEDS MAKING**, inline SVG.
- **Speaker notes:** "Fiber-seq gives us individual molecules, so the end goal is to decode
  single fibers and group them. First the model has to work on the aggregate signal, and we need
  to know what the fiber layer adds on top of sequence. Today is about that last step of box one."
- **Source:** `whattodo.md` "Phased strategy" and Tier 3 #8.
- **Priority:** core.

---

### 4. ABF1 calls: fiber only vs fiber + sequence (S9) — CAVEAT
**The user examines these runs before any claim is made.** No conclusion bullet goes on the
slide.

- **Message (neutral):** ABF1 calls, sites found and enrichment against MacIsaac on chrI and
  chrXIV, for fiber-only and fiber + sequence decodes.
- **On slide:** a table.

  | ABF1 vs MacIsaac (±20 bp) | MacIsaac sites | fiber only: found / calls / enrichment | fiber + sequence: found / calls / enrichment |
  |---|---|---|---|
  | chrI | 5 | 0 / 422 / 1.2× | 2 / 83 / 33.6× |
  | chrXIV | 19 | 7 / 1,429 / 6.2× | 8 / 327 / 33.3× |

  - Footnote: "Enrichment = mean ABF1 posterior at sites ÷ background. Untuned priors (λ = 1).
    chrI n = 5."
  - Banner, for the user during review only; remove before presenting: "CAVEAT: runs under
    examination."
- **Build steps:**
  1. Table header and the chrI row
  2. chrXIV row
  3. Footnote
- **Figure source path:** none (an inline HTML table).
- **Speaker notes (neutral, to finalise after the user's check):** "Here are ABF1 calls from the
  untuned model on two chromosomes, fiber layer alone versus fiber plus sequence. The columns are
  how many MacIsaac sites are recovered, how many calls are made, and how concentrated the
  posterior is at the sites. chrI only has five sites, so chrXIV is the more informative row."
- **Source (VERIFIED):**
  - `layer_scores/report_fib.json`: abf1 tp 0, n_pred 422, enrichment 1.198
  - `layer_scores/report_fib+seq.json`: tp 2, n_pred 83, enrichment 33.587
  - `layerXIV_scores/report_fib.json`: tp 7, n_pred 1,429, enrichment 6.181
  - `layerXIV_scores/report_fib+seq.json`: tp 8, n_pred 327, enrichment 33.287
  - Run dirs: `robocop_{chrI,chrXIV}_fib`, `robocop_{chrI,chrXIV}_fib_seq`
- **Note for the user's check:** these decodes use the fiber-only-trained trainDir with λ = 1
  and the old λ_unknown default. They predate tuner v2 (`HANDOFF.md` §1.4 "Other open threads").
- **Priority:** core (wording gated).

---

### 5. One ABF1 locus up close (S10) — PENDING
**Investigation running.** Do **not** state the mechanism claim ("a stronger fiber signal off
the motif outvotes the sequence; the shipped ABF1 matrix's spacer makes it worse") until the
coordinator sends confirmed findings.

- **Message (until confirmed):** chrI around ERV46, as drawn by the occupancy browser, at one
  MacIsaac ABF1 hit and one miss.
- **On slide:** the RoboCOP Occupancy Browser's tracks, redrawn inline (canvas) for
  chrI:60,900–62,900 (inside the browser's erv46 region chrI:60,001–65,000). Rows: ABF1
  posterior + nucleosome occupancy for runs `fib`, `fib+seq`, `seq` (browser labels); m6A per A
  (Watson/Crick, bg 0.138; same reads in every run); genes; axis. MacIsaac ABF1 bands at
  61,164–61,177 (miss: fib+seq ABF1 0.00) and 62,658–62,671 (hit: fib+seq ABF1 0.99), 1-based
  as in the browser. No logos on this slide (they stay on B3).
- **Build steps:**
  1. Legend, `fib` row, m6A, genes, axis
  2. `fib+seq` and `seq` rows
  3. Highlight the two MacIsaac sites (hit / miss chips)
- **Figure source path (VERIFIED):** data sliced from `analysis/posterior_viewer_all.html`
  (region erv46; identical to the published browser RYgeitfcR21v7zwY8XpTmN for these runs),
  built by `make_posterior_viewer.py` from `robocop_chrI_fib`, `robocop_chrI_fib_seq`,
  `robocop_chrI_seq`; embedded as JSON in the talk.
- **Speaker notes (neutral):** "2 kb of chrI around ERV46, drawn as the occupancy browser draws
  it: per run, ABF1 posterior and nucleosome occupancy; below, m6A per A per strand. Fiber
  plus sequence calls the MacIsaac site at 62.7 kb and not the one at 61.2 kb. [Mechanism
  sentence to be added once the investigation reports.]"
- **Source:**
  - Decodes: `robocop_chrI_fib`, `robocop_chrI_fib_seq`, `robocop_chrI_seq` (untuned, λ = 1).
  - Claim numbers held back until confirmed: `HANDOFF.md` §4.6 (LR 1e9–1e10, 8–91 bp) and §6
    (FIMO 5/5 vs 2/5, 3/5 in decode).
- **Priority:** condensable. If the investigation has not reported by the dry run, hide this
  slide (D6).

---

### 7. Ground truth, and the tuning set (S11) — was slide 6 until v4.5
- **Message:** we tune against MacIsaac and validate on held-out Rossi ChIP-exo, restricted to
  the 58 factor groups the two share.
- **On slide:**
  - Comparison table:

    | | MacIsaac (p005, c1) | Rossi ChExMix (ChIP-exo) |
    |---|---|---|
    | role | **tuning target** | **held-out validation** |
    | size | 25,104 merged sites · 119 TFs | 182,582 peaks · 378 TFs |
    | nature | motif + cross-species conservation, so a lower bound | in vivo binding summits |

  - Overlap: only **17.0%** of MacIsaac sites have a Rossi peak within 30 bp (**55%** for the
    12 fitted-footprint TFs).
  - **Decision: tune on the common set.**
    - **58 groups** (61 motifs) present in both MacIsaac and Rossi.
    - **55 are tuned.** ARO80, RFX1 and RPH1 have no MacIsaac site on the tuning chromosomes and
      stay fixed.
    - **The other 92 motifs are masked.**
  - Circularity (v4.3): "Rossi scores include all 58 groups. For 9 of them (ABF1, CIN5, FHL1,
    FKH1, MCM1, RAP1, REB1, SKO1, UME6) the Fiber-seq footprint was estimated at Rossi peaks, so
    their Rossi scores in fiber-layer runs are optimistic; sequence-only runs are unaffected."
- **Build steps:**
  1. The comparison table
  2. The overlap line
  3. "Tuning set: 58 groups / 61 motifs, 55 tuned, 92 masked"
  4. Circularity caveat
- **Figure source path:** none (inline table). Optional thumbnail:
  `/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/rossi_genic/genic_bars.png`.
- **Speaker notes:** "Two references. MacIsaac requires conservation, so it's strict and
  incomplete, and precision against it is a floor. Rossi's ChIP-exo is a much larger in vivo set
  we never tuned on. They overlap less than you'd expect, so we tuned only on the 58 factor
  groups both references cover, masked everything else, and score Rossi on all 58 groups. For nine
  of them the fiber footprint was estimated at Rossi peaks, so Rossi is optimistic for the runs with
  a fiber layer; the sequence-only run does not use those footprints." (v4.3)
- **Source:**
  - Sizes: HANDOFF §5.1, §5.2.
  - 17.0% / 55%: VERIFIED from `CT/rossi_validation/references.tsv` column
    `macisaac_in_rossi_cx` (2,065/12,124; 1,150/2,086).
  - 58 groups / 61 motifs / 55 tuned / 92 masked / ARO80, RFX1, RPH1 fixed: coordinator, plus
    `CT/PLAN_2026-09-14_tuner_v2.md` (lines 52, 295).
  - 9 fitted / 49 non-fitted: `CT/tw_legacy_validation.tsv` and
    `CT/{fw01,sw01,bw01}/validation.tsv` (the `fitted` column; the fitted row has
    n_groups = 9).
  - ⚠ Do not use HANDOFF §5.2's "77.5% inside Rossi".
- **Priority:** core.

---

### H6a. Why the priors need tuning (S12) — HIDDEN (after slide 6; was slide 7 until v4.5)
- **Message:** the per-factor priors are never learned, EM collapses them, and the decoder only
  ever sees one weight per factor, w = K_d·λ.
- **On slide:**
  - Starting weight = the motif's K_d, which encodes **motif sharpness, not abundance**. Nhp6a
    (7 bp) starts **2,478×** above Abf1 (14 bp).
  - Upstream EM is off here (`iterations = 0`). Forced on for 10 iterations: **28** factors
    pinned at the cap, **66** driven to exactly 0, ABF1 moved **3,723×** up; chrXIV ABF1 **0/19**
    sites.
  - Figure: EM trajectory of the transition prior. Log y vs EM iteration 0–10, ABF1 / REB1 /
    NHP6A highlighted, red cap line at 6.69e-4; log-likelihood panel below.
  - **Audit finding, one line:** "The decoder uses only w = K_d·λ per factor. The paper's α₀^L
    normalisation is bookkeeping. Our old tuner held the stored nucleosome prior fixed, which
    distorted the model, so it was rebuilt (slide 8)."
- **Build steps:**
  1. The K_d bullet
  2. EM figure plus the EM bullet
  3. The audit line (highlighted box)
- **Figure source path:**
  `/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/em_trace_robocop_train_em10_chrII_fiber.png`.
- **Speaker notes:** "Each factor's starting weight is its motif's K_d, which cancels the
  motif's own score, so the weights reflect how sharp a motif is, not how much protein there is.
  EM is the textbook fix, and it fails here: in one iteration most factors hit the cap or zero,
  and ABF1 goes the wrong way. An audit of our first tuner also showed the decoder only ever sees
  one weight per factor, and that holding the nucleosome's stored prior fixed had been silently
  moving its real weight, so we rebuilt the tuner around that weight directly."
- **Source:**
  - 2,478×, 28 / 66, 3,723×, cap 6.69e-4: HANDOFF §2.
  - chrXIV 0/19: VERIFIED, `layerXIV_scores/report_seq+em10.json`.
  - Audit line: VERIFIED by the coordinator from trainDirs: w_nuc = p_nuc/p_bg^147, 35 in
    `robocop_train_fiberonly/HMMconfig.pkl` vs 0.04047 / 748.5 / 13.77 / 0.4216 in
    `robocop_train_ct_u001_00..03/HMMconfig.pkl`. Full detail on B1.
- **Priority:** core. **S14b / audit decision:** folded as one line here; the full content stays
  on backup B1.

---

### 8. Tuning method, v2 (S13)
- **Message:** each factor group's call count is matched to MacIsaac on two chromosomes, with a
  third held out, using identical guarded rules for three layer configurations.
- **On slide:**
  - **Left:** loop diagram. Decode chrXIV + chrII → count calls per group (posterior ≥ 0.10) →
    compare to MacIsaac count → step log w → repeat, max 8 rounds. chrIV is decoded at round 0
    and at the final round only.
  - **Right, three columns:**
    - **Same for all three:**
      - fiber-only **fw01** · sequence-only **sw01** · both **bw01**
      - 58 MacIsaac∩Rossi groups, other motifs masked
      - `unknown` fixed at w = 1e-3
      - nucleosome weight 35, monitored only
    - **Removed:**
      - the nucleosome hold
      - the nucleosome stop rule
      - λ bounds
    - **Guards:**
      - step ≤ 10× per round
      - zero-count group → plain 10× step
      - per-bp weight cap w^(1/L) ≤ 0.70, a length-scaled concentration ceiling
      - ±1.25× deadband, with bisection once the target is bracketed
      - β (count elasticity) seeded at 0.5
- **Build steps:**
  1. Loop diagram
  2. chrIV holdout box added to the diagram
  3. "Same for all three" column
  4. "Removed" column
  5. "Guards" column
- **Figure source path:** **NEEDS MAKING**, loop diagram as inline SVG.
- **Speaker notes:** "The tuner decodes two chromosomes, counts each factor group's calls,
  compares with MacIsaac, and moves that group's weight on a log scale. Chromosome IV is never
  tuned on, only scored before and after. We removed everything that distorted the old loop and
  kept only guards against runaway: at most a tenfold step, a ceiling on per-base weight, a
  deadband, and bisection once a target is bracketed. The same rules run for fiber-only,
  sequence-only and both layers, so they are directly comparable."
- **Source:**
  - `CT/PLAN_2026-09-14_tuner_v2.md`; `tune_w.py` (VERIFIED in code: `BETA0 = 0.5`,
    `STEP_CAP = LN10`, `RHO_MAX = 0.70` "the ONLY bound", `DEADBAND = log(1.25)`,
    `MAX_ROUNDS = 8`, w_unknown 1e-3, "no hold, no stop rule, w_nuc = 35").
  - ⚠ `PLAN_2026-09-14_tuner_v2.md` line 121 still lists "λ ∈ [1e-6, 1e3]". That line is stale:
    `tune_w.py` says RHO_MAX is the only bound, and fw01 reached λ = 1e-7. Cite `tune_w.py` for
    "no λ bounds".
  - Also in the code but not on the slide: α = 0.7, halved on gap sign change, floor 0.2.
- **Priority:** core.

---

### 9. Results per campaign (S14)
- **Message:** all three campaigns move their counts toward target to different degrees, and
  their accuracy moves very differently.
- **On slide:**
  - **Chart** (inline, from §5.2): three small panels, one per campaign.
    - X = round, 0–7 (sw01 stops at 5).
    - Y-left = groups within 2× of target, of 41.
    - Y-right = MacIsaac F1 (all 58 groups), on the tuning chromosomes.
  - **Table:** tuning chromosomes chrXIV + chrII, round 0 → final.

    | run | stop reason | within 2× (of 41) | MacIsaac P / R / F1 (all) | nucleosome copies |
    |---|---|---|---|---|
    | fw01 fiber | 8-round limit | 12 → 30 | 0.49% / 10.2% / 0.0094 → 0.77% / 5.8% / 0.0135 | 8,800 → 8,918 (+1.3%) |
    | sw01 sequence | converged after round 5 | 8 → 41 | 2.62% / 3.6% / 0.0304 → 6.71% / 13.2% / 0.0890 | 9,259 → 9,192 (−0.7%) |
    | bw01 both | 8-round limit | 9 → 39 | 1.61% / 13.6% / 0.0288 → 3.52% / 10.1% / 0.0522 | 8,919 → 8,944 (+0.3%) |

  - Footnotes:
    - "41 = groups with ≥ 5 MacIsaac sites on chrXIV + chrII."
    - "MacIsaac precision is a floor (conservation-filtered)."
    - "Nucleosome weight fixed at 35; copies monitored."
- **Build steps:**
  1. Table rows appear one per step (fw01, sw01, bw01)
  2. Chart appears
  3. Footnotes
- **Figure source path:** **NEEDS MAKING**, inline chart from the §5.2 per-round data.
  `CT/tw_summary.png` does not exist yet; if the coordinator runs
  `python tw_summary.py fw01 sw01 bw01` before the build, embed that instead after checking its
  panels.
- **Speaker notes:** "Sequence-only converged: all 41 groups within 2× by round 4, and it stopped
  itself after round 5. Both-layers got 39 of 41 in eight rounds. Fiber-only got 30 and kept
  moving. Precision and recall went up together for sequence-only, while for fiber-only recall
  fell as counts were pushed down. Nucleosome counts stayed within about 1% in all three, so the
  nucleosome model was stable throughout."
- **Source (VERIFIED):** `CT/{fw01,sw01,bw01}/validation.tsv`, rows `set=tune`,
  `ref=macisaac`, `fitted=all`, round 0 and final (columns P, R, F1, within2x, nuc_copies).
  Stop reasons are from `CT/{fw01,sw01,bw01}/STOPPED`. I re-read both today.
- **Priority:** core.

---

### H9a. Fiber-only detail: why fw01 stalls — HIDDEN
- **Message:** fiber-only's unconverged groups split into fitted-footprint TFs the prior cannot
  push down, and a bisection artefact of the tuner.
- **On slide:**
  - F1 (all) peaked at **round 3 (0.0151)** and then slowly fell. Precision stayed flat at
    ~0.8% while recall fell from **11.7% (round 1) to 5.8% (round 7)**.
  - **11 of the 41 scored groups** (T ≥ 5) were still outside 2× at round 7 (41 − 30), plus
    PDR3 (target 2, outside the 41). This corrects the coordinator's "13".
    - **Six fiber-fitted TFs** (ABF1, RAP1, REB1, UME6, SKO1, FKH1) sit at λ = 1e-7 and are
      still 3–19× over target (ABF1 521 copies vs target 58). In these runs the fitted fiber
      footprint keeps calling them despite a tiny prior.
    - **Six others** (STE12, PHD1, SKN7, RTG3, HAP1, plus PDR3) are stuck near λ ≈ 1 by the
      bisection bracket: a **tuner artefact, to fix**.
    - Also over by gap, but frozen with T = 0 by design and not a tuning failure: RFX1, ARO80
      and RPH1.
  - Table (from `report_07.tsv`): group · target · copies · fold over · λ · fitted.

    | group | target | copies | fold | λ | fitted |
    |---|---|---|---|---|---|
    | SKO1 | 5 | 93.5 | 18.7 | 1e-7 | yes |
    | RAP1 | 29 | 436.5 | 15.1 | 1e-7 | yes |
    | ABF1 | 58 | 520.7 | 9.0 | 1e-7 | yes |
    | UME6 | 19 | 155.8 | 8.2 | 1e-7 | yes |
    | HAP1 | 19 | 99.9 | 5.3 | 1.10 | no |
    | REB1 | 44 | 215.7 | 4.9 | 1e-7 | yes |
    | RTG3 | 27 | 100 | 3.7 | 1.02 | no |
    | SKN7 | 50 | 166.5 | 3.3 | 1.03 | no |
    | PHD1 | 88 | 284.2 | 3.2 | 1.01 | no |
    | FKH1 | 23 | 72.2 | 3.1 | 1e-7 | yes |
    | STE12 | 195 | 580.3 | 3.0 | 1.04 | no |
    | PDR3 | 2 | 6.2 | 3.1 | 1.00 | no |

  - The table is complete: every group with |gap| > ln 2 in `report_07.tsv` except the three
    frozen T = 0 groups.
- **Build steps:**
  1. F1 / precision / recall bullet
  2. Table
  3. The two-group interpretation
- **Figure source path:** none (inline table).
- **Speaker notes:** "If anyone asks why fiber-only didn't converge. Half the stragglers are the
  factors with fitted fiber footprints: their prior went down to ten to the minus seven and they
  are still over-called. The other half are stuck at their starting weight because of how the
  bisection bracket was set, which is ours to fix."
- **Source:**
  - VERIFIED from `CT/fw01/report_07.tsv` (columns T, E, fold, lambda, why, fitted) and
    `CT/fw01/validation.tsv`.
  - Peak 0.0151 at round 3 (tune, macisaac, all) and final recall 5.77% verified.
  - Recall 11.67% is the round 1 value; the round 0 value is 10.15%.
- **Priority:** hidden.

---

### 10. Layer comparison (S15) — the payoff
- **Message:** in these runs, sequence-only tuning improved precision and recall, and the gain
  held on held-out chrIV and against Rossi. Adding the fiber layer lowered accuracy relative to
  sequence-only. Fiber-only matched counts but not accuracy.
- **On slide:**
  - **Table** (v4.3; round 0 → final, pooled over all 58 groups, `fitted=all`):

    | run | Tuning chrXIV+chrII: MacIsaac F1 | Tuning chrXIV+chrII: Rossi F1 | Held-out chrIV: MacIsaac F1 | Held-out chrIV: Rossi F1 |
    |---|---|---|---|---|
    | fiber only (fw01) | 0.0094 → 0.0135 | 0.0143 → 0.0111 | 0.0086 → 0.0106 | 0.0133 → 0.0111 |
    | sequence only (sw01) | 0.0304 → **0.0890** | 0.0228 → **0.0852** | 0.0246 → **0.0785** | 0.0194 → **0.0887** |
    | both layers (bw01) | 0.0288 → 0.0522 | 0.0338 → 0.0381 | 0.0244 → 0.0470 | 0.0356 → 0.0402 |

  - **Chart:** grouped bars of final F1 on held-out chrIV, three runs × {MacIsaac all, Rossi all},
    with the round-0 value as a tick on each bar; y-axis 0–0.10.
  - Three "in these runs" bullets:
    - Sequence-only: precision **and** recall rose. F1 roughly tripled or better (2.9–4.6×), and
      held on held-out chrIV and on Rossi.
    - Adding the fiber layer lowered accuracy relative to sequence-only.
    - Fiber-only matched counts, not accuracy.
  - Caveat strip (small):
    - 2 tuning chromosomes
    - MacIsaac precision is a floor
    - Rossi: all 58 groups; see slide 7 on circularity
    - count-matching sets operating points
    - fw01 bisection artefact (hidden slide H9a)
    - locus mechanism under investigation
- **Build steps:**
  1. Table with the fw01 row only
  2. sw01 row
  3. bw01 row
  4. Held-out chart
  5. The three bullets, one per step
  6. Caveat strip
- **Figure source path:** **NEEDS MAKING**, inline chart from the §5.3 numbers.
- **Speaker notes:** "This is what the tuning was for. With identical rules, the sequence-only
  model improved the most: both precision and recall went up, and the gain held on a chromosome
  we never tuned on and against Rossi's peaks across all 58 groups. For the two runs with a fiber
  layer those Rossi numbers are, if anything, optimistic, because nine footprints were estimated at
  Rossi peaks; sequence-only doesn't use them. Adding the fiber layer on top of sequence gave lower accuracy than sequence alone in
  these runs, and fiber alone matched counts without getting more accurate. I want to be careful:
  two tuning chromosomes, conservative references, and we're still examining the runs, so read
  this as where the evidence currently points, not a verdict on Fiber-seq."
- **Source (VERIFIED, v4.3):** `CT/{fw01,sw01,bw01}/validation.tsv`, rows `ref∈{macisaac, rossi_cx}`,
  `fitted=all`, `set∈{tune, holdout}`, round 0 and final (fw01 r7, sw01 r5, bw01 r7). Values are
  re-read by `scratchpad/vals.py`; the per-cell old → new table is in §9 "v4.3".
  - "Roughly tripled or better" (sw01, all groups): 2.93× tune MacIsaac, 3.74× tune Rossi,
    3.19× holdout MacIsaac, 4.58× holdout Rossi.
  - Pre-v4.3 (non-fitted) wording was "roughly doubled to tripled" (2.4 / 2.4 / 3.2 / 2.2 / 2.0×).
- **Priority:** core (payoff). Wording level is D4.

---

### H10a. Legacy comparison: old tuner, same scope — HIDDEN
- **Message:** the old genome-wide both-layer campaigns, scored on the same 58 groups and the
  same chromosomes, reached F1 in the same range as bw01.
- **On slide:**

  | run | tuner | tuning F1 (MacIsaac, all), chrXIV+chrII | holdout F1 chrIV | within 2× (of 41) |
  |---|---|---|---|---|
  | u001 | old (genome-wide, all motifs live, nucleosome hold) | 0.036 → 0.064 | 0.032 → 0.060 | 9 → 33 |
  | m001 | old (genome-wide, 69 masked, hold) | 0.031 → 0.057 | 0.027 → 0.052 | 9 → 34 |
  | bw01 | v2 (chrXIV+chrII, 92 masked, no hold) | 0.0288 → 0.0522 | 0.0244 → 0.0470 | 9 → 39 |

  - Note line: "Different tuning data (whole genome vs 2 chromosomes), masking, and priors —
    not a controlled comparison."
- **Build steps:** 1 (static).
- **Figure source path:** none (inline table).
- **Speaker notes:** "If asked whether the old campaigns did better: scored on the same scope,
  the old both-layer runs land a little above bw01. They were tuned on the whole genome with
  different masking and the nucleosome hold, so this isn't a like-for-like test of the tuner."
- **Source:**
  - VERIFIED by the coordinator: `CT/tw_legacy_validation.tsv` (computed by `tw_validate.py`
    from stored call positions). u001: tune 0.0362 → 0.0636, holdout 0.0322 → 0.0598, within 2×
    9 → 33. m001: tune 0.0315 → 0.0570, holdout 0.0273 → 0.0521, within 2× 9 → 34.
  - bw01 row VERIFIED from `CT/bw01/validation.tsv`.
- **Priority:** hidden. See D5: this is the most likely hard question.

---

### 11. Lessons and next steps (S16) — rewritten in v4.6
- **Message:** "Where these runs point so far, and what to try on the fiber layer."
- **On slide** (user's wording, lightly tightened):
  1. EM doesn't seem to help.
  2. Tuning to MacIsaac brings site counts closer, but with the fiber layer on, accuracy barely
     improves.
  3. Tuning the sequence layer alone seems to give better accuracy; not really for the fiber
     layer.
  4. The fiber layer seems to carry more weight than the sequence layer.
  5. → In aggregate, the fiber layer may only do so much, or we can try to fix its emission:
     coverage weighting · footprint width · the fallback rate for the 141 un-fitted TFs
- **Build steps:** 5, one bullet per step (bullet 5 arrives with its three sub-bullets).
- **Figure:** none (the compact roadmap was removed).
- **Speaker notes:** one or two sentences per step pointing to the supporting slide (step 1 →
  6; step 2 → 9, 10, B2; step 3 → 9, 10; step 4 → 5b, 5d, 10b, 10; step 5 → the emission
  fixes), then the removed content: the tuner ✔ line (slides 8–9); examine the runs at the locus level; fix the tuner's
  bisection bracket (H9a); re-tune on more chromosomes; single-fiber decoding; cluster fibers
  (slide 3 roadmap); aggregate-phase status (fiber layer, sequence layer, prior tuning done;
  separating each layer's contribution partly done).
- **Source:** user request (v4.6); supporting numbers are those already on slides 5b, 5d, 6,
  9, 10, 10b and B2. Emission fixes: `HANDOFF.md` §4.6, `whattodo.md` Tier 1 #3.
- **Priority:** core.

---

### B1. What the tuning audit found (backup)
- **Message:** the decoder uses only w = p/p_bg^L = K_d·λ per factor. Holding the stored
  nucleosome prior fixed random-walked the nucleosome weight, which barely affected fiber-only
  and badly distorted sequence-only.
- **On slide:**
  - w_nuc by round, u001: pristine **35** → **0.04047, 748.5, 13.77, 0.4216** (rounds 0–3).
    Chart: log y.
  - Held vs unheld at round 0:

    | | nucleosome copies | calls | F1 |
    |---|---|---|---|
    | fiber: fm001 (held) vs fm002 (unheld) | 65,529 vs 65,812 | 233,658 vs 232,495 | 0.0096 vs 0.0096 |
    | sequence: sm001 (held) vs sm002 (unheld) | 40,498 vs 69,249 | 129,446 vs 21,357 | 0.0261 vs 0.0273 |

  - Fix: tune log w directly, no hold (tuner v2, slide 8).
- **Build steps:**
  1. The w formula
  2. w_nuc chart
  3. The held-vs-unheld table
- **Figure source path:** **NEEDS MAKING**, inline chart from the five w_nuc values.
- **Speaker notes:** "All priors share one normalisation, and the decoder only sees each factor's
  weight relative to unbound DNA. Our old loop pinned the nucleosome's stored prior, so its real
  weight swung over four orders of magnitude. Fiber data are strong enough on nucleosomes to
  swamp it; sequence alone is not, which is why the first single-layer runs were discarded."
- **Source:**
  - w values: VERIFIED by the coordinator from `robocop_train_fiberonly/HMMconfig.pkl` (35),
    `robocop_train_ct_u001_00..03/HMMconfig.pkl`, `robocop_train_ct_fm002_00` (35); w_ABF1 =
    K_d·λ checked on `robocop_train_ct_fm001_02` (4.729e-13).
  - Held vs unheld: VERIFIED from `CT/{fm001,fm002,sm001,sm002}/state.json`
    `history[0].nucleosome_copies` (re-read by me) and `trajectory.tsv` round 0.
  - Audit files: `/usr/project/xtmp/nd141/scratch_tunelogic/`, `/usr/project/xtmp/nd141/scratch_nuchold/`.
- **Priority:** backup. This is the only place fm001 / sm001 / fm002 / sm002 numbers may appear.

### B2. Legacy genome-wide campaigns u001 / m001 (backup; old S14)
- **Message:** under the old loop, counts converged genome-wide, but F1 improved only in the
  first step.
- **On slide:**

  | round | u001 within 2× (of 81) | u001 MacIsaac F1 | u001 Rossi-CX F1 | m001 MacIsaac F1 | m001 Rossi-CX F1 |
  |---|---|---|---|---|---|
  | 0 | 16 | 0.029 | 0.044 | 0.025 | 0.037 |
  | 1 | 27 | 0.043 | 0.051 | 0.040 | 0.045 |
  | 3 | 60 | 0.048 | 0.053 | 0.041 | 0.046 |
  | 7 | 79 | 0.049 | 0.053 | 0.042 | 0.050 |

  - Caveats (v4.3): the old loop carried the nucleosome hold; both campaigns use the fiber layer,
    so the Rossi column is optimistic for the 9 fitted-footprint groups. Footnote: "Rossi: all 58
    groups; see slide 7 on circularity. MacIsaac F1: all 81 groups."
  - Rossi-CX values were already pooled over the 58 groups with a Rossi CX file (re-derived
    v4.3 from `match_*.tsv`: u001 0.04405 / 0.05143 / 0.05253 / 0.05329; m001 0.03711 / 0.04468 /
    0.04610 / 0.04994), so no number changed.
- **Build steps:** 1.
- **Figure source path:** optional screenshot of the Concentration Sheet `f7ff1c1b-…`, "What
  eight rounds of λ bought". Must be saved as a PNG to embed; **NEEDS MAKING** (manual
  screenshot).
- **Speaker notes:** "For context, the earlier genome-wide campaigns under the old loop."
- **Source:** VERIFIED, pooled from `CT/{u001,m001}/trajectory.tsv` and
  `CT/rossi_validation/match_{u001,m001}.tsv` (ref `rossi_cx`).
- **Priority:** backup.

### B3. Motif audit, "Odd One Out" (backup)
- **Message:** 7 of 153 shipped PWMs are the outlier against JASPAR and Rossi, and ABF1's
  verdict is unanimous.
- **On slide:**
  - Shortlist: Abf1_murphy, Rap1_motif2, Rap1_telomeric, Rap1_zhu, Pdr1_badis, Rap1_motif1,
    Cad1_murphy.
  - ABF1 FIMO: JASPAR 5/5 vs Murphy 2/5.
  - Logos figure.
- **Build steps:** 1.
- **Figure source path:**
  `/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/abf1_murphy_vs_jaspar_logos.png`.
  Optionally also a screenshot of artifact `7c63a39b-…`.
- **Speaker notes:** "Backup if PWM quality comes up."
- **Source:** HANDOFF §6.
- **Priority:** backup. Coordinate with slide 5's pending investigation before showing it.

---

## 3. Deferred (not in this talk; content kept so nothing is lost)

- **S4. What Fiber-seq measures.**
  - m6A marks accessible A's; the pileup gives k methylated of n reads per A per strand.
  - Figures: `chrI_meth_distribution.png` (per-bp m6A ratio histogram, chrI, ≥20 A-trials,
    n = 131,796, shipped background 0.138 marked); site #4 meth/A strip from
    `abf1_5sites_seq_abf1_revfix_layers.png`.
- **S5. How Fiber-seq enters the model.**
  - Emission = sequence × Fiber Watson (layer 5) × Fiber Crick (layer 6).
  - P(k | n, state j) = Binomial(k; n, p_j).
  - Background p 0.138; nucleosome p 0.024–0.14; fitted 14-column p-vectors for 12 TFs; 141 TFs
    share `combined_low_count`.
  - Sources: HANDOFF §4.1, §5.3; `whattodo.md` Tier 1 #3; `FIBERSEQ_CHANGES.md` §4.
- **S6. Code changes.**
  - `config.ini`: `tech2`, `pileupFile`, `nucleotide`
  - `getReads.getFiber_seq`: pileup → info.h5
  - `robocop.py`: 5 → 7 layers, binomial emission, longlong
  - `robocopExtras.py`: layer wiring, 1e-30 floor
  - `robocop_em.py`: iterations 10 → 0
  - Source: `FIBERSEQ_CHANGES.md`.
- **S7. Fitted footprints are the weak link.**
  - ABF1 fitted p 0.029–0.137, all below background 0.138.
  - Fallback 0.249/0.265, above background.
  - 14 columns vs a ~18–21 bp footprint.
  - Plot from `inputs/all_TFs_1000pealVal_params_pseudo.pkl` + `inputs/bg_params.pkl`
    (read-only; key layout unverified).
  - Source: HANDOFF §4.6, §5.3.
- **S8. Nucleosomes.**
  - Fiber-only full chrI: Chereji recall 0.79–0.81, median dyad error 6.5–7 bp, period
    171–172 bp.
  - Figures: `chrI_5run_recall.png`, `dyad_distance_chrI.png`.
  - Sources: `chrI_5run_metrics.json` (pre-revfix runs), `layer_scores/report_fib*.json`.
  - Tuner v2 nucleosome copies were stable (±1.3%) in all three campaigns (slide 9), which is a
    one-line substitute.

---

## 4. Checklist (now ~22:10; tonight + tomorrow morning)

### Tonight
- [ ] **A1 (10 min, user or coordinator). Settle decisions D1–D6** (§6), at least D2, D4 and
  D6.
- [ ] **A2 (agent, ~90 min; user not needed). Artifact build from PLAN.md.**
  - Build 11 visible slides, plus hidden H1a, H9a and H10a, plus backups B1–B3. Implement the
    hidden-slide key toggle, and one build step per scroll notch or arrow key.
  - Embed these PNGs:
    - `em_trace_robocop_train_em10_chrII_fiber.png`
    - `abf1_murphy_vs_jaspar_logos.png`
    - optionally `rossi_genic/genic_bars.png`
  - Draw inline: the H1a and slide 3 diagrams, the slide 5 browser tracks, the slide 8 loop diagram, the slide 9 and 10
    charts, and the B1 chart. Data is in §5; every number goes in as written there.
  - The agent must load the `artifact-design` skill first. It must keep the local HTML file
    (for offline presenting) and must not publish over any existing artifact URL.
- [ ] **A3 (30 min, user). Examine the slide 4 runs before finalising wording.** Runs:
  `robocop_{chrI,chrXIV}_fib`, `robocop_{chrI,chrXIV}_fib_seq`. Then replace slide 4's neutral
  message and notes, or keep them neutral.
- [ ] **A4 (coordinator → agent_ppt, when it arrives). Slide 5 investigation findings.** Unlock
  or rewrite slide 5 build step 3, and B3.

### Tomorrow morning
- [ ] **M1 (30 min, user). Review the artifact.** Note comments; the agent republishes to the
  **same URL**.
- [ ] **M2 (15 min). Finalise gated wording:** slide 4 (from A3), slide 5 (from A4, or hide it),
  slide 10 bullet strength (D4).
- [ ] **M3 (45 min, user). Dry run in the browser** on the presenting machine. Check scroll and
  arrow steps, the hidden toggle, projector legibility of the tables on slides 9 and 10, and the
  offline local-HTML fallback.
- [ ] **M4 (15 min). Number audit.** Every number must match §5 or its listed source.
  - Grep for "77.5".
  - Grep for fm001/sm001/fm002/sm002 numbers outside B1.
  - Check that every Rossi number is the all-58-groups value (v4.3) and carries the footnote.

**Cut-line if time runs short:** drop B2 and B3 → hide slide 5 → replace the slide 9 chart
with the table only → simplify the H1a diagram to bullets.

---

## 5. Verified data for the artifact build

### 5.1 Slide 4 (S9)
See the slide 4 source line: `layer_scores/` and `layerXIV_scores/` `report_{fib,fib+seq}.json`.

### 5.2 Per-round data for the slide 9 chart
Source: `CT/<run>/validation.tsv`, `set=tune`, chrXIV+chrII, VERIFIED. F1 columns are MacIsaac
F1 over all groups, MacIsaac F1 non-fitted, and Rossi F1 non-fitted.

| run | round | within 2× (of 41) | P (all) | R (all) | F1 MacIsaac all | F1 MacIsaac non-fitted | F1 Rossi non-fitted | nucleosome copies |
|---|---|---|---|---|---|---|---|---|
| fw01 | 0 | 12 | 0.0049 | 0.1015 | 0.0094 | 0.0074 | 0.0050 | 8,800 |
| fw01 | 1 | 8 | 0.0077 | 0.1167 | 0.0145 | 0.0165 | 0.0055 | 8,828 |
| fw01 | 2 | 15 | 0.0079 | 0.1008 | 0.0147 | 0.0150 | 0.0032 | 8,851 |
| fw01 | 3 | 21 | 0.0083 | 0.0890 | 0.0151 | 0.0137 | 0.0027 | 8,872 |
| fw01 | 4 | 28 | 0.0076 | 0.0723 | 0.0138 | 0.0112 | 0.0023 | 8,884 |
| fw01 | 5 | 29 | 0.0077 | 0.0660 | 0.0138 | 0.0105 | 0.0023 | 8,898 |
| fw01 | 6 | 29 | 0.0078 | 0.0625 | 0.0139 | 0.0115 | 0.0029 | 8,908 |
| fw01 | 7 | 30 | 0.0077 | 0.0577 | 0.0135 | 0.0115 | 0.0031 | 8,918 |
| sw01 | 0 | 8 | 0.0262 | 0.0361 | 0.0304 | 0.0283 | 0.0153 | 9,259 |
| sw01 | 1 | 29 | 0.0618 | 0.0869 | 0.0722 | 0.0553 | 0.0312 | 9,256 |
| sw01 | 2 | 37 | 0.0650 | 0.1167 | 0.0835 | 0.0671 | 0.0369 | 9,213 |
| sw01 | 3 | 39 | 0.0659 | 0.1265 | 0.0866 | 0.0667 | 0.0370 | 9,198 |
| sw01 | 4 | 41 | 0.0671 | 0.1320 | 0.0890 | 0.0677 | 0.0372 | 9,192 |
| sw01 | 5 | 41 | 0.0671 | 0.1320 | 0.0890 | 0.0677 | 0.0372 | 9,192 |
| bw01 | 0 | 9 | 0.0161 | 0.1362 | 0.0288 | 0.0267 | 0.0151 | 8,919 |
| bw01 | 1 | 24 | 0.0225 | 0.1376 | 0.0387 | 0.0382 | 0.0173 | 8,926 |
| bw01 | 2 | 31 | 0.0261 | 0.1216 | 0.0430 | 0.0429 | 0.0193 | 8,931 |
| bw01 | 3 | 35 | 0.0279 | 0.1084 | 0.0444 | 0.0427 | 0.0190 | 8,935 |
| bw01 | 4 | 37 | 0.0311 | 0.1056 | 0.0481 | 0.0459 | 0.0190 | 8,939 |
| bw01 | 5 | 37 | 0.0340 | 0.1070 | 0.0517 | 0.0475 | 0.0196 | 8,942 |
| bw01 | 6 | 37 | 0.0350 | 0.1049 | 0.0525 | 0.0472 | 0.0195 | 8,943 |
| bw01 | 7 | 39 | 0.0352 | 0.1008 | 0.0522 | 0.0474 | 0.0203 | 8,944 |

- sw01 rounds 4 and 5 are identical: it converged, with every tuned group taking a zero step.
- The fw01 within-2× count dips from 12 to 8 at round 1.

### 5.3 Held-out chrIV (slide 10)
Source: `CT/<run>/validation.tsv`, `set=holdout`, VERIFIED. **Superseded for slide 10 by §9 "v4.3"** (Rossi all groups); the non-fitted columns below are kept for the notes.

| run | round | P (all) | R (all) | F1 MacIsaac all | F1 MacIsaac non-fitted | F1 Rossi non-fitted | nucleosome copies |
|---|---|---|---|---|---|---|---|
| fw01 | 0 | 0.0045 | 0.0990 | 0.0086 | 0.0075 | 0.0051 | 8,432 |
| fw01 | 7 | 0.0060 | 0.0487 | 0.0106 | 0.0069 | 0.0047 | 8,561 |
| sw01 | 0 | 0.0207 | 0.0305 | 0.0246 | 0.0255 | 0.0181 | 8,884 |
| sw01 | 5 | 0.0589 | 0.1173 | 0.0785 | 0.0559 | 0.0360 | 8,826 |
| bw01 | 0 | 0.0135 | 0.1226 | 0.0244 | 0.0218 | 0.0212 | 8,570 |
| bw01 | 7 | 0.0312 | 0.0952 | 0.0470 | 0.0443 | 0.0262 | 8,592 |

---

## 6. Decisions (open)

- **D1. Title slide.** SETTLED (2026-09-15): the title slide is visible and the RoboCOP recap is
  hidden (H1a).
- **D2. With S4–S8 deferred, the audience never sees how Fiber-seq enters the model**, and
  slides 4, 5 and 10 assume it. Options:
  - (a) Add build step 4 on H1a ("+ two Fiber-seq layers, binomial m6A per state").
    Recommended.
  - (b) Build deferred S5 as a hidden slide.
  - (c) Leave it out and say it aloud.
- **D3. Audit placement.** My call: one line on hidden H6a (was slide 7), full content on backup B1. Promote B1
  to visible after slide 8 if the audience is methods-heavy.
- **D4. Strength of the slide 10 wording.** It currently uses the coordinator-allowed "in these
  runs" phrasing. The user may soften it further after examining the runs (A3). Also decide
  whether "adding the fiber layer lowered accuracy" goes on the slide or only in the notes.
- **D5. Legacy comparison (H10a).** Hidden by default. The old u001 holdout F1 (0.060) is
  above bw01's (0.047) on the same scope. Choose one:
  - Keep it hidden and answer if asked (recommended).
  - Show it, to pre-empt the question.
- **D6. Slide 5 if the investigation hasn't reported by the dry run.** Choose one:
  - Hide it (recommended).
  - Show the panels with the neutral caption only.

---

## 7. Open risks

- **R1. Slide 4 wording is gated on the user's examination (A3).** The decodes predate tuner v2
  and use λ = 1.
- **R2. The slide 5 mechanism is unconfirmed.** Nothing about "fiber outvotes sequence" or the
  Murphy spacer may be said until A4 arrives.
- **R3. Hard question: "is Fiber-seq hurting?"** The slide 10 answer is scoped to these runs:
  - 2 tuning chromosomes
  - count-matching objective
  - fiber emission known to be weak (HANDOFF §4.6)
  - fw01 bisection artefact
- **R4. Hard question: "the old tuner did better?"** See H10a / D5. The comparison is not
  controlled.
- **R5. Circularity.** (v4.3, user decision) Rossi numbers use all 58 groups; the 9 fitted-footprint
  groups make Rossi optimistic for fiber-layer runs. Stated on slide 7 and footnoted on slides 10 and B2.
- **R6. Stale line in `CT/PLAN_2026-09-14_tuner_v2.md`** (line 121, "λ ∈ [1e-6, 1e3]").
  `tune_w.py` has no λ bounds. Cite the code.
- **R7. MacIsaac precision looks tiny (0.5–7%).** Say "floor" once, on slide 7 or 9.
- **R8. Artifact delivery.** Presenting depends on claude.ai access. Keep the local HTML file
  and open it from disk as the fallback (M3). PNGs must be embedded, not linked.
- **R9. `tw_summary.{tsv,png}` has not been generated.** Charts are built from `validation.tsv`
  numbers (§5) instead.

---

## 8. Later / future work (not in this talk)

- **MNase-RoboCOP comparison**, needed eventually for the "better than MNase" goal. Nothing in
  this repo scores an MNase decode. The only one on disk is upstream's tutorial run
  (`/usr/project/xtmp/nd141/programs/RoboCOP/analysis/robocop_all/`: 10 × 5 kb windows, DM504
  subset, includes chrI:60,001–65,000 = ERV46), never scored.
- Fix the tuner v2 bisection bracket (fw01's six groups stuck near λ ≈ 1).
- Re-tune the three campaigns on more chromosomes, or genome-wide.
- Fiber emission fixes (`whattodo.md` Tier 1 #3). PWM shortlist (`HANDOFF.md` §6). Retrain with
  the sequence layer on. Single-fiber decoding. Fiber clustering.

  Slide 10: the claim is mostly false as worded

  Why the fiber layer has more power than sequence:
  - Sequence has a ceiling, fiber has none. A perfect ABF1 motif is worth at most 10^6.7 in likelihood ratio. ABF1's prior needs about
    10^6.4 to reach posterior 0.5, and real sites score only 10^1–10^6.4 (median 10^2.2). Sequence alone can almost never call a 
    site: the sequence-only decode gets 0 of 24.
  - Fiber evidence grows with read depth. Each read contributes very little, but the binomial treats every read at every A/T as
    independent evidence. With ~600 reads over a site, the fiber term swings between about 10^−29 and 10^+13 at true sites. Thinning
    reads to 10% cuts it to exactly 10%.
  - So fiber decides, sequence nudges. The rule "sequence + fiber at the site ≥ 6.4" predicts the combined decode's hit or miss at 24
    of 24 sites. Fiber alone predicts it at 22 of 24; sequence alone at 18.

  What's wrong with the claim:
  - "Outvotes by a signal a few bp off": mostly not. In the combined decode, 14 of 24 sites are missed. At 13 of those the ABF1
    posterior just disappears; it doesn't move somewhere else. Only chrI:108788 relocates, 73 bp upstream at posterior 0.14 (the right
    panel in the figure). The fiber vetoes the site itself: missed sites have m6A above background (median 0.19, against 0.056 at
    hits). "A few bp off" only describes the fiber-only decode, which shifts ABF1 −7 to +6 bp.
  - Murphy spacer flaw: true but secondary. The spacer costs real sites about 2 log units, and fixing it rescues 2 sites (chrI:61163
    goes 0.002 → 0.98 with JASPAR). A perfect motif still can't beat a fiber veto of −5 or worse.
  - Some vetoes may be right: 7 of the 14 missed sites have no Rossi ChIP-exo peak and are heavily methylated, so they may simply be
    unbound.

---

## 9. v4.2 additions (2026-09-15): locus slides, 7b (now 6), 8b, 10b, 10c

All locus slides are built by `presentation/templates/locus_slide/build_locus_slide.py`, which imports `analysis/make_posterior_viewer.py`. Build commands, JSON payloads and the verification scripts are in `/usr/project/xtmp/nd141/scratch_slide5/`. The chip hit/miss rule is the call run's maximum posterior ≥ 0.10 within ±20 bp of the site. This is not the slide-4 scorer rule; see the template README.

### 5b / 5c / 5d (hidden, after 5) — VERIFIED from decodes
The per-run value is the maximum posterior within ±20 bp of the site. Site m6A is Watson + Crick over the motif, from the fib run's counts; the background is 0.138.

| slide | window | site(s) (1-based) | Rossi CX summit | fib / fib+seq / seq | site m6A |
|---|---|---|---|---|---|
| 5b | chrXIV:186,900–188,900 | ABF1 187,879–187,892 | 187,885 | 1.00 / 1.00 / 0.00 (hit) | 0.005 |
|    |                        | ABF1 187,697–187,710 | none within 30 bp (181 bp) | 0.00 / 0.00 / 0.00 (miss) | 0.22 |
| 5c | chrXIV:465,700–467,700 | ABF1 466,725–466,738 | 466,731 | 0.00 / 0.00 / 0.00 (miss) | 0.21 |
| 5d | chrXIV:416,650–418,650 | REB1 417,645–417,652 | 417,649 | 0.72 / 1.00 / 0.06 (hit) | 0.30 |

Sites come from `inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1_match_PWM.bed`, the browser's bands; each matches a merged c1 gff interval within 1 bp. Rossi summits are from `/usr/project/xtmp/nd141/projects/data/rossi_strand/{Abf1,Reb1}_CX.bed`. On 5d the REB1 line is drawn in #1fa81c because the browser's #d9f3e2 is unreadable on the light plot.

### 6 (was 7b) — EM-trained priors at one ABF1 locus (visible; replaces hidden H6a) — VERIFIED
- **Runs:** `robocop_chrXIV_fib_seq`, `_fib_seq_em10` (trainDir `robocop_train_em10_chrII`), `_fib`, `_fib_em10` (`…_chrII_fiber`).
- **Window:** chrXIV:166,964–168,964; the site is MacIsaac ABF1 167,958–167,971, with its Rossi summit at 167,964.
- **Site posterior:** fib+seq 1.00, fib+seq+em10 1.00, fib 1.00, fib+em10 0.375.
- **Runs ≥ 0.30 in the window:** 1 → 4 (fib+seq) and 3 → 8 (fib). This is representative: chrXIV-wide, EM multiplies calls (fib+seq 327 → 1,198) while found rises only 8 → 11.
- **Strip, chrXIV, `score_robocop` rule:**
  - fib+seq: 8/327/F1 0.046 → 11/1,198/0.018
  - fib: 7/1,429/0.0097 → 5/1,640/0.0060
  - seq: 1/1/0.100 → 0/0/0
  - Sources: `layerXIV_scores/report_{fib+seq,fib,seq,seq+em10}.json`, `chrXIV_scores/report_{seqem10,fibem10}.json` (old outDir names).
- **Priors, re-read from HMMconfig.pkl `tf_prob` (153 motifs + unknown):**
  - em10_chrII: 28 at the cap 6.69e-4, 1 above (Skn7 6.93e-4), 66 at 0, ABF1 ×3,723. This matches HANDOFF §2.
  - `_fiber`: 54 at the cap, 9 above, 54 at 0, ABF1 ×3,723.
  - `_seqonly`: 6 at the cap, 61 at 0, ABF1 → 0.
  - The cap equals the mean + 2 SD of the initial non-unknown priors.
- **Note:** hidden H6a's (was slide 7) "chrXIV ABF1 0/19" belongs to the seq-only EM run, whose ABF1 prior is 0. Its other three numbers are from the fib+seq EM trainDir. fib+seq+em10 finds 11/19.

### 8 / 8b
Slide 8 keeps the loop diagram and "Same for all three" (3 steps). 8b (hidden) lists the guards: step ≤ 10×, zero-count → 10× step, per-bp cap w^(1/L) ≤ 0.70, ±1.25× deadband + bisection, β seeded 0.5, unknown fixed at w = 1e-3, nucleosome weight 35 monitored. It adds one plain-language sentence.

### 9
The table has 30 px column padding and 20 px text, and the charts are capped at 258 px tall. Foot: "P (precision): of the model's calls, the fraction within 30 bp of a known site. R (recall): of the known sites, the fraction the model finds. F1: one score combining both, high only when both are high." (The 30 bp is the tuning match, `count_calls.py` MATCH_TOL.)

### 10 / 10b / 10c
- **Slide 10** compares the separately tuned campaigns fw01 / sw01 / bw01, each at its own concentrations. The title now says so.
- **10b** (visible, companion) uses the job-C genome decodes `robocop_genome_tw_{fw01_07,sw01_05,bw01_07}` (48/48 info files each, chrI window fully covered, posteriors in [0,1]). At ERV46:
  - 61,164: fw01 0.00 / sw01 0.00 / bw01 0.00
  - 62,658: fw01 0.00 / sw01 0.71 / bw01 0.00
  - The chips use bw01, so both read miss.
- **10c** (hidden) uses job A, sw01 round-5 weights with each layer setup: `robocop_chrXIV_chrII_tw_sw01_05` (seq), `_twS05_both`, `_twS05_fib`. It shows the same site as slide 6 (was 7b; chrXIV 167,958):
  - site: seq 0.00 / fib+seq 1.00 / fib 1.00
  - runs ≥ 0.10 in the window: 0 / 5 / 14
  - This matches job A's genome pattern for ABF1: calls 95 → 3,188 with fiber added, 27 of the added calls MacIsaac-supported.
  - Job A has no chrI decode, so ERV46 is not available for it.

### v4.3 — Rossi on all 58 groups (user decision, 2026-09-15)
Every Rossi P/R/F1 on the deck now uses `fitted=all`. Values were read programmatically from the TSVs; each old slide value was first matched to its non-fitted row.

| slide | label | old value | new value | source row |
|---|---|---|---|---|
| 10 table + chart | fw01 tune MacIsaac F1 | 0.0074 → 0.0115 (nonfitted) | 0.0094 → 0.0135 | `CT/fw01/validation.tsv` round 0 and 7, set=tune, ref=macisaac, fitted=all |
| 10 table + chart | fw01 tune Rossi F1 | 0.0050 → 0.0031 (nonfitted) | 0.0143 → 0.0111 | `CT/fw01/validation.tsv` round 0 and 7, set=tune, ref=rossi_cx, fitted=all |
| 10 table + chart | fw01 holdout MacIsaac F1 | 0.0086 → 0.0106 (all) | 0.0086 → 0.0106 | `CT/fw01/validation.tsv` round 0 and 7, set=holdout, ref=macisaac, fitted=all |
| 10 table + chart | fw01 holdout MacIsaac F1 | 0.0075 → 0.0069 (nonfitted) | column dropped (use MacIsaac all) | `CT/fw01/validation.tsv` round 0 and 7, set=holdout, ref=macisaac, fitted=all |
| 10 table + chart | fw01 holdout Rossi F1 | 0.0051 → 0.0047 (nonfitted) | 0.0133 → 0.0111 | `CT/fw01/validation.tsv` round 0 and 7, set=holdout, ref=rossi_cx, fitted=all |
| 10 table + chart | sw01 tune MacIsaac F1 | 0.0283 → 0.0677 (nonfitted) | 0.0304 → 0.0890 | `CT/sw01/validation.tsv` round 0 and 5, set=tune, ref=macisaac, fitted=all |
| 10 table + chart | sw01 tune Rossi F1 | 0.0153 → 0.0372 (nonfitted) | 0.0228 → 0.0852 | `CT/sw01/validation.tsv` round 0 and 5, set=tune, ref=rossi_cx, fitted=all |
| 10 table + chart | sw01 holdout MacIsaac F1 | 0.0246 → 0.0785 (all) | 0.0246 → 0.0785 | `CT/sw01/validation.tsv` round 0 and 5, set=holdout, ref=macisaac, fitted=all |
| 10 table + chart | sw01 holdout MacIsaac F1 | 0.0255 → 0.0559 (nonfitted) | column dropped (use MacIsaac all) | `CT/sw01/validation.tsv` round 0 and 5, set=holdout, ref=macisaac, fitted=all |
| 10 table + chart | sw01 holdout Rossi F1 | 0.0181 → 0.0360 (nonfitted) | 0.0194 → 0.0887 | `CT/sw01/validation.tsv` round 0 and 5, set=holdout, ref=rossi_cx, fitted=all |
| 10 table + chart | bw01 tune MacIsaac F1 | 0.0267 → 0.0474 (nonfitted) | 0.0288 → 0.0522 | `CT/bw01/validation.tsv` round 0 and 7, set=tune, ref=macisaac, fitted=all |
| 10 table + chart | bw01 tune Rossi F1 | 0.0151 → 0.0203 (nonfitted) | 0.0338 → 0.0381 | `CT/bw01/validation.tsv` round 0 and 7, set=tune, ref=rossi_cx, fitted=all |
| 10 table + chart | bw01 holdout MacIsaac F1 | 0.0244 → 0.0470 (all) | 0.0244 → 0.0470 | `CT/bw01/validation.tsv` round 0 and 7, set=holdout, ref=macisaac, fitted=all |
| 10 table + chart | bw01 holdout MacIsaac F1 | 0.0218 → 0.0443 (nonfitted) | column dropped (use MacIsaac all) | `CT/bw01/validation.tsv` round 0 and 7, set=holdout, ref=macisaac, fitted=all |
| 10 table + chart | bw01 holdout Rossi F1 | 0.0212 → 0.0262 (nonfitted) | 0.0356 → 0.0402 | `CT/bw01/validation.tsv` round 0 and 7, set=holdout, ref=rossi_cx, fitted=all |
| B2 table | u001 Rossi-CX F1, rounds 0/1/3/7 | 0.044 / 0.051 / 0.053 / 0.053 (already all 58) | unchanged | `CT/rossi_validation/match_u001.tsv`, ref=rossi_cx, pooled over the 58 groups with a CX file |
| B2 table | m001 Rossi-CX F1, rounds 0/1/3/7 | 0.037 / 0.045 / 0.046 / 0.050 (already all 58) | unchanged | `CT/rossi_validation/match_m001.tsv`, same pooling |

- The slide 10 tuning MacIsaac column also moved from non-fitted to all (fw01 0.0094 → 0.0135, sw01 0.0304 → 0.0890, bw01 0.0288 → 0.0522; same rows as slide 9) so both references share one scope.
- Claims re-checked against all-groups numbers: sw01 is still best in every column; bw01 is below sw01 in every column ("adding fiber lowered accuracy" holds); fw01's Rossi F1 still falls on tune and holdout ("matched counts, not accuracy" holds). Only the sw01 ratio wording changed (now 2.9–4.6×).
- Not validation numbers, so unchanged: Rossi summit coordinates on 5b, 5c, 5d, 7b, 10b, 10c.
- **Slide 5 Rossi summits (added v4.3).** `/usr/project/xtmp/nd141/projects/data/rossi_strand/Abf1_CX.bed`, `chr1` rows with starts 61170 and 62664 (0-based 1 bp summits) → 1-based 61,171 and 62,665 via the template's `load_rossi_summits` (start + 1), run on the trk5 window chrI:60,900–62,900. One summit per MacIsaac site (61,164–61,177 miss; 62,658–62,671 hit), each 0.5 bp from the site centre. Re-running the same function on 7b (chrXIV 167,964), 10b (61,171, 62,665) and 10c (167,964) reproduces their existing `rossi` arrays, so those slides were left alone. Chips are unchanged (the chip text in 5b–5d does not mention Rossi either); Rossi support is in slide 5's notes.
- Not changed but worth a look: slide 7 (was 6) says "55% for the 12 fitted-footprint TFs"; the 2,086-site denominator is the 9 fitted groups in the 58-group set.
- Job A (`overnight/A_validation.tsv`), job C (`overnight/C_validation.tsv`), `CT/tw_legacy_validation.tsv` and the tempered runs bt02/bt05/bt10 have no Rossi P/R/F1 on any slide.
- Offline copy regenerated from talk.html (skeleton + base64 PNGs), byte-identical procedure to the v4.2 build.

### v4.4 — slide order, 7b simplified, slide 9 definitions (user request, 2026-09-15)
- **5b, 5c, 5d unhidden**: now main slides right after 5 (eyebrow "Part 2", hidden tags and "[Hidden backup locus.]" note prefixes removed; content unchanged). Outline Part 2 reads "slides 4–5d"; the progress bar and counter derive from `data-kind` automatically.
- **Slide 7 hidden** (`data-kind="hidden"`, tag "Hidden · replaced by slide 7b"); reachable with `h` / overview. Notes prefixed accordingly.
- **Slide 7b replaces 7**: ALT tag removed; title "EM makes the calls worse", message "EM training adds many calls but few correct ones." Two rows only (fib+seq, fib+seq+em10, 130 px each), 3 build steps (untrained row → EM row → site outline/chips + numbers line + prior line). Numbers line (fib+seq, chrXIV, 19 sites): found 8 → 11, calls 327 → 1,198, F1 0.046 → 0.018, re-read from `layerXIV_scores/report_fib+seq.json` (tp 8, n_pred 327, f1 0.04624) and `chrXIV_scores/report_seqem10.json` (tp 11, n_pred 1198, f1 0.01808). Prior line: "EM pinned 28 factors at the cap and drove 66 to zero" (`robocop_train_em10_chrII`, §9). The `trk7b-data` payload lost the `fib` / `fib+em10` runs (`runOrder`, `call.also`, `fiberRun` → fib+seq). Fiber-only and sequence-only numbers, their EM prior counts, the 0/19 detail, fib+em10 site max 0.375 and the window call counts moved to 7b's speaker notes.
- **Slide 9**: P / R / F1 definitions moved from the step-5 footnote to a compact line above the table at build step 1.
- Offline copy regenerated with `scratchpad/mk_offline.py`. Visible order: T, 1, 3, 4, 5, 5b, 5c, 5d, 6, 7b, 8, 9, 10, 10b, 11.
