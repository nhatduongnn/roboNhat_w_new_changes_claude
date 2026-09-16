# Plan after the 2026-09-15 lab meeting

_Written 2026-09-16 15:10. This is a read-only plan: nothing has been run or edited. Numbers come from
the files cited. A few counts were computed here from committed tables; those are marked
"(computed)"._

Abbreviations:
- **F1** is pooled over the 58 groups: calls with posterior ≥ 0.10, matched one-to-one within 30 bp.
- **chrIV** is the held-out chromosome. **tune** means chrXIV+chrII.
- **φ** is the Fiber-seq temper: the fiber log-evidence is divided by φ.

---

## 1. Where things stand

The concentration tuner works. It hits the MacIsaac site counts, and running it longer or with a
tighter target no longer changes accuracy. The biggest lever found so far is not concentration: it
is how loudly the Fiber-seq evidence speaks. Untempered, the fiber term swamps sequence. Dividing
the fiber log-evidence by φ = 10 raises held-out chrIV F1 from 0.047 (bw01) to 0.129 (bt10). That
beats sequence alone (sw01, 0.080). The cost is nucleosome recall on chrXIV: 0.812 → 0.687.

Accuracy was still rising and recall still falling at φ = 10, the largest value tried. Only
sw01, fw01 and bw01 have been decoded genome-wide. For sw01, tuned and never-tuned chromosomes score
the same (0.080 vs 0.079), so overfitting to the two tuning chromosomes is not a visible problem.

Fiber-only ABF1 calls are still mostly wrong:
- chrI, untuned: 422 calls, 0/5 MacIsaac sites (`analysis/layer_scores/report_fib.json`).
- fw01, held-out chrIV: ABF1 F1 0.031 (`conc_tuning/fw01/validation_groups.tsv`, r14).
- bt10, held-out chrIV: ABF1 F1 0.390, 24 of 44 sites (`conc_tuning/bt10/validation_groups.tsv`, r5).

Sources: `presentation/continue_results.md`, `presentation/overnight_results.md`, and the scratch
directories `scratch_fiber_vs_seq{,2}/`.

---

## 2. The note items, marked

| # | note | status | evidence |
|---|---|---|---|
| 1 | Stack the 422 calls on slide 4; look at their m6A | **OPEN** | No stack exists. The closest is the chrI matched-filter scan (HANDOFF §8). It found a generic "protected notch in an accessible region": RAP1 scored better than ABF1. Its code was never saved; only `analysis/abf1_profile_pm100_agentA.npz` survives. The 422 calls come from `robocop_chrI_fib` (fiber only, λ = 1, all 153 motifs live), threshold 0.30 × the chrI maximum (`score_robocop.abf1_call_threshold`). |
| 2 | Run longer than 8 iterations; tighten 2× → 1.5× → 1.25× | **DONE** | The tuner's stop target was already 1.25×: `tune_w.py` sets `DEADBAND = ln 1.25`, and 2× is only a reporting column. All six campaigns were continued at **1.1×, up to 15 rounds** (`presentation/continue_results.md`). Held-out chrIV F1 moved by ≤ 0.005 in every run: bt10 0.128 → 0.129, bt05 0.108 → 0.108, sw01 0.078 → 0.080, bt02 0.075 → 0.073, bw01 0.047 → 0.047, fw01 0.011 → 0.009. |
| 3 | Run more than 8 iterations | **DONE** (same as #2) | bt10, bt05, sw01 and bt02 converged at 1.1× (55/55 groups). bw01 and fw01 reached 15 rounds and are running on to 25: `state.json` shows `max_rounds` 25 and `iter` 19 at 15:09. bw01 has 3 groups outside 1.1× pinned by the bisection bracket (BAS1, RTG3, SKN7); fw01 has 13. bw01 accuracy was identical from r11 to r14 (0.050 on the tuning chromosomes). |
| 4 | Train on odd, validate on even chromosomes; tune to Rossi | **OPEN** (partly covered) | Rossi `_CX` is already scored every round as a validation reference (`conc_tuning/<run>/validation.tsv`). The genome-wide tuned vs never-tuned check exists for sw01, fw01 and bw01 (overnight job C). No Rossi-targeted or odd/even campaign has been run. |
| 5 | Mask everything except ABF1 and tune; or keep only the 12 TFs | **OPEN** | Older evidence exists but was not tuned under v2. ABF1-only mask on chrI (untuned, before the revfix): fiber-only recall 2/5 with the mask vs 0/5 without (memory `chrI-and-seqlayer-results`). Under the old tuner, masking hurt slightly (u001 0.049 vs m001 0.042), because the kept factors absorbed the calls and got stuck at the λ floor (`count-tuning-accuracy-result`). The v2 tuner has no λ floor. |

---

## 3. Recommended order

Cluster costs assume:
- ~16–18 min per round for chrXIV+chrII (400 windows, 40 tasks × 4 CPUs = 160 CPUs);
- a genome decode is 48 tasks × 4 CPUs = 192 CPUs, ~45 min per task on fitz or ~2 h on linux3x;
- the cap is 500 CPUs. bw01 and fw01 hold ~322 CPUs until about 17:00.

### Step 1: Stack the fiber-only ABF1 calls (note 1)

**What to run.** This is local analysis. No decode is needed.
1. **Calls.** Re-derive the 422 chrI calls with `score_robocop.call_abf1` from `robocop_chrI_fib`.
   - The caller returns a centre but no strand.
   - Orient each call by comparing the ABF1 forward-state and reverse-state posterior mass over the
     call (states 1–14 / 15–28, HANDOFF §4.3). Check the split against `dshared` first.
2. **m6A.** Read per-position k/n on the A channel, Watson and Crick, over ±150 bp.
   - Source: `make_params_pm50.Pileup(PILEUP)`.
   - For reverse calls, mirror the window **and** swap Watson/Crick (the revfix rule in
     `slide_abf1_profile.py`).
   - Plot the aggregate m6A fraction (Σk/Σn per offset), plus a per-call heatmap sorted by distance
     to the nearest decoded nucleosome dyad.
3. **Comparison stacks on the same axes:**
   - (a) all 300 MacIsaac ABF1 sites genome-wide, oriented by motif strand, with the bed used as
     shipped. chrI has only 5, so use the genome.
   - (b) +1 nucleosome edges, Chereji dyad ± 73 (the "nucleosome edge" control);
   - (c) MacIsaac REB1 and RAP1 sites (the "other notch-in-NDR" control, §8);
   - (d) random positions in accessible regions;
   - overlay the fitted 14-column ABF1 p-vector, the background p 0.1383 and the genome-wide pooled
     rate 0.0790 (§8).
4. **Annotate each call:** a MacIsaac site of any TF within 30 bp? A Rossi `_CX` summit of any TF
   within 30 bp? The distance to the nearest nucleosome?
5. **Extension (what makes it current).** Run the same stack on ABF1 calls from the tuned finals,
   split into true and false positives:
   - fw01 final, bt10_05 and sw01_07, on chrXIV+chrII+chrIV;
   - calls are in `conc_tuning/counts_tw_<run>_NN/calls/`.

**Why first.**
- It is free and runs alongside everything else.
- It answers the open mechanism question: are fiber-only false calls real ABF1 footprints that
  MacIsaac lacks, nucleosome edges, or a generic protected notch?
- The answer steers step 5: masking down to ABF1 only helps if fiber can tell ABF1 from other
  notches.
- It also steers the emission fix (HANDOFF §4.6).

**Cost.** CPU-minutes, plus maybe one 32–64 G task for the pileup reads. About 2–3 h of work.

**What would change the next step.**

| if the false calls show… | then |
|---|---|
| ABF1-like flank rims plus a notch | MacIsaac may be incomplete. Weigh Rossi more in step 6. |
| a nucleosome-edge profile | Nucleosome boundaries leak into TF states: an emission problem. Favour a beta-binomial arm in step 3. |
| the same notch as REB1/RAP1 sites | Identity must come from sequence. Drop fiber-only arms from step 5. |

**Risks.**
- The 422 calls come from an untuned, all-motif decode, so the extension on the tuned runs is what
  counts.
- Call centres are run midpoints, not motif-registered, so the profile blurs by a few bp.

### Step 2 (supporting): Close out bw01 and fw01

**What.** Either cancel them now or let them reach round 25 (~17:00). Record them as stalled with
bracket-pinned groups. **Do not fix the bracket now.**

**Why.**
- Neither configuration is recommended going forward: fiber-only F1 0.009, untempered 0.047.
- bw01 accuracy has been flat since r11.
- The pinned groups (bw01 3, fw01 13) will not move with more rounds.
- Cancelling frees ~320 CPUs for steps 3 and 4 two hours earlier.

**Fix later, only if a carried-forward campaign shows pinned groups.** When the bracket is narrower
than 0.1 decade but the gap is still outside the deadband, freeze the group at whichever bracketed
value has the smaller gap, and count it as converged. This is a proposal, not tested.

**Cost.** None.

### Step 3 (supporting, needed before notes 4 and 5): Extend the φ sweep beyond 10 on the current setup

**What.**
- New campaigns bt20 and bt50: both layers, `mr58` mask, MacIsaac targets, tune chrXIV+chrII,
  hold out chrIV, deadband 1.1×, cold start.
- Build `pkgvar/seq_maskoff_mr58_phi20` and `_phi50` as copies of `_phi10`, changing only φ
  (rule 3), with matching drivers.
- Run the emission probe gate against `_phi10`, as in PLAN §7.3.
- **Optional third arm:** beta-binomial ρ = 0.1 in place of φ, frozen from the scratch package
  (`scratch_fiber_vs_seq2/pkgvar_fvs2_temper`, env `BB_RHO`; `fb_conditions.tsv`).
- Score Chereji +1/−1 recall on chrXIV for each run, as the campaigns already do.

**Why here.**
- φ is the one knob that has moved accuracy: holdout F1 0.047 → 0.073 → 0.108 → 0.129 for
  φ = 1, 2, 5, 10.
- The continuation moved accuracy by ≤ 0.005.
- Accuracy had not peaked at 10, and sequence alone (φ → ∞) gives 0.080, so there is an optimum
  somewhere above 10.
- Steps 5 and 6 both need a fixed φ.
- Finding φ on 2 chromosomes costs ~2 h. Finding it on half the genome costs 3× more per round.

**Why beta-binomial is worth including.**
- Real ABF1 sites vary in m6A far more than the binomial allows (~83×, scratch).
- ρ ≈ 0.1 ranked sites about as well as φ = 10 (`q5_sites_summary.tsv`: AUROC 0.950 vs 0.983).
- It is the principled version of the HANDOFF §4.6 fix. φ is a blunt one.

**Cost.**
- About 5–6 rounds × 17 min ≈ 1.7 h per campaign, plus a ~10 min holdout.
- Two campaigns take 320 CPUs; three take 480, which only fits once bw01 and fw01 are gone.
- Code: copy the pkgvar trees and drivers, then run the probe gate. About 1 h.

**What would change the next step.**

| outcome | then |
|---|---|
| F1 peaks at 20 or 50 | carry that φ |
| F1 is flat from 10 to 50 while recall keeps falling | carry 10 |
| beta-binomial matches φ* with better nucleosome recall | carry beta-binomial |

**Risks.**
- Nucleosome recall may slide toward sw01's 0.137. The user needs a floor (decision D2).
- Chereji recall is measured on chrXIV only.

### Step 4 (supporting): Genome-wide decode of bt10_05 (later also φ*)

**What.**
- Run `sbatch_genome_decode.sh` with the bt10_05 trainDir and the `mr58_phi10` driver, 48 tasks,
  throttled to `%40` if it runs next to step 3.
- Score it the way overnight job C was scored (`overnight_score.py`): all 16 chromosomes, the
  tuned pair, never-tuned, per chromosome.
- Also report **odd vs even halves separately** and Chereji recall genome-wide.

**Why here.**
- The best model has only been scored on 2+1 chromosomes, while sw01, fw01 and bw01 have
  genome-wide numbers (0.080 / 0.011 / 0.044).
- This gives the baseline step 6 must beat, on exactly the odd/even split, for free.
- It needs nothing from steps 1–3, so it can start as soon as there are CPUs.

**Cost.** 192 CPUs (160 throttled), about 1–2 h wall.

**What would change the next step.**
- If bt10's never-tuned chromosomes match its tuned ones, as sw01's did, the odd/even split in step
  6 adds little protection against overfitting. Its value then rests on the Rossi target and on
  having more groups with targets.
- If they drop, step 6 matters more.

**Risk.** chrXII rDNA positions will be dropped (`n_excl` was ~20k for the fiber runs in job C).

### Step 5: Masked campaigns: the 12 fitted TFs, and ABF1 only (note 5)

**What.** Two campaigns at φ* (both layers, tempered), MacIsaac targets, tune chrXIV+chrII, hold out
chrIV:
- **m12:** keep the 12 fitted motifs plus `unknown`; mask everything else.
  - 9 of the 12 have MacIsaac targets.
  - NHP6A, SPT15 and TBF1 have none. Either freeze them at λ = 1, as ARO80/RFX1/RPH1 are now, or
    mask them, which makes it m9 (decision D3).
- **m01:** keep ABF1 plus `unknown`.
- Needs new pkgvar trees (copies of `seq_maskoff_mr58_phi<φ*>` with a new keep-list).
- `tune_w.py` must be generalised, preferably as a copy with a `--factor-set` option. It
  currently hard-codes 58 groups / 61 motifs, `KEEP_MR58` and `EXPECT_FROZEN`.
- **Comparison:** per-group holdout F1 for the same 9 groups inside bt10
  (`conc_tuning/bt10/validation_groups.tsv`, r5):

  | ABF1 | UME6 | REB1 | FKH1 | CIN5 | SKO1 | RAP1 | MCM1 | FHL1 |
  |---|---|---|---|---|---|---|---|---|
  | 0.390 | 0.407 | 0.347 | 0.263 | 0.241 | 0.125 | 0.099 | 0.056 | 0.000 |

- Report the fitted-only pooled F1 against MacIsaac, and against Rossi with the circularity caveat.

**Why here.**
- The question: does removing the 49 non-fitted competitors help the factors whose fiber footprint
  is actually fitted? The other 141 motifs share one `combined_low_count` emission (HANDOFF §5.3).
- It is cheap: 2 chromosomes, and fewer states, so decodes should be faster.
- It reuses the setup whose numbers already exist.
- It needs φ* from step 3, and step 1 tells whether ABF1-only is sensible.
- It comes before step 6 so the odd/even campaign can inherit the mask decision.

**Cost.**
- Code: tuner copy, two pkgvar trees, gates. About 2–3 h.
- Runs: about 6 rounds × ≤ 17 min for two campaigns, ~2 h, 320 CPUs. A good overnight job.

**What would change the next step.**

| outcome | then |
|---|---|
| m12's fitted-group F1 clearly above bt10's same groups | carry the 12-TF mask into step 6 |
| equal or worse, as masking was under u001/m001 | keep 58 groups |
| m01 ABF1 F1 well above 0.390 | the ABF1 bottleneck is competition, not emission |

**Risks.**
- With competitors gone, the kept factors absorb other factors' sites. REB1/RAP1 notches become
  ABF1 calls: step 1's "generic notch" outcome predicts this.
- Fitted-group numbers are small (SKO1 has 5 sites, MCM1 6 on chrIV).
- Rossi scores for the fitted TFs are circular (below).

### Step 6: Odd/even split, tuned to Rossi (note 4)

**What.** Two campaigns at φ*, with the mask chosen in step 5:
- **Arm R:** targets are Rossi `_CX` per-group counts summed over the odd chromosomes (I, III, V,
  VII, IX, XI, XIII, XV).
- **Arm M:** targets are MacIsaac counts over the same chromosomes.
- **Validation:**
  - even half (II, IV, VI, VIII, X, XII, XIV, XVI), scored against **both** references;
  - Chereji recall on even chromosomes;
  - round 0 and final only, or a single genome decode at the final round (it covers both halves).
- **Cold start**, not from bt10 weights. bt10 was tuned on chrXIV+chrII, which are even
  chromosomes. So were chrIV and the whole current setup, so a warm start leaks validation-half
  counts into the weights.
- **Circularity fix (recommended):** refit the 12 fiber footprints and the shared
  `combined_low_count` rate from **odd-chromosome Rossi peaks only**.
  - Use `make_params_pm50`'s `fit_group` machinery, as the LOCO refit in `slide_abf1_profile.py`
    does.
  - Write the result to a **new** pkl filename (rule 2) and load it from a new pkgvar tree.
  - The pkl is loaded at decode time: `seq_maskoff_12tfs` differs only in which pkl it loads.
- **Code:** generalise the tuner copy from step 5 to take chromosome sets, a target source
  (MacIsaac / Rossi `_CX` / Rossi `_motif`) and a group set.

**Why last.**
- It is the most expensive step: the odd half is 1,335 windows vs 400 now (computed from
  `coord_genome_full.tsv`); the even half is 1,686.
- It depends on φ* (step 3) and the mask choice (step 5).
- Count tuning moves the operating point, not accuracy: continuation ≤ 0.005; u001/m001 flat
  after step 1. So switching the target alone is unlikely to move F1 much.
- **Its real payoffs:**
  1. A held-out estimate over 8 chromosomes instead of 1.
  2. More groups with usable targets. Groups with ≥ 5 sites: odd half has MacIsaac 50/58 and
     Rossi `_CX` 55/58, vs MacIsaac 41/58 on chrXIV+chrII now (computed).
  3. A route to tuning TFs that MacIsaac lacks (NHP6A, SPT15, TBF1, and the other Rossi-only
     factors).
- Running both arms is the only way to tell "Rossi is a better target" apart from "more
  chromosomes helped".

**Correcting one premise.** For the 58 tuning groups, Rossi does not have many more sites than
MacIsaac. Odd half: Rossi `_CX` 5,475 vs MacIsaac 5,958. Even half: 6,310 vs 6,166 (computed with
`rossi_validate.load_references`). Rossi's size advantage is across its 378 TFs, not within these
58.

**Cost.**
- Round time: with ~100 tasks (400 CPUs) roughly 25–35 min per round for one campaign; with two
  campaigns at ~60 tasks each, roughly 40–50 min. This is an estimate scaled from 400 windows.
- Rounds: about 6–8 cold, so **~4–6 h wall for the two arms**, plus a validation decode of ~1 h.
- The refit and the tuner generalisation are about half a day of code and gates.

**What would change what follows.**

| outcome | meaning |
|---|---|
| Arm R ≈ Arm M on even-half F1 against both references | the target choice does not matter; keep MacIsaac (independent of the footprint training) as the validation reference |
| Arm R wins against MacIsaac as well as Rossi | Rossi is the better target |
| Arm R wins only against Rossi | suspect circularity or a target/validation match, not better calls |

**Risks.**
- **Circularity (HANDOFF §5.2).** The footprints were fit at Rossi peaks genome-wide
  (`inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed`). Tuning and validating on
  Rossi without the odd-only refit flatters every fiber-containing run for the 12 fitted TFs.
- MacIsaac overlaps Rossi only weakly: 2,065 of 12,124 MacIsaac sites in the 58 groups have a
  same-TF `_CX` summit (17%, `rossi_validation/references.tsv`), so the two references disagree.
- `_CX` includes peaks without a motif and is gene-body-heavy for some factors. `_motif` removes
  genic peaks preferentially (Fhl1 53.2 → 17.9%, §5.2), so the choice changes the targets.
- chrXII (rDNA, invalid posteriors) sits in the even half and is dropped by `count_calls.py`.

---

## 4. Decisions needed, by step

| before | decision | options (recommendation first) |
|---|---|---|
| Step 2 (now) | **D1**: bw01/fw01 | cancel now and record them as stalled; or let them finish at ~17:00 |
| Step 3 | **D2**: φ grid and pick rule | φ 20 and 50 (+ beta-binomial ρ 0.1). Pick the φ with the highest holdout F1 **subject to** chrXIV Chereji recall ≥ a floor the user names (bt10 0.687, bt05 0.736, bw01 0.812). An untested alternative: temper only TF states, not nucleosomes; this would need a new pkgvar. |
| Step 4 | **D2b**: which runs go genome-wide | bt10_05 now; φ* once chosen; bt05 optional |
| Step 5 | **D3**: mask arms | m12 + m01 both. NHP6A/SPT15/TBF1: frozen at λ = 1 (recommended, like ARO80/RFX1/RPH1) or masked (→ m9). Layers: tempered both at φ* only. No fiber-only masked arm unless step 1 shows ABF1-specific footprints in the false calls. |
| Step 6 | **D4**: target and validation | Arms R (Rossi `_CX`) and M (MacIsaac) both on odd; validate on even against both, with MacIsaac as the primary independent reference |
| Step 6 | **D5**: circularity | refit the 12 footprints plus `combined_low_count` on odd-only Rossi (recommended); or keep the shipped pkl and treat Rossi validation for fitted TFs as non-independent |
| Step 6 | **D6**: `_CX` vs `_motif` as the Rossi target | `_CX` (matches the current validation) or `_motif` (motif-anchored, fewer genic peaks) |
| Step 6 | **D7**: start point and factor set | cold start (recommended). Groups: the 58, the step 5 mask, or widen to Rossi-only TFs (only with Arm R). |

Rule 7 reminder: each campaign above changes one thing from its comparator:
- step 3: φ only;
- step 5: the mask only;
- step 6: the target and chromosome set (with the refit as a stated extra).

State the differences before launch and again next to the results.

---

## 5. Parallel vs sequential, and a 2-day schedule

**Run in parallel.** Steps 1, 3 and 4 have no dependencies on each other. Step 5's code and step 6's
refit and tuner code can be written while those run.

**Run in sequence.**
- Step 3 → step 5 (needs φ*).
- Step 1 → step 5 (whether ABF1-only is sensible).
- Steps 3 and 5 → step 6 (φ* and the mask).
- Step 4 on φ* waits for step 3.

**CPU plan.** Never more than 480 at once:
- bt20 + bt50 use 320;
- the genome decode throttled `%40` uses 160;
- the beta-binomial arm is added only after bw01/fw01 end.

| when | cluster | analyst |
|---|---|---|
| **Wed 15:30–17:00** | D1: cancel bw01/fw01 (or wait) | Step 1 stack (chrI 422, then the tuned TP/FP extension). Build `_phi20`/`_phi50` pkgvar + drivers; probe gate. |
| **Wed ~17:00** | Launch bt20 + bt50 (chained, cold) and the bt10_05 genome decode (`%40`). Add beta-binomial ρ 0.1 if D2 says so and CPUs allow. | Finish step 1 figures. Start the `tune_w` copy with `--factor-set`. |
| **Wed ~19:00–21:00** | bt20/bt50 converge plus holdout; the genome decode finishes (~19:00 on fitz, ~21:00 on linux3x) | Score job-C style plus odd/even plus Chereji. **D2**: choose φ*. Build m12/m01 pkgvar at φ*; gates. |
| **Wed night** | m12 + m01 (~2 h). Genome decode of φ* if ≠ 10. | — |
| **Thu morning** | — | Read step 5; **D3/D4/D5/D6**. Odd-only refit of the 12 footprints + `combined_low_count` (new pkl, new pkgvar); tuner chromosome/target options; dry run on existing counts (PLAN §7.4 style). |
| **Thu afternoon → Fri morning** | Arms R and M on the odd half (~4–6 h), then the final even-half or genome validation decode (~1 h) | Refresh the continuation board / talk numbers once results land. |

If Thursday's code slips, the schedule compresses cleanly by dropping Arm M. That leaves the target
effect confounded with the chromosome change, so say so next to the result.
