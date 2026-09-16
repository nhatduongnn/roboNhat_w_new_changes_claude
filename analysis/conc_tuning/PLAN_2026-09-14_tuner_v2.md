# Tuner v2: weight-space concentration tuning, three layer configurations, 58 MacIsaac∩Rossi groups

Status: **PLAN for review**. Nothing has been built, copied or submitted. Scratch evidence is in
`/usr/project/xtmp/nd141/scratch_tunerv2/`: `chroms.py`, `refs.json`, `rule.py`, `replay.py`,
`closedloop.py`.

## 0. Recommendation in one paragraph

Tune on **chrXIV + chrII** and hold out **chrIV**. Match counts on `occ`, the existing signal, which
is realistic tonight. Report P/R/F1 against MacIsaac and Rossi `_CX` for every round, so that
choosing a per-group best-F1 round is a free post-hoc stage 2.

Write a **new** tuner (`tune_w.py`) that works in θ = ln w. It has:
- a 1-decade step cap;
- a per-bp weight cap (w^{1/L} ≤ 0.70);
- censoring of E below the truncation floor;
- secant steps safeguarded by bisection inside a bracket;
- β seeded at 0.5.

A **new** row writer (`make_w_trainDir.py`) replaces the bit-exact fidelity gate with a check that
the written row reproduces the requested w.

Three masked pkgvar copies (`*_mr58`) run concurrently, on fitz nodes only. Estimates:
- ~15–20 min per round;
- 8 rounds in ~2.5–3 h;
- launch ~21:00, results ~00:30.

## 1. Factor set (verified)

`references.tsv` rows with a MacIsaac target and `n_cx > 0`: **58 groups → 61 motifs**. RAP1 is the
only multi-motif group (Rap1_zhu, Rap1_motif1, Rap1_motif2, Rap1_telomeric).

The 23 groups dropped for having no Rossi `_CX` file are ADR1, CST6, DAL80, DAL82, GAT1, GAT3, HAC1,
MOT3, MSN4, PHO2, PHO4, RDS1, RGT1, RIM101, SIP4, SMP1, TEC1, TYE7, UGA3, XBP1, YAP3, YAP6 and YOX1.

The 61 kept motifs:

> Abf1_murphy Ace2_badis Aft2_badis Aro80_zhu Azf1_badis Bas1_zhu Cad1_murphy Cbf1_zhu Cha4_zhu
> Cin5_murphy Fhl1_zhu Fkh1_zhu Fkh2_zhu Gal4_zhu Gcn4_zhu Gcr1_murphy Gln3_badis Gzf3_zhu
> Hap1_murphy Hsf1_badis Leu3_zhu Mbp1_zhu Mcm1_zhu Met31_badis Met32_badis Mig1_zhu Msn2_badis
> Nrg1_zhu Pdr1_badis Pdr3_murphy Phd1_zhu Put3_badis Rap1_motif1 Rap1_motif2 Rap1_telomeric
> Rap1_zhu Reb1_badis Rfx1_badis Rox1_badis Rph1_badis Rpn4_badis Rtg3_zhu Sfp1_zhu Skn7_badis
> Sko1_murphy Sok2_badis Stb4_murphy Stb5_murphy Ste12_murphy Stp1_murphy Stp4_zhu Sum1_zhu
> Sut1_murphy Swi4_badis Swi5_badis Ume6_zhu Yap1_zhu Ydr520c_badis Yml081w_zhu Yrr1_zhu
> Zap1_murphy

- **Mask:** keep 61 + `unknown` = 62 states; **92 masked** (assumption, **D1**).
- **Masking is exact in w-space.** The decoder only sees w_k = p_k/p_bg^{L_k}, so a masked motif's
  leftover prior is pure gauge: it has no effect on the posteriors. That also means m001's "masked
  motifs keep their prior mass" never mattered.
- **Precedent against masking:** genome F1 was u001 (live, untuned) 0.049 vs m001 (masked) 0.042.
- **Three groups have no MacIsaac site on chrXIV+chrII:** ARO80, RFX1, RPH1. They stay live but are
  **frozen at λ = 1**, so **55 groups are tuned**. Nine more have T of 1–2: MET32, MIG1, PDR3, PUT3,
  STP4, YDR520C, YML081W, YRR1, ZAP1. They are tuned, but left out of the within-2× metric
  (T ≥ 5 only).
- **`unknown` stays at λ 0.01, i.e. w = 0.1·0.01 = 1e-3** (L = 10, per-bp 0.50, below every tuned
  TF's cap). Decoded, it is a small sink: 7,119 copies genome-wide in fm002_00 (0.6% of bp) and 471
  in sm002_00. In w-space, 0.01 still means "a weak catch-all that cannot tile". Keep it (**D6**).

## 2. Chromosomes (D2, D3)

Counts are for the 58 groups. The "groups with ≥ N" columns count sites on the pair (chrXIV + X).

| second chrom | kb | windows | pair MacIsaac | MacIsaac ≥1/3/5/10 | Rossi `_CX` ≥1/3/5/10 |
|---|---|---|---|---|---|
| **chrII** | 813 | 204 | 1,439 | 55 / 46 / 41 / 35 | 58 / 57 / 53 / 43 |
| chrIV | 1,532 | 383 | 1,947 | 57 / 47 / 42 / 37 | 58 / 56 / 51 / 43 |
| chrV | 576 | 144 | 1,455 | 54 / 45 / 39 / 31 | 55 / 50 / 46 / 36 |
| chrXVI | 948 | 237 | 1,708 | 53 / 47 / 41 / 35 | 57 / 56 / 51 / 41 |
| chrXIV alone | 784 | 196 | 634 | 50 / 36 / 29 / 24 | 52 / 40 / 33 / 19 |

- **Pick chrII.** Its coverage is within one or two groups of chrIV's at about half the decode cost.
  It also has the best Rossi coverage.
- **The pair:** 400 windows; 1,439 MacIsaac and 1,441 Rossi `_CX` sites; median T per group 15.
- **Invalid posteriors:** avoid chrXII (rDNA). The count logs show single invalid positions
  elsewhere, e.g. chrVIII 562,238, chrII ×2 and chrXIV ×2. Those are dropped loudly and are
  harmless.
- **Held-out chrIV: recommend yes.**
  - Decode it at round 0 and at the final round for each campaign: 383 windows, ~8 min at 40 tasks.
  - chrIV has 1,313 MacIsaac / 1,264 Rossi sites, with 41 / 47 groups ≥ 5.
  - Optional afterwards: decode chrIV for every round (24 decodes, ~45 min of cluster time), enabling
    a held-out best-F1 round per group.
- **Circularity caveat:** the 12 fitted fiber footprints were trained on Rossi peak locations
  genome-wide. Rossi scores for fiber-containing configurations are flattering for those 12 on
  chrIV too. Report fitted and non-fitted groups separately.

## 3. Objective and update rule (D4)

### Objective

**Tonight: match counts on `occ`.** For each group g:
- T_g = MacIsaac sites on chrXIV + chrII, taken once per group from the per-chrom columns of
  `inputs/conc_targets_macisaac_c1.tsv`. These were checked equal to `load_macisaac_c1_sites`.
- E_g = Σ `occ` over g's motifs on the two chromosomes.

Calls (posterior ≥ 0.10) are used only for the P/R/F1 reporting.

An F1 or precision objective is not realistic tonight:
- F1 vs λ is non-monotone and noisy at a median T of 15, so it would need a per-group line search;
- u001/m001 showed that the count step sets an operating point, not accuracy.

**Stage 2** is free given the per-round tables: pick each group's best-F1 round on the tuning
chromosomes and score it on chrIV (needs the optional all-round chrIV decodes).

### Update, per tuned group

The tuned variable is the group offset δ_g = ln λ_g, so θ_k = ln Kd_k + δ_g for every motif in g.

```
y      = ln(E + 1),  yT = ln(T + 1),  gap = yT - y          # +1 pseudo-count: finite at E = 0
censor = E < 0.5 copies                                     # below save_sparse_posterior's 1e-4 floor
beta   : seed 0.5 (all three configurations); after a round with |Δδ| ≥ 0.1 and both E ≥ 0.5,
         beta ← clip(0.5·beta + 0.5·Δy/Δδ, 0.10, 1.0); censored or no-move rounds keep beta
alpha  : 0.7; halve on gap sign change, floor 0.2
step   : 0                            if |gap| ≤ ln 1.25  (deadband)
         sign(gap)·ln10               if censored          (no slope: one decade toward the target)
         alpha·gap/beta               otherwise
         clipped to ±ln10             (≤ 1 decade per round)
bracket: δ_under = highest δ measured under target, δ_over = lowest δ measured over;
         if both exist and the step leaves (δ_under, δ_over) → δ = midpoint   (safeguarded secant)
bounds : λ ∈ [1e-6, 1e3]  AND  per motif  w_k^{1/L_k} ≤ 0.70   (group hi = min over its motifs)
flags  : at_hi (cap or ρ-cap) and still > 2× under; at_lo and still > 2× over → report, keep going
```

**β seed.** The measured round-0→1 secant medians are u001 0.52, m001 0.48, fm001 0.65 and
sm001 0.64. The old seed of **0.23** therefore made first steps 2–3× too large, which pushed
factors into the zero zone.

**Why ρ_max = 0.70.** ρ = w^{1/L} is the prior odds per bp against background, so a TF needs an
emission likelihood ratio of at least 1/ρ per bp to win.
- Healthy decodes: every final tuned motif has ρ ≤ 0.54 (u001/m001); the pristine maximum TF is
  Gal4 at 0.59; the highest round with no pathology is u001_01's Pho2 at 0.71.
- Both runaways sat above 1: Sum1 1.04 (fm001 → D1) and Rox1 1.15 (sm001).
- w ≤ 1 (ρ ≤ 1) is not enough, because short motifs gain per-bp likelihood ratio from sequence.
- Where it binds: 0.70 is tighter than λ = 1e3 for 18 of the 61 motifs. The tightest are Gal4 at
  λ ≈ 26, Sum1 at 41, Rox1 at 82 and Msn2 at 218; the rest sit at 300–970. A cap of 0.8 would bind
  for 7 motifs. No finished u001/m001 motif needed ρ > 0.54, so the 0.70 cap costs nothing there.

**Nucleosome.** No hold, so w_nuc = 35 every round. There is no stop rule. The report prints
copies against the campaign's own round 0 and **WARNs** beyond ±10%.

**Stop rules.** Rounds 0–7 (MAX_ROUNDS = 8); or a round in which every tuned group is in the
deadband or at a bound; or any exception (writes `STOPPED`). The "within-2× stalled" rule is
dropped.

### Replay gate, already run in scratch

**Open loop** (`replay.py`): the new step applied to each recorded round's (λ, E).

| campaign | old rule, round 0: median / max \|Δlog10 λ\| | old rule, round 0: steps ≥ 3 decades | new rule, all rounds: max \|Δlog10 λ\| | new rule: max ρ proposed |
|---|---|---|---|---|
| fm001 | 2.10 / 6.0 | 30 | 1.0 | 0.53 |
| sm001 | 2.13 / 6.0 | 25 | 1.0 | 0.60 |
| u001 | 2.01 / 6.0 | 29 | 1.0 | 0.64 |

- **fm001 round 2**, which sent SUM1, ARO80, GAL4 and STB4 from about 1e-5 to **1e3**: the new rule
  gives at most +1 decade, or bisects inside the bracket. 19–28 groups are handled as censored
  rather than as a huge gap.
- **Late rounds of u001/m001**: the new steps are ≤ the old ones (median 0.00–0.05). The deadband
  freezes converged groups.

**Closed loop** (`closedloop.py`, a toy with E = E0·λ^β and a truncation cliff at 1 copy): within-2×
by round 7 is fm001 46 → 51, sm001 54 → 56 and u001 55 → 56 of 58, reached faster. The 1-decade cap
costs no rounds.

## 4. Code, file by file (all new; no legacy file edited)

| file | what |
|---|---|
| `make_w_trainDir.py` | `--src robocop_train_fiberonly --out D --weights w.json`, where the JSON holds {motif: λ} plus `unknown`. It builds w (TFs Kd·λ; unknown 0.1·λ; background 1; nucleosome 35), solves α by `brentq` on Σ w_k α^{L_k} = 1 (in log space, xtol 1e-15) and sets p_k = w_k α^{L_k}. It then calls `make_conc_trainDir.apply_probs` (import only), which runs `set_transition` / `set_initial_probs` / `set_end_probs`, copies the SIDECARS and writes `w_patch.json`. |
| ↳ gates | **(a) Source:** λ = 1 (unknown 1) reproduces the src decoder weights and row to rel ≤ 1e-9 (np.roots vs brentq, so not bit-exact). **(b) Legacy cross-check:** {unknown: 0.01} matches `robocop_train_ct_fm002_00` (legacy, no hold) to ≤ 1e-9. **(c) Blast radius:** only row `silent_states_begin` of `transition_matrix` changes. **(d) w-reproduction:** re-read the written pkl and require \|ln(tf_prob/bg^L) − θ_requested\| ≤ 1e-9 for every DBF plus the nucleosome, and ρ ≤ 0.70 for the tuned motifs. Any failure aborts. |
| `tune_w.py` | `init / build / submit [--chain] / update / next / status / holdout`, with `--run`. State lives in `conc_tuning/<run>/state.json`: config, driver, tune chroms, holdout chrom, the 58 groups → motifs, T per group, frozen groups, δ, β, α, brackets, flags, history. The update is §3. `update` also calls `tw_validate.py` for that round. |
| `tw_validate.py` | Imports `rossi_validate.load_references`, `fast_match` and `load_groups`, restricted to the 58 groups. It reads `counts_tw_<run>_NN/calls/<chrom>.tsv` and appends to `conc_tuning/<run>/validation.tsv`: round, set (tune/holdout), ref (macisaac/rossi_cx), fitted (all/fitted/nonfitted), sites, calls, matched, P, R, F1; plus within-2× (T ≥ 5), n_at_hi, n_at_lo, nucleosome copies and Δ vs round 0. `--legacy u001=0,7 m001=0,7` scores the existing `calls_ct_*` on the same chromosomes and groups, giving a baseline row. |
| `tw_summary.py` | Takes `fw01 sw01 bw01` and writes `conc_tuning/tw_summary.{tsv,png}`: 4 panels (F1 vs MacIsaac, F1 vs Rossi, within-2×, nucleosome Δ%) by round, one line per configuration, plus u001/m001 reference points. |
| `sbatch_tw_count.sh` | A copy of `sbatch_count_calls.sh` with `--calls`, `CHROMS` required and output to `conc_tuning/counts_${TAG}/`. |
| decode | **Reuse `sbatch_genome_decode.sh` unchanged.** It already takes `COORDS` and `NTASK`, and has the config.ini retry. Submit with `--array=0-39 --exclude=linux[31-40] --mem=24G` (the rule-7 table notes these). |

Merging counts is the existing `count_calls.py --merge` over the 2 tables.

## 5. pkgvar copies and drivers

```bash
for v in fiber_maskoff seqonly_maskoff seq_maskoff; do
  cp -a pkgvar/${v}_macisaac pkgvar/${v}_mr58 && find pkgvar/${v}_mr58 -name __pycache__ -exec rm -rf {} +
  sed "s/${v}_macisaac/${v}_mr58/g" run_split_revfix_${v}_macisaac.py > run_split_revfix_${v}_mr58.py
done
```

Then edit **only** the keep block in each `pkgvar/*_mr58/robocop/utils/robocopExtras.py`:
- replace the set with `KEEP_MR58` (61 motifs + `unknown`) and update its comment;
- change the print to `mr58 mask: kept 62 ... | masked 92`.

In `seqonly_maskoff_mr58` the block stays **after** the `data_emission_matrix[5|6][:] = 1` lines.
Verify with `diff -r pkgvar/X_macisaac pkgvar/X_mr58`: only that file, only that block.

## 6. Coords, names and launch

```bash
awk 'NR==1||$1=="chrXIV"||$1=="chrII"' coord_genome_full.tsv > coord_tw_chrXIV_chrII.tsv   # 400 windows, same tiling as genome runs
awk 'NR==1||$1=="chrIV"'               coord_genome_full.tsv > coord_tw_chrIV.tsv          # 383
```

| campaign | config | driver |
|---|---|---|
| `fw01` | Fiber only | `run_split_revfix_fiber_maskoff_mr58.py` |
| `sw01` | sequence only | `run_split_revfix_seqonly_maskoff_mr58.py` |
| `bw01` | both layers | `run_split_revfix_seq_maskoff_mr58.py` |

Per campaign and round:
- trainDir `robocop_train_tw_<run>_NN`
- decode `robocop_chrXIV_chrII_tw_<run>_NN`; holdout `robocop_chrIV_tw_<run>_NN`
- counts `conc_tuning/counts_tw_<run>_NN/`
- label `tw_<run>_NN`, so rule 6 holds (directory = label + region)

```bash
python tune_w.py init  --run fw01 --driver run_split_revfix_fiber_maskoff_mr58.py --tune chrXIV,chrII --holdout chrIV
python tune_w.py build --run fw01 --iter 0 && python tune_w.py submit --run fw01 --iter 0 --chain
python tune_w.py holdout --run fw01 --iter 0          # round-0 chrIV decode + count, in parallel
# same for sw01, bw01; `next` submits `holdout --iter <last>` when it writes STOPPED
python tw_validate.py --run fw01 sw01 bw01 --legacy u001=0,7 m001=0,7 && python tw_summary.py fw01 sw01 bw01
```

## 7. Gates before launch (~30 min)

1. **Rule replay:** done (§3). Re-run `replay.py` against the final `tune_w.py` update function
   (import it), expecting the same table.
2. **Writer:** build `robocop_train_tw_fw01_00` and see gates (a)–(d) pass. Then build a scratch
   trainDir with SUM1 at λ = 1e3 and check that the ρ cap clips it and the assert fires if bypassed.
3. **Emission probe** for each `*_mr58` tree: copy `scratch_fm_sm/gateA/{probe.py,probe.sh,check.py}`
   to `scratch_tunerv2/gateB/`, add `seq_maskoff`, and compare each `_mr58` against its `_macisaac`
   source on the ERV46 window.
   - kept 62 blocks plus bg/nuc columns bit-identical;
   - the 92 masked blocks exactly 0 on layers 5/6;
   - layer 0 == 1 for fiber, untouched for seq/both;
   - seq-only kept columns on 5/6 == 1;
   - stdout shows `kept 62 … masked 92`.
4. **Tuner dry run** in namespace `tw_dry`: feed the existing `counts_ct_m001_00/{chrXIV,chrII}.tsv`
   and `calls_ct_m001_00/calls/` to `update`, then check report, validation.tsv and STOPPED paths
   with no submission.
5. **No separate chrXIV gate decode.** Round 0 is itself a 10-minute decode. Read its report before
   round 1 lands, and cancel the chain if E, nucleosome copies or invalid-position counts look off.

## 8. Timeline (now 18:10; fitz 1,132 CPUs idle; fm002/sm002 cancelled at 18:04)

- **Per round (fitz only):** build 1 min; decode 400 windows / 40 tasks × ~0.7 min/window ≈ 7–9 min
  (+ queue); count 2 tasks ~4 min; next ~2 min. **≈ 15–20 min.** On linux3x it would be ≈ 2.5×.
- **Concurrency:** 3 campaigns × 40 × 4 CPUs = 480 CPUs.
- **Schedule:**
  - implement 18:30–20:30
  - gates 20:30–21:00
  - launch 21:00
  - 8 rounds by ~23:30–00:30
  - holdout plus validation +20 min
- **Fallbacks if partial:**
  - show round 0 vs latest per configuration: P/R/F1 vs MacIsaac and Rossi on chrXIV+chrII,
    within-2×, nucleosome Δ, flags, next to u001/m001 restricted to the same groups and chromosomes;
  - mark chrIV "pending";
  - round 0 alone already shows the three-configuration contrast.

## 9. Validation outputs (slide 15)

**`conc_tuning/tw_summary.tsv`**: one row per (campaign, round, set, ref), with calls, sites,
matched, P, R, F1, within-2× (T ≥ 5, of N), n_at_hi, n_at_lo and nucleosome Δ% vs r0.

**`tw_summary.png`**: the four panels of §4.

Slide table: rows fw01 / sw01 / bw01 / u001 / m001 (58 groups, chrXIV+chrII); columns F1 MacIsaac
r0 → final, F1 Rossi r0 → final, chrIV F1 r0 → final, within-2×, nucleosome Δ.

Expectation to state up front: fiber-only identity is not identifiable (75/84 motifs share the
`combined_low_count` emission), so fw01 F1 should stay near chance.

## 10. Decisions for the user

- **D1 — Non-common motifs: masked (assumed) or live-untuned.** Masked is cleaner: the 58 factors
  alone, identical across configurations. u001 > m001 on F1 argues for live. Cheap option: a 4th
  campaign `bu01` on the existing unmasked `seq_maskoff` driver, +160 CPUs.
- **D2 — Second chromosome:** chrII (recommended), chrIV (+1 group, 2× cost) or chrV (cheapest).
- **D3 — Held-out chrIV decode** at r0 and final: yes (recommended). Also every round?
- **D4 — Objective:** `occ` count-matching tonight plus post-hoc best-F1 round (recommended), or
  defer to F1 tuning later.
- **D5 — Reference status:** these 2-chromosome, 58-group runs supersede u001/m001 as the tuner of
  record. u001/m001 stay the genome-wide reference and are re-scored on the same groups and
  chromosomes for comparison.
- **D6 — `unknown`:** keep λ 0.01 (w 1e-3) live; alternatively mask it too.
- **D7 — ρ_max = 0.70** (vs 0.8 or w ≤ 1); keep the λ floor at 1e-6 (flag floor-stuck) or lower it
  to 1e-8.

## 11. Rule 7: what differs

| parameter | m001 | fm001 / sm001 | tuner v2 (fw01 / sw01 / bw01) |
|---|---|---|---|
| layers | fib+seq | fiber / seq-only | fiber / seq-only / both |
| live motifs | 84 + unknown | 84 + unknown | **61 + unknown (58 groups)** |
| tuned groups | 81 | 81 | **55** (ARO80, RFX1, RPH1 frozen, T = 0) |
| region / targets | genome, genome counts | genome | **chrXIV+chrII, per-chrom sums; chrIV holdout** |
| nucleosome | held; stop at ±5% | held; stop off | **no hold (w_nuc 35); warn at ±10% vs r0** |
| build | λ → `make_conc_trainDir` (bit-exact gate, drift abort) | same | **w → `make_w_trainDir` (w-reproduction gate)**; same decoder weights as legacy no-hold |
| gap | ln T − ln E | same | **ln(T+1) − ln(E+1); E < 0.5 censored** |
| β | seed 0.23, raw secant [0.1, 1] | same | **seed 0.5, averaged secant, no update when censored** |
| α | 0.7, halve on flip, floor 0.1 | same | 0.7, halve on flip, **floor 0.2** |
| step cap | none | none | **1 decade** |
| deadband / bracket | none | none | **ln 1.25 / bisection** |
| bounds | λ ∈ [1e-6, 1e3] | same | same **+ w^{1/L} ≤ 0.70** |
| flags | cap-stuck only | same | **cap-, ρ-cap- and floor-stuck** |
| stop rules | nuc ±5%, within-2× stall, 8 rounds | stall, 8 | **all frozen, 8 rounds, error** |
| counting | occ, no calls | same | occ **+ `--calls`**, Rossi validation every round |
| scheduling | 48 tasks, 48 G, any node | same | 40 tasks, 24 G, fitz only (no effect on results) |
