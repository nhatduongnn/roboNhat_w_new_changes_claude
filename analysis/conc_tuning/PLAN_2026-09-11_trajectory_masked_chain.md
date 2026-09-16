# Concentration tuning: what each λ step buys, a masked-TF companion run, auto-chained rounds

## Context

Campaign `u001` is running:
- λ_unknown = 0.01 and the nucleosome prior held;
- the 81 MacIsaac groups (84 motifs) are tuned toward raw MacIsaac counts;
- round 0 is done, round 1 has decoded and is being counted.

Each round's `conc_tuning/u001/report_NN.tsv` records, per group, the MacIsaac count, decoded
copies, calls (posterior ≥ 0.10) and calls matched within 30 bp of a MacIsaac site.

The user wants:
1. **Per-step accuracy accounting.** When λ moves, did the count get closer to MacIsaac, and at
   what gain or cost in precision and recall? Which round, and which λ step for which TF,
   changed accuracy most?
2. **A companion run with the 69 non-MacIsaac motifs masked out**, using the ABF1-only mask
   mechanism, so they can't compete with the tuned TFs. Decision: **mask only**. Their prior is
   left in place, so the 84 tuned TFs keep exactly u001's priors and any difference between the
   runs comes from removing the competitors. The masked motifs hold 1.1% of the prior at every
   entry, more than all 84 kept TFs combined (0.96%). `unknown` stays live at 0.01.
3. Decision: **auto-chain rounds** on the cluster, because rounds take ~2 h (the linux3x nodes
   set the pace) and there are now two campaigns.

## Part 1: per-step accuracy trajectory

**New script `analysis/tuning_trajectory.py --run <run>`.** It reads every
`conc_tuning/<run>/report_NN.tsv`, so it also covers round 0, which already exists. No state
changes are needed.

Per group per round, it computes:
- precision = matched / calls
- recall = matched / MacIsaac
- F1
- count fold-off = decoded / MacIsaac

**Step effects**, comparing round t with round t−1 for each group:

| column | meaning |
|---|---|
| Δlog10 λ | the step taken |
| Δ\|log count gap\| | did the count get closer to MacIsaac? |
| Δcalls, Δmatched | |
| Δprecision, Δrecall, ΔF1 | |
| marginal precision = Δmatched / Δcalls | of the calls this step added (or removed), what fraction were at MacIsaac sites |

Marginal precision is the direct "at what cost" number:
- for a raise, near 0 means the step added junk;
- for a lowering, a high value means the step threw away real sites.

**Output**
- `conc_tuning/<run>/trajectory.tsv`: long format, one row per group per round.
- Printed summary:
  - pooled precision, recall and F1 per round, with the change from the previous round;
  - the latest step's top F1 gains and losses, with each group's λ step and marginal precision;
  - per group, the round whose step changed F1 most.
- `tune_concentrations.py update` calls it at the end of each round.

**Caveat:** MacIsaac requires conservation, so precision is a lower bound. It's meaningful for
comparing rounds and steps, not as an absolute number (the same stance as `score_factors.py`).

## Part 2: masked companion run `m001`

1. **New package copy `analysis/pkgvar/seq_maskoff_macisaac/`**
   - Copy `pkgvar/seq_maskoff/` (772 KB) and change only `robocop/utils/robocopExtras.py`.
   - After the 1e-30 floor (around line 106), insert the keep-list block modelled on
     `pkgvar/seq_maskoff_12tfs/robocop/utils/robocopExtras.py:108-140`:
     - keep set = the 84 motifs with `has_target=1` in `inputs/conc_targets_macisaac_c1.tsv`,
       plus `unknown`, hardcoded (pkgvar copies bake their state in);
     - assert every name exists in `dshared['tfs']`;
     - zero Fiber layers 5 and 6 over each other motif's `tf_starts .. +2*tf_lens` state block;
     - print the kept/masked summary.
   - Background and nucleosome states are outside `dshared['tfs']`, so they're untouched.
2. **New driver `analysis/run_split_revfix_seq_maskoff_macisaac.py`**: a copy of
   `run_split_revfix_seq_maskoff.py` with the pkgvar path changed.
3. **`tune_concentrations.py`**: the driver becomes per-campaign.
   - `init_state` records `driver` (new `--driver` flag, defaulting to the current `DRIVER`).
   - `cmd_submit` uses `st.get("driver", DRIVER)`, so u001's existing state is unaffected.
4. **Start m001 at round 0**: every TF at λ = 1, unknown 0.01, nucleosome held, masked driver,
   same 81 targets. The masked motifs have no targets, so the loop never moves them.

## Part 3: auto-chaining

- **New `tune_concentrations.py next --iter t`**:
  1. runs `update --iter t` (report + trajectory);
  2. evaluates the stop rules;
  3. if none fire: `build --iter t+1`, then `submit --iter t+1 --chain`.
- **Stop rules**, each written to `conc_tuning/<run>/STOPPED` with the reason:
  - nucleosome copies more than 5% from 66,521 (pause for the user's review);
  - within-2x count not risen for 2 rounds;
  - round 8 reached;
  - any exception.
- **`submit --chain`** adds a third job after the count job (`afterok`):
  - new `sbatch_tune_next.sh` (1 CPU, 8 GB, 1 h) runs `next --iter t --run R`;
  - the job is named `ctNext_<run>_NN` so it's visible in `squeue`;
  - with `afterok`, a failed count leaves the chain stopped rather than proceeding.

## Execution

0. Save a copy of this plan in the repo for the user to revisit:
   `analysis/conc_tuning/PLAN_2026-09-11_trajectory_masked_chain.md`.
1. Implement Parts 1–3.
2. Run the checks (Verification 1–4).
3. u001: `update --iter 1` (round 1 counts are ready) → report to the user → `build --iter 2` →
   `submit --iter 2 --chain`.
4. m001: `build --iter 0` → `submit --iter 0 --chain`.
5. As reports land, summarise both runs side by side:
   - within-2x, precision/recall/F1 per round;
   - biggest step effects;
   - the can't-fix list;
   - nucleosome copies.

## Verification

1. `tuning_trajectory.py --run u001` on round 0: pooled precision 1.7% (1,517 / 86,838) and
   recall 8.4% (1,517 / 17,997) match the round-0 printout. After round 1's update, the delta
   columns are filled.
2. Masked package, via a quick local run of the mask function on a toy window or a 1-segment
   decode (chrI, first coordinate), with the masked driver:
   - the printout says 85 kept (84 + unknown) and 69 masked;
   - in the optable, all 69 masked columns are exactly 0 and the kept columns are non-zero;
   - background and nucleosome columns are present.
3. Chaining: a trivial test job that runs `sbatch --test-only` from a compute node confirms
   jobs can be submitted from inside a job.
4. `next` logic dry-run in a throwaway namespace (as with the earlier self-test): feed the
   existing round-0 counts, and confirm each stop rule writes `STOPPED` and skips submission.
5. After m001 round 0: the decode's masked-TF columns sum to 0 in `counts_ct_m001_00.tsv`
   (occ = 0 for all 69), and the can't-fix list and report look sane.
