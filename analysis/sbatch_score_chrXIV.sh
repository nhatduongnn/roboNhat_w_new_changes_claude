#!/bin/bash
#SBATCH --job-name=XIVscore
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --array=0-6
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/XIVscore_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/XIVscore_%A_%a.err

# Score one chrXIV decode per array task: the ABF1/nucleosome headline via
# score_robocop.py, and the per-factor grid via score_factors.py. ~10 min per run on
# chrXIV (measured 2m51s on chrI, 3.4x the sequence), so seven of them in parallel
# rather than 70 minutes of serial waiting.
#
# 200G because score_robocop.py holds the whole chromosome's state posterior at once:
# region_optable allocates positions x n_states = 784,333 x 3,485 float64 = 22 GB, and
# get_posterior_binding_probability_df builds its DataFrame on top. chrI needed 6.4 GB
# and fit in 48G; chrXIV does not. Chunking it with --regions would be cheaper but
# WRONG -- the ABF1 call threshold is 0.30 x the max over the scored region, so
# per-chunk regions would silently recalibrate the threshold and the chrXIV numbers
# would stop being comparable to the chrI ones. score_factors.py already chunks
# internally (CHUNK = 200_000) precisely because it computes its own global max first.
#
# Reads the (label, outDir) pairs from chrXIV_runs.tsv -- the same table score_factors.py
# takes with --runs-from, so the run set is written down in exactly one place.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg

mapfile -t ROWS < <(grep -v '^#' chrXIV_runs.tsv | grep -v '^$')
ROW="${ROWS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${ROW%%$'\t'*}"
OUTDIR="${ROW##*$'\t'}"
echo "task $SLURM_ARRAY_TASK_ID -> label=$LABEL outDir=$OUTDIR  $(date)"

mkdir -p chrXIV_scores
python score_robocop.py "$OUTDIR" --label "$LABEL" \
       --out "chrXIV_scores/report_${LABEL}.json"
python score_factors.py --chrom chrXIV --min-sites 5 \
       --run "${LABEL}=${OUTDIR}" --out "chrXIV_scores/factors_${LABEL}"
echo "done $LABEL $(date)"
