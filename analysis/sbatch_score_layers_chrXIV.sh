#!/bin/bash
#SBATCH --job-name=layerXIVScore
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --array=0-12
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/layerXIVScore_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/layerXIVScore_%A_%a.err

# Score one point of the ABF1 concentration scan per array task.
#
# The whole grid runs concurrently -- that is the point. Each lambda is an independent
# decode, so eight of them cost one wait instead of eight, and the output is the entire
# F1-vs-lambda CURVE rather than a single argmax. A flat curve would say concentration is
# not the lever, which is worth learning before building anything.
#
# 96G rather than chrXIV's 200G: chrI is 230 kb against chrXIV's 784 kb, and the scorer
# holds n_positions x n_states.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

mkdir -p layerXIV_scores

# ARRAY BOUND GUARD. The --array bound above is a literal, not derived from layer_runs_chrXIV.tsv.
# A stale bound does not error -- the extra rows simply never get a task, and the missing
# reports look exactly like reports nobody asked for. Fail loudly instead.
NROWS=$(grep -v '^[[:space:]]*#' layer_runs_chrXIV.tsv | grep -v '^[[:space:]]*$' | wc -l)
if [ "${SLURM_ARRAY_TASK_MAX:-0}" -ne "$((NROWS - 1))" ]; then
    echo "ERROR: --array=0-${SLURM_ARRAY_TASK_MAX} but layer_runs_chrXIV.tsv has $NROWS rows;" >&2
    echo "       widen the #SBATCH --array bound to 0-$((NROWS - 1))." >&2
    exit 1
fi

ROW=$(grep -v '^[[:space:]]*#' layer_runs_chrXIV.tsv | grep -v '^[[:space:]]*$' \
      | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
LABEL=$(echo "$ROW" | awk '{print $1}')
OUTDIR=$(echo "$ROW" | awk '{print $2}')
LAM=na

echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
echo "LABEL=$LABEL  OUTDIR=$OUTDIR  lambda=$LAM"

python score_robocop.py "$OUTDIR" --label "$LABEL" \
       --out "layerXIV_scores/report_${LABEL}.json"

echo "done $LABEL $(date)"
