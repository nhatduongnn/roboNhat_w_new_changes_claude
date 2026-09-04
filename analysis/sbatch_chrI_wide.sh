#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.err

# chrI decode with widened TF footprints, 6-way split -- same shape as
# sbatch_revfix_seq_maskoff.sh so the comparison against
# robocop_chrI_seq_maskoff_revfix is one-variable. Parameterized by
# VARIANT/TRAINDIR/OUTDIR; launched by run_wide_all.sh.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${VARIANT:?VARIANT not exported}"
: "${TRAINDIR:?TRAINDIR not exported}"
: "${OUTDIR:?OUTDIR not exported}"

echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
echo "VARIANT=$VARIANT  TRAINDIR=$TRAINDIR  OUTDIR=$OUTDIR"
python run_split_variant_wide.py coord_chrI_full.tsv "$TRAINDIR" "./$OUTDIR/" \
       "$SLURM_ARRAY_TASK_ID" 6 "$VARIANT"
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
