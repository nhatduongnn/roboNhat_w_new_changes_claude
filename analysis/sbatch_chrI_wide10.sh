#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.err

# chrI decode, 6-way split -- same shape as sbatch_revfix_seq_maskoff.sh so the comparison
# against robocop_chrI_seq_maskoff_revfix is one-variable apart from the widening itself.
#
# Parameterized by DRIVER/TRAINDIR/OUTDIR, matching sbatch_chrXIV.sh's interface so the two
# chromosomes are launched the same way. Launched by run_widememe_all.sh, which is the only
# place the (DRIVER, TRAINDIR, OUTDIR) mapping is written down.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${DRIVER:?DRIVER not exported}"
: "${TRAINDIR:?TRAINDIR not exported}"
: "${OUTDIR:?OUTDIR not exported}"
: "${VARIANT:?VARIANT not exported}"

echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
echo "DRIVER=$DRIVER  VARIANT=$VARIANT  TRAINDIR=$TRAINDIR  OUTDIR=$OUTDIR"
python "$DRIVER" coord_chrI_full.tsv "$TRAINDIR" "./$OUTDIR/" "$SLURM_ARRAY_TASK_ID" 6 "$VARIANT"
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
