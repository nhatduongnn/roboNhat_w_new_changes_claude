#!/bin/bash
#SBATCH --job-name=chrI_maskon_split
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/chrI_maskon_split_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/chrI_maskon_split_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# chrI, split into total=6 groups (array 0-5), Fiber-only, HARD ABF1-forbid mask ON.
# State frozen at import: seq neutralizer (robocopExtras:101) ACTIVE = seq OFF;
#   hard-forbid block (robocopExtras, post-1e-30-floor, zeros Fiber ch5/ch6 states 29:nuc_start)
#   commented IN = TRUE ABF1-only. Launch the whole array with this state; do not flip until done.
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
python run_fiberonly_split.py coord_chrI_full.tsv robocop_train_fiberonly ./robocop_chrI_maskon/ "$SLURM_ARRAY_TASK_ID" 6
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
