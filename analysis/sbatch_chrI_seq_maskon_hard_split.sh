#!/bin/bash
#SBATCH --job-name=chrI_seq_maskhard
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/chrI_seq_maskhard_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/chrI_seq_maskhard_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# chrI, split into total=6 groups (array 0-5), Fiber + SEQUENCE layer ON, ABF1 HARD-forbid mask ON.
# State frozen at import: robocopExtras.py:101 (seq neutralizer) commented OUT = seq ON;
#   robocopExtras.py hard-forbid block (post-floor, zeros Fiber ch5/ch6 states 29:nuc_start)
#   commented IN = TRUE ABF1-only (every other TF posterior == 0, not 1e-30).
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
python run_fiberonly_split.py coord_chrI_full.tsv robocop_train_fiberonly ./robocop_chrI_seq_maskon_hard/ "$SLURM_ARRAY_TASK_ID" 6
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
