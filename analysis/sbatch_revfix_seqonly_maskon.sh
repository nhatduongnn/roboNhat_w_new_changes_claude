#!/bin/bash
#SBATCH --job-name=revfix_seqonly_maskon
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/revfix_seqonly_maskon_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/revfix_seqonly_maskon_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# chrI, 6-way split. SEQUENCE LAYER ONLY -- Fiber channels neutralised to 1.0.
# ABF1-only hard mask, carried on layer 0 (see the variant's robocopExtras.py).
# Layer/mask state is BAKED INTO pkgvar/seqonly_maskon/ -- nothing to comment in or
# out in pkg/, and no race with concurrently running arrays.
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
python run_split_revfix_seqonly_maskon.py coord_chrI_full.tsv robocop_train_fiberonly ./robocop_chrI_seqonly_maskon_revfix/ "$SLURM_ARRAY_TASK_ID" 6
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
