#!/bin/bash
#SBATCH --job-name=revfix_seq_maskon
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/revfix_seq_maskon_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/revfix_seq_maskon_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# chrI, 6-way split. Fiber + sequence layer, ABF1-only hard mask.
# WITH the fiber reverse-block channel-cross fix (robocop.py:648-649 and :662).
# Layer/mask state is BAKED INTO pkgvar/seq_maskon/robocop/utils/robocopExtras.py --
# nothing to comment in or out, and no race with concurrently running arrays.
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
python run_split_revfix_seq_maskon.py coord_chrI_full.tsv robocop_train_fiberonly ./robocop_chrI_seq_maskon_revfix/ "$SLURM_ARRAY_TASK_ID" 6
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
