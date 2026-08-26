#!/bin/bash
#SBATCH --job-name=var_lowabf1
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/var_lowabf1_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/var_lowabf1_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg

# VARIANT 2 low_abf1: combined_low_count lowered to ABF1's mean footprint (0.0843/0.0797 vs 0.2489/0.2647), background left as shipped
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
python run_split_variant_lowabf1.py coord_chrI_full.tsv robocop_train_fiberonly ./robocop_chrI_seq_maskoff_lowabf1/ "$SLURM_ARRAY_TASK_ID" 6
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
