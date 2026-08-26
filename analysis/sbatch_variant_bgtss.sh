#!/bin/bash
#SBATCH --job-name=var_bgtss
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/var_bgtss_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/var_bgtss_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg

# VARIANT 1 bg_tss: background re-estimated from ORF-oriented TSS-200..-100 open regions (inputs/bg_params_tss.pkl, p=0.1606 vs 0.1383 shipped)
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
python run_split_variant_bgtss.py coord_chrI_full.tsv robocop_train_fiberonly ./robocop_chrI_seq_maskoff_bgtss/ "$SLURM_ARRAY_TASK_ID" 6
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
