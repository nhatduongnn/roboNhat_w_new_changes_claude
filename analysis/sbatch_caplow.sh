#!/bin/bash
#SBATCH --job-name=caplow
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/caplow_%A_%a.out
#SBATCH --error=logs/caplow_%A_%a.err
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
echo "VARIANT=$VARIANT  task=$SLURM_ARRAY_TASK_ID  host=$(hostname)  $(date)"
python run_split_variant_${VARIANT}.py coord_chrI_full.tsv robocop_train_fiberonly \
       ./robocop_chrI_seq_maskoff_${VARIANT}/ "$SLURM_ARRAY_TASK_ID" 6
echo "DONE $(date)"
