#!/bin/bash
#SBATCH --job-name=var_12tfs
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/var_12tfs_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/var_12tfs_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg

# VARIANT 3 12_tfs: only the 12 TFs with fitted footprints occupiable; the other 141 plus 'unknown' hard-masked
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
python run_split_variant_12tfs.py coord_chrI_full.tsv robocop_train_fiberonly ./robocop_chrI_seq_maskoff_12tfs/ "$SLURM_ARRAY_TASK_ID" 6
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
