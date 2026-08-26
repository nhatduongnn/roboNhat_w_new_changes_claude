#!/bin/bash
#SBATCH --job-name=chrI_em10
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/chrI_em10_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/chrI_em10_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# chrI, 6-way split. Fiber + sequence layer, all TFs. Same package variant, emission
# model and reverse-strand fix as sbatch_revfix_seq_maskoff.sh -- the ONLY difference is
# trainDir: robocop_train_em10_chrII (10 Baum-Welch iterations on the transition prior,
# fitted on held-out chrII) instead of robocop_train_fiberonly (unfitted calculateKD).
# So any change against robocop_chrI_seq_maskoff_revfix is attributable to the prior.
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
python run_split_em10_decode.py coord_chrI_full.tsv robocop_train_em10_chrII ./robocop_chrI_seq_maskoff_em10/ "$SLURM_ARRAY_TASK_ID" 6
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
