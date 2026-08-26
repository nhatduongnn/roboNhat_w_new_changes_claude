#!/bin/bash
#SBATCH --job-name=jaspar_seq_maskon
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/jaspar_seq_maskon_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/jaspar_seq_maskon_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# chrI, 6-way split. Fiber + sequence, ABF1-only hard mask.
# trainDir = robocop_train_jaspar -> HMMconfig.pkl built from the JASPAR MA0265.3
# (reverse-complemented) ABF1 matrix. The decode NEVER re-reads the meme file; it
# takes pwm_emission/tf_prob/transition_matrix from that HMMconfig (robocop_no_em.py:51),
# which is why the retrain was required.
# Layer/mask state is BAKED INTO the pkgvar/ variant the driver imports.
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
python run_split_revfix_seq_maskon.py coord_chrI_full.tsv robocop_train_jaspar ./robocop_chrI_seq_maskon_JASPAR/ "$SLURM_ARRAY_TASK_ID" 6
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
