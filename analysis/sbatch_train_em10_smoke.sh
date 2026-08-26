#!/bin/bash
#SBATCH --job-name=em10_smoke
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/em10_smoke_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/em10_smoke_%j.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# 2 windows x 2 iterations. update_transition_probs has never executed in this codebase
# (robocop_em.py ships iterations = 0), so this proves the Baum-Welch step runs and behaves
# before the full job spends 2 hours. em_smoke_gate.py exits non-zero on any violation,
# which is what the full job's --dependency=afterok keys off.
echo "Host: $(hostname)  Start: $(date)"
rm -rf ./robocop_train_em10_smoke/
ROBOCOP_EM_ITERS=2 python train_em10.py coord_smoke_chrII_2.tsv ./robocop_train_em10_smoke/
echo "End: $(date)"

python em_smoke_gate.py robocop_train_em10_smoke --iters 2
