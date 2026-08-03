#!/bin/bash
#SBATCH --job-name=seqlayer_maskoff_noem
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/seqlayer_maskoff_noem_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/seqlayer_maskoff_noem_%j.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# Fiber-seq + SEQUENCE ON, ALL TFs (mask OFF), 5-window set, WITHOUT-EM path (keeps tmpDir).
# Code state at import: robocopExtras.py ~101 neutralizer OUT (seq ON); robocop.py ~797 mask OUT (mask OFF).
echo "Host: $(hostname)  Start: $(date)"
python run_fiberonly_split.py coord_all_fiber.tsv robocop_train_fiberonly ./robocop_seqlayer_maskoff/ 0 1
echo "End: $(date)"
