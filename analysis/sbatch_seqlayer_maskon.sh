#!/bin/bash
#SBATCH --job-name=robocop_seqlayer_maskon
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/seqlayer_maskon_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/seqlayer_maskon_%j.err

# Activate conda BEFORE strict mode (its scripts reference unbound vars).
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# EXPERIMENT: Fiber-seq + SEQUENCE layer ON, ABF1-only (mask ON), 5-window set.
# Isolates how well ABF1 alone is positioned when sequence helps. Code state required at
# import: robocopExtras.py ~101 neutralizer commented OUT (seq ON); robocop.py ~797 mask
# commented IN (mask ON).

echo "Host: $(hostname)  Start: $(date)"
python run_fiberonly.py coord_all_fiber.tsv config_fiberonly.ini ./robocop_seqlayer_maskon/
echo "End: $(date)"
