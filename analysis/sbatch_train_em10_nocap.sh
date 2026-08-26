#!/bin/bash
#SBATCH --job-name=em10_nocap
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/em10_nocap_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/em10_nocap_%j.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg

# CAP-OFF variant of the em10 run: same variant, coords and iterations, but the
# constrained-EM ceiling is lifted. In the capped run 28 of 154 states ended up pinned at
# mean+2sd = 6.69e-4 -- Abf1 and Nhp6a among them -- so their 1.0x prior ratio was imposed,
# not fitted. With the cap off, EM places each prior where the data actually puts it.
echo "Host: $(hostname)  Start: $(date)"
ROBOCOP_EM_CAP=off python train_em10.py coord_train_chrII_20.tsv ./robocop_train_em10_chrII_nocap/
echo "End: $(date)"

python em_trace_report.py robocop_train_em10_chrII_nocap
python em_smoke_gate.py robocop_train_em10_chrII_nocap --iters 10
