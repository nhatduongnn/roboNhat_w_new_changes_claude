#!/bin/bash
#SBATCH --job-name=em10_fiber
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/em10_fiber_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/em10_fiber_%j.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg

# FIBER-ONLY control on the em10 run: same coords, same iterations, same cap -- the ONLY
# difference is that pkgvar/fiber_maskoff_em10 has the sequence layer off. Tests whether a
# trained prior can rescue ABF1 identity when no localising signal exists (untrained
# fiber-only scores 0 TP / 422 predictions / 1.08x enrichment).
echo "Host: $(hostname)  Start: $(date)"
python train_em10_fiber.py coord_train_chrII_20.tsv ./robocop_train_em10_chrII_fiber/
echo "End: $(date)"

python em_trace_report.py robocop_train_em10_chrII_fiber
python em_smoke_gate.py robocop_train_em10_chrII_fiber --iters 10
