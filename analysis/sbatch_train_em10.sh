#!/bin/bash
#SBATCH --job-name=train_em10
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/train_em10_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/train_em10_%j.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

# 10 Baum-Welch iterations on the transition prior, using pkgvar/seq_maskoff_em10 --
# pkgvar/seq_maskoff (what the revfix decodes use, reverse-strand fix included) with
# iterations flipped on and a compact per-iteration trace.
#
# Training coords are 20 x 5 kb on chrII (median A-trials 66-89), so chrI stays a fully
# held-out test set. Single job, no array: EM is serial across iterations, and every
# segment's emission tensor is held live at once (~1 GB per 5 kb window at n_states=3485),
# which is what sets --mem.
echo "Host: $(hostname)  Start: $(date)"
python train_em10.py coord_train_chrII_20.tsv ./robocop_train_em10_chrII/
echo "End: $(date)"

python em_trace_report.py robocop_train_em10_chrII

# Same gate as the smoke test, at full length. Exits non-zero if the likelihood ever
# decreased, if iteration 0 is not the unfitted calculateKD prior, or if nothing moved --
# which is what the chrI decode's --dependency=afterok keys off, so a bad fit is never
# decoded and mistaken for a result.
python em_smoke_gate.py robocop_train_em10_chrII --iters 10
