#!/bin/bash
#SBATCH --job-name=postViewer
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/postViewer_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/postViewer_%j.err

# Rebuild the combined (chrI + chrXIV) posterior browser.
#
# Off the login node because the run list now includes wide150, whose HMMconfig.pkl is
# 914 MB -- 20 runs x 3 regions loads one of those per run, and the wide variants' pkls
# dominate. Regions are only 5 kb, so the posterior tables themselves are small.
#
# The runs come from viewer_runs_chrI.tsv / viewer_runs_chrXIV.tsv via viewer_regions.tsv.
# Their label sets must be IDENTICAL -- the label is the join key that preserves the
# selected run across a region switch; make_posterior_viewer.py prints a note per label
# that is not shared, and a clean run prints none.
#
# No --pkg: every widened variant here is widememe-method and keeps the shipped
# sum_for_dbf_probs, so the baseline ../pkg collapse is the correct one (the block-wide
# plateau is the intended rendering -- see README_wide_implementations.md).

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

python make_posterior_viewer.py \
    --regions viewer_regions.tsv \
    --out posterior_viewer_all.html

ls -la posterior_viewer_all.html
