#!/bin/bash
#SBATCH --job-name=pm200
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/pm200_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/pm200_%j.err

# Re-fit the per-TF Fiber-seq binomial p over motif centre +/- 200 bp (401 columns) and
# replot, out to the pm200 folder. Same machinery as the +/-50 round -- only --half moves.
#
# Two envs on purpose: the refit needs fiber_params_lib (robocop-2024, R_HOME), the
# plotter needs logomaker (pyranges_env3). Refit first; it self-gates on the register
# check, so a failure here must stop before any plot is written.

set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg

PKL=inputs/all_TFs_1000pealVal_params_pseudo_pm200bp.pkl
OUTDIR=robocop_all_abf1/figures/factor_p_values_pm200

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh

conda activate robocop-2024
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
echo "== refit, half=200 =="
python make_params_pm50.py --half 200 --out "$PKL"

conda deactivate
conda activate pyranges_env3
echo "== plot =="
python plot_factor_p_values.py --pm50 "$PKL" --outdir "$OUTDIR"

echo "done $(date)"
ls -l "$OUTDIR"
