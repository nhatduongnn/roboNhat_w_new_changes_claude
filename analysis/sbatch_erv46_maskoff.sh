#!/bin/bash
#SBATCH --job-name=robocop_erv46_maskoff
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/erv46_maskoff_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/erv46_maskoff_%j.err

# Activate conda BEFORE strict mode (its scripts reference unbound vars).
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

# Headless plotting
export MPLBACKEND=Agg
# ABF1-focus mask OFF -> all TFs participate in the Fiber emission
export ROBOCOP_ABF1_MASK=0

echo "Host: $(hostname)  Start: $(date)"
python run_fiberonly.py coord_erv46.tsv config_fiberonly.ini ./robocop_erv46_maskoff/
echo "End: $(date)"
