#!/bin/bash
#SBATCH --job-name=robocop_chrI_maskon
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/chrI_maskon_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/chrI_maskon_%j.err

# Activate conda BEFORE strict mode (its scripts reference unbound vars).
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

# Headless plotting
export MPLBACKEND=Agg

# NOTE: mask ON/OFF is set in code (pkg/robocop/robocop.py line ~797, the ABF1-focus
# mask). This job MUST be launched with that line commented IN (mask ON). Once this
# job has imported robocop, the state is frozen in memory and the line may be toggled
# for the mask-OFF job. See sbatch_chrI_maskoff.sh.

echo "Host: $(hostname)  Start: $(date)"
python run_fiberonly.py coord_chrI_full.tsv config_fiberonly.ini ./robocop_chrI_maskon/
echo "End: $(date)"
