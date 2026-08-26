#!/bin/bash
#SBATCH --job-name=slide_abf1
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/slide_abf1_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/slide_abf1_%j.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg
# fiber_params_lib -> the generator imports rpy2/fitdistrplus
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
echo "Host: $(hostname)  Start: $(date)"
python slide_abf1_profile.py --halves 7,10,12,25,50 --tol 20
echo "End: $(date)"
