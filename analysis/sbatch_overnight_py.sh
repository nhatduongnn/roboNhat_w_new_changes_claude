#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --exclude=linux[31-40]
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.err

# Overnight 2026-09-15: run one light python step (scoring, summary, B watcher) as a Slurm job.
#   PYARGS="overnight_summary.py" sbatch sbatch_overnight_py.sh

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
: "${PYARGS:?PYARGS not exported}"
echo "Host: $(hostname)  PYARGS=$PYARGS  Start: $(date)"
python $PYARGS
echo "End: $(date)"
