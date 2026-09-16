#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.err

# One link of the concentration-tuning auto-chain. Submitted by
# `tune_concentrations.py submit --chain` with afterok on the round's count job, it runs
# `next`: update (report + trajectory) -> stop rules -> build + submit the following round,
# which attaches the next link. A stop writes conc_tuning/<run>/STOPPED with the reason;
# every link appends to conc_tuning/<run>/chain.log.
#
# Launch (normally never by hand):  RUN=u001 ITER=2 sbatch sbatch_tune_next.sh

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${RUN:?RUN not exported}"
: "${ITER:?ITER not exported}"

echo "Host: $(hostname)  RUN=$RUN  ITER=$ITER  Start: $(date)"
python tune_concentrations.py next --run "$RUN" --iter "$ITER"
echo "End: $(date)"
