#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.err

# One link of the tuner-v2 chain (tune_w.py). CMD=next: update -> validate -> stop rules ->
# build + submit the next round (or STOPPED + chrIV holdout). CMD=validate-holdout: score the
# chrIV holdout calls of round ITER. Every link appends to conc_tuning/<run>/chain.log.
#
# Launch (normally never by hand):  RUN=fw01 ITER=2 CMD=next sbatch sbatch_tw_next.sh

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${RUN:?RUN not exported}"
: "${ITER:?ITER not exported}"
: "${CMD:?CMD not exported}"

echo "Host: $(hostname)  RUN=$RUN  ITER=$ITER  CMD=$CMD  Start: $(date)"
python tune_w.py "$CMD" --run "$RUN" --iter "$ITER" ${TW_ROOT:+--root "$TW_ROOT"}
echo "End: $(date)"
