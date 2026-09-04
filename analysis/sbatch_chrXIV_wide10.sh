#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --array=0-11
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.err

# One parameterized chrXIV decode array. Which model configuration runs is decided
# entirely by the three exported variables -- DRIVER (which pkgvar gets sys.path'd),
# TRAINDIR (which fitted/unfitted prior), OUTDIR. That keeps seven configurations in
# one script instead of seven near-identical copies, following the VARIANT pattern
# already used by sbatch_caplow.sh.
#
# Launch via run_chrXIV_all.sh -- it carries the (DRIVER, TRAINDIR, OUTDIR) table and
# is the only place that mapping is written down.
#
# chrXIV: 196 windows (coord_chrXIV_full.tsv), 12-way split -> ~16 windows/task.
# chrI ran ~10 windows/task in 7-9 min, so expect ~15-20 min here.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${DRIVER:?DRIVER not exported}"
: "${TRAINDIR:?TRAINDIR not exported}"
: "${OUTDIR:?OUTDIR not exported}"
: "${VARIANT:?VARIANT not exported}"

echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
echo "DRIVER=$DRIVER  VARIANT=$VARIANT  TRAINDIR=$TRAINDIR  OUTDIR=$OUTDIR"
python "$DRIVER" coord_chrXIV_full.tsv "$TRAINDIR" "./$OUTDIR/" "$SLURM_ARRAY_TASK_ID" 12 "$VARIANT"
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
