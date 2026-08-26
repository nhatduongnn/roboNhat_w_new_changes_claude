#!/bin/bash
#SBATCH --job-name=conc_sweep
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/conc_%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/conc_%x_%A_%a.err

# ABF1 concentration sweep. Decodes whole chrI with a trainDir whose ONLY difference from
# robocop_train_fiberonly is that dbf_conc['Abf1_murphy'] was scaled by LAM (one row of the
# transition matrix -- see make_conc_trainDir.py).
#
# Usage:
#   sbatch --export=ALL,LAM=30,VARIANT=seq_maskon  sbatch_conc_sweep.sh
#   sbatch --export=ALL,LAM=30,VARIANT=seq_maskoff sbatch_conc_sweep.sh
#
# VARIANT selects the frozen package copy under pkgvar/ (layer + mask state baked in), via
# the existing run_split_revfix_<VARIANT>.py driver -- unchanged, it already takes trainDir
# as argv[2]. Nothing here edits pkg/ or pkgvar/.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg

: "${LAM:?set LAM, e.g. --export=ALL,LAM=30,VARIANT=seq_maskon}"
: "${VARIANT:?set VARIANT to seq_maskon or seq_maskoff}"

TRAINDIR="robocop_train_conc${LAM}"
OUTDIR="./robocop_chrI_${VARIANT}_conc${LAM}/"

echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  Start: $(date)"
echo "LAM=$LAM  VARIANT=$VARIANT  TRAINDIR=$TRAINDIR  OUTDIR=$OUTDIR"
test -f "$TRAINDIR/HMMconfig.pkl" || { echo "missing $TRAINDIR/HMMconfig.pkl"; exit 1; }
cat "$TRAINDIR/conc_patch.json"

python "run_split_revfix_${VARIANT}.py" coord_chrI_full.tsv "$TRAINDIR" "$OUTDIR" \
       "$SLURM_ARRAY_TASK_ID" 6

echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
