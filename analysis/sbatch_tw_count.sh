#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.err

# Tuner v2 count step: a copy of sbatch_count_calls.sh with --calls always on, CHROMS REQUIRED
# (a subset decode must never fall back to all 16 chromosomes), one chromosome per array task,
# output to ${COUNTS_ROOT:-conc_tuning}/counts_${TAG}/<chrom>.tsv (+ macisaac/ and calls/ sidecars).
#
# Launch (normally from tune_w.py):
#   OUTDIR=<decode> TAG=tw_fw01_00 CHROMS="chrXIV chrII" sbatch --array=0-1 sbatch_tw_count.sh

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${OUTDIR:?OUTDIR not exported}"
: "${TAG:?TAG not exported}"
: "${CHROMS:?CHROMS not exported (required for tuner v2 counts)}"
COUNTS_ROOT=${COUNTS_ROOT:-conc_tuning}

mkdir -p "${COUNTS_ROOT}/counts_${TAG}"

CHROM=$(echo $CHROMS | tr " " "\n" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$CHROM" ]; then
    echo "ERROR: array task $SLURM_ARRAY_TASK_ID has no chromosome in CHROMS='$CHROMS'." >&2
    exit 1
fi

echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  CHROM=$CHROM  OUTDIR=$OUTDIR  Start: $(date)"
python count_calls.py "$OUTDIR" --chrom "$CHROM" --calls \
       --out "${COUNTS_ROOT}/counts_${TAG}/${CHROM}.tsv"
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
