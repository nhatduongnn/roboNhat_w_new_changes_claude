#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-15
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.err

# Count predicted sites per factor, one chromosome per array task.
#
# WHY PER-CHROMOSOME. Counting genome-wide in one process would be a ~40 min serial pass
# over 12 Mb x 3485 states, which would dominate a tuning iteration whose decode is itself
# only ~45 min. Split 16 ways it finishes in the time of the largest chromosome (chrIV,
# 1.53 Mb, ~2x chrXIV), so ~5-8 min. count_calls.py --merge then sums the tables.
#
# This also makes `n_pred_adaptive` well defined: score_robocop's threshold is
# 0.30 x the factor's max over the span being scored, and per-chromosome is exactly the span
# score_factors.py uses -- which is what lets the two tools be cross-checked. See the
# count_calls.py docstring for why the tuning loop uses `occ` and not either call count.
#
# Memory: holds one chromosome's tracks for ~160 factors (chrIV: 160 x 1.53e6 x 4 B ~ 1 GB)
# plus one 200 kb optable chunk (200000 x 3485 x 8 B ~ 5.6 GB). 64G is ample.
#
# Launch:  OUTDIR=<decode> TAG=<name> sbatch sbatch_count_calls.sh

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${OUTDIR:?OUTDIR not exported}"
: "${TAG:?TAG not exported}"

mkdir -p conc_tuning/counts_${TAG}

# CHROMS lets a subset decode count only the chromosomes it actually covers;
# default is all 16 nuclear chromosomes, in chrom.sizes order.
CHROMLIST=${CHROMS:-$(awk '$1!="chrM"{print $1}' inputs/sacCer3.chrom.sizes | tr "\n" " ")}
CHROM=$(echo $CHROMLIST | tr " " "\n" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$CHROM" ]; then
    echo "ERROR: array task $SLURM_ARRAY_TASK_ID has no chromosome in CHROMLIST." >&2
    exit 1
fi

echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  CHROM=$CHROM  Start: $(date)"
python count_calls.py "$OUTDIR" --chrom "$CHROM" \
       --out "conc_tuning/counts_${TAG}/${CHROM}.tsv"
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
