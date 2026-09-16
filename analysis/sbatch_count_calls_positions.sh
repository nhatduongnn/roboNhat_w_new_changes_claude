#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.err

# Re-count finished decodes with count_calls.py --calls, to recover call POSITIONS.
#
# The tuning loop's count jobs (sbatch_count_calls.sh) kept only per-factor counts and the
# MacIsaac match, so scoring the same rounds against another reference (Rossi) needs the calls
# themselves. Same code, same resources as sbatch_count_calls.sh, one (decode, chromosome) per
# array task, but everything goes to a NEW directory conc_tuning/calls_<TAG>/:
#     <chrom>.tsv            counts table  -- must be byte-identical to counts_<TAG>/<chrom>.tsv
#     macisaac/<chrom>.tsv   MacIsaac match -- must be byte-identical to counts_<TAG>/macisaac/
#     calls/<chrom>.tsv      every fixed-threshold call: factor chrom center start end
# The originals are never written to; the byte-identity is checked afterwards with diff.
#
# Launch:  TAGS="ct_u001_00 ct_u001_01 ..." sbatch --array=0-$((16*N-1)) sbatch_count_calls_positions.sh
#          task id = decode_index * 16 + chromosome_index (chrom.sizes order, chrM excluded)

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${TAGS:?TAGS not exported}"

CHROMLIST=$(awk '$1!="chrM"{print $1}' inputs/sacCer3.chrom.sizes | tr "\n" " ")
TAGARR=($TAGS)
DI=$((SLURM_ARRAY_TASK_ID / 16))
CI=$((SLURM_ARRAY_TASK_ID % 16))
TAG=${TAGARR[$DI]}
CHROM=$(echo $CHROMLIST | tr " " "\n" | sed -n "$((CI + 1))p")
if [ -z "$TAG" ] || [ -z "$CHROM" ]; then
    echo "ERROR: array task $SLURM_ARRAY_TASK_ID maps to no (tag, chromosome)." >&2
    exit 1
fi
OUTDIR=robocop_genome_${TAG}

mkdir -p conc_tuning/calls_${TAG}
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  TAG=$TAG  CHROM=$CHROM  Start: $(date)"
python count_calls.py "$OUTDIR" --chrom "$CHROM" --calls \
       --out "conc_tuning/calls_${TAG}/${CHROM}.tsv"
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
