#!/bin/bash
# Log-space twin of sbatch_agg_baseline.sh: the AGGREGATE (pileup, binomial) decode of the prototype
# window under pkgvar/seq_maskoff_logfull (= seq_maskoff + the logfull edits) and the SAME trainDir as
# the per-molecule runs. Reference for pooled_emission_check_logfull.py and checks.py.
#
#   ROBOCOP_LOG_EMISSION=exact sbatch --export=ALL singlefiber/sbatch_agg_baseline_logfull.sh
#
#SBATCH --job-name=permol_agg_logfull
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/usr/project/xtmp/nd141/permol_proto_logfull/logs/agg_logfull_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/permol_proto_logfull/logs/agg_logfull_%j.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg
export R_HOME=$CONDA_PREFIX/lib/R
export ROBOCOP_ABF1_MASK=0
: "${ROBOCOP_LOG_EMISSION:?set ROBOCOP_LOG_EMISSION to floored or exact}"
export ROBOCOP_LOG_EMISSION

OUT=/usr/project/xtmp/nd141/permol_proto_logfull/agg_logfull_${ROBOCOP_LOG_EMISSION}
mkdir -p "$OUT"
echo "Host: $(hostname)  Start: $(date)  ROBOCOP_LOG_EMISSION=$ROBOCOP_LOG_EMISSION"
/usr/bin/time -v python run_split_seq_maskoff_logfull.py \
    coord_erv46.tsv robocop_train_fiberonly "$OUT" 0 1
python write_run_info.py --outdir "$OUT" --driver run_split_seq_maskoff_logfull.py \
    --traindir robocop_train_fiberonly --coords coord_erv46.tsv --source decode-time --force
echo "End: $(date)"
