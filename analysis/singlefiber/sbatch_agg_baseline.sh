#!/bin/bash
# Check-1 baseline: an AGGREGATE-binomial decode of the prototype window under the SAME
# package tree (sequence layer ON) and the SAME trainDir as the per-molecule runs, so the
# only difference between this and arm A is the emission form.
#
# Note on the comparator already on disk: robocop_erv46_maskoff decodes the same window
# but was produced by run_fiberonly_noem.py, which does sys.path.insert(0, '../pkg/'),
# and pkg/robocop/utils/robocopExtras.py:101 has data_emission_matrix[0][:] = 1 LIVE --
# i.e. it is fiber-only, sequence layer OFF. Comparing against it would vary two things
# at once, so this run exists instead. (robocop_erv46_maskoff is still reported, labelled.)
#
#SBATCH --job-name=permol_agg_baseline
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/usr/project/xtmp/nd141/permol_proto/logs/agg_baseline_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/permol_proto/logs/agg_baseline_%j.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg
export R_HOME=$CONDA_PREFIX/lib/R
export ROBOCOP_ABF1_MASK=0

OUT=/usr/project/xtmp/nd141/permol_proto/agg_baseline
mkdir -p "$OUT"
echo "Host: $(hostname)  Start: $(date)"
# UNMODIFIED driver, UNMODIFIED tree pkgvar/seq_maskoff, idx=0 total=1 so the info file
# is named info_0_1.h5 (what write_factor_table.py expects).
/usr/bin/time -v python run_split_revfix_seq_maskoff.py \
    coord_erv46.tsv robocop_train_fiberonly "$OUT" 0 1
echo "End: $(date)"
