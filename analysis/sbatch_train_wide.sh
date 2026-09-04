#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.err

# Build a trainDir carrying the WIDENED TF geometry. Parameterized by VARIANT/OUTDIR so
# the wide and widenull models share one script; launched by run_wide_all.sh.
#
# This is NOT a fit -- robocop_em.py:116 hardcodes iterations = 0, so it is a config build
# plus one forward-backward over the 20 chrII training windows. It exists only because
# widening changes tf_lens/tf_starts/n_states/pwm_emission/transition_matrix, all of which
# live in HMMconfig.pkl; capA/capB/12tfs needed no trainDir because they only touched
# parameter pkls read at runtime.
#
# Memory matches sbatch_train_em10.sh: every training segment's emission tensor is held
# live at once (~1 GB per 5 kb window at n_states ~3.5k).
#
# tf_pads.tsv is read HERE and frozen into HMMconfig.pkl as dshared['tf_pads']; decodes
# inherit it and never re-read the tsv.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${VARIANT:?VARIANT not exported}"
: "${OUTDIR:?OUTDIR not exported}"

echo "Host: $(hostname)  Start: $(date)"
echo "VARIANT=$VARIANT  OUTDIR=$OUTDIR"
cat tf_pads.tsv | grep -v '^[[:space:]]*#' | grep -v '^[[:space:]]*$' || true

python train_wide.py coord_train_chrII_20.tsv "./$OUTDIR/" --variant "$VARIANT"

# Gate the built geometry before anything decodes against it. Catches a stale params pkl
# or an edited tf_pads.tsv between build and use, which would otherwise surface only as
# quietly wrong numbers.
python check_wide_traindir.py "$OUTDIR" --variant "$VARIANT"
echo "End: $(date)"
