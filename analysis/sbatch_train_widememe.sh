#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.err

# Build the trainDir for the meme-file-widened model.
#
# Unlike sbatch_train_wide.sh, nothing here reads tf_pads.tsv at run time: the geometry is
# already baked into inputs/motifs_meme_wide.txt (built once by make_meme_wide.py), and
# get_transition_matrix_info derives tf_lens from pwm[tf].shape[1] with no code change.
# pkgvar/seq_maskoff_widememe differs from pkgvar/seq_maskoff by exactly one line, the
# Fiber params filename.
#
# This is NOT a fit -- robocop_em.py:116 hardcodes iterations = 0, so it is a config build
# plus one forward-backward over the 20 chrII training windows. It exists only because
# widening changes tf_lens/tf_starts/n_states/pwm_emission/transition_matrix, all of which
# live in HMMconfig.pkl; capA/capB/12tfs needed no trainDir because they only touched
# parameter pkls read at runtime.
#
# Memory matches sbatch_train_em10.sh: every training segment's emission tensor is held
# live at once (~1 GB per 5 kb window at n_states ~3.5k).

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

VARIANT="${VARIANT:-seq_maskoff_widememe}"
OUTDIR="${OUTDIR:-robocop_train_widememe}"
CONFIG="${CONFIG:-config_fiberonly_widememe.ini}"

echo "Host: $(hostname)  Start: $(date)"
echo "VARIANT=$VARIANT  OUTDIR=$OUTDIR  CONFIG=$CONFIG"
grep '^pwmFile' "$CONFIG"

python train_widememe.py coord_train_chrII_20.tsv "./$OUTDIR/" \
       --variant "$VARIANT" --config "$CONFIG"

# Gate the built geometry before anything decodes against it: a bad pad slice would shift
# ABF1's whole Fiber-seq footprint relative to its motif and still decode without error.
python check_widememe_traindir.py "$OUTDIR"
echo "End: $(date)"
