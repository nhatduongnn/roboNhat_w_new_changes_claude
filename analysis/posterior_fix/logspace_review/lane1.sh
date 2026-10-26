#!/bin/bash
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg R_HOME=$CONDA_PREFIX/lib/R
set -x
python posterior_fix/harness_capture.py pkgvar/seq_maskoff_fit9_bgopen posterior_fix/logspace_review/coord_chrXIV_412001.tsv posterior_fix/logspace_review/scratch/caplin412_out posterior_fix/logspace_review/scratch/cap_lin_412001.npz --traindir robocop_train_tw_bo09_48 > posterior_fix/logspace_review/scratch/cap_lin_412001.log 2>&1
python posterior_fix/harness_capture.py pkgvar/seq_maskoff_fit9_bgopen posterior_fix/logspace_review/coord_chrXIV_408001.tsv posterior_fix/logspace_review/scratch/caplin408_out posterior_fix/logspace_review/scratch/cap_lin_408001.npz --traindir robocop_train_tw_bo09_48 > posterior_fix/logspace_review/scratch/cap_lin_408001.log 2>&1
ROBOCOP_LOG_EMISSION=floored python posterior_fix/harness_capture.py pkgvar/seq_maskoff_fit9_bgopen_logfull posterior_fix/logspace_review/coord_chrXIV_412001.tsv posterior_fix/logspace_review/scratch/caplogf412_out posterior_fix/logspace_review/scratch/cap_logfloored_412001.npz --traindir robocop_train_tw_bo09_48 > posterior_fix/logspace_review/scratch/cap_logfloored_412001.log 2>&1
echo lane1 done
