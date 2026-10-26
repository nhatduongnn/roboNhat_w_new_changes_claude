#!/bin/bash
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg R_HOME=$CONDA_PREFIX/lib/R
export ROBOCOP_LOG_EMISSION=exact
set -x
python run_split_seq_maskoff_fit9_bgopen_logfull.py posterior_fix/logspace_review/coord_chrXIV_412001.tsv robocop_train_tw_bo09_48 posterior_fix/logspace_review/scratch/exact_bo09_412001 0 1 > posterior_fix/logspace_review/scratch/exact_bo09_412001.log 2>&1
python run_split_seq_maskoff_all153_logfull.py posterior_fix/logspace_review/coord_chrXIV_412001.tsv robocop_train_tw_bu01_11 posterior_fix/logspace_review/scratch/exact_bu01_412001 0 1 > posterior_fix/logspace_review/scratch/exact_bu01_412001.log 2>&1
python run_split_seq_maskoff_fit9_bgopen_logfull.py posterior_fix/coord_rdna_chrXII_seg2.tsv robocop_train_tw_bo09_48 posterior_fix/logspace_review/scratch/exact_bo09_rdna2 0 1 > posterior_fix/logspace_review/scratch/exact_bo09_rdna2.log 2>&1
echo lane2 done
