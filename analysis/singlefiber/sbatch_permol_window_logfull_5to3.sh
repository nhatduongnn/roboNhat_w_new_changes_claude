#!/bin/bash
#SBATCH --job-name=permol5to3
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/usr/project/xtmp/nd141/permol_proto_logfull/filt/slurm_%A_%a.out
# Single-fiber decode with singlefiber/run_permol_window_logfull_5to3.py (2026-11-02).
# The sbatch for the CURRENT driver; sbatch_permol_window_logfull.sh runs the old driver/model.
#   0: the 10-30 all7 model via explicit flags, --cols all --consensus off  -> must match to the 10-30 run
#      /usr/project/xtmp/nd141/permol_proto_logfull/all7/full/all7_5to3 (regression gate for the edits)
#   1: the current driver defaults (filtered, rDNA-free model) at ERV46
# Usage (from analysis/): sbatch --array=0 singlefiber/sbatch_permol_window_logfull_5to3.sh
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg R_HOME=$CONDA_PREFIX/lib/R ROBOCOP_LOG_EMISSION=exact
O=/usr/project/xtmp/nd141/permol_proto_logfull/filt
D=singlefiber/run_permol_window_logfull_5to3.py
FILT_FLAGS=${FILT_FLAGS:-}
i=${SLURM_ARRAY_TASK_ID:-0}
echo "$(hostname) task $i $(date)"
case $i in
  0) python $D --arm A --cols all --consensus off --pkg-tree pkgvar/permol_seq_maskoff_logfull_nucbrogsym_all7 --nuc-params nucleosome_params_brogaard_sym_all7.pkl --bg-params bg_params_all7.pkl --tf-params all_TFs_1000pealVal_params_pseudo.pkl --min-mapq 0 --min-aln-len 0 --min-qs -1 --keep-reads none --out $O/gate_all7_defaults ;;
  1) python $D --arm A $FILT_FLAGS --out $O/erv46_5to3_nordna ;;
esac
echo "End $(date)"
