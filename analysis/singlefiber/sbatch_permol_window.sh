#!/bin/bash
# One task per (footprint arm x efficiency mode) for the single-fiber prototype window.
#
#   sbatch --array=0            singlefiber/sbatch_permol_window.sh   # arm A, eff off
#   sbatch --array=1-9%5        singlefiber/sbatch_permol_window.sh   # the rest
#
# Task -> setting (index = SLURM_ARRAY_TASK_ID):
#   0  A                eff off      5  B pi=0.5          eff on
#   1  A                eff on       6  B pi=0.7          eff off
#   2  B pi=0.3         eff off      7  B pi=0.7          eff on
#   3  B pi=0.3         eff on       8  C flat 0.024286   eff off
#   4  B pi=0.5         eff off      9  C flat 0.024286   eff on
#
#SBATCH --job-name=permol_window
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/usr/project/xtmp/nd141/permol_proto/logs/permol_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/permol_proto/logs/permol_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg
export R_HOME=$CONDA_PREFIX/lib/R

ARMS=(A       A      B       B      B       B      B       B      C      C)
PIS=(""      ""     0.3     0.3    0.5     0.5    0.7     0.7    ""     "")
EFFS=(0       1      0       1      0       1      0       1      0      1)
NAMES=(armA_effoff armA_effon armB_pi0p3_effoff armB_pi0p3_effon \
       armB_pi0p5_effoff armB_pi0p5_effon armB_pi0p7_effoff armB_pi0p7_effon \
       armC_effoff armC_effon)

i=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$i]}; PI=${PIS[$i]}; EFF=${EFFS[$i]}; NAME=${NAMES[$i]}
OUT=/usr/project/xtmp/nd141/permol_proto/$NAME
PIARG=""
if [ -n "$PI" ]; then PIARG="--pi $PI"; fi

echo "Host: $(hostname)  Task: $i  arm=$ARM pi=${PI:-none} eff=$EFF  out=$OUT"
echo "Start: $(date)"
/usr/bin/time -v python singlefiber/run_permol_window.py \
    --arm "$ARM" $PIARG --eff "$EFF" \
    --window chrI:60001-65000 --traindir robocop_train_fiberonly --out "$OUT"
echo "End: $(date)"
