#!/bin/bash
# Log-space twin of sbatch_permol_window.sh. One task per (arm x efficiency), same table:
#   0 A effoff  1 A effon  2 B0.3 off  3 B0.3 on  4 B0.5 off  5 B0.5 on  6 B0.7 off  7 B0.7 on  8 C off  9 C on
#
#   ROBOCOP_LOG_EMISSION=exact   sbatch --export=ALL --array=0 singlefiber/sbatch_permol_window_logfull.sh
#   ROBOCOP_LOG_EMISSION=floored MAXMOL=20 sbatch --export=ALL --array=0 singlefiber/sbatch_permol_window_logfull.sh
#
#SBATCH --job-name=permol_logfull
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/usr/project/xtmp/nd141/permol_proto_logfull/logs/permol_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/permol_proto_logfull/logs/permol_%A_%a.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg
export R_HOME=$CONDA_PREFIX/lib/R
: "${ROBOCOP_LOG_EMISSION:?set ROBOCOP_LOG_EMISSION to floored or exact}"
export ROBOCOP_LOG_EMISSION
MAXMOL=${MAXMOL:-0}

ARMS=(A       A      B       B      B       B      B       B      C      C)
PIS=(""      ""     0.3     0.3    0.5     0.5    0.7     0.7    ""     "")
EFFS=(0       1      0       1      0       1      0       1      0      1)
NAMES=(armA_effoff armA_effon armB_pi0p3_effoff armB_pi0p3_effon \
       armB_pi0p5_effoff armB_pi0p5_effon armB_pi0p7_effoff armB_pi0p7_effon \
       armC_effoff armC_effon)

i=${SLURM_ARRAY_TASK_ID:-0}
ARM=${ARMS[$i]}; PI=${PIS[$i]}; EFF=${EFFS[$i]}; NAME=${NAMES[$i]}_${ROBOCOP_LOG_EMISSION}
if [ "$MAXMOL" != "0" ]; then NAME=${NAME}_first${MAXMOL}; fi
OUT=/usr/project/xtmp/nd141/permol_proto_logfull/$NAME
PIARG=""
if [ -n "$PI" ]; then PIARG="--pi $PI"; fi

echo "Host: $(hostname)  Task: $i  arm=$ARM pi=${PI:-none} eff=$EFF mode=$ROBOCOP_LOG_EMISSION maxmol=$MAXMOL out=$OUT"
echo "Start: $(date)"
/usr/bin/time -v python singlefiber/run_permol_window_logfull.py \
    --arm "$ARM" $PIARG --eff "$EFF" --max-molecules "$MAXMOL" \
    --window chrI:60001-65000 --traindir robocop_train_fiberonly --out "$OUT"
echo "End: $(date)"
