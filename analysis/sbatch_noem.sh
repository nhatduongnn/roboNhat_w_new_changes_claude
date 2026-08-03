#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
# job-name and output/error are set per-submission via sbatch CLI flags.
#
# Args (passed after the script name to sbatch):
#   $1 = ROBOCOP_ABF1_MASK value (1 = mask on, 0 = mask off)
#   $2 = coord file
#   $3 = output dir
# Fixed: trainDir = robocop_train_fiberonly, plot region = chrI 60500 64500

# Activate conda BEFORE strict mode (its scripts reference unbound vars).
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export ROBOCOP_ABF1_MASK=$1

echo "Host: $(hostname)  Start: $(date)"
echo "MASK=$1  COORDS=$2  OUTDIR=$3"
python run_fiberonly_noem.py "$2" robocop_train_fiberonly "$3" chrI 60500 64500
echo "End: $(date)"
