#!/bin/bash
#SBATCH --job-name=train_jaspar
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/train_jaspar_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/train_jaspar_%j.err

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
# parameterize.getParamsMNase calls fitdistrplus via rpy2; the package lives in the
# robocop-2024 env's R library, but R_HOME otherwise resolves to the system R whose
# library is empty -> PackageNotInstalledError.
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

# Builds robocop_train_jaspar/HMMconfig.pkl + pwm.p from the JASPAR ABF1 matrix.
# robocop_em.py:116 has iterations = 0, so this is a config BUILD, not an EM fit.
# Nothing about this run touches pkg/ -- the layer/mask toggles only matter at decode.
echo "Host: $(hostname)  Start: $(date)"
python run_fiberonly.py coord_train.tsv config_jaspar.ini ./robocop_train_jaspar/
echo "End: $(date)"
