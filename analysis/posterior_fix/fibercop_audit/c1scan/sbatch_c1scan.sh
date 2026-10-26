#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-47
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.err
# Genome-wide exact C1 check for one model (scan_genome.py). Export MODEL, TREE, TRAINDIR.
# Stagger as in sbatch_genome_decode.sh (rpy2 package:utils transient at R startup).
sleep $(( RANDOM % 45 ))
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
: "${MODEL:?}" "${TREE:?}" "${TRAINDIR:?}"
C=posterior_fix/fibercop_audit/c1scan
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID/48  Model: $MODEL  Start: $(date)"
python $C/scan_genome.py $TREE $TRAINDIR coord_genome_full.tsv $C/work_$MODEL $C/out_$MODEL $SLURM_ARRAY_TASK_ID 48
echo "End: $(date)"
