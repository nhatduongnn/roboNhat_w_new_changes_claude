#!/bin/bash
#SBATCH --job-name=score_conc
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/scoreconc_%x_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/scoreconc_%x_%j.err

# Whole-chrI score of ONE decode -> conc_sweep_scores/<label>.json
# Usage: sbatch --export=ALL,OUTDIR=robocop_chrI_seq_maskon_conc30 sbatch_score_conc.sh
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg
: "${OUTDIR:?set OUTDIR}"
echo "Host: $(hostname)  Start: $(date)  OUTDIR=$OUTDIR"
python score_conc_sweep.py --only "$OUTDIR"
echo "End: $(date)"
