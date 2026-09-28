#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-47
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.err

# One parameterized WHOLE-GENOME decode array, same (DRIVER, TRAINDIR, OUTDIR) contract as
# sbatch_chrXIV.sh -- which model runs is decided entirely by the exported variables.
#
# coord_genome_full.tsv is 3,021 windows over the 16 nuclear chromosomes (chrM excluded:
# neither MacIsaac nor Rossi assays it, so a decode there could never be scored). At 48-way
# that is ~63 windows/task. chrI ran ~10 windows/task in 7-9 min and chrXIV ~16 in 11-12,
# i.e. ~0.75 min/window, so expect ~45-50 min/task. Memory is flat at ~10.5 GB regardless of
# region size -- the cost scales with windows, not chromosome length -- so the 48G request is
# the same generous headroom every other baseline decode uses.
#
# WHY 48 TASKS. The split is stride-interleaved (run_robocop.py:93-97,
# `indices = range(idx, len(coords), total)`), so any task count divides the work evenly and
# no merge step is needed -- each task writes its own tmpDir/info_<idx>_<total>.h5 and the
# scorer globs them all. 48 keeps per-task wall-clock under an hour while staying polite on
# a shared partition.
#
# Launch via tune_concentrations.py, which owns the (TRAINDIR, OUTDIR) mapping per iteration.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

: "${DRIVER:?DRIVER not exported}"
: "${TRAINDIR:?TRAINDIR not exported}"
: "${OUTDIR:?OUTDIR not exported}"

NTASK=${NTASK:-48}
COORDS=${COORDS:-coord_genome_full.tsv}   # subset sweeps pass coord_sweep.tsv

echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID/$NTASK  Start: $(date)"
echo "DRIVER=$DRIVER  TRAINDIR=$TRAINDIR  OUTDIR=$OUTDIR  COORDS=$COORDS"
# One retry. Every array task runs `cp trainDir/config.ini outDir/config.ini` and then parses
# it (run_robocop.py:60), and cp truncates before writing, so a task that reads while another
# is copying sees an empty file -> NoSectionError: 'main' within seconds of starting. That
# killed u001 round 3 task 32 (2026-09-11). The retry re-copies the same file, so it is safe;
# a genuine failure fails twice and still fails the job.
if ! python "$DRIVER" "$COORDS" "$TRAINDIR" "./$OUTDIR/" "$SLURM_ARRAY_TASK_ID" "$NTASK"; then
    echo "Task $SLURM_ARRAY_TASK_ID failed; retrying once in 60 s: $(date)" >&2
    sleep 60
    python "$DRIVER" "$COORDS" "$TRAINDIR" "./$OUTDIR/" "$SLURM_ARRAY_TASK_ID" "$NTASK"
fi
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
