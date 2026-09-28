#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=vwFT
#SBATCH --output=/usr/project/xtmp/nd141/scratch_viewer/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/scratch_viewer/logs/%x_%A_%a.err
# Factor-table BACKFILL for existing decodes (viewer site, Step 0b). One array task = LINES_PER_TASK
# rows of the task list written by `build_run_matrix.py --tasks FILE`; each row is
#     decode_dir <TAB> idx <TAB> total <TAB> driver <TAB> traindir <TAB> tree <TAB> seg,seg,...
# and runs write_factor_table.py on just those (window-overlapping) segments, with the tree and real
# trainDir the registry resolved (so renamed / widened / decoy decodes collapse with the right slices).
# Submit with a concurrency cap so live tuning chains are never starved, e.g.
#   sbatch --array=0-N%40 --export=ALL,TASKS=/usr/project/xtmp/nd141/scratch_viewer/backfill_tasks.tsv sbatch_factor_backfill.sh
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
: "${TASKS:?TASKS not exported}"
PER=${LINES_PER_TASK:-4}
first=$(( SLURM_ARRAY_TASK_ID * PER + 1 ))
last=$(( first + PER - 1 ))
rc=0
sed -n "${first},${last}p" "$TASKS" | while IFS=$'\t' read -r dir idx total driver traindir tree segs; do
    echo "== $dir info_${idx}_${total} tree=$tree traindir=$traindir segs=$segs  $(date +%T)"
    python write_factor_table.py --outdir "$dir" --driver "$driver" --traindir "$traindir" --tree "$tree" \
        --idx "$idx" --total "$total" --segments "$segs" || { echo "FAILED $dir $idx" >&2; rc=1; }
done
echo "done $(date)"
