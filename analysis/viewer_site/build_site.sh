#!/bin/bash
# Build / refresh the viewer site at /usr/project/xtmp/nd141/viewer_site (not in git).
#   bash analysis/viewer_site/build_site.sh            # registry, backfill (sbatch, waits), extract, check, web
#   NO_BACKFILL=1 bash analysis/viewer_site/build_site.sh   # skip the factor-table backfill step
# Steps (plan okay-so-i-want-humming-hearth.md):
#   1  build_run_matrix.py      -> runs.json / runs.tsv, RUN_INFO.json backfill, backfill task list
#   0b sbatch_factor_backfill   -> factor_tables/part_*.npz for window segments still missing (%40, 1 CPU)
#   3  extract_shared.py        -> shared/<chrom>.json.gz, colors.json, scratch fiber_ref/
#   2  extract_run.py --all     -> runs/<run>/<chrom>.bin.gz (incremental)
#   4  fiber_check.py           -> fiber_check.json
#   5  copy web/ + build_info.json
set -eo pipefail
A=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
SITE=/usr/project/xtmp/nd141/viewer_site
SCR=/usr/project/xtmp/nd141/scratch_viewer
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
export MPLBACKEND=Agg R_HOME=$CONDA_PREFIX/lib/R
cd "$A"
mkdir -p "$SITE" "$SCR/logs"
python viewer_site/build_run_matrix.py --out "$SITE" --backfill-run-info --tasks "$SCR/backfill_tasks.tsv"
N=$(wc -l < "$SCR/backfill_tasks.tsv")
if [ "$N" -gt 0 ] && [ -z "$NO_BACKFILL" ]; then
    # FREEZE the list: array tasks read it by line number, so a later rebuild of
    # backfill_tasks.tsv must never change what a running array sees.
    T="$SCR/backfill_tasks_$(date +%Y%m%d_%H%M%S).tsv"; cp "$SCR/backfill_tasks.tsv" "$T"; chmod a-w "$T"
    PER=4; NT=$(( (N + PER - 1) / PER ))
    echo "backfill: $N rows -> $NT array tasks (%40, 1 CPU each) from $T; waiting"
    sbatch --wait --array=0-$((NT - 1))%40 --export=ALL,TASKS="$T",LINES_PER_TASK=$PER \
        viewer_site/sbatch_factor_backfill.sh || echo "some backfill tasks failed; see $SCR/logs" >&2
fi
[ -s "$SITE/colors.json" ] && [ -z "$FORCE_SHARED" ] || python viewer_site/extract_shared.py --out "$SITE"
python viewer_site/extract_run.py --all --out "$SITE"
python viewer_site/fiber_check.py --out "$SITE"
cp viewer_site/web/index.html viewer_site/web/app.js viewer_site/web/style.css "$SITE/"
python - "$SITE" <<'PY'
import json, sys, time, subprocess, pandas as pd
site = sys.argv[1]
A = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis"
w = pd.read_csv(A + "/viewer_site/windows.tsv", sep="\t")
sizes = dict((l.split()[0], int(l.split()[1])) for l in open(A + "/inputs/sacCer3.chrom.sizes"))
reg = json.load(open(site + "/runs.json"))
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=A, capture_output=True, text=True).stdout.strip()
json.dump(dict(built=time.strftime("%Y-%m-%d %H:%M:%S"), git=head, n_runs=len(reg["runs"]),
               windows=w.to_dict("records"), chrom_sizes=sizes), open(site + "/build_info.json", "w"), indent=1)
print("build_info.json written")
PY
echo "site ready: $SITE  (serve: bash $A/viewer_site/serve.sh)"
