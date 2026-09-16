#!/bin/bash
#SBATCH --job-name=fimoGenome
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-15
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%A_%a.err

# Genome-wide FIMO scan of all 153 RoboCOP motifs.
#
# WHY. 69 of the 153 motifs have no MacIsaac target and no Rossi row, so the concentration
# loop has nothing to tune them against -- yet they carry 17% of all TF occupancy, and the
# single worst over-caller (Nhp6a, occ 1592 on chrXIV) is one of them. A motif scan gives
# every motif a number.
#
# WHAT IT IS AND IS NOT. FIMO counts motif MATCHES, not bound sites. At p<1e-4 over 12 Mb on
# two strands a motif collects ~2400 hits by chance, far above any real site count, so these
# numbers are NOT targets as they stand. They are calibrated against MacIsaac on the 84
# motifs that have both, and the fitted mapping is what supplies targets for the rest
# (fimo_targets.py). MacIsaac is itself a motif scan plus a cross-species conservation
# filter, so the two are the same kind of object -- which is what makes that calibration
# meaningful, and it leaves Rossi entirely free for validation.
#
# THRESHOLD. Scanned at a permissive 1e-3 with --text so any stricter cut can be applied
# afterwards without rescanning; --text also skips q-value computation, which is the slow
# part and which we do not use. Expect ~3.7M hits total (2 x 12e6 x 153 x 1e-3 / 153 per
# motif). chrM is scanned but dropped at counting time -- neither MacIsaac nor Rossi assays
# it.
#
# BACKGROUND. The order-0 background is taken from a decode's own pwm.p, not MEME's uniform
# default, so FIMO log-odds are on the same scale RoboCOP scores with. Same recipe as
# compare_fimo_macisaac_chrI.py:run_fimo.
#
# SPLIT. 153 motifs over 16 array tasks, ~10 motifs each, passed as repeated --motif flags.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024

set -eo pipefail

cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis

FIMO=/home/users/nd141/miniconda3/envs/meme/bin/fimo
MEMEFILE=inputs/motifs_meme.txt
GENOME=inputs/SacCer3.fa
OUT=fimo_genome
NTASK=16

mkdir -p $OUT

# order-0 background from the decode's pwm.p (built once; task 0 wins the race harmlessly)
BG=$OUT/robocop_bg.txt
if [ ! -s "$BG" ]; then
  python - <<'EOF'
import pickle, numpy as np, os
b = np.ravel(pickle.load(open("robocop_chrI_fib_seq/pwm.p", "rb"))["background"])[:4]
os.makedirs("fimo_genome", exist_ok=True)
with open("fimo_genome/robocop_bg.txt", "w") as fh:
    fh.write("# order 0\n")
    for letter, v in zip("ACGT", b):
        fh.write("%s %.8f\n" % (letter, v))
EOF
fi

# this task's slice of the motif list
MOTIFS=$(grep '^MOTIF' $MEMEFILE | awk '{print $2}' \
         | awk -v i=$SLURM_ARRAY_TASK_ID -v n=$NTASK 'NR % n == i')
if [ -z "$MOTIFS" ]; then echo "no motifs for task $SLURM_ARRAY_TASK_ID"; exit 0; fi

ARGS=""
for m in $MOTIFS; do ARGS="$ARGS --motif $m"; done
echo "Host: $(hostname)  Task: $SLURM_ARRAY_TASK_ID  motifs: $(echo $MOTIFS | wc -w)  Start: $(date)"

$FIMO --bfile "$BG" --thresh 1e-3 --text $ARGS "$MEMEFILE" "$GENOME" \
  > "$OUT/hits_${SLURM_ARRAY_TASK_ID}.tsv"

echo "rows: $(wc -l < $OUT/hits_${SLURM_ARRAY_TASK_ID}.tsv)"
echo "Task: $SLURM_ARRAY_TASK_ID  End: $(date)"
