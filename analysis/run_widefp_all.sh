#!/bin/bash
# Build and launch the per-TF-footprint widened run, `widefp`.
#
# Same 12 individually-fitted TFs as `wide10`, but the pads are the USER-SPECIFIED
# per-TF footprint widths (tf_pads_widefp.tsv) instead of a uniform +/-10. 140 extra
# motif columns -> 280 extra states -> n_states 3765 (wide10 is 3965, baseline 3485).
#
# One trainDir, four decodes: the sequence layer is switched on/off at DECODE time by one
# line in robocopExtras, so fib+seq and fib share robocop_train_widefp byte-for-byte and
# differ only in which pkgvar the decode imports.
#
#   fibseq+widefp   pkgvar/seq_maskoff_widefp     robocop_{chrI,chrXIV}_widefp
#   fib+widefp      pkgvar/fiber_maskoff_widefp   robocop_{chrI,chrXIV}_fib_widefp
#
# Job ids land in .widefp_jobids.

set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
mkdir -p logs
: > .widefp_jobids

RUN=widefp
PADS="tf_pads_${RUN}.tsv"
MEME="inputs/motifs_meme_${RUN}.txt"
PARAMS="inputs/all_TFs_1000pealVal_params_pseudo_${RUN}.pkl"
TRAINDIR="robocop_train_${RUN}"

for f in "$PADS" "$MEME" "$PARAMS" "config_fiberonly_${RUN}.ini" \
         "pkgvar/seq_maskoff_${RUN}" "pkgvar/fiber_maskoff_${RUN}"; do
    [ -e "$f" ] || { echo "missing $f -- build it first"; exit 1; }
done
# Never silently overwrite a finished run.
for d in "$TRAINDIR" robocop_chrI_${RUN} robocop_chrXIV_${RUN} \
         robocop_chrI_fib_${RUN} robocop_chrXIV_fib_${RUN}; do
    [ -e "$d" ] && { echo "ERROR: $d already exists; move it aside first."; exit 1; }
done

TRAIN_ID=$(VARIANT="seq_maskoff_${RUN}" OUTDIR="$TRAINDIR" \
           CONFIG="config_fiberonly_${RUN}.ini" \
           PADS="$PADS" MEME="$MEME" PARAMS="$PARAMS" \
           sbatch --parsable --job-name="${RUN}Train" sbatch_train_wide10.sh)
echo "train              $TRAIN_ID"
echo "$RUN train $TRAIN_ID" >> .widefp_jobids

for CHR in chrI chrXIV; do
    for LAYER in seq fib; do
        if [ "$LAYER" = seq ]; then
            VAR="seq_maskoff_${RUN}";   OUT="robocop_${CHR}_${RUN}"
        else
            VAR="fiber_maskoff_${RUN}"; OUT="robocop_${CHR}_fib_${RUN}"
        fi
        JID=$(DRIVER=run_split_widememe.py TRAINDIR="$TRAINDIR" \
              OUTDIR="$OUT" VARIANT="$VAR" \
              sbatch --parsable --job-name="${RUN}${LAYER}${CHR}" \
                     --dependency=afterok:$TRAIN_ID "sbatch_${CHR}_wide10.sh")
        printf '%-18s %s  -> %s\n' "$LAYER $CHR" "$JID" "$OUT"
        echo "$RUN ${LAYER}_${CHR} $JID" >> .widefp_jobids
    done
done

echo
echo "watch:  squeue -u \$USER"
echo "gate:   logs/${RUN}Train_*.out  (check_widememe_traindir.py must pass)"
