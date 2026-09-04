#!/bin/bash
# Build and launch the +/-150 widened run, `wide150`.
#
# Same 12 individually-fitted TFs as `wide10`/`widefp`, but padded by 150 columns on each
# side instead of 10. MOTIVATION: at +/-200 bp the Fiber-seq accessibility profile around a
# motif is hyper-methylated (hyper-accessible) far past the motif edge, and a 20 bp pad
# cannot see it. 12 TFs x 300 extra columns = 3600 -> 7200 extra states -> n_states 10685
# (baseline 3485, wide10 3965, widefp 3765, wide10all 9605).
#
# Nothing in RoboCOP changed: get_transition_matrix_info derives tf_lens from
# pwm[tf].shape[1], so the widened MEME file propagates on its own. The Fiber p-vector is
# sliced from the 401-column pm200 refit (make_params_wide.py --src ... --half 200) rather
# than the 101-column pm50 one, because +/-150 does not fit in +/-50.
#
# One trainDir, four decodes; the sequence layer is switched on/off at DECODE time by one
# line in robocopExtras, so fib+seq and fib share robocop_train_wide150 byte-for-byte and
# differ only in which pkgvar the decode imports.
#
#   fibseq+wide150   pkgvar/seq_maskoff_wide150     robocop_{chrI,chrXIV}_wide150
#   fib+wide150      pkgvar/fiber_maskoff_wide150   robocop_{chrI,chrXIV}_fib_wide150
#
# MEMORY. wide10all (n_states 9605) peaked at 75 GB in training and 20 GB per decode task;
# the dominant term is the (7, n_obs, n_states) emission tensor held live for each of the
# 20 training windows, which is linear in n_states. wide150 is 1.11x that, so the shipped
# 128G/96G would very likely do -- but the sbatch scripts are shared with other variants
# and must not be edited, so the headroom is added on the sbatch command line instead.
#
# Job ids land in .wide150_jobids.

set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
mkdir -p logs
: > .wide150_jobids

RUN=wide150
PADS="tf_pads_${RUN}.tsv"
MEME="inputs/motifs_meme_${RUN}.txt"
PARAMS="inputs/all_TFs_1000pealVal_params_pseudo_${RUN}.pkl"
TRAINDIR="robocop_train_${RUN}"
TRAIN_MEM=200G
TRAIN_TIME=8:00:00
DECODE_MEM=128G
DECODE_TIME=10:00:00

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
           sbatch --parsable --job-name="${RUN}Train" \
                  --mem=$TRAIN_MEM --time=$TRAIN_TIME sbatch_train_wide10.sh)
echo "train              $TRAIN_ID"
echo "$RUN train $TRAIN_ID" >> .wide150_jobids

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
                     --mem=$DECODE_MEM --time=$DECODE_TIME \
                     --dependency=afterok:$TRAIN_ID "sbatch_${CHR}_wide10.sh")
        printf '%-18s %s  -> %s\n' "$LAYER $CHR" "$JID" "$OUT"
        echo "$RUN ${LAYER}_${CHR} $JID" >> .wide150_jobids
    done
done

echo
echo "watch:  squeue -u \$USER"
echo "gate:   logs/${RUN}Train_*.out  (check_widememe_traindir.py must pass)"
