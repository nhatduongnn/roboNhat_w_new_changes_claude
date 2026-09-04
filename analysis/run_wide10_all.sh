#!/bin/bash
# Build and launch the two +/-10bp widened runs.
#
#   wide10      the 12 individually-fitted TFs widened; each uses its OWN flank
#               composition and its OWN +/-50bp Fiber profile.       n_states 3965
#   wide10all   all 153 motifs widened; the 12 as above, the other 141 share the
#               pooled low-count estimate on BOTH layers -- flanks stacked from the
#               942 sites of the 62 sub-threshold TFs, and a combined_low_count
#               +/-50bp slice cut to each TF's own length.            n_states 9605
#
# The >=50/<50 split is RoboCOP's own
# (abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:54,147), not a new invention.
#
# Everything is data: two meme files, two params pkls, and one changed line per pkgvar
# (the params filename). No RoboCOP logic is modified.
#
#   ./run_wide10_all.sh            both runs, both chromosomes
#   ./run_wide10_all.sh wide10     just run A
#
# Job ids land in .wide10_jobids.

set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
mkdir -p logs
: > .wide10_jobids

WHICH="${1:-both}"

for RUN in wide10 wide10all; do
    [ "$WHICH" != "both" ] && [ "$WHICH" != "$RUN" ] && continue
    PADS="tf_pads_${RUN}.tsv"
    MEME="inputs/motifs_meme_${RUN}.txt"
    PARAMS="inputs/all_TFs_1000pealVal_params_pseudo_${RUN}.pkl"
    TRAINDIR="robocop_train_${RUN}"

    echo "===== $RUN ====="
    for f in "$PADS" "$MEME" "$PARAMS"; do
        [ -f "$f" ] || { echo "missing $f -- build it first"; exit 1; }
    done

    TRAIN_ID=$(VARIANT="seq_maskoff_${RUN}" OUTDIR="$TRAINDIR" \
               CONFIG="config_fiberonly_${RUN}.ini" \
               PADS="$PADS" MEME="$MEME" PARAMS="$PARAMS" \
               sbatch --parsable --job-name="${RUN}Train" sbatch_train_wide10.sh)
    echo "  train         $TRAIN_ID"
    echo "$RUN train $TRAIN_ID" >> .wide10_jobids

    for CHR in chrI chrXIV; do
        SCRIPT="sbatch_${CHR}_wide10.sh"
        JID=$(DRIVER=run_split_widememe.py TRAINDIR="$TRAINDIR" \
              OUTDIR="robocop_${CHR}_${RUN}" VARIANT="seq_maskoff_${RUN}" \
              sbatch --parsable --job-name="${RUN}${CHR}" \
                     --dependency=afterok:$TRAIN_ID "$SCRIPT")
        echo "  $CHR decode  $JID"
        echo "$RUN $CHR $JID" >> .wide10_jobids
    done
done

echo
echo "watch:  squeue -u \$USER"
echo "gates:  logs/*Train_*.out"
