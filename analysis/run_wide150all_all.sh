#!/bin/bash
# Build and launch the +/-150-on-EVERYTHING widened run, `wide150all`.
#
# All 153 motifs padded by 150 columns each side -- the 12 individually-fitted TFs from
# their own Rossi flanks and their own +/-200bp Fiber profile, the other 141 from the
# pooled `combined_low_count` profile and the pooled Rossi flank stack.
# 153 x 300 = 45900 extra columns -> 91800 extra states -> n_states 95285.
#
# ============================ DO NOT SUBMIT AS-IS ============================
# BLOCKED: librobocop.so linearizes every 2-D/3-D index in 32-bit `int`:
#
#     int I(int row, int col, int ncol) { return row * ncol + col; }   (pkg/robocop/bc.c:14)
#
# The last element of an n_states x n_states matrix is at (n-1)*n + (n-1). That exceeds
# INT_MAX for any n_states >= 46341, so at n_states 95285 the true index 9,079,231,224
# comes back as 489,296,632 -- verified empirically against the shipped .so by
# sbatch_wide150all_memcheck.sh. construct_transition_matrix, find_parents_and_children,
# set_initial_probs and the forward/backward passes would all write outside their buffers:
# a segfault if you are lucky, silently wrong posteriors if you are not. The emission
# tensor index I3(6, n_obs-1, n_obs, n-1, n_states) = 3,334,974,999 overflows too.
#
# The safe ceiling with the shipped library is n_states <= 46340, i.e. a uniform pad of at
# most ~70 columns on all 153 motifs. wide150 (12 TFs, n_states 10685) is far inside it.
#
# Fixing this means widening the index type in pkg/robocop/bc.c + bc.h + algo.c to
# `long`/`size_t` and rebuilding a SEPARATE librobocop.so, then pointing this variant's
# `cshared` at it (config_fiberonly_wide150all.ini already carries that field, so no other
# variant is disturbed). That is a change to RoboCOP's numerical core and has to be
# reviewed and re-validated against an existing run before it is trusted -- it is not
# something this script should do on its own.
#
# MEMORY, once the index width is fixed, is also severe (measured model: wide10all at
# n_states 9605 peaked at 75 GB training / 20 GB per decode, dominated by the
# (7, n_obs, n_states) emission tensor held live for all 20 training windows):
#   t_mat + parents_mat + children_mat   3 x 72.6 GB  = 218 GB   (always resident)
#   emission tensor, per segment                        26.7 GB
#   training holds 20 segments live                    534 GB
#   -> train peak ~760 GB, decode peak ~256 GB, HMMconfig.pkl ~73 GB on disk
# Only the 1,150,000 MB compsci-cluster-fitz-* nodes can hold the training job at all.
# TRAIN_MEM/DECODE_MEM below are set accordingly.
#
# Set I_KNOW_INDEX_OVERFLOW=1 to submit anyway.
# =============================================================================
#
#   fibseq+wide150all  pkgvar/seq_maskoff_wide150all    robocop_{chrI,chrXIV}_wide150all
#   fib+wide150all     pkgvar/fiber_maskoff_wide150all  robocop_{chrI,chrXIV}_fib_wide150all
#
# Job ids land in .wide150all_jobids.

set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
mkdir -p logs

RUN=wide150all
PADS="tf_pads_${RUN}.tsv"
MEME="inputs/motifs_meme_${RUN}.txt"
PARAMS="inputs/all_TFs_1000pealVal_params_pseudo_${RUN}.pkl"
TRAINDIR="robocop_train_${RUN}"
N_STATES=95285
INDEX_CEILING=46340
TRAIN_MEM=900G
TRAIN_TIME=24:00:00
DECODE_MEM=400G
DECODE_TIME=24:00:00

if [ "$N_STATES" -gt "$INDEX_CEILING" ] && [ "${I_KNOW_INDEX_OVERFLOW:-0}" != "1" ]; then
    echo "REFUSING TO SUBMIT: n_states $N_STATES > $INDEX_CEILING, the largest state space"
    echo "librobocop.so can index with 32-bit int (pkg/robocop/bc.c:14). See the header of"
    echo "this script and run sbatch_wide150all_memcheck.sh for the empirical proof."
    echo "Everything else for this variant is built and ready:"
    ls -d "$PADS" "$MEME" "$PARAMS" "config_fiberonly_${RUN}.ini" \
          "pkgvar/seq_maskoff_${RUN}" "pkgvar/fiber_maskoff_${RUN}"
    exit 1
fi

: > .${RUN}_jobids
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
echo "$RUN train $TRAIN_ID" >> .${RUN}_jobids

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
        echo "$RUN ${LAYER}_${CHR} $JID" >> .${RUN}_jobids
    done
done

echo
echo "watch:  squeue -u \$USER"
echo "gate:   logs/${RUN}Train_*.out  (check_widememe_traindir.py must pass)"
