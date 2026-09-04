#!/bin/bash
# Fiber-only decodes of the three widened models.
#
# WHY. Every wide run so far is fib+seq. With the sequence layer on, ABF1's enrichment is
# 33x and the widening is competing against a strong PWM signal. With it OFF, the ONLY
# thing distinguishing one TF state from another is its Fiber-seq footprint -- which makes
# this the most direct test there is of whether widening improves the footprint, the
# emission side we established is actually the bottleneck (AUROC ~0.58 at every prior).
#
# NO NEW TRAINDIR. `fib` and `seq` already share robocop_train_fiberonly: the sequence
# layer is switched off entirely at decode time by one line in robocopExtras
# (data_emission_matrix[0][:] = 1, applied per segment). So the widened trainDirs built for
# the fib+seq runs are reused byte-for-byte, and these decodes differ from their fib+seq
# twins ONLY in that one masking line. Same geometry, same prior, same Fiber parameters.
#
# Baseline for comparison is the existing `fib` run (unwidened, sequence off), which scores
# enrichment 6.2 / F1 0.010 on chrXIV against seq's 33.3 / 0.046.
#
#   ./run_fibwide_all.sh              all three, both chromosomes
#   ./run_fibwide_all.sh wide10all    just the all-153 one
#
# Job ids land in .fibwide_jobids.

set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
mkdir -p logs
: > .fibwide_jobids

WHICH="${1:-all}"

for RUN in widememe wide10 wide10all; do
    [ "$WHICH" != "all" ] && [ "$WHICH" != "$RUN" ] && continue
    TRAINDIR="robocop_train_${RUN}"
    VARIANT="fiber_maskoff_${RUN}"
    [ -d "$TRAINDIR" ] || { echo "missing $TRAINDIR"; exit 1; }
    [ -d "pkgvar/$VARIANT" ] || { echo "missing pkgvar/$VARIANT"; exit 1; }

    echo "===== fib+$RUN  (trainDir reused: $TRAINDIR) ====="
    for CHR in chrI chrXIV; do
        JID=$(DRIVER=run_split_widememe.py VARIANT="$VARIANT" TRAINDIR="$TRAINDIR" \
              OUTDIR="robocop_${CHR}_fib_${RUN}" \
              sbatch --parsable --job-name="fw${RUN}${CHR}" "sbatch_${CHR}_wide10.sh")
        echo "  $CHR  $JID"
        echo "fib_$RUN $CHR $JID" >> .fibwide_jobids
    done
done
echo
echo "watch:  squeue -u \$USER"
