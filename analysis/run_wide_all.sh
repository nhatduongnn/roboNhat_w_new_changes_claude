#!/bin/bash
# Build both widened trainDirs and chain the chrI decodes onto them.
#
# Two configurations, identical geometry, differing ONLY in what the pad columns emit in
# the Fiber layers:
#
#   wideA      pkgvar/seq_maskoff_wide      fitted +/-50bp p-values in the pads
#   wideAnull  pkgvar/seq_maskoff_widenull  background rate in the pads   <- CONTROL
#
# Why the control is not optional: a longer state block is cheaper than the background it
# displaces (background costs background_prob = 0.9332 per bp; motif-internal transitions
# cost 1.0), so widening ABF1 by 9 bp hands it a ~1.9x implicit prior boost regardless of
# what the pads emit. Both runs carry that boost identically, so it cancels and
# `wideA > wideAnull` isolates the footprint's own contribution. `wideA > baseline` alone
# would not distinguish the two.
#
# Decodes are --dependency=afterok on their build, so a failed geometry gate never gets
# decoded and mistaken for a result.
#
# Usage: ./run_wide_all.sh [wideA|wideAnull|all]
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs

submit_one() {
  local label="$1" variant="$2" traindir="$3" outdir="$4"
  local tj dj
  tj=$(sbatch --parsable --job-name="trainwide_$label" \
        --export=ALL,VARIANT="$variant",OUTDIR="$traindir" sbatch_train_wide.sh)
  dj=$(sbatch --parsable --dependency=afterok:"$tj" --job-name="chrI_$label" \
        --export=ALL,VARIANT="$variant",TRAINDIR="$traindir",OUTDIR="$outdir" sbatch_chrI_wide.sh)
  printf '%-10s train=%-10s decode=%-10s  %s -> %s\n' "$label" "$tj" "$dj" "$traindir" "$outdir"
  echo "$label train $tj decode $dj" >> .wide_jobids
}

: > .wide_jobids
case "${1:-all}" in
  wideA)     submit_one wideA     seq_maskoff_wide     robocop_train_wideA     robocop_chrI_wideA ;;
  wideAnull) submit_one wideAnull seq_maskoff_widenull robocop_train_wideAnull robocop_chrI_wideAnull ;;
  all)
    submit_one wideA     seq_maskoff_wide     robocop_train_wideA     robocop_chrI_wideA
    submit_one wideAnull seq_maskoff_widenull robocop_train_wideAnull robocop_chrI_wideAnull ;;
  *) echo "usage: $0 [wideA|wideAnull|all]" >&2; exit 2 ;;
esac
echo "job ids in .wide_jobids"
