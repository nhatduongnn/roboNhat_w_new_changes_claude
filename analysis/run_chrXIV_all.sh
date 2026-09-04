#!/bin/bash
# Submit all seven chrXIV decodes.
#
# chrI carries only 5 MacIsaac ABF1 sites, so TP/recall there move in 20% steps and
# only FP count and enrichment have been trustworthy. chrXIV carries 19 ABF1 + 11 REB1
# at the genome's highest ABF1 density (2.42/100 kb) with the same Fiber-seq depth
# (50.9x vs chrI's 49.4x), so nothing is confounded by coverage.
#
# The (driver, trainDir) pairing IS the experiment -- the driver picks the pkgvar, which
# fixes the emission model (sequence layer on/off, capA/capB Fiber-seq params); the
# trainDir picks the transition prior (unfitted calculateKD, or one of three EM fits).
# Note cap-off is a TRAINING-time toggle: em10nocap decodes under the plain seq_maskoff
# like every other sequence-layer run, and differs only in its trainDir.
set -euo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
mkdir -p logs

submit () {  # name driver traindir outdir
  local jid
  jid=$(sbatch --parsable --job-name="XIV_$1" \
        --export=ALL,DRIVER="$2",TRAINDIR="$3",OUTDIR="$4" sbatch_chrXIV.sh)
  printf '%-12s %-38s %-32s %s\n' "$1" "$3" "$4" "job $jid"
  echo "$jid" >> .chrXIV_jobids
}

: > .chrXIV_jobids
printf '%-12s %-38s %-32s %s\n' LABEL TRAINDIR OUTDIR JOB

# fiber layer only ---------------------------------------------------------------
submit fib       run_split_revfix_fiber_maskoff.py robocop_train_fiberonly          robocop_chrXIV_maskoff_fib
submit fibem10   run_split_fiber_em10_decode.py    robocop_train_em10_chrII_fiber   robocop_chrXIV_maskoff_em10
# fiber + sequence layer ---------------------------------------------------------
submit seq       run_split_revfix_seq_maskoff.py   robocop_train_fiberonly          robocop_chrXIV_seq_maskoff_revfix
submit seqem10   run_split_em10_decode.py          robocop_train_em10_chrII         robocop_chrXIV_seq_maskoff_em10
submit em10nocap run_split_em10_decode.py          robocop_train_em10_chrII_nocap   robocop_chrXIV_seq_maskoff_em10nocap
# fiber + sequence, alternative Fiber-seq background ------------------------------
submit capA      run_split_variant_capA.py         robocop_train_fiberonly          robocop_chrXIV_seq_maskoff_capA
submit capB      run_split_variant_capB.py         robocop_train_fiberonly          robocop_chrXIV_seq_maskoff_capB

echo
echo "84 tasks submitted. Job ids in .chrXIV_jobids"
