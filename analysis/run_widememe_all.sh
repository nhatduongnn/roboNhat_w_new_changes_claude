#!/bin/bash
# Build the meme-file-widened trainDir, then decode chrI and chrXIV against it.
#
# The whole widening is data: inputs/motifs_meme_wide.txt (7 left / 2 right estimated
# flank columns on Abf1_murphy) + inputs/all_TFs_1000pealVal_params_pseudo_wide.pkl (the
# matching 23 bp Fiber-seq vector) + ONE changed line in pkgvar/seq_maskoff_widememe
# naming that pkl. No RoboCOP logic was modified.
#
# The trainDir must exist before either decode starts -- the geometry lives in
# HMMconfig.pkl -- so both decode arrays are chained with --dependency=afterok. If the
# train job's gates fail, neither decode runs.
#
#   ./run_widememe_all.sh          submit everything
#   ./run_widememe_all.sh chrI     train + chrI only
#
# Job ids land in .widememe_jobids.

set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
mkdir -p logs

WHICH="${1:-all}"
DRIVER=run_split_widememe.py
TRAINDIR=robocop_train_widememe
: > .widememe_jobids

echo "== building the widened meme file =="
python make_meme_wide.py

echo
echo "== train =="
TRAIN_ID=$(VARIANT=seq_maskoff_widememe OUTDIR=$TRAINDIR \
           sbatch --parsable --job-name=wmTrain sbatch_train_widememe.sh)
echo "train            $TRAIN_ID"
echo "train $TRAIN_ID" >> .widememe_jobids

echo
echo "== decodes (afterok:$TRAIN_ID) =="
CHRI_ID=$(DRIVER=$DRIVER TRAINDIR=$TRAINDIR OUTDIR=robocop_chrI_widememe \
          sbatch --parsable --job-name=wmChrI \
                 --dependency=afterok:$TRAIN_ID sbatch_chrI_widememe.sh)
echo "chrI  (0-5)      $CHRI_ID"
echo "chrI $CHRI_ID" >> .widememe_jobids

if [ "$WHICH" = "all" ]; then
    CHRXIV_ID=$(DRIVER=$DRIVER TRAINDIR=$TRAINDIR OUTDIR=robocop_chrXIV_widememe \
                sbatch --parsable --job-name=wmChrXIV \
                       --dependency=afterok:$TRAIN_ID sbatch_chrXIV.sh)
    echo "chrXIV (0-11)    $CHRXIV_ID"
    echo "chrXIV $CHRXIV_ID" >> .widememe_jobids
fi

echo
echo "watch:  squeue -u \$USER"
echo "gates:  logs/wmTrain_${TRAIN_ID}.out"
