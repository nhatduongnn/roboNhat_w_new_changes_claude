"""Decode with the meme-file-widened model -- Fiber-seq + sequence layers, MASK OFF.

ABF1's state block is 23 bp (7 + 14 + 2) because inputs/motifs_meme_wide.txt gives it 23
columns, not because any RoboCOP code was changed. The 7 left and 2 right columns carry the
real flanking base composition estimated from its 341 Rossi sites, and the Fiber-seq layers
read the matching 23 bp slice of the +/-50bp refit
(inputs/all_TFs_1000pealVal_params_pseudo_wide.pkl) -- the one line
pkgvar/seq_maskoff_widememe changes relative to pkgvar/seq_maskoff.

The geometry lives in HMMconfig.pkl, so this REQUIRES its own trainDir:

    seq_maskoff_widememe  <->  robocop_train_widememe

MASK OFF: the ABF1-only mask would forbid the competing TF states outright and hide whether
the wider footprint actually wins competitions it was previously losing.

TWO THINGS TO REMEMBER WHEN READING THE OUTPUT:
  * ABF1's prior is ~0.105x baseline (see train_widememe.py), so detection is suppressed for
    reasons unrelated to the footprint.
  * The shipped sum_for_dbf_probs sums the WHOLE 23 bp block, so ABF1's posterior is a 23 bp
    plateau, not a 14 bp peak. That inflates the genome-wide mean ~1.6x and deflates
    enrichment by the same factor. This is deliberate -- the widened plateau is visible in
    the posterior viewer -- but score_robocop.py numbers need that correction before being
    compared to baseline or wideA.

Usage: python run_split_widememe.py <coordFile> <trainDir> <outDir> <idx> <total> [variant]
"""
import os
import sys

variant = sys.argv[6] if len(sys.argv) > 6 else os.environ.get(
    "WIDEMEME_VARIANT", "seq_maskoff_widememe")
sys.path.insert(0, 'pkgvar/%s/' % variant)
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_widememe ===")
print("pkg variant:", os.path.abspath('pkgvar/%s/' % variant))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== done (idx %d/%d) ===" % (idx, total))
