"""chrI decode, 12tfs variant -- Fiber-seq + sequence, MASK OFF.

VARIANT 3 12_tfs: only the 12 TFs with fitted footprints occupiable; the other 141 plus 'unknown' hard-masked

NO RETRAIN IS NEEDED for any of these variants. HMMconfig.pkl holds only PWM-derived
quantities (pwm_emission, tf_prob, transition_matrix); the Fiber-seq parameter pkls are
read at RUNTIME by robocop.py:598/606 inside the emission build, and that emission matrix
is never persisted. So every variant decodes against robocop_train_fiberonly and shares the
baseline's exact HMMconfig -- the comparison against robocop_chrI_seq_maskoff_revfix is
strictly one-variable, with no nucleosome-model drift.

MASK OFF is required: variants 1 and 2 only alter parameters used by the ~142 fallback TFs,
which an ABF1-only mask would forbid outright, hiding the effect entirely.
"""
import sys, os
sys.path.insert(0, 'pkgvar/seq_maskoff_12tfs/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_variant_12tfs ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff_12tfs/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== done (idx %d/%d) ===" % (idx, total))
