"""Decode with WIDENED TF footprint states -- Fiber-seq + sequence layers, MASK OFF.

The TF state block is longer than its motif: pad columns emit the fitted +/-50bp Fiber-seq
p-values in layers 5/6 and pwm['background'] in layer 0. Because the background state emits
that same vector, the pads cancel in the sequence likelihood ratio -- the motif is untouched
and only the Fiber layers see the wider window. Pads come from tf_pads.tsv, frozen into the
trainDir by train_wide.py.

Unlike capA/capB/12tfs this REQUIRES its own trainDir (the geometry lives in HMMconfig.pkl),
so pair the variant with the matching trainDir:

    seq_maskoff_wide      <-> robocop_train_wideA        fitted pad columns
    seq_maskoff_widenull  <-> robocop_train_wideAnull    pad columns = background rate

widenull is the CONTROL. A longer block gets an implicit prior boost of
background_prob^-(pads) (0.9332^-9 = 1.9x for ABF1) because covering a base with a longer TF
block is free while covering it with background is not. Both runs carry an identical boost,
so it cancels and `wide > widenull` isolates the footprint's own contribution.

MASK OFF: the ABF1-only mask would forbid the competing TF states outright and hide whether
the wider footprint actually wins competitions it was previously losing.

Usage: python run_split_variant_wide.py <coordFile> <trainDir> <outDir> <idx> <total> [variant]
"""
import sys, os

variant = sys.argv[6] if len(sys.argv) > 6 else os.environ.get("WIDE_VARIANT", "seq_maskoff_wide")
sys.path.insert(0, 'pkgvar/%s/' % variant)
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_variant_wide ===")
print("pkg variant:", os.path.abspath('pkgvar/%s/' % variant))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== done (idx %d/%d) ===" % (idx, total))
