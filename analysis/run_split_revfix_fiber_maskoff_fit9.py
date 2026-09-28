"""chrI decode for the reverse-complement fix, variant: fiber_maskoff_fit9 (sequence layer OFF; all motifs except the 9 fitted MacIsaac TFs, and unknown, hard-masked).

Identical to run_fiberonly_split.py except it imports the package variant
pkgvar/fiber_maskoff_fit9/, which has the layer/mask toggles BAKED IN rather than
hand-commented. That removes the flip-and-wait race: every task of every array
imports exactly the state it is supposed to, no matter when slurm starts it,
and all four configurations can run concurrently.
"""
import sys, os
sys.path.insert(0, 'pkgvar/fiber_maskoff_fit9/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_revfix_fiber_maskoff_fit9 ===")
print("pkg variant:", os.path.abspath('pkgvar/fiber_maskoff_fit9/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_revfix_fiber_maskoff_fit9 done (idx %d/%d) ===" % (idx, total))
