"""chrI fiber-only decode using the EM-trained transition prior.

Same shape as run_split_em10_decode.py, but imports pkgvar/fiber_maskoff/ (sequence layer
off) so it matches the emission model its trainDir was fitted under. As there, the decode
uses the UNMODIFIED variant -- pkgvar/fiber_maskoff, not _em10 -- so the only difference
against robocop_chrI_maskoff_revfix is the trainDir.
"""
import sys, os
sys.path.insert(0, 'pkgvar/fiber_maskoff/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_fiber_em10_decode ===")
print("pkg variant:", os.path.abspath('pkgvar/fiber_maskoff/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_fiber_em10_decode done (idx %d/%d) ===" % (idx, total))
