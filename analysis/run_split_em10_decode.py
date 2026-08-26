"""chrI decode using the EM-trained transition prior.

Identical to run_split_revfix_seq_maskoff.py -- same package variant, same emission model,
same reverse-strand fix -- and differs ONLY in the trainDir passed on the command line.
That is the point: the decode must use the unmodified pkgvar/seq_maskoff (NOT
pkgvar/seq_maskoff_em10, which only exists to turn training on), so the single variable
between this run and robocop_chrI_seq_maskoff_revfix is the fitted concentrations.
"""
import sys, os
sys.path.insert(0, 'pkgvar/seq_maskoff/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_em10_decode ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_em10_decode done (idx %d/%d) ===" % (idx, total))
