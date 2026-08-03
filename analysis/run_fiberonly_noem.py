"""
Fiber-seq-only RoboCOP inference (NO EM) + plot.

Uses run_robocop_without_em, which does inference against a pre-trained HMMconfig.pkl and
does NOT delete tmpDir -- so tmpDir/info.h5 (the decode) persists for plotting/analysis.

The ABF1-focus mask is controlled by env var ROBOCOP_ABF1_MASK ('1' = restrict Fiber
emission to background + ABF1 states; unset/0 = all TFs participate). See pkg/robocop/robocop.py.

Usage:
    python run_fiberonly_noem.py <coordFile> <trainDir> <outDir> <chrm> <start> <end>
"""
import sys
import os

sys.path.insert(0, '../pkg/')
from run_robocop import run_robocop_without_em, plot_robocop_output

coordFile = sys.argv[1]
trainDir  = sys.argv[2]
outDir    = sys.argv[3]
chrm      = sys.argv[4]
start     = int(sys.argv[5])
end       = int(sys.argv[6])

print("=== run_fiberonly_noem ===")
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("plot region:", chrm, start, end)
print("ROBOCOP_ABF1_MASK =", os.environ.get('ROBOCOP_ABF1_MASK'))
sys.stdout.flush()

run_robocop_without_em(coordFile, trainDir, outDir)

try:
    plot_robocop_output(outDir, chrm, start, end)
    print("Plot written under", outDir + "figures/")
except Exception as e:
    print("Plotting failed (decode output in tmpDir/info.h5 is still valid):", repr(e))

print("=== run_fiberonly_noem done ===")
