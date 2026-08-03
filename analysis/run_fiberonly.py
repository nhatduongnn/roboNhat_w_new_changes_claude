"""
Fiber-seq-only RoboCOP driver (no EM).

Runs run_robocop_with_em with iterations=0 (set in robocop_em.py), i.e. inference-only,
using only the Fiber-seq emission layer. The ABF1-focus mask is controlled by the env var
ROBOCOP_ABF1_MASK (set to '1' to restrict emission to background + ABF1 states; unset/0 to
let all TFs participate). See pkg/robocop/robocop.py.

Usage:
    python run_fiberonly.py <coordFile> <configFile> <outDir>
"""
import sys
import os

sys.path.insert(0, '../pkg/')
from run_robocop import run_robocop_with_em, plot_robocop_output

coordFile = sys.argv[1]
configFile = sys.argv[2]
outDir = sys.argv[3]

print("=== run_fiberonly ===")
print("coordFile :", coordFile)
print("configFile:", configFile)
print("outDir    :", outDir)
print("ROBOCOP_ABF1_MASK =", os.environ.get('ROBOCOP_ABF1_MASK'))
sys.stdout.flush()

run_robocop_with_em(coordFile, configFile, outDir)

# Plot a representative region that is present in the coords (chrI 60001-65000 segment).
try:
    plot_robocop_output(outDir, "chrI", 60500, 64500)
except Exception as e:
    print("Plotting failed (decode output in tmpDir/info.h5 is still valid):", repr(e))

print("=== run_fiberonly done ===")
