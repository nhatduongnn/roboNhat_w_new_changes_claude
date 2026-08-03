"""
Fiber-seq-only RoboCOP posterior decoding on a SUBSET of segments (no EM), for
memory-bounded parallelization across a slurm array.

Uses run_robocop_without_em(..., idx, total): the coord file's segments are interleaved
into `total` groups and only group `idx` (segments idx, idx+total, idx+2*total, ...) is
decoded, writing tmpDir/info_<idx>_<total>.h5. Running idx=0..total-1 in parallel decodes
all segments while each task holds only ~1/total of them in memory. score_robocop.py globs
tmpDir/info*.h5, so it reads every shard back automatically.

The ABF1-focus mask (pkg/robocop/robocop.py ~797) is a code toggle frozen at import; launch
all tasks of one array with the intended mask state and do not flip it until the array is done.

Usage:
    python run_fiberonly_split.py <coordFile> <trainDir> <outDir> <idx> <total>
"""
import sys
import os

sys.path.insert(0, '../pkg/')
from run_robocop import run_robocop_without_em

coordFile = sys.argv[1]
trainDir  = sys.argv[2]
outDir    = sys.argv[3]
idx       = int(sys.argv[4])
total     = int(sys.argv[5])

print("=== run_fiberonly_split ===")
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
sys.stdout.flush()

run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)

print("=== run_fiberonly_split done (idx %d/%d) ===" % (idx, total))
