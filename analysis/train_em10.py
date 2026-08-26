"""Train the RoboCOP transition prior (the per-DBF "concentration") with EM.

Upstream never fits it: `parameterize.getDBFconc` sets tf_prob from `calculateKD` -- the Kd
of each motif's consensus, a pure function of motif length and information content -- and
`robocop_em.py` hardcodes `iterations = 0`, so the Baum-Welch loop below it never runs.
The shipped prior therefore gives Nhp6a (7 bp, degenerate) 7.2239e-4 and Abf1 (14 bp,
informative) 1.7974e-7, a 4019x gap, while the model's own posterior implies only ~6.9x.

This driver imports pkgvar/seq_maskoff_em10/, which is pkgvar/seq_maskoff/ (the variant the
revfix decodes use, reverse-strand fix included) with EM turned on and a compact
per-iteration trace of the prior written to <outDir>/em_trace/iter{i}.npz.

Train off the decode chromosome so chrI stays held out.

Usage
-----
    python train_em10.py coord_train_chrII_20.tsv ./robocop_train_em10_chrII/
    ROBOCOP_EM_ITERS=2 python train_em10.py coords.tsv ./smoke/     # short smoke test
"""
import os
import sys

sys.path.insert(0, 'pkgvar/seq_maskoff_em10/')
from run_robocop import run_robocop_with_em

if len(sys.argv) not in (3, 4):
    sys.exit("usage: python train_em10.py <coordFile> <outDir> [configFile]")

coordFile, outDir = sys.argv[1], sys.argv[2]
configFile = sys.argv[3] if len(sys.argv) == 4 else "config_fiberonly.ini"

print("=== train_em10 ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff_em10/'))
print("coordFile:  ", coordFile)
print("configFile: ", configFile)
print("outDir:     ", outDir)
print("iterations: ", os.environ.get("ROBOCOP_EM_ITERS", "10 (default)"))
sys.stdout.flush()

run_robocop_with_em(coordFile, configFile, outDir)
print("=== train_em10 done ===")
