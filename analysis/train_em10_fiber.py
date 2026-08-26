"""EM training with the SEQUENCE LAYER OFF -- the fiber-only control on train_em10.py.

Why: with the sequence layer on, EM raised Abf1's prior 3720x and ABF1 became the #1 TF
state on chrI. Two explanations compete. (H1) the prior was genuinely miscalibrated and
fixing it is what helped. (H2) the sequence layer already localises ABF1, the tiny prior
was suppressing it, and EM merely let existing signal express.

Fiber-only separates them. EM fits ONE SCALAR per factor and never touches emissions, so
with sequence off the only per-factor signal left is the fiber protection profile -- which
is an unlocalised "something is protected here" detector. If a trained prior still leaves
ABF1 near its untrained 1.08x enrichment, that is H2: the prior only helps where a
localising signal already exists, and the real lever is the emission model.

Imports pkgvar/fiber_maskoff_em10/, which differs from pkgvar/seq_maskoff_em10/ by exactly
one line (robocopExtras.py:101, the sequence-layer switch), so the EM code is provably the
same and the layer is the only variable.

Usage
-----
    python train_em10_fiber.py coord_train_chrII_20.tsv ./robocop_train_em10_chrII_fiber/
"""
import os
import sys

sys.path.insert(0, 'pkgvar/fiber_maskoff_em10/')
from run_robocop import run_robocop_with_em

if len(sys.argv) not in (3, 4):
    sys.exit("usage: python train_em10_fiber.py <coordFile> <outDir> [configFile]")

coordFile, outDir = sys.argv[1], sys.argv[2]
configFile = sys.argv[3] if len(sys.argv) == 4 else "config_fiberonly.ini"

print("=== train_em10_fiber (SEQUENCE LAYER OFF) ===")
print("pkg variant:", os.path.abspath('pkgvar/fiber_maskoff_em10/'))
print("coordFile:  ", coordFile)
print("outDir:     ", outDir)
print("iterations: ", os.environ.get("ROBOCOP_EM_ITERS", "10 (default)"))
print("cap:        ", os.environ.get("ROBOCOP_EM_CAP", "mean+2sd (default)"))
sys.stdout.flush()

run_robocop_with_em(coordFile, configFile, outDir)
print("=== train_em10_fiber done ===")
