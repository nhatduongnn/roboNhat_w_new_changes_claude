"""EM training with the FIBER LAYERS OFF -- the sequence-only mirror of train_em10_fiber.py.

Why this exists
---------------
train_em10_fiber.py asked what EM does when the sequence layer is removed. This asks the
complementary question: what does the model do on SEQUENCE ALONE, with no chromatin data
at all? That is the floor every Fiber-seq run has to beat. If fib+seq scores no better
than seq by itself, the Fiber-seq layers are contributing nothing to TF identification and
the whole premise needs revisiting.

It also fixes an asymmetry in the existing run matrix. `fibem10` decodes against
robocop_train_em10_chrII_fiber -- an EM fit performed with the sequence layer off, matching
the layers its decode uses. There is no equivalent for sequence-only, so borrowing
robocop_train_em10_chrII (fitted with BOTH layers on) would fit the prior under one
emission model and decode under another. This trains the matching one.

Imports pkgvar/seqonly_maskoff_em10/, which differs from pkgvar/seq_maskoff_em10/ by
exactly the two lines that neutralise the Fiber channels (robocopExtras.py, layers 5 and 6
set to 1.0, applied after the 1e-30 floor so the floor cannot re-populate them). The EM
code is therefore provably identical and the layer set is the only variable.

Usage
-----
    python train_em10_seqonly.py coord_train_chrII_20.tsv ./robocop_train_em10_chrII_seqonly/
"""
import os
import sys

sys.path.insert(0, 'pkgvar/seqonly_maskoff_em10/')
from run_robocop import run_robocop_with_em

coordFile, outDir = sys.argv[1], sys.argv[2]
print("=== train_em10_seqonly ===")
print("pkg variant:", os.path.abspath('pkgvar/seqonly_maskoff_em10/'))
print("coordFile:  ", coordFile)
print("outDir:     ", outDir)
print("EM iterations:", os.environ.get("ROBOCOP_EM_ITERS", "10 (default)"))
sys.stdout.flush()

run_robocop_with_em(coordFile, "config_fiberonly.ini", outDir)
print("=== done ===")
