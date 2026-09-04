"""Build the trainDir for the meme-file-widened model.

Why this step exists, when capA/capB/12tfs did not need one
-----------------------------------------------------------
Those variants only change parameter pkls that robocop.py reads at RUNTIME while building
the emission matrix, so they all reuse robocop_train_fiberonly. Widening does not: it
changes tf_lens, tf_starts, n_states, pwm_emission and transition_matrix, and all five are
frozen inside trainDir/HMMconfig.pkl (robocop_no_em.py:51). So the widened model needs its
own trainDir.

The build is cheap and is NOT a fit: robocop_em.py:116 hardcodes `iterations = 0`, so this
is a config build plus one forward-backward over the training windows.

How this differs from the earlier train_wide.py
-----------------------------------------------
train_wide.py drove pkgvar/seq_maskoff_wide, which carried ~124 lines of edits implementing
per-TF pads read from tf_pads.tsv at train time. This one drives pkgvar/seq_maskoff_widememe,
which differs from pkgvar/seq_maskoff by exactly ONE line (the Fiber params filename). All
the geometry comes from the widened MEME file named in config_fiberonly_widememe.ini --
`get_transition_matrix_info` already derives tf_lens from `pwm[tf].shape[1]`, so the shipped
code widens the model on its own.

EXPECT THE PRIOR TO MOVE. getDBFconc derives tf_prob from calculateKD(pwm), and the PWM is
no longer the shipped one, so ABF1's prior lands at ~0.105x baseline (0.537x from the
0.9332^9 length cancellation, 5.12x further down because calculateKD reads the estimated pad
columns as tighter binding). This is known and accepted; check_widememe_traindir.py records
it. Undo later without retraining via make_conc_trainDir.py --tf Abf1_murphy --lam 5.1185.

Usage
-----
    python train_widememe.py coord_train_chrII_20.tsv ./robocop_train_widememe/
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("coordFile")
ap.add_argument("outDir")
ap.add_argument("--variant", default="seq_maskoff_widememe")
ap.add_argument("--config", default="config_fiberonly_widememe.ini")
a = ap.parse_args()

sys.path.insert(0, 'pkgvar/%s/' % a.variant)
from run_robocop import run_robocop_with_em

print("=== train_widememe ===")
print("pkg variant:", os.path.abspath('pkgvar/%s/' % a.variant))
print("coordFile:  ", a.coordFile)
print("configFile: ", a.config)
print("outDir:     ", a.outDir)
sys.stdout.flush()

run_robocop_with_em(a.coordFile, a.config, a.outDir)
print("=== done ===")
