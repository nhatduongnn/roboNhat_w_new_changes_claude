"""Build a trainDir whose HMM geometry carries the WIDENED TF footprints.

Why this step exists, when capA/capB/12tfs did not need one
-----------------------------------------------------------
Every previous variant only changed parameter pkls that robocop.py reads at RUNTIME while
building the emission matrix, so they all reuse robocop_train_fiberonly. Widening does not:
it changes tf_lens, tf_starts, n_states, pwm_emission and transition_matrix, and all five
are frozen inside trainDir/HMMconfig.pkl (robocop_no_em.py:51). So the widened model needs
its own trainDir.

The build is cheap and is NOT a fit: robocop_em.py:116 hardcodes `iterations = 0`, so this
is a config build plus one forward-backward over the training windows. And because
parameterize.getDBFconc derives tf_prob from calculateKD(pwm) -- and the PWMs are untouched
-- the resulting prior is identical to the baseline's, keeping the comparison one-variable.

tf_pads.tsv is read HERE, once, and frozen into HMMconfig.pkl as dshared['tf_pads'].
Decodes inherit the pads from the trainDir and never re-read the tsv, so editing it later
cannot desynchronise a decode from the geometry it depends on.

Usage
-----
    python train_wide.py coord_train_chrII_20.tsv ./robocop_train_wideA/
    python train_wide.py coord_train_chrII_20.tsv ./robocop_train_wideAnull/ \
           --variant seq_maskoff_widenull
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("coordFile")
ap.add_argument("outDir")
ap.add_argument("--variant", default="seq_maskoff_wide")
ap.add_argument("--config", default="config_fiberonly.ini")
a = ap.parse_args()

sys.path.insert(0, 'pkgvar/%s/' % a.variant)
from run_robocop import run_robocop_with_em

print("=== train_wide ===")
print("pkg variant:", os.path.abspath('pkgvar/%s/' % a.variant))
print("coordFile:  ", a.coordFile)
print("configFile: ", a.config)
print("outDir:     ", a.outDir)
print("tf_pads:    ", os.environ.get("ROBOCOP_TF_PADS", "tf_pads.tsv"))
sys.stdout.flush()

run_robocop_with_em(a.coordFile, a.config, a.outDir)
print("=== done ===")
