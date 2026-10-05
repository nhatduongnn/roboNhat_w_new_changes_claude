"""Decode driver, variant: seq_maskoff_all153 (campaign bu01).

ALL 153 motifs + `unknown` are live. The tree's KEEP_ALL154 set keeps every name, so the mask
loop masks nothing and asserts so; it exists only to satisfy tune_w.driver_keep (tune_w.py:341-350),
which requires exactly one KEEP_* set and raises on an unmasked tree.

What actually differs from pkgvar/seq_maskoff -- two lines, both emission parameters:

  robocop.py:598  all_TFs_1000pealVal_params_pseudo.pkl -> ..._clc08.pkl
                  combined_low_count 0.24894/0.26468 -> 0.08/0.08.
                  `unknown` inherits this: it is a tf_prob key absent from the pkl, so it takes
                  the else branch at robocop.py:671-676. 142 of 154 TF states use it.
  robocop.py:606  bg_params.pkl -> bg_params_open.pkl
                  background 0.13827/0.13841 -> 0.24894/0.26468, the level combined_low_count
                  used to occupy (open promoter DNA rather than nucleosome-flanking linker).

Rationale: shipped, the fallback for 142 TF states sat 1.80x ABOVE background, so "a TF is bound
here" emitted MORE m6A than background. Here background means open DNA and the fallback means a
bound footprint.

NO RETRAIN NEEDED. HMMconfig.pkl holds only PWM-derived quantities -- no fiber `p` vector of any
kind. Both pkls are read at RUNTIME by robocop.py:598/602/606 while the emission matrix is built,
once per segment per strand, and that matrix is never persisted. Same argument as
run_split_variant_capA.py:14-19 and run_split_variant_bgtss.py.

Rule 7: this differs from bw01 in THREE things at once -- background level, combined_low_count /
unknown level, and the mask (92 motifs masked -> live). Attribution between them is not available
from this run alone.
"""
import sys, os
sys.path.insert(0, 'pkgvar/seq_maskoff_all153/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_revfix_seq_maskoff_all153 ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff_all153/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_revfix_seq_maskoff_all153 done (idx %d/%d) ===" % (idx, total))
