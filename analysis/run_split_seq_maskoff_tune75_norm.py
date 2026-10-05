"""Decode driver, variant: seq_maskoff_tune75 (campaign bu01).

Only the 75 groups that HAVE a MacIsaac target on chrXIV+chrII are live -- exactly the groups
tune_w can tune (one motif per group; RAP1 -> Rap1_telomeric). The 78 untunable motifs and
`unknown` are hard-masked, because they are frozen at lambda=1 and in bu01 they took the calls:
on chrXIV at the final round, unknown 3109.7, Nhp6b 1578.8, Sig1 1059.7, Nhp6a 851.7, against
Abf1 28.4. User decision 2026-10-01: mask what cannot be tuned.

Differs from pkgvar/seq_maskoff_all153 in the keep-set ONLY -- same tree, same two pkls -- so
campaign bv75 is a ONE-VARIABLE comparison against bu01.

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

Rule 7: vs bu01 this differs ONLY in the mask (154 live -> 75 live). vs bw01 it still differs in
three things (background, combined_low_count/unknown, and which motifs are live).
"""
import sys, os
sys.path.insert(0, 'pkgvar/seq_maskoff_tune75_norm/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_seq_maskoff_tune75_norm ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff_tune75_norm/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_seq_maskoff_tune75_norm done (idx %d/%d) ===" % (idx, total))
