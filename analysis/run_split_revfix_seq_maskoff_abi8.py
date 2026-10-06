"""Decode driver, variant: seq_maskoff_abi8 -- pkgvar/seq_maskoff ported to the 8-arg C ABI.

2026-10-06 binary cutover (POSTERIOR_SCALING.md): librobocop.so now exports only
posterior_decoding_v2(..., int silent_states_begin, double *posterior_table), so the 7-arg
pkgvar/seq_maskoff can no longer decode. pkgvar/seq_maskoff_abi8 is a cp -a of it whose ONLY
change is that call site (robocop/robocop.py:973-978). Same layers, mask, pkls -- same model;
the posterior is now normalised per position in C (fix B). Default driver of tune_concentrations.py.

Original docstring follows.
chrI decode for the reverse-complement fix, variant: seq_maskoff.

Identical to run_fiberonly_split.py except it imports the package variant
pkgvar/seq_maskoff_abi8/, which has the layer/mask toggles BAKED IN rather than
hand-commented. That removes the flip-and-wait race: every task of every array
imports exactly the state it is supposed to, no matter when slurm starts it,
and all four configurations can run concurrently.
"""
import sys, os
sys.path.insert(0, 'pkgvar/seq_maskoff_abi8/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_revfix_seq_maskoff ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff_abi8/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_revfix_seq_maskoff done (idx %d/%d) ===" % (idx, total))
