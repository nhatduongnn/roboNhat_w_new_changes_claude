"""Decode driver, variant: seq_maskoff_logfull_nucbrogsym_all7 (2026-10-30) -- copy of
run_split_seq_maskoff_logfull.py pointing at pkgvar/seq_maskoff_logfull_nucbrogsym_all7/, whose only change
is the m6A files (inputs/nucleosome_params_brogaard_sym_all7.pkl, inputs/bg_params_all7.pkl). Use with
trainDir robocop_train_fiberonly_nucbrogsym (passed as argv[2]; not hardcoded here).

Original docstring follows.
seq_maskoff_logfull -- log-space twin of run_split_revfix_seq_maskoff.py (2026-10-25).

Made for the single-fiber log-space port (singlefiber/LOGFULL_PORT.md): the aggregate (pileup) reference
decode for the pooled-emission check. pkgvar/seq_maskoff_logfull/ is a copy of pkgvar/seq_maskoff/ (the
parent of pkgvar/permol_seq_maskoff) with the same three edits as seq_maskoff_tune75_logfull (cshared ->
librobocop_logfull.so; robocop.py log path; robocopExtras.py 1e-30 floor deleted). seq_maskoff has no hard
mask, so there is no mask line to convert. Model, emission pkls and layers are those of seq_maskoff, unchanged.
Requires env ROBOCOP_LOG_EMISSION in {floored, exact} (exact = binom.logpmf, no floor).
"""
import sys, os
sys.path.insert(0, 'pkgvar/seq_maskoff_logfull_nucbrogsym_all7/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_seq_maskoff_logfull_nucbrogsym_all7 ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff_logfull_nucbrogsym_all7/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
print("ROBOCOP_LOG_EMISSION:", os.environ.get("ROBOCOP_LOG_EMISSION"))
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_seq_maskoff_logfull_nucbrogsym_all7 done (idx %d/%d) ===" % (idx, total))
