"""Decode driver, variant: seq_maskoff_all153_logfull -- log-space re-decode of campaign bu01 (2026-10-23).

Copy of run_split_seq_maskoff_tune75_logfull.py pointing at pkgvar/seq_maskoff_all153_logfull/: the full log-space
decoder (librobocop_logfull.so: fward_log / bward_log / posterior_decoding_log). The tree is a copy
of pkgvar/seq_maskoff_all153/ with the same three edits as seq_maskoff_tune75_logfull (cshared -> logfull .so;
robocop.py log path; robocopExtras.py 1e-30 floor deleted and hard masks -> -inf). Model, mask and
emission pkls are those of the bu01 decode driver run_split_revfix_seq_maskoff_all153.py, unchanged.
Requires env ROBOCOP_LOG_EMISSION in {floored, exact} (exact = binom.logpmf, no floor).
"""
import sys, os
sys.path.insert(0, 'pkgvar/seq_maskoff_all153_logfull/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_seq_maskoff_all153_logfull ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff_all153_logfull/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
print("ROBOCOP_LOG_EMISSION:", os.environ.get("ROBOCOP_LOG_EMISSION"))
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_seq_maskoff_all153_logfull done (idx %d/%d) ===" % (idx, total))
