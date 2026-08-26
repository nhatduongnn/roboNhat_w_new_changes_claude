"""chrI decode, variant: seqonly_maskon -- SEQUENCE LAYER ONLY, no Fiber-seq.

The third leg of the comparison. Existing runs:
    robocop_chrI_maskon_revfix          Fiber only          (layer 0 neutralised)
    robocop_chrI_seq_maskon_revfix      Fiber + sequence    (all three live)
    robocop_chrI_seqonly_maskon_revfix  sequence only       <- this one

Purpose: isolate what the sequence layer alone decides, so it can be compared
directly against the FIMO scan of the same Abf1_murphy PWM. If the sequence layer
is doing what FIMO does, the two should agree; where they disagree, the difference
is the HMM's transition prior and inter-state competition rather than the motif.

Layer/mask state is BAKED IN to pkgvar/seqonly_maskon/robocop/utils/robocopExtras.py,
so nothing in pkg/ is touched and this can run concurrently with anything else.
Note that variant puts the ABF1-only mask on LAYER 0, because layers 5/6 are set to
1.0 here and a mask on them would be silently discarded.
"""
import sys, os
sys.path.insert(0, 'pkgvar/seqonly_maskon/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_revfix_seqonly_maskon ===")
print("pkg variant:", os.path.abspath('pkgvar/seqonly_maskon/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_revfix_seqonly_maskon done (idx %d/%d) ===" % (idx, total))
