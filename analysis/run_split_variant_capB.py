"""chrI decode, variant capB -- all TFs, Fiber-seq + sequence layers, MASK OFF.

Both variants cap combined_low_count at the merged-pileup background
(inputs/all_TFs_1000pealVal_params_pseudo_caplow.pkl, 0.2489/0.2647 -> 0.1548/0.1527), so
the 141 TFs without an individual Fiber-seq fit stop out-competing background on
methylation LEVEL alone. They differ only in the background rate:

  capA  bg = 0.154806 / 0.152661   inputs/bg_params_merged.pkl
        the honest re-fit: identical linker definition to the shipped bg, but on the
        MERGED 7-barcode pileup the TF params were fit on, instead of 03202025_barcode01
        alone (shipped 0.138268 / 0.138406).
  capB  bg = 0.440177 / 0.442727   inputs/bg_params_tss_top10.pkl
        the drastic option: mean of the top 10% of each ORF-oriented TSS-200..-100 window.

NO RETRAIN NEEDED. HMMconfig.pkl holds only PWM-derived quantities; the Fiber-seq
parameter pkls are read at RUNTIME by robocop.py:598/606 while the emission matrix is
built, and that matrix is never persisted. Both variants therefore decode against
robocop_train_fiberonly and share the baseline's exact HMMconfig, so the comparison
against robocop_chrI_seq_maskoff_revfix is strictly one-variable.

MASK OFF is required: the change only affects the ~141 fallback TFs, which an ABF1-only
mask would forbid outright, hiding the effect entirely.
"""
import sys, os
sys.path.insert(0, 'pkgvar/seq_maskoff_capB/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_variant_capB ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff_capB/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== done (idx %d/%d) ===" % (idx, total))
