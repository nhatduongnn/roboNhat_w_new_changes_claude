"""Decode driver, variant: seq_maskoff_fit9_bgopen.

Identical to run_split_revfix_seq_maskoff_fit9.py in every respect except the background
Fiber-seq parameter file. The pkgvar tree pkgvar/seq_maskoff_fit9_bgopen/ differs from
pkgvar/seq_maskoff_fit9/ in ONE line, robocop/robocop.py:606:

    inputs/bg_params.pkl        0.13826787 W / 0.13840569 C   (Chereji linker band,
                                dyad+/-73..88, barcode01-only pileup)
 -> inputs/bg_params_open.pkl   0.24893964 W / 0.26467511 C   (= combined_low_count,
                                bit-exact; see make_bg_open.py)

WHY: the shipped background is nucleosome-flanking LINKER, not open DNA, so it sits between
the protected and the accessible level. Moving it to the combined_low_count level makes the
background state mean "unbound, accessible DNA", which is what the decode needs it to mean.

MASK: inherited unchanged from the parent -- KEEP_FIT9 keeps the 9 fitted motifs that have a
MacIsaac target (Abf1, Cin5, Fhl1, Fkh1, Mcm1, Rap1_telomeric, Reb1, Sko1, Ume6) and masks
everything else, including `unknown` and the three fitted-but-untargetable TFs (Nhp6a, Spt15,
Tbf1). Applied after the 1e-30 floor in robocopExtras.py, so masked states stay EXACTLY 0.

NO RETRAIN NEEDED. HMMconfig.pkl holds only PWM-derived quantities -- its key list contains no
fiber `p` vector of any kind. The Fiber-seq parameter pkls are read at RUNTIME by
robocop.py:598/602/606 while the emission matrix is built, once per segment per strand, and
that matrix is never persisted. Background is state 0, the only state below nuc_start that no
TF block covers, and it takes its p from the default fill at robocop.py:631
(`ps[:] = bg_params['p'][strand]['A']`), so swapping this pkl changes exactly the background
state and nothing else. Same argument as run_split_variant_capA.py:14-19 and
run_split_variant_bgtss.py.

Campaign: bo09. Comparator: bf09 (identical but for this one pkl).
"""
import sys, os
sys.path.insert(0, 'pkgvar/seq_maskoff_fit9_bgopen/')
from run_robocop import run_robocop_without_em

coordFile, trainDir, outDir = sys.argv[1], sys.argv[2], sys.argv[3]
idx, total = int(sys.argv[4]), int(sys.argv[5])
print("=== run_split_revfix_seq_maskoff_fit9_bgopen ===")
print("pkg variant:", os.path.abspath('pkgvar/seq_maskoff_fit9_bgopen/'))
print("coordFile:", coordFile, "| trainDir:", trainDir, "| outDir:", outDir)
print("idx:", idx, "total:", total)
sys.stdout.flush()
run_robocop_without_em(coordFile, trainDir, outDir, idx=idx, total=total)
print("=== run_split_revfix_seq_maskoff_fit9_bgopen done (idx %d/%d) ===" % (idx, total))
