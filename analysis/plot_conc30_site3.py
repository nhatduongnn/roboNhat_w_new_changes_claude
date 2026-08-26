"""Native RoboCOP decode plots at MacIsaac ABF1 site #3 (chrI:61163-61177, mid 61170).

Uses the shipped plotting entry point -- run_robocop.plot_robocop_output ->
plotRoboCOP.plot_output -- the same one regen_plots.py drives. No custom plotting.

Renders the lambda=1 baseline and the lambda=30 run at the same locus so the ABF1 posterior
can be compared directly. Every position in [start, end] must be covered by a decoded
segment or plot_output sys.exit()s; 61170 lies in segment 60001-65000.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pkg")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_robocop import plot_robocop_output

SITE3 = 61170
RUNS = [
    ("robocop_chrI_seq_maskon_revfix",  "lam1"),
    ("robocop_chrI_seq_maskon_conc30",  "lam30"),
    ("robocop_chrI_seq_maskoff_revfix", "maskoff_lam1"),
    ("robocop_chrI_seq_maskoff_conc30", "maskoff_lam30"),
]
# (half-width, tag). Tight = read the posterior at the motif; wide = ERV46 context window
# regen_plots.py uses, so these are comparable to the existing figures.
WINDOWS = [(275, "tight"), (2000, "wide")]

for outDir, tag in RUNS:
    if not os.path.isdir(outDir):
        print("SKIP (missing):", outDir)
        continue
    for half, wtag in WINDOWS:
        lo, hi = SITE3 - half, SITE3 + half
        print("=== %s  chrI:%d-%d (%s) ===" % (outDir, lo, hi, wtag))
        try:
            plot_robocop_output(outDir, "chrI", lo, hi)
        except SystemExit:
            print("  plot_output exited: coordinates not fully covered")
            continue
        except Exception as e:
            print("  FAILED:", repr(e))
            continue
        finally:
            plt.close("all")   # HANDOFF: matplotlib figure leak OOM'd whole-chrI plot loops
        src = os.path.join(outDir, "RoboCOP_chrI_%d_%d.png" % (lo, hi))
        dst = "site3_decode_%s_%s.png" % (tag, wtag)
        if os.path.isfile(src):
            import shutil
            shutil.copy2(src, dst)
            print("  ->", dst)
        else:
            cands = [f for f in os.listdir(outDir) if f.endswith(".png")]
            print("  produced:", cands[-3:])
print("DONE")
