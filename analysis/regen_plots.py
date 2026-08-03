"""Regenerate the ERV46 (chrI 60500-64500) decode plots for every decodable
folder, so all plots use the new deterministic per-name color map."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pkg")))
from run_robocop import plot_robocop_output

CHRM, START, END = "chrI", 60500, 64500
FOLDERS = [
    "robocop_all_abf1", "robocop_all_fiber",
    "robocop_seqlayer_maskon", "robocop_seqlayer_maskoff",
    "robocop_chrI_maskon", "robocop_chrI_maskoff",
    "robocop_erv46_maskon", "robocop_erv46_maskoff",
]
for f in FOLDERS:
    d = os.path.join(os.path.dirname(__file__), f)
    if not os.path.isdir(d):
        print("SKIP (missing):", f); continue
    print("=== regenerating:", f, "===")
    try:
        plot_robocop_output(d, CHRM, START, END)
    except Exception as e:
        print("  FAILED:", f, "->", repr(e))
print("DONE")
