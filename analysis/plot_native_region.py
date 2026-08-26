"""RoboCOP's OWN decoding plot (plotRoboCOP.plot_output) for a region, per run.

Writes to <outDir>/figures/robocop_output_<chrm>_<start>_<end>.png -- the standard
location and the standard figure, so it is directly comparable to the ones already
in robocop_chrI_maskon_revfix/figures/.

Two caveats that are properties of these runs, not of this script:

  * ABF1-ONLY MASK. Every *_maskon_* run forbids states 29..nuc_start, i.e. every TF
    except ABF1 and the generic `unknown`. So "which TFs got assigned" can only ever
    answer ABF1. A run without the mask would be needed to see the other 152.

  * DISPLAY CUTOFF. visualization.plot_occupancy_profile is called with
    threshold=0.1 (plotRoboCOP.py:186); preprocess_occupancy_profile then drops any
    DBF column whose max over the window is < 0.1 AND zeroes every remaining value
    < 0.1. Anything fainter than 0.1 is invisible here by construction -- read the
    posterior directly (see plot_site5_decoded_fimo.py) for weak signal.

    python plot_native_region.py --start 190366 --end 190766
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from robocop.utils import plotRoboCOP

RUNS = [
    ("robocop_chrI_maskon_revfix", "Fiber-seq only"),
    ("robocop_chrI_seq_maskon_revfix", "Fiber-seq + sequence"),
    ("robocop_chrI_seqonly_maskon_revfix", "Sequence only"),
    ("robocop_chrI_seq_maskon_JASPAR", "Fiber-seq + sequence, JASPAR ABF1"),
    ("robocop_chrI_seqonly_maskon_JASPAR", "Sequence only, JASPAR ABF1"),
    ("robocop_chrI_seq_maskoff_JASPAR", "Fiber-seq + sequence, JASPAR ABF1, MASK OFF"),
    ("robocop_chrI_seq_maskoff_revfix", "BASELINE Murphy fiber+seq mask off"),
    ("robocop_chrI_seq_maskoff_bgtss", "V1 bg_tss"),
    ("robocop_chrI_seq_maskoff_lowabf1", "V2 low_abf1"),
    ("robocop_chrI_seq_maskoff_12tfs", "V3 12_tfs"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default="chrI")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="subset of output dirs; default = all three")
    args = ap.parse_args()

    runs = [r for r in RUNS if args.runs is None or r[0] in args.runs]
    for outDir, label in runs:
        if not os.path.isdir(outDir):
            print("SKIP %-38s (missing)" % outDir)
            continue
        print("\n=== %s  (%s) ===" % (outDir, label), flush=True)
        try:
            plotRoboCOP.plot_output(outDir, args.chrom, args.start, args.end, save=True)
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, e))
            continue
        finally:
            # whole-chrI decodes previously OOM'd from a matplotlib figure leak;
            # close explicitly rather than relying on plot_output to do it
            plt.close("all")
        out = "%s/figures/robocop_output_%s_%d_%d.png" % (
            outDir.rstrip("/"), args.chrom, args.start, args.end)
        print("  wrote %s  (%s)" % (out, "exists" if os.path.isfile(out) else "MISSING"))


if __name__ == "__main__":
    main()
