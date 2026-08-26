"""Stage-0 readout: did raising ABF1's concentration bring back MacIsaac site #3?

Windowed score only -- returns in seconds, no whole-chrI pass. Invokes
score_robocop.score(..., return_abf1_tracks=True) so the numbers and the calls come from
the scorer's single peak-calling code path (call_abf1), not a re-implementation.

`abf1_global_max=1.0` is the quick proxy for the call threshold: every chrI run's ABF1
posterior max is ~1.0, so abf1_call_threshold(1.0) = 0.30 is the run's real threshold to
within rounding. The definitive per-run threshold comes from the whole-chrI gmax in the
Stage-1 scoring.

Usage:
    python check_conc_site3.py                      # default: lam30 vs the lam=1 baselines
    python check_conc_site3.py robocop_chrI_seq_maskon_conc30 ...
"""
import sys
import numpy as np
import score_robocop as S

# MacIsaac ABF1 sites on chrI (centers), from inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed
SITES = {1: 45325, 2: 45505, 3: 61170, 4: 62664, 5: 108795}
PAD = 300

# lambda=1 baselines are the existing *_revfix runs: the lambda=1 patch is bit-identical to
# robocop_train_fiberonly, which is the trainDir those runs already used.
DEFAULT_RUNS = [
    ("seq_maskon  lam=1 ", "robocop_chrI_seq_maskon_revfix"),
    ("seq_maskon  lam=30", "robocop_chrI_seq_maskon_conc30"),
    ("seq_maskoff lam=1 ", "robocop_chrI_seq_maskoff_revfix"),
    ("seq_maskoff lam=30", "robocop_chrI_seq_maskoff_conc30"),
    ("seq_maskon  JASPAR", "robocop_chrI_seq_maskon_JASPAR"),
    ("seq_maskoff JASPAR", "robocop_chrI_seq_maskoff_JASPAR"),
]


def probe(outDir):
    """Return {site: (local_max_posterior, hit_bool, threshold)} for the 5 sites."""
    out = {}
    for n, c in SITES.items():
        r = S.score(outDir, regions=[("chrI", c - PAD, c + PAD)],
                    abf1_global_max=1.0, return_abf1_tracks=True)
        det = r["_per_region"][0]["abf1_detail"]
        track = np.array(det["track"])
        pos = np.array(det["pos"])
        m = (pos >= c - 25) & (pos <= c + 25)
        # a hit = one of the scorer's own calls lands within tol of the reference center
        hit = any(abs(call["center"] - c) <= 20 for call in det["calls"])
        out[n] = (float(track[m].max()), hit, det["threshold"])
    return out


def main():
    runs = ([(a, a) for a in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_RUNS)
    import os
    print("MacIsaac ABF1 sites on chrI. Value = max ABF1 posterior within +/-25 bp; "
          "* = scorer called it (within 20 bp).")
    print("Fiber-seq says: #3 protected (LR 4.3e+07 for ABF1); #2 and #5 ~30%% methylated "
          "(LR 2.6e-08 / 3.9e-03).")
    print("Note stored posteriors are floored: save_sparse_posterior zeroes anything < 1e-4.\n")
    hdr = "%-20s" % "run" + "".join("%14s" % ("site#%d" % n) for n in SITES) + "%10s" % "recall"
    print(hdr)
    print("-" * len(hdr))
    for label, outDir in runs:
        if not os.path.isdir(outDir):
            print("%-20s  (not on disk yet)" % label)
            continue
        try:
            res = probe(outDir)
        except Exception as e:
            print("%-20s  FAILED: %r" % (label, e))
            continue
        cells = "".join("%13.4f%s" % (res[n][0], "*" if res[n][1] else " ") for n in SITES)
        rec = sum(res[n][1] for n in SITES)
        print("%-20s%s%8d/5" % (label, cells, rec))


if __name__ == "__main__":
    main()
