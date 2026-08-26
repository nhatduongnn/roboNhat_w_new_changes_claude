"""Sample training windows on a held-out chromosome, filtered by Fiber-seq depth.

Why this exists
---------------
`coord_train.tsv` (the RoboCOP paper's training set) is 5 x 5 kb on chrI -- the same
chromosome every decode here is scored on, and two of its windows sit directly on ERV46.
For an EM run whose whole point is to see how the fitted concentrations change the chrI
decode, training on chrI makes the result partly circular.

This samples windows on a different chromosome instead, rejecting any window whose
Fiber-seq A-coverage is too thin to constrain the emission model. Depth is read straight
off the modkit pileup named in the config, using the same column convention as
`robocop/utils/getReads.py:188 getValuesFiber_seqOneFileNucleotide`:

    [0] chrom  [1] start(0-based)  [3] mod base  [5] strand  [9] "trials pct count ..."

so field 9's first space-separated subfield is `valid_coverage`, i.e. the A-trials at that
position on that strand. Watson and Crick are summed, matching the "depth" track in
`make_posterior_viewer.py`.

The scan is done with awk rather than pandas: the pileup is ~823 MB and we only need three
numbers per row, so streaming it keeps this to a few seconds and a few MB.

Usage
-----
    python make_train_coords.py                       # chrII, 20 x 5 kb, seed 0
    python make_train_coords.py --chrom chrIII --n 12 --out coord_train_chrIII_12.tsv
    python make_train_coords.py --report-only         # just print the depth landscape
"""
import argparse
import configparser
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def chrom_length(sizes_file, chrom):
    for line in open(sizes_file):
        f = line.split()
        if f and f[0] == chrom:
            return int(f[1])
    sys.exit("chromosome %s not found in %s" % (chrom, sizes_file))


def a_trials(pileup, chrom, length, nucleotide="a"):
    """Per-position A-trials on `chrom`, Watson + Crick summed. 0-based index."""
    awk = (
        'BEGIN{FS="\\t"} $1=="%s" && tolower($4)=="%s" '
        '{split($10, a, " "); print $2"\\t"a[1]}' % (chrom, nucleotide)
    )
    p = subprocess.Popen(["awk", awk, pileup], stdout=subprocess.PIPE, text=True)
    depth = np.zeros(length, dtype=np.int32)
    n_rows = 0
    for line in p.stdout:
        pos, tr = line.split("\t")
        i = int(pos)
        if 0 <= i < length:
            depth[i] += int(tr)
            n_rows += 1
    p.wait()
    if p.returncode != 0:
        sys.exit("awk failed on %s" % pileup)
    return depth, n_rows


def window_stats(depth, start0, width):
    """Median A-trials over the covered A positions in a window, and how many there are."""
    w = depth[start0:start0 + width]
    nz = w[w > 0]
    if nz.size == 0:
        return 0.0, 0
    return float(np.median(nz)), int(nz.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default="chrII", help="chromosome to sample from")
    ap.add_argument("--n", type=int, default=20, help="number of windows")
    ap.add_argument("--width", type=int, default=5000, help="window width in bp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-depth", type=float, default=20.0,
                    help="reject a window whose median A-trials is below this")
    ap.add_argument("--min-a-frac", type=float, default=0.15,
                    help="reject a window where fewer than this fraction of bp carry any A-trial")
    ap.add_argument("--edge", type=int, default=10000,
                    help="bp to exclude at each chromosome end (sub-telomeric)")
    ap.add_argument("--config", default=os.path.join(_HERE, "config_fiberonly.ini"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="print the depth landscape and exit without writing")
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    pileup = cfg.get("main", "pileupFile")
    sizes = os.path.join(_HERE, cfg.get("main", "chrSizesFile"))
    nucleotide = cfg.get("main", "nucleotide")

    out = args.out or os.path.join(
        _HERE, "coord_train_%s_%d.tsv" % (args.chrom, args.n))

    length = chrom_length(sizes, args.chrom)
    print("%s: %s bp" % (args.chrom, f"{length:,}"))
    print("scanning %s ..." % os.path.basename(pileup))
    depth, n_rows = a_trials(pileup, args.chrom, length, nucleotide.lower())
    covered = depth > 0
    print("  %s pileup rows, %s positions with A-trials (%.1f%% of the chromosome)"
          % (f"{n_rows:,}", f"{int(covered.sum()):,}", 100 * covered.mean()))
    print("  A-trials over covered positions: median %.0f  p10 %.0f  p90 %.0f"
          % (np.median(depth[covered]), np.percentile(depth[covered], 10),
             np.percentile(depth[covered], 90)))

    # candidate windows on a non-overlapping grid, away from the chromosome ends
    grid = []
    s = args.edge
    while s + args.width <= length - args.edge:
        med, na = window_stats(depth, s, args.width)
        grid.append((s, med, na / float(args.width)))
        s += args.width
    meds = np.array([g[1] for g in grid])
    print("  %d candidate %d bp windows; median-depth across them: "
          "min %.0f  median %.0f  max %.0f"
          % (len(grid), args.width, meds.min(), np.median(meds), meds.max()))

    ok = [g for g in grid if g[1] >= args.min_depth and g[2] >= args.min_a_frac]
    print("  %d pass (median depth >= %.0f and A-fraction >= %.2f), %d rejected"
          % (len(ok), args.min_depth, args.min_a_frac, len(grid) - len(ok)))
    if args.report_only:
        return
    if len(ok) < args.n:
        sys.exit("only %d windows pass the depth filter; need %d. Lower --min-depth "
                 "or pick a chromosome with deeper coverage." % (len(ok), args.n))

    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(ok), size=args.n, replace=False).tolist())
    chosen = [ok[i] for i in pick]

    with open(out, "w") as f:
        f.write("chr\tstart\tend\n")
        for s0, med, afrac in chosen:
            # coords are 1-based inclusive, matching coord_train.tsv and the
            # minStart-1 convention in getValuesFiber_seqOneFileNucleotide
            f.write("%s\t%d\t%d\n" % (args.chrom, s0 + 1, s0 + args.width))
    print("\nwrote %s (%d windows, %s bp total)"
          % (out, len(chosen), f"{len(chosen) * args.width:,}"))

    side = {
        "chrom": args.chrom, "n": args.n, "width": args.width, "seed": args.seed,
        "min_depth": args.min_depth, "min_a_frac": args.min_a_frac, "edge": args.edge,
        "pileup": pileup, "candidates": len(grid), "passed": len(ok),
        "windows": [{"start": s0 + 1, "end": s0 + args.width,
                     "median_A_trials": med, "A_fraction": round(afrac, 4)}
                    for s0, med, afrac in chosen],
    }
    with open(out.replace(".tsv", ".json"), "w") as f:
        json.dump(side, f, indent=2)
    print("wrote %s" % out.replace(".tsv", ".json"))

    print("\n  window                     median A-trials   A-fraction")
    for s0, med, afrac in chosen:
        print("  %s:%s-%s %14.0f %12.2f"
              % (args.chrom, f"{s0 + 1:,}", f"{s0 + args.width:,}", med, afrac))


if __name__ == "__main__":
    main()
