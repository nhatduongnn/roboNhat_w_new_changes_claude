"""Tile a whole chromosome into overlapping decode windows.

Why this exists
---------------
`coord_chrI_full.tsv` -- the 59 x 5 kb tiling every chrI decode runs on -- has no
generator in the repo; it was produced by hand. Extending the analysis to another
chromosome meant reproducing an undocumented convention, so this pins it down:

    5000 bp windows on a 4000 bp stride (1000 bp overlap), 1-based inclusive
    coordinates, final window clipped to the chromosome end.

Run against chrI it reproduces `coord_chrI_full.tsv` exactly (see --verify), which is
the guarantee that a chrXIV or chrIV decode is tiled the same way the chrI results were.

The overlap is not decoration. `region_optable` in score_robocop.py stitches per-window
posteriors back together, and a window's first and last few hundred bases are the least
constrained -- the HMM has no context beyond the segment edge. The 1 kb overlap means
every interior base is covered by two windows.

Usage
-----
    python make_chrom_coords.py --chrom chrXIV            # -> coord_chrXIV_full.tsv
    python make_chrom_coords.py --chrom chrIV --out foo.tsv
    python make_chrom_coords.py --chrom chrI --verify coord_chrI_full.tsv
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHROM_SIZES = os.path.join(HERE, "inputs", "sacCer3.chrom.sizes")


def chrom_length(chrom, path=CHROM_SIZES):
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            name, size = line.split()[:2]
            if name == chrom:
                return int(size)
    raise KeyError("%s not in %s" % (chrom, path))


def windows(length, window=5000, stride=4000):
    """1-based inclusive (start, end) pairs tiling [1, length].

    The last window is clipped to `length` rather than padded, and is dropped if it
    would be wholly contained in its predecessor -- otherwise a chromosome whose
    length lands just past a stride boundary gets a duplicate tail.
    """
    out = []
    start = 1
    while start <= length:
        end = min(start + window - 1, length)
        if out and end <= out[-1][1]:
            break
        out.append((start, end))
        if end == length:
            break
        start += stride
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--window", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=4000)
    ap.add_argument("--chrom-sizes", default=CHROM_SIZES)
    ap.add_argument("--out", default=None, help="default coord_<chrom>_full.tsv")
    ap.add_argument("--verify", default=None,
                    help="compare against an existing coord file instead of writing")
    args = ap.parse_args()

    length = chrom_length(args.chrom, args.chrom_sizes)
    wins = windows(length, args.window, args.stride)
    lines = ["chr\tstart\tend"] + ["%s\t%d\t%d" % (args.chrom, s, e) for s, e in wins]
    text = "\n".join(lines) + "\n"

    if args.verify:
        have = open(args.verify).read()
        if have == text:
            print("MATCH: %s reproduces %s exactly (%d windows)"
                  % (args.chrom, args.verify, len(wins)))
            return 0
        print("MISMATCH against %s" % args.verify)
        hl, tl = have.splitlines(), text.splitlines()
        print("  existing %d lines, generated %d lines" % (len(hl), len(tl)))
        for i, (a, b) in enumerate(zip(hl, tl)):
            if a != b:
                print("  first diff at line %d: %r vs %r" % (i + 1, a, b))
                break
        return 1

    out = args.out or "coord_%s_full.tsv" % args.chrom
    with open(out, "w") as fh:
        fh.write(text)
    print("%s: %d bp -> %d windows of %d bp (stride %d) -> %s"
          % (args.chrom, length, len(wins), args.window, args.stride, out))
    print("  first: %s  last: %s" % (lines[1], lines[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
