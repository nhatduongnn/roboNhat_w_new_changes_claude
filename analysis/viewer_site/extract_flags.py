#!/usr/bin/env python
"""Flagged positions (reads disagree with the reference) per chromosome, for the Run Browser.

    python viewer_site/extract_flags.py [chrom ...] [--out DIR] [--flags-dir DIR]

Writes <out>/shared/<chrom>.flags.json.gz = flagged_positions.flags_payload() over the whole
chromosome (columnar: p k r a ac dc ic is c; k = 0 alt base, 1 indel, 2 low coverage), plus
"kinds" (legend labels) and "source" (the TSV read). Rule and scan: analysis/flagged_positions.py.
A chromosome without a flags TSV gets no file; app.js then draws no flag strip.
"""
import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.dirname(HERE)
sys.path.insert(0, A)
import flagged_positions as FP  # noqa: E402

OUT_DEFAULT = "/usr/project/xtmp/nd141/viewer_site"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chroms", nargs="*")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--flags-dir", default=FP.FLAGS_DIR)
    a = ap.parse_args()
    sizes = dict((l.split()[0], int(l.split()[1])) for l in open(os.path.join(A, "inputs", "sacCer3.chrom.sizes")))
    os.makedirs(os.path.join(a.out, "shared"), exist_ok=True)
    for chrom in a.chroms or sorted(sizes):
        df = FP.load_flags(chrom, 1, sizes[chrom], a.flags_dir)
        if df is None:
            print("%-8s no flags file, skipped" % chrom)
            continue
        pl = FP.flags_payload(df)
        pl["kinds"] = [lab for _, lab in FP.KINDS]
        pl["source"] = FP.flags_path(chrom, a.flags_dir)
        dst = os.path.join(a.out, "shared", "%s.flags.json.gz" % chrom)
        with gzip.open(dst + ".part", "wt", encoding="utf-8") as fh:
            json.dump(pl, fh, separators=(",", ":"))
        os.replace(dst + ".part", dst)
        print("%-8s %s -> %s" % (chrom, FP.counts(pl), dst))


if __name__ == "__main__":
    main()
