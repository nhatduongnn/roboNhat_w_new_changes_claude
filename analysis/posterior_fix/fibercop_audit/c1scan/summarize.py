"""Per-chromosome summary of the genome C1 scan (out_<model>/seg_*.tsv, events_*.tsv).

    cd analysis && python posterior_fix/fibercop_audit/c1scan/summarize.py bo09_48 bu01_11 bv75_49

Writes c1scan/summary_by_chrom.tsv. Positions are counted once: segments overlap by 1,000 bp, so
bad rows are reported per segment (as stored) and C1 events are de-duplicated by (chrom, pos).
"""
import sys, glob, os
import pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
rows = []
order = ["chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI", "chrVII", "chrVIII", "chrIX", "chrX", "chrXI",
         "chrXII", "chrXIII", "chrXIV", "chrXV", "chrXVI"]
for m in sys.argv[1:]:
    seg = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(H, "out_" + m, "seg_*.tsv"))])
    evf = [pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(H, "out_" + m, "events_*.tsv"))]
    ev = pd.concat(evf) if evf else pd.DataFrame(columns=["chrom", "pos", "dnats"])
    print(m, "segments", len(seg), "unique segments", seg.segment.nunique())
    for c in order:
        s = seg[seg.chrom == c]; e = ev[ev.chrom == c]
        rows.append(dict(model=m, chrom=c, segments=len(s), segs_with_c1=int((s.n_events > 0).sum()),
                         c1_positions=int(e.drop_duplicates(["chrom", "pos"]).shape[0]),
                         max_dnats=float(e.dnats.abs().max()) if len(e) else 0.0,
                         rows_bad_orig=int(s.bad_orig.sum()), rows_bad_c1_only=int(s.bad_c1.sum()),
                         rows_bad_fixed=int(s.bad_fixed.sum())))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(H, "summary_by_chrom.tsv"), sep="\t", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda v: "%.3g" % v))
