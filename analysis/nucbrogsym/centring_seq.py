#!/usr/bin/env python
"""Sequence symmetry vs centre offset, Brogaard top-2000 (and Chereji as a reference).

offset o: centre = 1-based base (BED start + o). See nucmodel.py.
Writes OUT/centring_seq.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nucmodel as NM  # noqa: E402

OUT = "/usr/project/xtmp/nd141/permol_proto_logfull/nucbrog"

g = NM.load_genome()
res = {}
for name, path in (("brogaard", NM.BROGAARD), ("chereji", NM.CHEREJI)):
    rows = NM.read_bed(path)
    res[name] = dict(n_rows=len(rows), by_offset={})
    for o in range(-3, 4):
        seqs, dropped = NM.extract(rows, g, o)
        c1, d1, c2, d2 = NM.symmetry_chi2(seqs)
        res[name]["by_offset"][o] = dict(n=len(seqs), dropped=dropped,
                                          chi2_base=c1, df_base=d1, chi2_dinuc=c2, df_dinuc=d2)
        print("%-8s o=%+d n=%d  chi2_base=%.1f (df %d)  chi2_dinuc=%.1f (df %d)" % (
            name, o, len(seqs), c1, d1, c2, d2))
        sys.stdout.flush()
json.dump(res, open(os.path.join(OUT, "centring_seq.json"), "w"), indent=1)
