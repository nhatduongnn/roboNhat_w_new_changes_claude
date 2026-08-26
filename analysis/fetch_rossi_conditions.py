#!/usr/bin/env python
"""Build inputs/rossi_sample_conditions.tsv -- the growth condition of every
Rossi ChIP-exo sample, which the downloaded data does NOT carry.

The per-sample bundles under rossi_strand/<id>/<id>_YEP/ are named for the
Yeast Epigenome Project, not for a growth medium; their README lists only file
formats and no metadata.  So the condition has to come from the deposit:

  GEO GSE147927 series matrix -> !Sample_title           (encodes <id>_<TF>_<rep>)
                             -> !Sample_characteristics  (treatment: N min heat shock)
  CEGRcode/2021-Rossi_Nature/02_References_and_Features_Files/sample-key.tab
                             -> sample id <-> GSM <-> SRX <-> replicate

Rossi et al. grew cells in YPD to OD600 0.8 at 25 C, then heat-shocked a subset
by adding an equal volume of 55 C YPD to reach 37 C for 3 or 6 minutes.  Of the
1251 deposited samples, 22 are heat shock; 21 of those are present on this disk.

Run once; the output is small and belongs in version control.
"""
import os
import re
import gzip
import json
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "inputs", "rossi_sample_conditions.tsv")
ROSSI_DIR = "/usr/xtmp/nd141/projects/data/rossi_strand"

GEO = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147927/matrix/"
       "GSE147927_series_matrix.txt.gz")
KEY = ("https://raw.githubusercontent.com/CEGRcode/2021-Rossi_Nature/master/"
       "02_References_and_Features_Files/sample-key.tab")


def fetch(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as fh:
        data = fh.read()
    return data if binary else data.decode("utf-8", "replace")


def main():
    raw = gzip.decompress(fetch(GEO, binary=True)).decode("utf-8", "replace")
    rec = {}
    for line in raw.split("\n"):
        if line.startswith("!Sample_"):
            p = line.rstrip().split("\t")
            rec.setdefault(p[0], []).append([v.strip('"') for v in p[1:]])

    titles = rec["!Sample_title"][0]
    gsms = rec["!Sample_geo_accession"][0]
    # the characteristics come as several parallel blocks; find the one that
    # carries 'treatment:' rather than assuming its index
    treat = None
    for block in rec.get("!Sample_characteristics_ch1", []):
        if any(v.startswith("treatment:") for v in block):
            treat = block
            break
    if treat is None:
        raise SystemExit("no treatment block found in the series matrix")

    rows = []
    for t, g, tr in zip(titles, gsms, treat):
        m = re.match(r"(\d+)_(.+?)_(rep\d+)_(\S+)$", t)
        if not m:
            continue                      # NoTag / BY4741 controls
        cond = tr.replace("treatment:", "").strip() or "normal"
        rows.append(dict(sample_id=m.group(1), target=m.group(2),
                         replicate=m.group(3), assay=m.group(4),
                         gsm=g, condition=cond))

    # sanity-check the ids against the lab's own key
    key_ids = set()
    try:
        for line in fetch(KEY).split("\n")[1:]:
            f = line.split("\t")
            if len(f) >= 5 and f[4].strip().isdigit():
                key_ids.add(f[4].strip())
    except Exception as exc:                      # offline: skip the cross-check
        print("warning: could not fetch sample-key.tab (%s)" % exc)
    if key_ids:
        got = {r["sample_id"] for r in rows}
        print("sample-key.tab ids: %d; parsed from GEO: %d; in key but not GEO: %d"
              % (len(key_ids), len(got), len(key_ids - got)))

    on_disk = set()
    if os.path.isdir(ROSSI_DIR):
        on_disk = {d for d in os.listdir(ROSSI_DIR) if d.isdigit()}
    for r in rows:
        r["on_disk"] = "yes" if r["sample_id"] in on_disk else ""

    cols = ["sample_id", "target", "replicate", "assay", "condition", "gsm",
            "on_disk"]
    with open(OUT, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: (x["target"], x["replicate"])):
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    from collections import Counter
    c_all = Counter(r["condition"] for r in rows)
    c_disk = Counter(r["condition"] for r in rows if r["on_disk"])
    print("wrote %s (%d samples)" % (OUT, len(rows)))
    print("condition, all deposited samples: %s" % dict(c_all))
    print("condition, samples present on this disk (%d): %s"
          % (sum(c_disk.values()), dict(c_disk)))
    hs = [r for r in rows if r["condition"] != "normal" and r["on_disk"]]
    print("\nheat-shock samples on disk (%d):" % len(hs))
    for r in sorted(hs, key=lambda x: (x["target"], x["replicate"])):
        print("   %-7s %-8s %-6s %s"
              % (r["sample_id"], r["target"], r["replicate"], r["condition"]))


if __name__ == "__main__":
    main()
