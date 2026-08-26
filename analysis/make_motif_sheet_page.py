#!/usr/bin/env python
"""Render the three-way motif comparison into the sortable HTML sheet.

    python motif_distance_sheet.py      # builds the TSV + JSON (incl. logo PWMs)
    python make_motif_sheet_page.py     # builds the page from the JSON

The JSON carries, per native motif, the aligned matrices for all three datasets
in the native frame -- each already reverse-complemented and offset exactly as it
entered its comparison -- so the logos the page draws are the matrices that were
actually compared.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "motif_distance_sheet.json")
TPL = os.path.join(HERE, "motif_sheet_template.html")
OUT = os.path.join(HERE, "motif_distance_sheet.html")


def main():
    rows = json.load(open(SRC))
    tpl = open(TPL).read()
    if "__DATA__" not in tpl:
        raise SystemExit("template is missing the __DATA__ placeholder")
    html = tpl.replace("__DATA__", json.dumps(rows, separators=(",", ":")))
    with open(OUT, "w") as fh:
        fh.write(html)
    print("wrote %s  (%d rows, %d KB)" % (OUT, len(rows), len(html) // 1024))


if __name__ == "__main__":
    main()
