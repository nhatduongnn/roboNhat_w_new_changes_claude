"""Build c1_report.html (the published C1-check report) from c1scan/summary_c1_rows.tsv.

    cd analysis && python posterior_fix/fibercop_audit/c1scan/report/make_report.py

Numbers in the prose come from c1scan/README.md (trace of bo09 seg 307) and extent/ (stored decodes).
"""
import os, re, html
import pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
t = pd.read_csv(os.path.join(H, "..", "summary_c1_rows.tsv"), sep="\t", index_col=0, dtype=str)
MODELS = [("bo09_48", "bo09 r48", "9 motifs live, ABF1 &lambda; 5.8e-45"),
          ("bu01_11", "bu01 r11", "all 153 motifs live"),
          ("bv75_49", "bv75 r49", "75 motifs live")]


def cell(v):
    m = re.match(r"(\d+) / (\d+)(?: \(\+(\d+) other\))?", v)
    segs, rows, other = int(m.group(1)), int(m.group(2)), m.group(3)
    lvl = 0 if rows == 0 else 1 if rows < 100 else 2 if rows < 2000 else 3
    o = '<span class="oth">+%s not C1</span>' % format(int(other), ",") if other else ""
    if rows == 0:
        return '<td class="l0"><span class="z">0</span></td>'
    return ('<td class="l%d"><span class="r">%s</span><span class="s">%d seg</span>%s</td>'
            % (lvl, format(rows, ","), segs, o))


rows = []
for chrom, r in t.iterrows():
    cls = ' class="tot"' if chrom == "total" else (' class="rd"' if "rDNA" in chrom else "")
    rows.append("<tr%s><th scope=\"row\">%s</th>%s</tr>" % (cls, html.escape(chrom), "".join(cell(r[m]) for m, _, _ in MODELS)))
head = "".join('<th scope="col">%s<small>%s</small></th>' % (lab, sub) for _, lab, sub in MODELS)
table = '<table class="res"><thead><tr><th scope="col">chromosome</th>%s</tr></thead><tbody>%s</tbody></table>' % (head, "\n".join(rows))

page = open(os.path.join(H, "template.html")).read().replace("%%TABLE%%", table)
out = os.path.join(H, "c1_report.html")
open(out, "w").write(page)
print("wrote", out, len(page), "bytes")
