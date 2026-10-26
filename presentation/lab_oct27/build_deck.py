"""Assemble the Oct 27 lab deck (lab_oct27_deck.html) from deck_template.html and the four locus payloads
written by build_payloads.py (locus_<key>.json: 6 rows, bo09/bu01/bv75 x before/after, both read the Run
Browser way). Adds display labels for the site chips and, where the dbf colour is too pale or too dark to
read on both themes, an override factor colour (stated in each slide's footnote).

    python make_template.py && python build_deck.py
"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))

COLOR = {"xiv724": "#7c4dcc",   # background: dbf/viewer colour #9aa7b4 is too close to the nucleosome grey
         "iv1200": "#0e9f8e"}   # Fhl1_zhu: dbf colour #42230b is unreadable on the dark theme


def load(key):
    P = json.load(open(os.path.join(H, "locus_%s.json" % key)))
    P["call"]["label"] = "bo09 after"
    P["call"]["also"] = ["bo09 before", "bv75 after"]
    P["call"]["alsoLabel"] = {"bo09 before": "bo09 before", "bv75 after": "bv75 after"}
    # positions where any column differs >= 0.1 (numbers.json, spans_ge_0p1), marked on each "after" row
    N = json.load(open(os.path.join(H, "numbers.json")))["loci"][key]["models"]
    for mod in ("bo09", "bu01", "bv75"):
        P["runs"][mod + " after"]["diff"] = N[mod]["spans_ge_0p1"]
    if key in COLOR:
        P["colors"]["tf"] = COLOR[key]
    return P


def js(d): return json.dumps(d, separators=(",", ":")).replace("</", "<\\/")


t = open(os.path.join(H, "deck_template.html"), encoding="utf-8").read()
for key, ph in (("rpc19", "%%LOC_RPC19%%"), ("xiv440", "%%LOC_XIV440%%"), ("xiv724", "%%LOC_XIV724%%"), ("iv1200", "%%LOC_IV1200%%")):
    t = t.replace(ph, js(load(key)))
assert "%%" not in t
out = os.path.join(H, "lab_oct27_deck.html")
open(out, "w", encoding="utf-8").write(t)
print("wrote", out, "%.1f KB" % (len(t.encode()) / 1e3), "slides", t.count('<section class="slide'))
