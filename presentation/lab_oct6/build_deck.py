"""Assemble the Oct 6 lab deck (artifact AJojw8M1SkhzVbsJnud6Pa) from deck_template.html and three
locus payloads built by presentation/templates/locus_slide/build_locus_slide.py (decode mode, which
reads posteriors via score_robocop.region_optable -- normalised per position, never clipped):

  rpc19_up.json    chrXIV:412410-412909  bo09_48, bu01_11, bv75_49 (500 bp centred on the poly-A)
  rpc19_down.json  chrXIV:413212-413711  same runs (500 bp centred on the poly-T)
  fixXIVup.json / fixXIVdown.json  the same two 500 bp windows, robocop_w30k_bv75_49_{Bprod,logfull}_both
  fixXII.json  chrXII:452401-454400  robocop_rdna_chrXII_{Bprod,logfull}_both

Adds: A/T homopolymer runs >= 10 bp (inputs/SacCer3.fa) to rpc19; the bg-open background rate
0.249 (inputs/bg_params_open.pkl, Watson) for the m6A dashed line; display labels for chips.
Drops the unused fiber arrays from the two fix payloads.  Output: lab_oct6_deck.html
"""
import argparse, json, os, re, sys
H = os.path.dirname(os.path.abspath(__file__))
A = os.path.abspath(os.path.join(H, "..", "..", "analysis"))
from Bio import SeqIO

ap = argparse.ArgumentParser()
ap.add_argument("--reader", choices=["corrected", "viewer"], default="corrected",
                help="slides 4-5 only: corrected = region_optable payloads (rpc19_<side>.json); viewer = the old Run "
                     "Browser reading (rpc19_<side>_viewer.json from viewer_reader.py). Slides 17-19 always use corrected.")
ARGS = ap.parse_args()

def load(n): return json.load(open(os.path.join(H, n + ".json")))
genome = {r.id: str(r.seq).upper() for r in SeqIO.parse(os.path.join(A, "inputs", "SacCer3.fa"), "fasta")}

rpc = {}
for side in ("up", "down"):
    r = load("rpc19_" + side + ("_viewer" if ARGS.reader == "viewer" else ""))
    seq = genome[r["chrom"]][r["s"] - 1:r["e"]]
    r["tracts"] = [[r["s"] + m.start(), r["s"] + m.end() - 1, m.group()[0]] for m in re.finditer(r"A{10,}|T{10,}", seq)]
    r["bg"] = 0.24893964
    r["call"]["label"] = "bo09"; r["call"]["alsoLabel"] = {"bu01_11": "bu01", "bv75_49": "bv75"}
    print("rpc19_%s tracts" % side, r["tracts"])
    rpc[side] = r

fx = {}
for side in ("up", "down"):
    f = load("fixXIV" + side); f.pop("fiber", None)
    f["tracts"] = rpc[side]["tracts"]
    f["call"]["label"] = "fix B"; f["call"]["alsoLabel"] = {"log": "log space"}
    fx[side] = f
fz = load("fixXII"); fz.pop("fiber", None)
names = {g["name"] for g in fz["genes"]}
if "RDN37-1" in names and "RDN25-1" in names:
    fz["genes"] = [g for g in fz["genes"] if g["name"] != "RDN25-1"]
    for g in fz["genes"]:
        if g["name"] == "RDN37-1": g["name"] = "RDN37-1 (35S, contains RDN25-1)"

def js(d): return json.dumps(d, separators=(",", ":")).replace("</", "<\\/")
t = open(os.path.join(H, "deck_template.html"), encoding="utf-8").read()
t = t.replace("%%RPC19UP%%", js(rpc["up"])).replace("%%RPC19DOWN%%", js(rpc["down"])).replace("%%FIXXIVUP%%", js(fx["up"])).replace("%%FIXXIVDOWN%%", js(fx["down"])).replace("%%FIXXII%%", js(fz))
assert "%%" not in t
out = os.path.join(H, "lab_oct6_deck.html")
open(out, "w", encoding="utf-8").write(t)
print("reader for slides 4-5:", ARGS.reader); print("wrote", out, "%.1f KB" % (len(t.encode()) / 1e3), "slides", t.count('<section class="slide'))
