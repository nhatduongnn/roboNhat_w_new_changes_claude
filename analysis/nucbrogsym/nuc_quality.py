#!/usr/bin/env python
"""Nucleosome quality at ERV46, old vs new model, all molecules (nucbrog/full/{old,new}).

  * checks.check2 (pooled per-read dyads vs Chereji +1/-1, <= 20 bp) unchanged;
  * per reference dyad (Chereji, and Brogaard top-2000 dyads in the window): median |nearest
    per-read call - ref| over reads with a call within 73 bp, and fraction of covering reads
    with a call within 20 bp;
  * read-averaged nucleosome occupancy and nuc_center: old vs new correlation, peaks vs refs;
  * posterior validity from run_stats.json.
Writes OUT/nuc_quality.json and OUT/nuc_quality.png.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ANALYSIS, "singlefiber"))
sys.path.insert(0, ANALYSIS)
os.chdir(ANALYSIS)
import checks as CK          # noqa: E402
import score_robocop as S    # noqa: E402
import nucmodel as NM        # noqa: E402

# ALL7 (2026-10-30): $NUCQ_OUT selects another results root (runs under $NUCQ_OUT/full); default unchanged.
OUT = os.environ.get("NUCQ_OUT", "/usr/project/xtmp/nd141/permol_proto_logfull/nucbrog")
# argv: run names under OUT/full (first two are compared and plotted); default old new
NAMES = sys.argv[1:] or ["old", "new"]
RUNS = {nm: os.path.join(OUT, "full", nm) for nm in NAMES}
TAG = "" if NAMES == ["old", "new"] else "_" + "_".join(NAMES)
CHROM = "chrI"


def per_read_calls(pm):
    j = pm["cols"].index("nuc_center")
    pos = np.arange(pm["start"], pm["end"] + 1)
    calls = []
    for i in range(pm["per_mol"].shape[0]):
        cv = pm["cov"][i]
        tr = pm["per_mol"][i, :, j].astype(float)
        tr[~cv] = 0
        mx = float(tr.max())
        h = max(0.02, 0.20 * mx) if mx > 0 else 0.02
        calls.append([int(pos[k]) for k in S.call_peaks(tr, height=h, distance=120)])
    return calls


def per_ref(pm, calls, refs):
    out = []
    for r in refs:
        k = r - pm["start"]
        covering = [i for i in range(len(calls)) if pm["cov"][i, k]]
        d = []
        n20 = 0
        for i in covering:
            near = [c for c in calls[i] if abs(c - r) <= 73]
            if near:
                dd = min(abs(c - r) for c in near)
                d.append(dd)
                n20 += dd <= 20
        out.append(dict(ref=int(r), n_covering=len(covering), n_with_call=len(d),
                        median_abs_dist=float(np.median(d)) if d else None,
                        frac_covering_within20=(n20 / len(covering)) if covering else None))
    return out


res = {}
pms = {}
for nm, d in RUNS.items():
    pm = CK.load_permol(d)
    pms[nm] = pm
    st = json.load(open(os.path.join(d, "run_stats.json")))
    c2 = CK.check2(pm, CHROM)
    calls = per_read_calls(pm)
    ch = S.load_chereji(CK.CHEREJI_BED)
    chref = sorted(ch[(ch.chr == CHROM) & (ch.dyad >= pm["start"]) & (ch.dyad <= pm["end"])].dyad)
    br = [s for c, s, e in NM.read_bed(NM.BROGAARD) if c == CHROM and pm["start"] <= s <= pm["end"]]
    avg = pm["avg"]
    jn, jc = pm["cols"].index("nucleosome"), pm["cols"].index("nuc_center")
    pk = [int(pm["start"] + k) for k in S.call_peaks(avg[:, jc], height=0.02, distance=120)]
    near = lambda refs: [min((abs(p - r) for p in pk), default=None) for r in refs]
    res[nm] = dict(n_molecules=st["n_molecules_decoded"], posterior_validity=st["posterior_validity"],
                   check2=dict(recall=c2["recall"], verdict=c2["verdict"],
                               n_pooled_calls=c2["n_pooled_calls"],
                               median_dyad_sd_bp=c2["median_dyad_sd_bp"],
                               median_dist=c2.get("match", {}).get("median_dist")),
                   chereji=per_ref(pm, calls, chref), brogaard=per_ref(pm, calls, sorted(br)),
                   avg_nuc_occ_mean=float(avg[:, jn].mean()),
                   avg_nuc_center_peaks=pk,
                   avg_peak_dist_to_chereji=near(chref), avg_peak_dist_to_brogaard=near(sorted(br)))
o, n = pms[NAMES[0]]["avg"], pms[NAMES[1]]["avg"]
jn = pms[NAMES[0]]["cols"].index("nucleosome")
res["avg_nucleosome_%s_vs_%s" % (NAMES[0], NAMES[1])] = dict(
    pearson=float(np.corrcoef(o[:, jn], n[:, jn])[0, 1]),
    mean_abs_diff=float(np.abs(o[:, jn] - n[:, jn]).mean()),
    max_abs_diff=float(np.abs(o[:, jn] - n[:, jn]).max()))
assert all(list(pms[k]["names"]) == list(pms[NAMES[0]]["names"]) for k in NAMES)
json.dump(res, open(os.path.join(OUT, "nuc_quality%s.json" % TAG), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))

if TAG:          # the figure is only drawn for the default old-vs-new comparison
    sys.exit(0)
x = np.arange(pms["old"]["start"], pms["old"]["end"] + 1)
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(x, o[:, jn], color="#888888", lw=1.5, label="old model (Chereji, shipped m6A)")
ax.plot(x, n[:, jn], color="#2a78d6", lw=1.5, label="new model (Brogaard, symmetric)")
for r in res["old"]["chereji"]:
    ax.axvline(r["ref"], color="#eb6834", lw=1, ls="--")
for r in res["old"]["brogaard"]:
    ax.axvline(r["ref"], color="#2a9d4a", lw=1, ls=":")
ax.set_xlabel("chrI position (orange dashed = Chereji +1/-1 dyads, green dotted = Brogaard top-2000)")
ax.set_ylabel("read-averaged P(nucleosome)")
ax.set_ylim(0, 1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("ERV46, %d reads, ref direction, start = any" % res["old"]["n_molecules"], fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "nuc_quality%s.png" % TAG), dpi=130)
