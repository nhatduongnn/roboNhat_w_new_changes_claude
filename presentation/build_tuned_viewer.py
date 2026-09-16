#!/usr/bin/env python3
"""Tuned Occupancy Browser: tuner-v2 campaigns fw01 / sw01 / bw01, round 0 vs final.

Two stages, so decode loading can run on compute nodes:

    python build_tuned_viewer.py region <key>     # -> scratch/<key>.json (one region payload)
    python build_tuned_viewer.py emit             # -> tuned_occupancy_viewer.html

Regions come from tuned_regions.tsv (select_tuned_regions.py), runs from tuned_runs.tsv
(label, tuning outDir for chrXIV/chrII, holdout outDir for chrIV). Reuses
make_posterior_viewer's encoding/colour/gene helpers and score_robocop's loaders unchanged;
differences from that viewer, all to fit the 16 MB budget and the tuning question:
  * tracks = the 61 tuned motifs + unknown + nucleosome + nuc_center (no background), plus one
    'other motifs' column = sum of the 92 untuned motifs (non-zero only in the legacy u001 run);
    the factor list is identical in every region and run;
  * Fiber-seq counts stored once per region (checked identical across every run first);
  * references are MacIsaac c1 merged-site centers and Rossi _CX summits for the 58 groups,
    exactly as rossi_validate.load_references builds them;
  * per run, calls = centers of runs with posterior >= 0.10 (score_robocop.call_abf1, the tuning
    loop's rule) for the 61 motifs, pooled per group, matched one-to-one within 30 bp
    (rossi_validate.fast_match). Calls are computed on the window padded by PAD bp so a
    footprint cut by the window edge keeps its true center, and counted by center in window.
"""
import csv, glob, json, os, sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.abspath(os.path.join(HERE, "..", "analysis"))
SCRATCH = "/usr/project/xtmp/nd141/scratch_tuneviewer/regions"
TEMPLATE = os.path.join(HERE, "tuned_occupancy_template.html")
OUT = os.path.join(HERE, "tuned_occupancy_viewer.html")
sys.path.insert(0, AN)
os.chdir(AN)
import make_posterior_viewer as MPV   # noqa: E402
import score_robocop as S             # noqa: E402
import rossi_validate as RV           # noqa: E402
import tune_w                          # noqa: E402

PAD = 250
CALL = 0.10
TOL = 30
QUANT = 100                 # 0.01 steps (the original viewer uses 1000; 10x fewer digits)
SPECIAL_COLS = ["nucleosome", "nuc_center", "unknown"]   # background dropped for size
OTHER = "other_motifs"


def read_regions():
    out = []
    for line in open(os.path.join(HERE, "tuned_regions.tsv")):
        if line.startswith("#") or not line.strip():
            continue
        k, lab, reg, view, _ = line.rstrip("\n").split("\t")
        out.append(dict(key=k, label=lab, region=reg, view=view))
    return out


def read_runs():
    out = []
    for line in open(os.path.join(HERE, "tuned_runs.tsv")):
        if line.startswith("#") or not line.strip():
            continue
        lab, tune, hold = line.rstrip("\n").split("\t")
        out.append((lab, tune, hold))
    return out


def run_dirs(chrom):
    return [(lab, hold if chrom == "chrIV" else tune) for lab, tune, hold in read_runs()]


def build(key):
    spec = {r["key"]: r for r in read_regions()}[key]
    chrom, ws, we = MPV.parse_region(spec["region"])
    start, end = max(1, ws - PAD), we + PAD
    n = end - start + 1
    G, _, _ = tune_w.factor_set()
    refs, _ = RV.load_references(G)
    motifs = [m for g in G for m in G[g]]
    m2g = {m: g for g in G for m in G[g]}

    runs = run_dirs(chrom)
    post, calls, fiber0, invalid, fiber_same = {}, {}, None, {}, {}
    for lab, d in runs:
        dec = S.load_decode(d)
        op, covered, _ = S.region_optable(dec, chrom, start, end)
        assert covered.all(), "%s does not cover %s:%d-%d" % (d, chrom, start, end)
        vals = op.values
        bad = ~(np.isfinite(vals).all(1) & (vals >= -1e-6).all(1) & (vals <= 1 + 1e-6).all(1))
        invalid[lab] = int(bad.sum())
        if bad.any():
            op.loc[bad, :] = 0.0          # same rule as count_calls: invalid positions dropped
        others = [c for c in op.columns if c not in motifs and c not in SPECIAL_COLS
                  and c not in ("nuc_padding", "nuc_start", "nuc_end")]
        tracks = {c: op[c].values for c in motifs + SPECIAL_COLS}
        others = [c for c in others if c != "background"]
        tracks[OTHER] = op[others].values.sum(1)
        post[lab] = tracks
        pos = np.arange(start, end + 1)
        cg = defaultdict(list)
        for m in motifs:
            for c in S.call_abf1(op[m].values, pos, CALL):
                if ws <= c["center"] <= we:
                    cg[m2g[m]].append(int(c["center"]))
        calls[lab] = cg
        fb = S.region_fiber_counts(dec, chrom, start, end)
        if fiber0 is None:
            fiber0 = fb
        fiber_same[lab] = bool(all(np.allclose(fb[k], fiber0[k]) for k in fb))
        print("  %-16s %s  invalid=%d  calls=%d  fiber_same=%s" % (
            lab, d, invalid[lab], sum(map(len, cg.values())), fiber_same[lab]), flush=True)

    # references in the loaded span (drawn) and in the window (counted)
    mac, ros = [], []
    for g in G:
        for p in refs[g]["macisaac"].get(chrom, []):
            if start <= p <= end:
                mac.append([int(p), g])
        for p in (refs[g]["rossi_cx"] or {}).get(chrom, []):
            if start <= p <= end:
                ros.append([int(p), g])
    mac.sort(); ros.sort()
    inwin = lambda L: [x for x in L if ws <= x[0] <= we]

    counts = {}
    for lab, _ in runs:
        cg = calls[lab]
        mref = defaultdict(list); rref = defaultdict(list)
        for p, g in inwin(mac): mref[g].append(p)
        for p, g in inwin(ros): rref[g].append(p)
        counts[lab] = dict(
            calls=sum(len(v) for v in cg.values()),
            mac=sum(RV.fast_match(cg.get(g, []), mref[g], TOL) for g in G),
            ros=sum(RV.fast_match(cg.get(g, []), rref[g], TOL) for g in G),
            invalid=invalid[lab],
            groups={g: len(v) for g, v in cg.items() if v})

    # encode (round-trip checked, as make_posterior_viewer does)
    enc = {}
    for lab, tracks in post.items():
        e = {}
        for f, v in tracks.items():
            q = np.rint(np.nan_to_num(v) * QUANT).astype(np.int32)
            r = MPV.rle(q)
            assert np.array_equal(MPV.unrle(r, n), q)
            assert np.abs(MPV.unrle(r, n) / QUANT - np.nan_to_num(v)).max() <= 0.5 / QUANT + 1e-6
            if r:
                e[f] = r
        enc[lab] = e

    import pysam
    seq = pysam.FastaFile(MPV.FASTA).fetch(chrom, start - 1, end).upper()
    assert len(seq) == n
    payload = dict(
        key=key, label=spec["label"], chrom=chrom, start=start, end=end, win=[ws, we],
        view=[ws, we], runOrder=[l for l, _ in runs], dirs={l: d for l, d in runs},
        post=enc, counts=counts, mac=mac, ros=ros,
        nMac=len(inwin(mac)), nRos=len(inwin(ros)),
        fiber={k2: [int(x) if float(x).is_integer() else round(float(x), 2) for x in fiber0[k1]]
               for k2, k1 in (("mw", "meth_watson"), ("mc", "meth_crick"),
                              ("aw", "A_watson"), ("ac", "A_crick"))},
        fiberSame=fiber_same, seq=seq, genes=MPV.load_genes(chrom, start, end))
    os.makedirs(SCRATCH, exist_ok=True)
    json.dump(payload, open(os.path.join(SCRATCH, key + ".json"), "w"), separators=(",", ":"))
    print("wrote %s  (%.0f kB)" % (key, os.path.getsize(os.path.join(SCRATCH, key + ".json")) / 1e3))


def emit():
    G, _, fitted = tune_w.factor_set()
    motifs = [m for g in G for m in G[g]]
    specs = read_regions()
    regions = {}
    for s in specs:
        regions[s["key"]] = json.load(open(os.path.join(SCRATCH, s["key"] + ".json")))
    labels0 = regions[specs[0]["key"]]["runOrder"]
    for k, r in regions.items():
        assert r["runOrder"] == labels0, "label set differs in %s" % k
        assert set(r["post"]) == set(labels0), "run data missing in %s" % k
    factors = motifs + [OTHER] + SPECIAL_COLS
    colors, labels = {}, {}
    for m in motifs:
        g = m2g = [gg for gg in G if m in G[gg]][0]
        labels[m] = g if len(G[g]) == 1 else "%s · %s" % (g, m.split("_", 1)[1])
        colors[m] = MPV.to_hex(MPV.color_for_name(m.split("_")[0].upper()))
    for f in SPECIAL_COLS:
        labels[f], colors[f] = f, MPV.SPECIAL[f]
    labels[OTHER], colors[OTHER] = "other motifs (92 untuned, summed)", "#7d6b91"
    group_of = {m: [g for g in G if m in G[g]][0] for m in motifs}
    group_color = {g: colors[G[g][0]] for g in G}
    runs = read_runs()
    data = dict(
        title="Tuned Occupancy Browser", regionOrder=[s["key"] for s in specs],
        regionLabels={s["key"]: s["label"] for s in specs}, regions=regions,
        factors=factors, colors=colors, labels=labels, groupOf=group_of,
        groupColor=group_color, fitted={g: int(v) for g, v in fitted.items()},
        pinned=["nucleosome", "nuc_center", "unknown", OTHER], quant=QUANT,
        runs=[dict(label=l, tune=t, hold=h) for l, t, h in runs],
        bg={"fib r0": 0.1383, "fib final": 0.1383, "both r0": 0.1383, "both final": 0.1383,
            "u001 r7 legacy": 0.1383},
        selection=json.load(open(os.path.join(HERE, "tuned_regions_selection.json"))))
    js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = open(TEMPLATE, encoding="utf-8").read().replace("{{DATA_JSON}}", js)
    open(OUT, "w", encoding="utf-8").write(html)
    size = os.path.getsize(OUT)
    print("page %.2f MB, %d regions x %d runs" % (size / 1e6, len(regions), len(labels0)))
    assert size < 16 * 1024 * 1024


if __name__ == "__main__":
    if sys.argv[1] == "region":
        for k in sys.argv[2:]:
            build(k)
    elif sys.argv[1] == "region-index":
        build(read_regions()[int(sys.argv[2])]["key"])
    else:
        emit()
