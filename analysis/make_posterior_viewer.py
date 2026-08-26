#!/usr/bin/env python3
"""Build a self-contained interactive posterior viewer for one genomic region.

The static plot_robocop_output PNGs cannot be zoomed and stack every factor into one
opaque fill. This emits an HTML page with the same posterior table on a pan/zoom canvas,
arbitrary factor toggles, and aligned m6A / depth / sequence / gene tracks -- enough to
answer, locus by locus, whether a call sits on flat unmethylated DNA or on genuinely
methylated DNA.

    python make_posterior_viewer.py \
        --region chrI:60001-65000 --view chrI:60500-64500 \
        --run revfix=robocop_chrI_seq_maskoff_revfix \
        --run capA=robocop_chrI_seq_maskoff_capA \
        --run capB=robocop_chrI_seq_maskoff_capB \
        --out posterior_viewer_erv46.html

Reuses score_robocop.load_decode / region_optable / region_fiber_counts for the data and
the decode's own dbf_color_map.pkl for colors, so factor colors match the PNGs exactly.
"""
import argparse, hashlib, json, os, pickle, random, sys, re

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_robocop as S

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "posterior_viewer_template.html")
FASTA = os.path.join(HERE, "inputs", "SacCer3.fa")
GTF = os.path.join(HERE, "inputs", "sacCer3.gtf")

KEEP_MIN = 1e-3          # drop factors that never reach this over the loaded region
QUANT = 1000             # posteriors stored as round(p * QUANT)
GAP = 4                  # merge nonzero runs separated by <= this many zeros
CALL = 0.30              # posterior threshold for the sites-of-interest list
MAX_BYTES = 16 * 1024 * 1024

# Fiber-seq background rates in play, for the m6A panel's reference lines.
# Background methylation rate per run, for the m6A panel's reference lines. em10 shares
# revfix's emission model exactly (both decode under pkgvar/seq_maskoff, reading the same
# inputs/bg_params.pkl) -- only the transition prior differs -- so its bg is revfix's.
BG = {"revfix": 0.1383, "capA": 0.1548, "capB": 0.4402, "em10": 0.1383}
NHP6A_RATE = 0.1098

# Colors for the non-TF states. 'nucleosome' 0.7 grey and 'unknown' #D3D3D3 are what
# colorMap() in pkg/robocop/utils/plotRoboCOP.py:81 uses.
SPECIAL = {
    "background": "#9aa7b4", "nucleosome": "#b3b3b3", "unknown": "#d3d3d3",
    "nuc_padding": "#cfcfcf", "nuc_center": "#8c8c8c",
    "nuc_start": "#a6a6a6", "nuc_end": "#a6a6a6",
}
PINNED = ["nucleosome", "background", "unknown", "Nhp6a_zhu", "Abf1_murphy"]


def color_for_name(name):
    """Byte-for-byte the recipe in plotRoboCOP._color_for_name (:77): md5 -> seeded RNG."""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(h)
    return (rng.random(), rng.random(), rng.random())


def to_hex(c):
    if isinstance(c, str):
        if c.startswith("#"):
            return c
        try:                                   # matplotlib grey string, e.g. '0.7'
            g = int(round(float(c) * 255))
            return "#%02x%02x%02x" % (g, g, g)
        except ValueError:
            return "#888888"
    r, g, b = c[0], c[1], c[2]
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in (r, g, b))


def parse_region(s):
    m = re.match(r"^([\w.]+):(\d[\d,]*)-(\d[\d,]*)$", s.strip())
    if not m:
        raise ValueError("region must look like chrI:60001-65000, got %r" % s)
    return m.group(1), int(m.group(2).replace(",", "")), int(m.group(3).replace(",", ""))


def rle(vals):
    """Run-length encode into [[startOffset, [v, ...]], ...] over nonzero stretches.

    Runs separated by <= GAP zeros are merged; splitting on every single zero would
    otherwise shred the noisy TF columns into thousands of two-element segments.
    """
    nz = np.flatnonzero(vals)
    if nz.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(nz) > GAP + 1)
    starts = np.concatenate(([nz[0]], nz[breaks + 1]))
    ends = np.concatenate((nz[breaks], [nz[-1]]))
    return [[int(a), [int(v) for v in vals[a:b + 1]]] for a, b in zip(starts, ends)]


def unrle(runs, n):
    out = np.zeros(n, dtype=np.int32)
    for off, vals in runs:
        out[off:off + len(vals)] = vals
    return out


def load_genes(chrom, start, end):
    """Same parse as plotRegion (pkg/robocop/utils/plotRoboCOP.py:108).

    Note the GTF names chromosomes 'I', not 'chrI' -- hence the chrom[3:].
    """
    if not os.path.isfile(GTF):
        return []
    a = pd.read_csv(GTF, sep="\t", header=None, comment="#")
    key = chrom[3:] if chrom.lower().startswith("chr") else chrom
    a = a[(a[0].astype(str) == key) & (a[3] <= end) & (a[4] >= start)]
    genes = {}
    for _, r in a.iterrows():
        if r[2] != "transcript":
            continue
        attrs = {}
        for g in str(r[8]).rstrip(";").split(";"):
            parts = g.split()
            if len(parts) >= 2:
                attrs[parts[0]] = parts[1].strip('"')
        name = attrs.get("gene_name") or attrs.get("gene_id") or "?"
        if name in genes:
            genes[name] = (min(genes[name][0], r[3]), max(genes[name][1], r[4]), r[6])
        else:
            genes[name] = (r[3], r[4], r[6])
    return [{"name": k, "start": int(v[0]), "end": int(v[1]), "strand": v[2]}
            for k, v in sorted(genes.items(), key=lambda kv: kv[1][0])]


def intervals(mask, positions_start, gap=20):
    """Merge a boolean mask into [(start, end)] genomic intervals, bridging small gaps."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(a) + positions_start, int(b) + positions_start) for a, b in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help="chrI:60001-65000 (data loaded)")
    ap.add_argument("--view", default=None, help="initial view, defaults to --region")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    chrom, start, end = parse_region(args.region)
    vchrom, vs, ve = parse_region(args.view) if args.view else (chrom, start, end)
    if vchrom != chrom:
        sys.exit("--view chromosome %s != --region chromosome %s" % (vchrom, chrom))
    vs, ve = max(vs, start), min(ve, end)
    n = end - start + 1
    runs = [tuple(r.split("=", 1)) for r in args.run]

    print("region %s:%d-%d  (%d bp), %d run(s)" % (chrom, start, end, n, len(runs)))

    tables, fibers, colormaps = {}, {}, {}
    for label, d in runs:
        print("  loading %-8s %s" % (label, d), flush=True)
        dec = S.load_decode(d)
        op, covered, _ = S.region_optable(dec, chrom, start, end)
        fb = S.region_fiber_counts(dec, chrom, start, end)
        if fb is None:
            sys.exit("%s carries no Fiber-seq datasets (tech2 unset)" % d)
        tables[label] = op
        fibers[label] = fb
        cm_path = os.path.join(d.rstrip("/"), "dbf_color_map.pkl")
        colormaps[label] = pickle.load(open(cm_path, "rb")) if os.path.isfile(cm_path) else {}
        print("     covered %d/%d bp" % (int(covered.sum()), n))

    # --- factor set: union across runs, so the sidebar is identical in every run ---
    keep = set()
    for label, op in tables.items():
        mx = op.max(0)
        keep |= {c for c in op.columns if mx[c] >= KEEP_MIN}
    factors = [c for c in tables[runs[0][0]].columns if c in keep]
    print("  keeping %d of %d factors (max >= %g)" % (len(factors), tables[runs[0][0]].shape[1], KEEP_MIN))

    colors, labels = {}, {}
    cm = colormaps[runs[0][0]]
    for f in factors:
        disp = f.split("_")[0].upper()
        labels[f] = disp if f not in SPECIAL else f
        if f in SPECIAL:
            colors[f] = SPECIAL[f]
        elif disp in cm:
            colors[f] = to_hex(cm[disp])
        else:
            colors[f] = to_hex(color_for_name(disp))

    # --- encode ---
    payload_runs, quantized = {}, {}
    for label, op in tables.items():
        post, q = {}, {}
        for f in factors:
            v = np.rint(np.nan_to_num(op[f].values) * QUANT).astype(np.int32)
            q[f] = v
            enc = rle(v)
            if enc:
                post[f] = enc
        quantized[label] = q
        fb = fibers[label]
        payload_runs[label] = {
            "post": post,
            "fiber": {
                "mw": [round(float(x), 2) for x in fb["meth_watson"]],
                "mc": [round(float(x), 2) for x in fb["meth_crick"]],
                "aw": [round(float(x), 2) for x in fb["A_watson"]],
                "ac": [round(float(x), 2) for x in fb["A_crick"]],
            },
        }

    # --- verification 1: the encoding round-trips ---
    for label in tables:
        for f, enc in payload_runs[label]["post"].items():
            back = unrle(enc, n)
            assert np.array_equal(back, quantized[label][f]), "RLE mismatch %s/%s" % (label, f)
            dq = back / QUANT
            orig = np.nan_to_num(tables[label][f].values)
            err = np.abs(dq - orig).max()
            assert err <= 0.5 / QUANT + 1e-9, "quantization error %g in %s/%s" % (err, label, f)
    print("  encode round-trip OK (max error <= %g)" % (0.5 / QUANT))

    # --- sites of interest ---
    sites = []
    for label, op in tables.items():
        for f in ("Nhp6a_zhu", "Abf1_murphy"):
            if f not in op.columns:
                continue
            for a, b in intervals(op[f].values >= CALL, start):
                pk = float(op[f].values[a - start:b - start + 1].max())
                sites.append({"start": a, "end": b, "peak": pk, "run": label, "factor": f,
                              "label": "%s  %s  %s:%s-%s  (%.2f)" %
                                       (label, labels.get(f, f), chrom, f"{a:,}", f"{b:,}", pk)})
    sites.sort(key=lambda s: (s["start"], s["run"]))
    print("  %d sites of interest (%s >= %.2f)" % (len(sites), "Nhp6a/Abf1", CALL))

    # --- sequence + genes ---
    import pysam
    seq = pysam.FastaFile(FASTA).fetch(chrom, start - 1, end).upper()
    assert len(seq) == n, "fasta returned %d bp, expected %d" % (len(seq), n)
    genes = load_genes(chrom, start, end)
    print("  %d genes in region" % len(genes))

    # --- reference lines for the m6A panel ---
    refs = [{"label": "Nhp6a fitted %.3f" % NHP6A_RATE, "value": NHP6A_RATE,
             "tone": "tf", "run": None}]
    for label, _ in runs:
        if label in BG:
            refs.append({"label": "%s bg %.3f" % (label, BG[label]), "value": BG[label],
                         "tone": "bg", "run": label})

    on0 = [f for f in ("nucleosome", "background", "Nhp6a_zhu", "Abf1_murphy") if f in factors]
    data = {
        "chrom": chrom, "start": start, "end": end, "view": [vs, ve],
        "factors": factors, "colors": colors, "labels": labels,
        "pinned": [f for f in PINNED if f in factors],
        "initialOn": on0,
        "runOrder": [label for label, _ in runs],
        "runs": payload_runs, "seq": seq, "genes": genes, "sites": sites, "refs": refs,
        "title": args.title or ("%s Occupancy Browser" % chrom),
        "subtitle": "%s:%s–%s loaded · %s bp · %d factors · %d runs · fiber + sequence layers"
                    % (chrom, f"{start:,}", f"{end:,}", f"{n:,}", len(factors), len(runs)),
        "provenance": "Built by <code>analysis/make_posterior_viewer.py</code> from "
                      + " · ".join("<code>%s</code>" % d.rstrip("/") for _, d in runs),
    }

    tpl = open(TEMPLATE, encoding="utf-8").read()
    js = json.dumps(data, separators=(",", ":"))
    html = tpl.replace("{{DATA_JSON}}", js).replace("{{TITLE}}", data["title"])
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    size = os.path.getsize(out)
    print("  payload %.2f MB · page %.2f MB" % (len(js) / 1e6, size / 1e6))
    assert size < MAX_BYTES, "page is %.1f MB, over the 16 MB artifact cap" % (size / 1e6)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
