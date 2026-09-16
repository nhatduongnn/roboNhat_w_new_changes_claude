#!/usr/bin/env python3
"""Build one "locus up close" slide for the talk deck: embedded JSON + the <section> markup.

The slide redraws the RoboCOP Occupancy Browser (analysis/posterior_viewer_template.html) for
one ~2 kb window: one row per run (one factor's posterior + nucleosome occupancy), one shared
m6A Watson/Crick panel, genes, a bp axis, and reference-site outlines labelled hit/miss.

All data come from analysis/make_posterior_viewer.py, imported and never modified:
  * decode mode (default): build_region(<window>, None, runs) reads the decodes directly
    (score_robocop.load_decode / region_optable / region_fiber_counts underneath);
  * viewer mode (--from-viewer HTML --viewer-region KEY): slices the payload already embedded
    in a built browser page. Same numbers, no decode access, seconds instead of minutes.

    python build_locus_slide.py --region chrXIV:186900-188900 --factor Abf1_murphy \
        --runs ../../../analysis/viewer_runs_chrXIV.tsv --labels fib,fib+seq,seq \
        --slide-id 5b --title "One ABF1 locus up close (2)" --kind hidden \
        --out-json 5b.json --out-section 5b.section.html

See README.md for the insertion recipe and the hit/miss rule.
"""
import argparse, html, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ANALYSIS = os.path.join(REPO, "analysis")
sys.path.insert(0, ANALYSIS)
os.environ.setdefault("MPLBACKEND", "Agg")
import make_posterior_viewer as V          # noqa: E402  (data functions, reused as-is)

SNIPPET = os.path.join(HERE, "locus_slide_snippet.html")

# swatch class + short description per run label (deck colours: --fiber / --both / --seq)
RUN_STYLE = {
    "fib": ("fiber", "fiber only"),
    "fib+seq": ("both", "fiber + sequence"),
    "seq": ("seq", "sequence only"),
    "fib+em10": ("fiber", "fiber only, EM 10 it."),
    "fib+seq+em10": ("both", "fiber + seq, EM 10 it."),
    "seq+em10": ("seq", "sequence only, EM 10 it."),
}
# MacIsaac bed name -> the viewer's reference-track key (REF_TRACKS in make_posterior_viewer)
REF_KEY = {bedname: key for key, bedname, _, _, _ in V.REF_TRACKS}
REF_COLOR = {bedname: col for _, bedname, col, _, _ in V.REF_TRACKS}
MACISAAC_C1 = ("/usr/project/xtmp/nd141/projects/replicate_prob_dyad_plot/data/ref-data/"
               "MacIsaac_p005_c1_V64_SGD.gff3")
ROMAN = {"chr%d" % i: "chr" + r for i, r in enumerate(
    "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI".split(), 1)}

CALL_RUN, CALL_THR, CALL_TOL = "fib+seq", 0.10, 20     # hit/miss rule, see README


def fmt(v):
    return f"{v:,}"


def load_c1_intervals(chrom, tf, s, e, slop=20, path=MACISAAC_C1):
    """Merged MacIsaac c1 site intervals (interval union, 20 bp slop: make_conc_targets' rule)."""
    import make_conc_targets as MCT
    spans = sorted(MCT._macisaac_intervals(path).get((chrom, tf), []))
    out, cur = [], None
    for a, b in spans:
        if cur is None or a - slop > cur[1]:
            if cur:
                out.append(cur)
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        out.append(cur)
    return [[a, b, "."] for a, b in out if b >= s and a <= e]


def load_rossi_summits(path, chrom, s, e):
    pts = []
    for line in open(path):
        f = line.split("\t")
        if len(f) < 3:
            continue
        c = ROMAN.get(f[0], f[0])
        if c == chrom:
            p = int(f[1]) + 1                        # 0-based bed start of a 1 bp summit
            if s <= p <= e:
                pts.append(p)
    return sorted(pts)


def from_decodes(region, runs, factor):
    """build_region on the window itself -> {label: (tf[], nuc[])}, fiber per label, genes, colours."""
    d = V.build_region(region, None, runs)
    n = d["end"] - d["start"] + 1
    post = {}
    for label, _ in runs:
        P = d["runs"][label]["post"]
        tf = V.unrle(P.get(factor, []), n).tolist()
        nuc = V.unrle(P.get("nucleosome", []), n).tolist()
        post[label] = (tf, nuc)
    return d, post, 0, n


def from_viewer(path, key, runs, factor, s, e):
    raw = open(path, encoding="utf-8").read()
    i = raw.find('{"regions"')
    data, _ = json.JSONDecoder().raw_decode(raw[i:])
    d = data["regions"][key]
    n = d["end"] - d["start"] + 1
    a, b = s - d["start"], e - d["start"] + 1
    if a < 0 or b > n:
        sys.exit("window %d-%d is outside viewer region %s (%d-%d)" % (s, e, key, d["start"], d["end"]))
    post = {}
    for label, _ in runs:
        P = d["runs"][label]["post"]
        post[label] = (V.unrle(P.get(factor, []), n)[a:b].tolist(),
                       V.unrle(P.get("nucleosome", []), n)[a:b].tolist())
    return d, post, a, b


def build(args):
    chrom, s, e = V.parse_region(args.region)
    labels = [x.strip() for x in args.labels.split(",")]
    table = dict(V.read_runs(args.runs)) if args.runs else {}
    for r in args.run or []:
        k, v = r.split("=", 1)
        table[k] = v
    runs = [(l, os.path.join(ANALYSIS, table[l]) if not os.path.isabs(table[l]) else table[l])
            for l in labels]
    ref_tf = (args.ref_tf or args.factor.split("_")[0]).upper()
    tf_label = args.tf_label or ref_tf

    if args.from_viewer:
        d, post, a, b = from_viewer(args.from_viewer, args.viewer_region, runs, args.factor, s, e)
        src_region = [d["start"], d["end"]]
    else:
        d, post, a, b = from_decodes("%s:%d-%d" % (chrom, s, e), runs, args.factor)
        src_region = [s, e]
    if args.browser_region:
        src_region = list(V.parse_region(args.browser_region)[1:])

    fr = args.fiber_run or labels[0]
    F = d["runs"][fr]["fiber"]
    fiber = {k: F[k][a:b] for k in ("mw", "mc", "aw", "ac")}

    if args.sites == "macisaac_bed":
        key = REF_KEY.get(ref_tf)
        if key is None:
            sys.exit("MacIsaac bed carries only %s; use --sites macisaac_c1" % sorted(REF_KEY))
        allsites = V.load_ref_sites(chrom, s, e)[key] if not args.from_viewer else d["refSites"][key]
        sites = [list(x) for x in allsites if x[1] >= s and x[0] <= e]
    else:
        sites = load_c1_intervals(chrom, ref_tf, s, e)
    genes = [g for g in d["genes"] if g["end"] >= s and g["start"] <= e]

    colors = {"tf": d["colors"].get(args.factor, "#17ef13"),
              "nuc": V.SPECIAL["nucleosome"],
              "ref": REF_COLOR.get(ref_tf, "#d4145a")}
    out = {
        "chrom": chrom, "s": s, "e": e, "region": src_region,
        "factor": args.factor, "tfLabel": tf_label, "refLabel": "MacIsaac " + ref_tf,
        "runOrder": labels,
        "runs": {l: {"tf": post[l][0], "nuc": post[l][1]} for l in labels},
        "fiberRun": fr, "fiber": fiber, "sites": sites,
        "genes": genes, "bg": V.BG.get(fr, 0.1383), "colors": colors,
        "call": {"run": args.call_run, "thr": CALL_THR, "tol": CALL_TOL},
    }
    if args.rossi:
        out["rossi"] = load_rossi_summits(args.rossi, chrom, s, e)
    for l in labels:
        assert len(out["runs"][l]["tf"]) == e - s + 1, "length mismatch in %s" % l
    return out


def site_calls(D):
    """(site, max call-run posterior within +-tol of the site, hit?) -- the chip rule, in Python."""
    arr, res = D["runs"][D["call"]["run"]]["tf"], []
    for st in D["sites"]:
        lo, hi = max(D["s"], st[0] - D["call"]["tol"]), min(D["e"], st[1] + D["call"]["tol"])
        mx = max(arr[p - D["s"]] for p in range(lo, hi + 1)) / 1000.0
        res.append((st, mx, mx >= D["call"]["thr"]))
    return res


def section(D, a):
    """Fill the <template> section of the snippet file."""
    snip = open(SNIPPET, encoding="utf-8").read()
    m = re.search(r'<template id="locus-section">\n(.*?)</template>', snip, re.S)
    tpl = m.group(1)
    labels = D["runOrder"]
    rows = []
    for i, l in enumerate(labels):
        sw, desc = RUN_STYLE.get(l, ("", l))
        if a.swatches:
            sw = a.swatches.split(",")[i]
        if a.descs:
            desc = a.descs.split("|")[i]
        stp = int(a.row_steps.split(",")[i]) if a.row_steps else (1 if i == 0 else 2)
        rows.append('      <div class="trk-lab" data-step="%d"><span class="rn"><span class="sw %s"></span>%s</span>'
                    '<span class="rd">%s</span></div>\n'
                    '      <canvas data-step="%d" data-trk="post" data-run="%s" data-h="%d"></canvas>\n'
                    % (stp, sw, html.escape(l), html.escape(desc), stp, html.escape(l), a.post_h))
    calls = site_calls(D)
    run_list = ", ".join(labels[:-1]) + " and " + labels[-1] if len(labels) > 1 else labels[0]
    sites_txt = " and ".join("%s–%s (%s %s posterior %.2f)" % (fmt(st[0]), fmt(st[1]), D["call"]["run"], D["tfLabel"], mx)
                             for st, mx, _ in calls)
    fmt_str = ("Occupancy browser tracks, %s:%s–%s. Rows: %s posterior and nucleosome occupancy for runs %s; "
               + ("" if a.meth_h == 0 else "m6A per A on Watson and Crick; ") + "genes %s. %s site%s at %s.")
    aria = fmt_str % (D["chrom"], fmt(D["s"]), fmt(D["e"]), D["tfLabel"], run_list,
                      ", ".join(g["name"] for g in D["genes"]) or "none", D["refLabel"],
                      "" if len(calls) == 1 else "s", sites_txt or "none")
    rossi_leg = ('\n      <span><span class="lk rossi"></span>Rossi %s summit</span>' % D["tfLabel"]) if "rossi" in D else ""
    vals = {
        "ID": a.slide_id, "KIND": a.kind, "SEC": a.sec, "TITLE": html.escape(a.title),
        "TAGATTR": (' data-tag="%s"' % a.tag) if a.tag else "",
        "EYEBROW": a.eyebrow, "MSG": a.msg, "TF": html.escape(D["tfLabel"]),
        "TFCOLOR": D["colors"]["tf"], "REFLABEL": html.escape(D["refLabel"]),
        "ARIA": html.escape(aria, quote=True), "ROWS": "".join(rows),
        "METHLEG": "" if a.meth_h == 0 else (
            '      <span><span class="lk dot" style="background:var(--wat)"></span>m6A Watson</span>\n'
            '      <span><span class="lk dot" style="background:var(--cri)"></span>m6A Crick</span>\n'
            '      <span><span class="lk dash"></span>background rate %.3f</span>\n' % D["bg"]),
        "METHROW": "" if a.meth_h == 0 else (
            '      <div class="trk-lab" data-step="1"><span class="rn">m6A per A</span><span class="rd">same reads in every run</span></div>\n'
            '      <canvas data-step="1" data-trk="meth" data-h="%d"></canvas>\n' % a.meth_h),
        "EXTRA": open(a.extra_html, encoding="utf-8").read() if a.extra_html else "", "DATAID": a.data_id, "FOOT": a.foot, "ROSSILEG": rossi_leg,
        "REFSTYLE": "" if D["colors"]["ref"] == "#d4145a" else ' style="--locus-ref:%s"' % D["colors"]["ref"],
    }
    out = tpl
    for k, v in vals.items():
        out = out.replace("{{%s}}" % k, v)
    assert "{{" not in out, re.findall(r"\{\{\w+\}\}", out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, help="window, chrom:start-end (1-based inclusive)")
    ap.add_argument("--factor", default="Abf1_murphy", help="optable column, e.g. Abf1_murphy, Reb1_badis")
    ap.add_argument("--tf-label", default=None, help="display name (default: factor prefix, upper)")
    ap.add_argument("--tf-color", default=None, help="override the factor line colour (browser colour too pale, e.g. REB1 #d9f3e2)")
    ap.add_argument("--ref-tf", default=None, help="MacIsaac TF name (default: factor prefix, upper)")
    ap.add_argument("--runs", default=None, help="label<TAB>outDir table (viewer_runs_*.tsv)")
    ap.add_argument("--run", action="append", metavar="LABEL=DIR", help="extra/override run")
    ap.add_argument("--labels", default="fib,fib+seq,seq", help="rows, in order")
    ap.add_argument("--fiber-run", default=None, help="run whose m6A counts feed the shared panel (default: first)")
    ap.add_argument("--call-run", default=CALL_RUN, help="run whose posterior labels the sites hit/miss")
    ap.add_argument("--sites", choices=["macisaac_bed", "macisaac_c1"], default="macisaac_bed")
    ap.add_argument("--rossi", default=None, help="Rossi <TF>_CX.bed; summits drawn as ticks")
    ap.add_argument("--from-viewer", default=None, help="built browser HTML to slice instead of decodes")
    ap.add_argument("--viewer-region", default=None, help="region key inside --from-viewer")
    ap.add_argument("--browser-region", default=None, help="region to cite in JSON 'region' (default: source)")
    ap.add_argument("--call-also", default=None, help="comma list of runs whose site max is appended to each chip")
    ap.add_argument("--descs", default=None, help="|-separated row descriptions (default: RUN_STYLE)")
    ap.add_argument("--swatches", default=None, help="comma list of swatch classes per row: fiber,both,seq")
    ap.add_argument("--row-steps", default=None, help="build step per run row, e.g. 1,2,1,2 (default: first 1, rest 2)")
    ap.add_argument("--post-h", type=int, default=100, help="height of each run row (px)")
    ap.add_argument("--meth-h", type=int, default=104, help="height of the m6A panel (px); 0 drops the panel and its legend entries")
    ap.add_argument("--extra-html", default=None, help="file with markup inserted between the tracks and the footnote")
    ap.add_argument("--slide-id", default="5x")
    ap.add_argument("--data-id", default=None, help="id of the JSON <script> (default trk<slide-id>-data)")
    ap.add_argument("--kind", default="hidden", choices=["main", "hidden", "backup"])
    ap.add_argument("--sec", default="2")
    ap.add_argument("--tag", default="", help="data-tag, e.g. PENDING")
    ap.add_argument("--title", default="One locus up close")
    ap.add_argument("--eyebrow", default='<span class="tag hid">Hidden · backup locus</span> A first look at TF calls')
    ap.add_argument("--msg", default="")
    ap.add_argument("--foot", default="")
    ap.add_argument("--from-json", default=None, help="reuse a payload built earlier (only re-fill the section)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-section", default=None)
    a = ap.parse_args()
    a.data_id = a.data_id or "trk%s-data" % a.slide_id
    D = json.load(open(a.from_json, encoding="utf-8")) if a.from_json else build(a)
    if a.call_also:
        D["call"]["also"] = a.call_also.split(",")
    if a.tf_color:
        D["colors"]["tf"] = a.tf_color
    js = json.dumps(D, separators=(",", ":"), ensure_ascii=False)
    open(a.out_json, "w", encoding="utf-8").write(js)
    for st, mx, hit in site_calls(D):
        print("  site %s:%d-%d  %s %s max(+-%d) = %.3f  -> %s" % (D["chrom"], st[0], st[1], D["call"]["run"],
              D["tfLabel"], D["call"]["tol"], mx, "hit" if hit else "miss"))
    print("wrote %s (%.1f KB)" % (a.out_json, len(js) / 1e3))
    if a.out_section:
        sec = section(D, a)
        sec = sec.replace("{{JSON}}", "")
        open(a.out_section, "w", encoding="utf-8").write(
            sec + '<script type="application/json" id="%s">%s</script>\n' % (a.data_id, js.replace("</", "<\\/")))
        print("wrote %s" % a.out_section)


if __name__ == "__main__":
    main()
