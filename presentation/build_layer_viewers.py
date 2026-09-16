#!/usr/bin/env python3
"""Two viewers on the Tuned Occupancy Browser machinery (build_tuned_viewer.py, reused unchanged).

    python build_layer_viewers.py <variant> region-index <i>   # one region payload -> scratch
    python build_layer_viewers.py <variant> emit [--drop 'label,label']

variant
  tempered    Tuned Occupancy Browser + bt02/bt05/bt10 finals (runs: tuned_runs_tempered.tsv)
  sameweights sw01 round-05 weights decoded seq-only / seq+fiber / fiber-only (runs:
              layer_same_weights_runs.tsv), plus per-window fiber veto / added calls from
              analysis/overnight/A_veto_calls.tsv and the pooled table from A_validation.tsv.

Same 42 regions (tuned_regions.tsv), same call rule, same encoding as build_tuned_viewer.py.
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_tuned_viewer as B   # noqa: E402  (chdirs into analysis/)

SCR = "/usr/project/xtmp/nd141/scratch_viewers2"
V = {
    "tempered": dict(runs="tuned_runs_tempered.tsv", template="tuned_occupancy_template_tempered.html",
                     out="tuned_occupancy_viewer_tempered.html", title="Tuned Occupancy Browser",
                     default="both φ1 final",
                     fiber_runs={"fib r0", "fib final", "both r0", "both φ1 final", "both φ2 final",
                                 "both φ5 final", "both φ10 final", "u001 r7 legacy"}),
    "sameweights": dict(runs="layer_same_weights_runs.tsv", template="layer_same_weights_template.html",
                        out="layer_same_weights_viewer.html", title="Same Weights, Three Layers",
                        default="seq + fiber", fiber_runs={"seq + fiber", "fiber only"}),
}
CMP = {"seq + fiber": "both_vs_seqonly", "fiber only": "fib_vs_seqonly"}


def setup(variant):
    cfg = V[variant]
    B.SCRATCH = os.path.join(SCR, variant, "regions")
    runs_path = os.path.join(HERE, cfg["runs"])

    def read_runs():
        out = []
        for line in open(runs_path, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            lab, tune, hold = line.rstrip("\n").split("\t")
            out.append((lab, tune, hold))
        return out
    B.read_runs = read_runs
    return cfg


def veto_payload(key):
    """Fiber veto / added calls (overnight_score.py part A) in the loaded span of one region."""
    spec = {r["key"]: r for r in B.read_regions()}[key]
    chrom, ws, we = B.MPV.parse_region(spec["region"])
    start, end = max(1, ws - B.PAD), we + B.PAD
    which = "holdout" if chrom == "chrIV" else "tune"
    calls, counts = [], {}
    for lab, comp in CMP.items():
        counts[lab] = dict(veto=0, added=0, veto_mac=0, veto_ros=0, added_mac=0, added_ros=0)
    inv = {v: k for k, v in CMP.items()}
    with open("overnight/A_veto_calls.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["set"] != which or r["chrom"] != chrom:
                continue
            p = int(r["center"])
            if not (start <= p <= end):
                continue
            lab = inv[r["comparison"]]
            calls.append([p, r["group"], r["kind"], lab, int(r["macisaac_support"]), int(r["rossi_support"])])
            if ws <= p <= we:
                c = counts[lab]; k = r["kind"]
                c[k] += 1
                c[k + "_mac"] += int(r["macisaac_support"])
                c[k + "_ros"] += int(r["rossi_support"])
    calls.sort()
    return dict(vetoCalls=calls, vetoCounts=counts)


def region(variant, i):
    setup(variant)
    key = B.read_regions()[i]["key"]
    B.build(key)
    if variant == "sameweights":
        p = os.path.join(B.SCRATCH, key + ".json")
        d = json.load(open(p))
        d.update(veto_payload(key))
        json.dump(d, open(p, "w"), separators=(",", ":"))
        print("veto payload:", d["vetoCounts"])


def summary_rows():
    """A_validation.tsv rows, copied verbatim (numbers as written in the file)."""
    rows = []
    with open("overnight/A_validation.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["status"] != "ok" or r["fitted"] not in ("all", "nonfitted"):
                continue
            rows.append({k: r[k] for k in ("config", "set", "chroms", "decode", "ref", "fitted",
                                           "n_groups", "sites", "calls", "matched", "P", "R", "F1")})
    return rows


def emit(variant, drop):
    cfg = setup(variant)
    B.TEMPLATE = os.path.join(HERE, cfg["template"])
    B.OUT = os.path.join(HERE, cfg["out"])
    G, _, fitted = B.tune_w.factor_set()
    motifs = [m for g in G for m in G[g]]
    specs = B.read_regions()
    runs = [r for r in B.read_runs() if r[0] not in drop]
    keep = [r[0] for r in runs]
    regions = {}
    for s in specs:
        r = json.load(open(os.path.join(B.SCRATCH, s["key"] + ".json")))
        assert all(l in r["runOrder"] for l in keep), (s["key"], r["runOrder"])
        r["runOrder"] = keep
        for fld in ("post", "counts", "dirs", "fiberSame"):
            r[fld] = {l: r[fld][l] for l in keep}
        regions[s["key"]] = r
    for k, r in regions.items():
        assert r["runOrder"] == keep and set(r["post"]) == set(keep), k
    factors = motifs + [B.OTHER] + B.SPECIAL_COLS
    colors, labels = {}, {}
    for m in motifs:
        g = [gg for gg in G if m in G[gg]][0]
        labels[m] = g if len(G[g]) == 1 else "%s · %s" % (g, m.split("_", 1)[1])
        colors[m] = B.MPV.to_hex(B.MPV.color_for_name(m.split("_")[0].upper()))
    for f in B.SPECIAL_COLS:
        labels[f], colors[f] = f, B.MPV.SPECIAL[f]
    labels[B.OTHER], colors[B.OTHER] = "other motifs (92 untuned, summed)", "#7d6b91"
    data = dict(
        title=cfg["title"], regionOrder=[s["key"] for s in specs],
        regionLabels={s["key"]: s["label"] for s in specs}, regions=regions,
        factors=factors, colors=colors, labels=labels,
        groupOf={m: [g for g in G if m in G[g]][0] for m in motifs},
        fitted={g: int(v) for g, v in fitted.items()},
        pinned=["nucleosome", "nuc_center", "unknown", B.OTHER], quant=B.QUANT,
        runs=[dict(label=l, tune=t, hold=h) for l, t, h in runs],
        bg={l: 0.1383 for l in keep if l in cfg["fiber_runs"]},
        defaultRun=cfg["default"], dropped=sorted(drop),
        selection=json.load(open(os.path.join(HERE, "tuned_regions_selection.json"))))
    if variant == "sameweights":
        data["summary"] = summary_rows()
    js = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = open(B.TEMPLATE, encoding="utf-8").read().replace("{{DATA_JSON}}", js)
    open(B.OUT, "w", encoding="utf-8").write(html)
    size = os.path.getsize(B.OUT)
    print("page %.2f MB, %d regions x %d runs: %s" % (size / 1e6, len(regions), len(keep), keep))
    assert size < 16 * 1024 * 1024


if __name__ == "__main__":
    variant, cmd = sys.argv[1], sys.argv[2]
    if cmd == "region-index":
        region(variant, int(sys.argv[3]))
    else:
        drop = set()
        if "--drop" in sys.argv:
            drop = {x for x in sys.argv[sys.argv.index("--drop") + 1].split(",") if x}
        emit(variant, drop)
