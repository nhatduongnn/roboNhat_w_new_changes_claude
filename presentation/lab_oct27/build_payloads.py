"""Build the four before/after locus payloads for the Oct 27 lab deck, and the numbers on their captions.

    cd analysis && python ../presentation/lab_oct27/build_payloads.py

BEFORE = the original decoder (librobocop_prenorm.so) as the old Run Browser drew it.
AFTER  = the log-space decoder, exact emission (robocop_tw3_<m>_logfull, librobocop_logfull.so).
Both sides are read the SAME way, the Run Browser way: <decode>/factor_tables/part_*.npz (clipped per
segment to [0, 1] by write_factor_table.py), overlapping segments averaged equally per position --
analysis/viewer_site/extract_run.py load_parts + stitch, imported read-only (the same functions
presentation/lab_oct6/viewer_reader.py re-implements and posterior_fix/fibercop_audit/logcompare/compare_old_log.py imports).

Per location writes locus_<key>.json (the drawLocus payload, 6 rows) and numbers.json (all caption
numbers). Also re-computes compare_old_log.py's diff_ge_0p1 for the four 30 kb viewer windows that hold
these locations and checks it against old_vs_log_windows.tsv (cross-check of the reader).
m6A counts: score_robocop.region_fiber_counts on the bo09 log-space decode (identical reads in every run).
Genes: make_posterior_viewer.load_genes. Read-only on every decode.
"""
import json, os, pickle, sys
import numpy as np, pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
A = os.path.abspath(os.path.join(H, "..", "..", "analysis"))
sys.path.insert(0, os.path.join(A, "viewer_site")); sys.path.insert(0, A)
os.chdir(A)
from extract_run import load_parts, stitch          # noqa: E402  the Run Browser reader
import build_run_matrix as BRM                        # noqa: E402  live (KEEP) motif sets per tree
import score_robocop as S                             # noqa: E402
import make_posterior_viewer as V                     # noqa: E402
sys.path.insert(0, os.path.join(H, "..", "templates", "locus_slide"))
import build_locus_slide as BLS                       # noqa: E402  load_c1_intervals only

MODELS = [("bo09", "bo09_48"), ("bu01", "bu01_11"), ("bv75", "bv75_49")]
OLD = {"chrXIV": "robocop_chrXIV_chrII_tw_%s", "chrII": "robocop_chrXIV_chrII_tw_%s", "chrIV": "robocop_chrIV_tw_%s"}
NEW = "robocop_tw3_%s_logfull"
NON_MOTIF = {"background", "unknown", "nuc_padding", "nucleosome", "nuc_center", "nuc_start", "nuc_end"}
LOCS = [  # key, chrom, start, end (1-based inclusive), forced factor (None = choose from data), 30 kb viewer window
    ("rpc19", "chrXIV", 412400, 412900, "Abf1_murphy", (400001, 430000)),
    ("xiv440", "chrXIV", 440001, 445000, None, (436001, 466000)),
    ("xiv724", "chrXIV", 724001, 727000, None, (715001, 745000)),
    ("iv1200", "chrIV", 1200001, 1202000, None, (1190001, 1220000)),
]
CM = pickle.load(open(os.path.join(A, "robocop_chrI_fib", "dbf_color_map.pkl"), "rb"))
_parts = {}


def parts(d, chrom):
    if (d, chrom) not in _parts:
        segs, cols, _ = load_parts(d, chrom)
        _parts[(d, chrom)] = (segs, cols)
    return _parts[(d, chrom)]


def read(d, chrom, s, e):
    segs, cols = parts(d, chrom)
    v, cov = stitch(segs, len(cols), s, e)
    assert cov.all(), "%s does not cover %s:%d-%d" % (d, chrom, s, e)
    return pd.DataFrame(v, columns=cols)


def keep_set(d):
    ri = json.load(open(os.path.join(A, d, "RUN_INFO.json")))
    k = BRM.tree_attrs(ri["pkg_tree"]).get("keep")
    return set(k) if k else None   # None = every motif live


def spans(mask, s):
    idx = np.flatnonzero(mask)
    if idx.size == 0: return []
    br = np.flatnonzero(np.diff(idx) > 1)
    st = np.concatenate(([idx[0]], idx[br + 1])); en = np.concatenate((idx[br], [idx[-1]]))
    return [[int(a) + s, int(b) + s] for a, b in zip(st, en)]


EXT = pd.concat([pd.read_csv(os.path.join(A, "posterior_fix/fibercop_audit/extent", f), sep="\t")
                 for f in ("rowsums_bo09_bu01.tsv", "rowsums_bv75_49.tsv")])


def broken_rows(d, chrom, s, e):
    """Old-decode rows with emitting row sum > 1 + 1e-6 (extent/scan_rowsums.py: first_bad..last_bad per segment)."""
    m = np.zeros(e - s + 1, bool); info = []
    for _, r in EXT[(EXT.dir == d) & (EXT.chrom == chrom) & (EXT.n_bad > 0) & (EXT.end >= s) & (EXT.start <= e)].iterrows():
        a, b = int(r.start) + int(r.first_bad), int(r.start) + int(r.last_bad)
        info.append(dict(segment=int(r.segment), rows=[a, b], n_bad=int(r.n_bad), rs_max=float(r.rs_max)))
        lo, hi = max(a, s), min(b, e)
        if lo <= hi: m[lo - s:hi - s + 1] = True
    return m, info


def color(f):
    if f in V.SPECIAL: return V.SPECIAL[f]
    disp = f.split("_")[0].upper()
    return V.to_hex(CM[disp]) if disp in CM else V.to_hex(V.color_for_name(disp))


def crosscheck():
    """compare_old_log.py's diff_ge_0p1 (both row classes summed) on the 4 viewer windows vs its TSV."""
    tsv = pd.read_csv(os.path.join(A, "posterior_fix/fibercop_audit/logcompare/old_vs_log_windows.tsv"), sep="\t")
    out = []
    for key, chrom, s, e, _, (ws, we) in LOCS:
        for lab, m in MODELS:
            o, n = read(OLD[chrom] % m, chrom, ws, we), read(NEW % m, chrom, ws, we)
            cols = [c for c in o.columns if c in n.columns]
            mine = int((np.abs(o[cols].values - n[cols].values).max(1) >= 0.1).sum())
            ref = int(tsv[(tsv.model == lab) & (tsv.window == "%s:%d-%d" % (chrom, ws, we))].diff_ge_0p1.sum())
            out.append(dict(window="%s:%d-%d" % (chrom, ws, we), model=lab, mine=mine, tsv=ref, match=mine == ref))
            print("  crosscheck %-22s %s  mine %4d  tsv %4d  %s" % (out[-1]["window"], lab, mine, ref, "OK" if mine == ref else "MISMATCH"))
    return out


def main():
    numbers = {"crosscheck": crosscheck(), "loci": {}}
    for key, chrom, s, e, forced, _ in LOCS:
        n = e - s + 1
        print("== %s %s:%d-%d" % (key, chrom, s, e))
        tabs, N = {}, {}
        motif_rank = {}
        for lab, m in MODELS:
            od, nd = OLD[chrom] % m, NEW % m
            o, nw = read(od, chrom, s, e), read(nd, chrom, s, e)
            cols = [c for c in o.columns if c in nw.columns]
            D = np.abs(o[cols].values - nw[cols].values)
            dmax = D.max(1)
            keep = keep_set(nd)
            live = [c for c in cols if c not in NON_MOTIF and (keep is None or c in keep)]
            per = {c: dict(n_ge_0p1=int((D[:, cols.index(c)] >= 0.1).sum()), max=float(D[:, cols.index(c)].max())) for c in cols}
            for c in live:
                r = motif_rank.setdefault(c, [0, 0.0]); r[0] += per[c]["n_ge_0p1"]; r[1] = max(r[1], per[c]["max"])
            # old segments (with their row-sum class) covering the window
            segs_o = [(a, b) for a, b, v, _, _ in parts(od, chrom)[0] if not (b < s or a > e)]
            N[lab] = dict(before=od, after=nd, n_cols_compared=len(cols), live_motifs=len(live),
                          any_ge_0p1=int((dmax >= 0.1).sum()), any_ge_0p5=int((dmax >= 0.5).sum()),
                          max_diff=float(dmax.max()), spans_ge_0p1=spans(dmax >= 0.1, s),
                          nuc_ge_0p1=per["nucleosome"]["n_ge_0p1"], bg_ge_0p1=per["background"]["n_ge_0p1"],
                          old_segments=segs_o,
                          old_broken=broken_rows(od, chrom, s, e)[1],
                          ge_0p1_in_broken=int(((dmax >= 0.1) & broken_rows(od, chrom, s, e)[0]).sum()),
                          ge_0p1_outside_broken=int(((dmax >= 0.1) & ~broken_rows(od, chrom, s, e)[0]).sum()),
                          spans_outside_broken=spans((dmax >= 0.1) & ~broken_rows(od, chrom, s, e)[0], s),
                          nuc_spans_ge_0p1=spans(D[:, cols.index("nucleosome")] >= 0.1, s),
                          top_cols=sorted(([c, per[c]["n_ge_0p1"], round(per[c]["max"], 4)] for c in cols if per[c]["n_ge_0p1"]),
                                          key=lambda x: -x[1])[:8])
            tabs[lab] = (o, nw)
            print("  %s any>=0.1 %d  >=0.5 %d  max %.4f  spans %s" % (lab, N[lab]["any_ge_0p1"], N[lab]["any_ge_0p5"],
                  N[lab]["max_diff"], N[lab]["spans_ge_0p1"][:6]))
            print("     top columns", N[lab]["top_cols"])
            print("     old broken rows", N[lab]["old_broken"], " >=0.1 in/outside broken", N[lab]["ge_0p1_in_broken"], N[lab]["ge_0p1_outside_broken"], N[lab]["spans_outside_broken"][:8])
        ranked = sorted(motif_rank.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))
        print("  live-motif ranking (positions >= 0.1 summed over models, max):", ranked[:6])
        factor = forced or (ranked[0][0] if ranked and ranked[0][1][0] > 0 else "background")
        print("  factor:", factor)
        # payload
        runs, order = {}, []
        for lab, _ in MODELS:
            o, nw = tabs[lab]
            for side, t in (("before", o), ("after", nw)):
                L = "%s %s" % (lab, side); order.append(L)
                runs[L] = {"tf": [int(round(x * 1000)) for x in t[factor].values],
                           "nuc": [int(round(x * 1000)) for x in t["nucleosome"].values]}
            N[lab]["factor_ge_0p1"] = int((np.abs(o[factor].values - nw[factor].values) >= 0.1).sum())
            N[lab]["factor_max_before"] = float(o[factor].max()); N[lab]["factor_max_after"] = float(nw[factor].max())
            N[lab]["nuc_mean_before"] = float(o["nucleosome"].mean()); N[lab]["nuc_mean_after"] = float(nw["nucleosome"].mean())
        dec = S.load_decode(NEW % "bo09_48")
        fb = S.region_fiber_counts(dec, chrom, s, e)
        fiber = {k2: [round(float(x), 2) for x in fb[k]] for k2, k in
                 (("mw", "meth_watson"), ("mc", "meth_crick"), ("aw", "A_watson"), ("ac", "A_crick"))}
        ref_tf = factor.split("_")[0].upper()
        # MacIsaac c1 sites (119-TF gff, merged by interval union with 20 bp slop), as on the Oct 6 RPC19 slide
        sites = BLS.load_c1_intervals(chrom, ref_tf, s, e) if factor not in NON_MOTIF else []
        print("  MacIsaac c1 %s sites:" % ref_tf, sites, " colour", color(factor))
        P = {"chrom": chrom, "s": s, "e": e, "region": [s, e], "factor": factor,
             "tfLabel": factor.split("_")[0].upper() if factor not in NON_MOTIF else factor,
             "refLabel": "MacIsaac " + ref_tf, "runOrder": order, "runs": runs,
             "fiberRun": "bo09 after", "fiber": fiber, "sites": sites,
             "genes": V.load_genes(chrom, s, e), "bg": 0.24893964,
             "colors": {"tf": color(factor), "nuc": V.SPECIAL["nucleosome"], "ref": "#d4145a"},
             "call": {"run": "bo09 after", "thr": 0.10, "tol": 20},
             "reader": "Run Browser reader on both sides: factor_tables (clipped per segment), overlapping segments averaged"}
        for L in order: assert len(P["runs"][L]["tf"]) == n
        if key == "rpc19":
            i = 412655 - s
            for L in order:
                print("  412,655 %s: %s %.3f" % (L, factor, runs[L]["tf"][i] / 1000))
            site_max = {}
            for st in sites:
                lo, hi = st[0] - 20 - s, st[1] + 20 - s
                for L in order: site_max.setdefault("%d-%d" % (st[0], st[1]), {})[L] = max(runs[L]["tf"][max(0, lo):hi + 1]) / 1000
            N["at_412655"] = {L: runs[L]["tf"][i] / 1000 for L in order}
            N["site_max_pm20"] = site_max
            print("  site maxima", site_max)
        numbers["loci"][key] = dict(chrom=chrom, s=s, e=e, factor=factor, ranking=[[c, r[0], round(r[1], 4)] for c, r in ranked[:8]], models=N)
        json.dump(P, open(os.path.join(H, "locus_%s.json" % key), "w"), separators=(",", ":"))
        print("  wrote locus_%s.json" % key)
    json.dump(numbers, open(os.path.join(H, "numbers.json"), "w"), indent=1)
    print("wrote numbers.json")


if __name__ == "__main__":
    main()
