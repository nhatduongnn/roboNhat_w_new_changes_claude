#!/usr/bin/env python
"""Before / local prior / raised global prior at ERV46 (2026-11-03, plan let-s-go-with-mad-snazzy-lollipop).

Numbers (compare_localprior/compare_summary.json) and the side-by-side figure
compare_abf1_3way_erv46.png (+ .pdf) in --figdir.

Per-read P(ABF1) at a site = max over the motif interval of the per-molecule Abf1_murphy column, over reads
covering the whole motif (as abf1sweep/analyze_sweep.py). Methylated motif As = site_mixture.load_site counts
(fl = 20; own-strand calls; the driver's read filters), matched by read name. site_mixture's per-read
post_bound: tf_compare/mixture_filtnordna_pb0p02/per_read.json (pooled f) and run_cv_per_read.json (f fitted
with the read's run held out). Read order of the figure: --order-from json (the 'before' run's clustering).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.chdir(ANALYSIS)
import fig_permol_fibers as FP     # noqa: E402  (drawers + loaders)
import site_mixture as SM          # noqa: E402

SITES = {"site1": ("mac_chrI_61165", 61165, 61178), "site2": ("mac_chrI_62659", 62659, 62672)}
CTRL = {"ctrl_gene_body_64001": (64001, 64014), "ctrl_nuc_core_61491": (61491, 61504)}
EXCL = 10
COLORS = {"before": "#2a78d6", "localprior": "#eb6834", "globalprior": "#1baf7a"}   # dataviz slots 1-3
MIX = "/usr/project/xtmp/nd141/tf_compare/mixture_filtnordna_pb0p02"


def site_vals(z, cols, w0, s, e):
    ia = cols.index("Abf1_murphy")
    c = z["per_mol_cov"][:, s - w0:e - w0 + 1].all(1)
    v = z["per_mol"][c, s - w0:e - w0 + 1, ia].max(1).astype(float)
    return c, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="/usr/project/xtmp/nd141/permol_proto_logfull/filt/erv46_5to3_nordna")
    ap.add_argument("--localprior", default="/usr/project/xtmp/nd141/permol_proto_logfull/compare_localprior/localprior")
    ap.add_argument("--globalprior", default="/usr/project/xtmp/nd141/permol_proto_logfull/abf1sweep/x1500")
    ap.add_argument("--order-from", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--out", default="/usr/project/xtmp/nd141/permol_proto_logfull/compare_localprior/compare_summary.json")
    ap.add_argument("--view", default="60901-62900", help="bp range drawn in the heatmap rows")
    a = ap.parse_args()
    runs = dict(before=a.before, localprior=a.localprior, globalprior=a.globalprior)
    Z = {k: np.load(os.path.join(v, "permol.npz")) for k, v in runs.items()}
    names = [str(n) for n in Z["before"]["names"]]
    for k in Z:
        assert [str(n) for n in Z[k]["names"]] == names, "%s: different reads / order" % k
    cols = [str(c) for c in Z["before"]["cols"]]
    w0 = int(Z["before"]["win"][0])
    mix = json.load(open(MIX + "/per_read.json"))
    rcv = json.load(open(MIX + "/run_cv_per_read.json"))
    counts = {}
    for k, (nm, s, e) in SITES.items():
        d = SM.load_site(SM.Site("chrI", s, e, "+", nm), fl=20)
        km, nmo, _, _ = SM._counts(d)
        counts[k] = {n: (int(x), int(y)) for n, x, y in zip(d.names, km, nmo)}
    out = dict(runs=runs, sites={}, controls={}, elsewhere={}, nucleosome_vs_before={})
    scatter = {}
    for k, (nm, s, e) in SITES.items():
        mb = dict(zip(mix[nm]["names"], mix[nm]["post_bound"]))
        mh = {}
        for rn, x in rcv.get(nm, {}).items():
            mh.update(dict(zip(x["names"], x["post_bound"])))
        out["sites"][k] = dict(name=nm, motif=[s, e])
        scatter[k] = {}
        for cond, z in Z.items():
            c, v = site_vals(z, cols, w0, s, e)
            nn = np.array(names)[c]
            km = np.array([counts[k].get(n, (np.nan, np.nan))[0] for n in nn], float)
            okc = ~np.isnan(km)
            pb = np.array([mb.get(n, np.nan) for n in nn], float)
            ph = np.array([mh.get(n, np.nan) for n in nn], float)
            okm, okh = ~np.isnan(pb), ~np.isnan(ph)
            r = dict(n=int(c.sum()), median=float(np.median(v)), max=float(v.max()), min=float(v.min()),
                     frac_gt_0p5=float((v > .5).mean()), n_gt_0p5=int((v > .5).sum()), sd=float(v.std()),
                     n_matched_calls=int(okc.sum()),
                     spearman_vs_meth_motif_As=float(spearmanr(v[okc], km[okc])[0]),
                     median_by_meth={int(m): float(np.median(v[okc][km[okc] == m])) for m in np.unique(km[okc])},
                     n_by_meth={int(m): int((km[okc] == m).sum()) for m in np.unique(km[okc])},
                     n_matched_mixture=int(okm.sum()),
                     spearman_vs_mixture_post=float(spearmanr(v[okm], pb[okm])[0]),
                     mean_abs_diff_vs_mixture_post=float(np.abs(v[okm] - pb[okm]).mean()),
                     call_agreement_vs_mixture_0p5=float(((v[okm] > .5) == (pb[okm] > .5)).mean()),
                     spearman_vs_mixture_post_heldout_run=float(spearmanr(v[okh], ph[okh])[0]) if okh.sum() > 2 else None,
                     mean_abs_diff_vs_mixture_post_heldout_run=float(np.abs(v[okh] - ph[okh]).mean()) if okh.any() else None,
                     call_agreement_vs_mixture_heldout_0p5=float(((v[okh] > .5) == (ph[okh] > .5)).mean()) if okh.any() else None)
            out["sites"][k][cond] = r
            scatter[k][cond] = (km[okc], v[okc])
        # site_mixture itself, on the same reads as 'before'
        c, _ = site_vals(Z["before"], cols, w0, s, e)
        nn = np.array(names)[c]
        pb = np.array([mb.get(n, np.nan) for n in nn], float)
        km = np.array([counts[k].get(n, (np.nan, np.nan))[0] for n in nn], float)
        ok = ~np.isnan(pb) & ~np.isnan(km)
        out["sites"][k]["site_mixture_pooled"] = dict(
            n=int(ok.sum()), median=float(np.median(pb[ok])), max=float(pb[ok].max()),
            frac_gt_0p5=float((pb[ok] > .5).mean()), sd=float(pb[ok].std()),
            spearman_vs_meth_motif_As=float(spearmanr(pb[ok], km[ok])[0]))
        scatter[k]["site_mixture"] = (km[ok], pb[ok])
    for k, (s, e) in CTRL.items():
        out["controls"][k] = {}
        for cond, z in Z.items():
            c, v = site_vals(z, cols, w0, s, e)
            out["controls"][k][cond] = dict(n=int(c.sum()), median=float(np.median(v)), max=float(v.max()),
                                            frac_gt_0p5=float((v > .5).mean()))
    ia, jn = cols.index("Abf1_murphy"), cols.index("nucleosome")
    for cond, z in Z.items():
        cov = z["per_mol_cov"]
        A = z["per_mol"][:, :, ia].astype(float)
        call = (A > .5) & cov
        mask = np.ones(A.shape[1], bool)
        for _, s, e in SITES.values():
            mask[s - EXCL - w0:e + EXCL - w0 + 1] = False
        el = call[:, mask]
        pos = np.arange(w0, w0 + A.shape[1])[mask]
        loci = []
        for p in pos[el.any(0)]:
            if loci and p - loci[-1][1] <= 1:
                loci[-1][1] = int(p)
            else:
                loci.append([int(p), int(p)])
        out["elsewhere"][cond] = dict(
            reads_with_call=int(el.any(1).sum()), call_bp_total=int(el.sum()), n_loci=len(loci),
            loci=[dict(start=s, end=e, n_reads=int(call[:, s - w0:e - w0 + 1].any(1).sum())) for s, e in loci],
            avg_abf1_bp_gt_0p5=int((z["avg_optable"][:, ia] > .5).sum()), avg_abf1_max=float(z["avg_optable"][:, ia].max()))
        if cond != "before":
            z0 = Z["before"]
            pn = np.where(cov, z["per_mol"][:, :, jn], np.nan)
            p0 = np.where(z0["per_mol_cov"], z0["per_mol"][:, :, jn], np.nan)
            out["nucleosome_vs_before"][cond] = dict(
                avg_mean_abs_diff=float(np.abs(z["avg_optable"][:, jn] - z0["avg_optable"][:, jn]).mean()),
                avg_max_abs_diff=float(np.abs(z["avg_optable"][:, jn] - z0["avg_optable"][:, jn]).max()),
                perread_mean_abs_diff=float(np.nanmean(np.abs(pn - p0))),
                perread_max_abs_diff=float(np.nanmax(np.abs(pn - p0))))
    for cond, d in runs.items():
        st = json.load(open(os.path.join(d, "run_stats.json")))
        out.setdefault("posterior_validity", {})[cond] = st.get("posterior_validity")
        out.setdefault("local_prior", {})[cond] = st.get("local_prior")
    # ------------------------------------------------------------------ figure
    P = FP.load_permol(a.before)
    win = P["win"]
    labels, o_shuf, o_clu, cinfo = FP.order_from_json(a.order_from, P["names"])
    out["figure_order"] = dict(order_from=os.path.abspath(a.order_from),
                               names_clustered_md5=__import__("hashlib").md5(
                                   "\n".join(P["names"][i] for i in o_clu).encode()).hexdigest())
    calls, ext0, _ = FP.load_calls("chrI", win, FP.BAM)
    T = {cond: FP.build_table(FP.load_permol(d), calls, ext0, win) for cond, d in runs.items()}
    v0, v1 = (int(x) for x in a.view.split("-"))
    vwin = (v0, v1)
    clu = labels[o_clu]
    sub = {cond: FP.subset(T[cond], o_clu, win, vwin) for cond in T}
    n = len(o_clu)
    oz = np.arange(n)
    sites = FP.macisaac_sites("chrI", win)
    vs = [s for s in sites if s[1] >= v0 and s[0] <= v1]
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 19), dpi=200)
    rows = [("R", "Fiber-seq raw m6A, fibers in the 'before' clustered order (no decode)"),
            ("before", "P(ABF1) BEFORE: current defaults (genome-wide prior 1.8e-7)"),
            ("localprior", "P(ABF1) LOCAL PRIOR: site weight w calibrated to f (each read's own run held out)"),
            ("globalprior", "P(ABF1) RAISED GLOBAL PRIOR: ABF1 x1500 genome-wide (abf1sweep/x1500)")]
    h, gap, top = 0.150, 0.022, 0.965
    boxes = {}
    y = top
    for k, _ in rows:
        y -= h
        boxes[k] = [FP.X0, y, FP.X1 - FP.X0, h]
        y -= gap
    for k, title in rows:
        ax = fig.add_axes(boxes[k])
        S = sub["before"]
        if k == "R":
            FP.draw_raw_fibers(ax, S["starts"], S["ends"], S["dot_rows"], S["dot_pos"], v0, v1, n,
                               tick_fs=FP.TICK_FS, line_lw=0.35, dot_s=3.0)
        else:
            im = FP.draw_posterior(ax, np.clip(sub[k]["abf"], 0, 1), v0, v1, cmap=FP.ABF1_CMAP,
                                   max_bins=10 ** 6, tick_fs=FP.TICK_FS, hide_xticklabels=(k != "globalprior"))
            FP._side_cbar(fig, im, boxes[k], "P(ABF1)", ["0", "1"])
        ax.set_title(title, fontsize=FP.TITLE_FS, loc="left", pad=3)
        FP._separators(ax, clu)
        FP.overlay(ax, vs, None)
        ax.set_xlim(v0, v1)
        if k != "globalprior":
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("chrI position (bp)", fontsize=FP.LAB_FS)
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        sb = fig.add_axes([FP.SB_X0, boxes[k][1], FP.SB_W, boxes[k][3]])
        FP.draw_sidebar(sb, strands=sub["before"]["strands"], clusters=FP._sidebar_clusters(clu), n_rows=n,
                        lab_fs=FP.LAB_FS, tick_fs=FP.TICK_FS)
        sb.set_yticks([]); sb.set_ylabel("Fibers", fontsize=FP.LAB_FS, labelpad=2)
    rng = np.random.default_rng(0)
    yb = y - 0.165
    for j, (k, (nm, s, e)) in enumerate(SITES.items()):
        ax = fig.add_axes([FP.X0 + j * 0.44, yb, 0.38, 0.15])
        off = {"before": -0.27, "localprior": -0.09, "globalprior": 0.09, "site_mixture": 0.27}
        for cond in ("before", "localprior", "globalprior", "site_mixture"):
            km, v = scatter[k][cond]
            ax.scatter(km + off[cond] + rng.uniform(-0.06, 0.06, len(km)), v, s=10,
                       color=COLORS.get(cond, "#7a7a7a"), alpha=0.75, linewidths=0,
                       label={"before": "before", "localprior": "local prior", "globalprior": "global x1500",
                              "site_mixture": "site_mixture post_bound"}[cond])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("methylated As in the motif (own strand)", fontsize=FP.LAB_FS)
        ax.set_ylabel("per-read P(ABF1)", fontsize=FP.LAB_FS)
        ax.set_title("%s %s:%d-%d" % (k, "chrI", s, e), fontsize=FP.TITLE_FS, loc="left")
        ax.tick_params(labelsize=FP.TICK_FS)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if j == 1:
            ax.legend(fontsize=FP.LEG_FS, frameon=False, loc="upper right")
    fig.text(FP.X0, 0.99, "ERV46 chrI:%d-%d, %d fibers, same rows in every panel (order from %s)" % (
        v0, v1, n, os.path.basename(a.order_from)), fontsize=FP.TITLE_FS, va="top")
    os.makedirs(a.figdir, exist_ok=True)
    pre = os.path.join(a.figdir, "compare_abf1_3way_erv46")
    fig.savefig(pre + ".png", dpi=200); fig.savefig(pre + ".pdf"); plt.close(fig)
    out["figure"] = pre + ".png"
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    for k in SITES:
        for cond in ("before", "localprior", "globalprior", "site_mixture_pooled"):
            r = out["sites"][k][cond]
            print("%s %-20s n=%3d median %.4g max %.4g frac>0.5 %.3f sd %.4f rho_meth %.3f %s" % (
                k, cond, r["n"], r["median"], r["max"], r["frac_gt_0p5"], r["sd"], r["spearman_vs_meth_motif_As"],
                "" if cond.startswith("site") else "rho_mix %.3f agree %.3f" % (r["spearman_vs_mixture_post"],
                                                                                 r["call_agreement_vs_mixture_0p5"])))
    print("elsewhere", {c: (v["reads_with_call"], v["n_loci"]) for c, v in out["elsewhere"].items()})
    print("nuc", out["nucleosome_vs_before"])
    print("wrote", a.out, pre + ".png")


if __name__ == "__main__":
    main()
