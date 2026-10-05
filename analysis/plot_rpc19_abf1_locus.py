"""Locus slide (house style, see presentation/templates/locus_slide/README.md) for the RPC19
promoter/terminator ABF1 calls in bo09 vs bu01.

Why a new script: the template builder emits an HTML/canvas <section> for the deck. This draws
the same component with matplotlib so the result is a PNG that can be read back, AND it carries
one thing the template cannot: bo09's segment 307 posterior is scaled by a constant 1.027e13
over chrXIV:412001-413407, so those positions are either re-derived from the overlapping clean
segment 306 (412001-413000) or marked INVALID (413001-413408). Colours, the grey nucleosome
area, the m6A Watson/Crick panel and the hit/miss rule (>=0.10 within +-20 bp of the site
interval) follow the template exactly.

    python plot_rpc19_abf1_locus.py            # -> figures/rpc19_abf1_locus_bo09_bu01.png
"""
import os, sys, argparse, contextlib
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import score_robocop as S
from Bio import SeqIO

TF_COLOR = "#17ef13"      # dbf_color_map.pkl ABF1
NUC_COLOR = "#b3b3b3"     # make_posterior_viewer.SPECIAL['nucleosome']
REF_COLOR = "#d4145a"     # MacIsaac ABF1
REB1_COLOR = "#0d7a8c"
WAT, CRI = "#1f77b4", "#ff7f0e"
GPLUS, GMINUS = "#87ceeb", "#f08080"
INK, MUTED, RULE = "#111111", "#6b6a66", "#d8d7d3"
BAD = "#f3d9d9"

RUNS = [("bo09_48", "robocop_chrXIV_chrII_tw_bo09_48", "fit9 + bg-open; 9 live TFs, unknown masked; ABF1 tf_prob 1.4e-51"),
        ("bu01_11", "robocop_chrXIV_chrII_tw_bu01_11", "all153 + clc 0.08; 154 live states; ABF1 tf_prob 3.37e-05")]
FACTOR = "Abf1_murphy"
SEG_TOL = 5e-2            # a segment position is used only if its state posterior sums to ~1

# references (1-based inclusive)
MAC_ABF1 = [(412615, 412627)]                   # MacIsaac_p005_c1_V64_SGD.gff3, chrXIV Name=ABF1
MAC_REB1 = [(412600, 412607)]
ROSSI_CX = [412620, 413359, 413414]             # rossi_strand/Abf1_CX.bed, chr14 (bed start+1)
GENES = [("YNL114C", 412684, 413055, "-"), ("RPC19 (YNL113W)", 412771, 413199, "+"),
         ("DBP2 (YNL112W)", 413639, 416281, "+")]
BG_W, BG_C = 0.24893964, 0.26467511             # inputs/bg_params_open.pkl
ABF1_W, ABF1_C = 0.084308, 0.079658             # mean of inputs/all_TFs_1000pealVal_params_pseudo.pkl Abf1


def region(dec, chrom, lo, hi):
    """Per-factor posterior + fiber counts, averaging only over numerically valid segments."""
    dsh, coords = dec["dshared"], dec["coords"]
    n = hi - lo + 1
    ptab = np.zeros((n, dsh["n_states"])); cnt = np.zeros(n)
    fib = {k: np.zeros(n) for k in ("meth_watson", "meth_crick", "A_watson", "A_crick")}
    fcnt = np.zeros(n); nt = np.full(n, -1, dtype=np.int64)
    for inf in dec["infofiles"]:
        with h5py.File(inf, "r") as f:
            for idx in S._seg_idxs(coords, chrom, lo, hi):
                k = "segment_%d" % idx
                if k not in f.keys():
                    continue
                dp = S._get_sparse_todense(f, k + "/posterior")
                ss, se = int(coords.loc[idx]["start"]), int(coords.loc[idx]["end"])
                ds, de = max(0, lo - ss), min(hi - ss + 1, se - ss + 1)
                ps = max(0, ss - lo); pe = ps + de - ds
                good = np.abs(dp[ds:de, :].sum(1) - 1.0) <= SEG_TOL
                sel = np.where(good)[0]
                ptab[ps + sel] += dp[ds + sel, :]; cnt[ps + sel] += 1
                nt[ps:pe] = S._get_sparse_todense(f, k + "/nucleotides")[ds:de]
                for w in fib:
                    fib[w][ps:pe] += S._get_sparse_todense(f, "%s/Fiber_count_%s" % (k, w))[ds:de]
                fcnt[ps:pe] += 1
    cov = cnt > 0
    ptab[cov] /= cnt[cov, None]
    for w in fib:
        fib[w][fcnt > 0] /= fcnt[fcnt > 0]
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        opt = S.get_posterior_binding_probability_df(dsh, ptab)
    cols = list(opt.columns); v = opt.values
    return dict(tf=v[:, cols.index(FACTOR)].astype(float),
                nuc=v[:, cols.index("nucleosome")].astype(float),
                cov=cov, nt=nt, **fib)


def bad_spans(cov, pos):
    out, i = [], 0
    while i < len(cov):
        if not cov[i]:
            j = i
            while j + 1 < len(cov) and not cov[j + 1]:
                j += 1
            out.append((int(pos[i]), int(pos[j]))); i = j + 1
        else:
            i += 1
    return out


def site_max(tr, cov, pos, a, b, tol=20):
    m = (pos >= a - tol) & (pos <= b + tol) & cov
    return float(tr[m].max()) if m.any() else float("nan")


def homopolymers(seq, off, minlen=8):
    import re
    return [(mo.group()[0], off + mo.start(), off + mo.end() - 1)
            for mo in re.finditer(r"A{%d,}|T{%d,}" % (minlen, minlen), seq)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="chrXIV:412300-414000")
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "rpc19_abf1_locus_bo09_bu01.png"))
    a = ap.parse_args()
    chrom, span = a.region.split(":"); lo, hi = (int(x) for x in span.split("-"))
    pos = np.arange(lo, hi + 1)
    gen = {r.id: str(r.seq).upper() for r in SeqIO.parse(os.path.join(HERE, "inputs", "SacCer3.fa"), "fasta")}[chrom]
    seq = gen[lo - 1:hi]

    D = {}
    for lab, d, _ in RUNS:
        D[lab] = region(S.load_decode(os.path.join(HERE, d)), chrom, lo, hi)
        print("%s: %d/%d positions valid; invalid spans %s"
              % (lab, D[lab]["cov"].sum(), len(pos), bad_spans(D[lab]["cov"], pos)))

    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(6, 1, left=0.165, right=0.985, top=0.885, bottom=0.095,
                          height_ratios=[0.22, 1.0, 1.0, 1.05, 0.26, 0.40], hspace=0.20)
    ax_chip = fig.add_subplot(gs[0])
    ax_bo, ax_bu, ax_m, ax_h, ax_g = (fig.add_subplot(gs[i]) for i in range(1, 6))
    axes = [ax_chip, ax_bo, ax_bu, ax_m, ax_h, ax_g]

    for ax, (lab, _, desc) in zip((ax_bo, ax_bu), RUNS):
        d = D[lab]; cov = d["cov"]
        tf = np.where(cov, d["tf"], np.nan); nuc = np.where(cov, d["nuc"], np.nan)
        ax.fill_between(pos, 0, nuc, color=NUC_COLOR, lw=0, zorder=1)
        ax.plot(pos, tf, color=TF_COLOR, lw=1.8, zorder=3)
        for (s0, s1) in bad_spans(cov, pos):
            ax.add_patch(Rectangle((s0, 0), s1 - s0 + 1, 1.05, color=BAD, lw=0, zorder=0))
            ax.text((s0 + s1) / 2, 0.38, "posterior invalid here\n(segment 307 scaled by 1.027e13)",
                    ha="center", va="center", fontsize=8, color="#9b3b3b", zorder=4)
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1.0])
        ax.axhline(0.10, color=MUTED, lw=0.8, ls=":", zorder=2)
        ax.set_ylabel("ABF1 posterior\nover nucleosome", fontsize=9, color=INK)
        ax.text(-0.158, 0.86, lab, transform=ax.transAxes, fontsize=11, color=INK,
                ha="left", va="center", weight="bold")
        ax.text(-0.158, 0.52, desc.replace("; ", "\n"), transform=ax.transAxes, fontsize=7.4,
                color=MUTED, ha="left", va="top")

    d = D["bo09_48"]
    isA = d["nt"] == 0; isT = d["nt"] == 3
    with np.errstate(divide="ignore", invalid="ignore"):
        fw = np.where(isA & (d["A_watson"] > 0), d["meth_watson"] / np.maximum(d["A_watson"], 1e-9), np.nan)
        fc = np.where(isT & (d["A_crick"] > 0), d["meth_crick"] / np.maximum(d["A_crick"], 1e-9), np.nan)
    ax_m.plot(pos, fw, ".", ms=2.8, color=WAT)
    ax_m.plot(pos, fc, ".", ms=2.8, color=CRI)
    ax_m.axhline(BG_W, color="#146b63", lw=1.0, ls="--")
    ax_m.axhline(ABF1_W, color="#0f9a0c", lw=1.0, ls="--")
    ax_m.text(lo + 8, BG_W + 0.02, "background model p = 0.249", fontsize=7.5, color="#146b63",
              bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax_m.text(lo + 8, ABF1_W + 0.02, "ABF1 model p = 0.084", fontsize=7.5, color="#0f9a0c",
              bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax_m.set_ylim(0, 0.62); ax_m.set_yticks([0, 0.25, 0.5])
    ax_m.set_ylabel("m6A fraction per A", fontsize=9, color=INK)
    ax_m.text(-0.158, 0.86, "m6A per A", transform=ax_m.transAxes, fontsize=11, color=INK,
              ha="left", va="center", weight="bold")
    ax_m.text(-0.158, 0.52, "Watson (blue) / Crick (orange)\nsame reads in every run",
              transform=ax_m.transAxes, fontsize=7.4, color=MUTED, ha="left", va="top")

    for base, s0, s1 in homopolymers(seq, lo, 8):
        ax_h.add_patch(Rectangle((s0, 0.10), s1 - s0 + 1, 0.55,
                                 color="#7b4fa8" if base == "A" else "#b06ccb", lw=0))
        ax_h.text((s0 + s1) / 2, 0.72, "%s%d" % (base, s1 - s0 + 1), fontsize=7.5,
                  ha="center", va="bottom", color="#5c3580")
    ax_h.set_ylim(0, 1.35); ax_h.set_yticks([])
    ax_h.text(-0.158, 0.60, "poly-A / poly-T (run \u2265 8 bp)", transform=ax_h.transAxes,
              fontsize=8.5, color=INK, ha="left", va="center")

    for name, gs_, ge, st in GENES:
        y = 0.58 if st == "+" else 0.10
        ax_g.add_patch(Rectangle((max(gs_, lo), y), min(ge, hi) - max(gs_, lo), 0.22,
                                 color=GPLUS if st == "+" else GMINUS, lw=0))
        ax_g.text((max(gs_, lo) + min(ge, hi)) / 2, y + 0.26, "%s %s" % (name, st),
                  fontsize=8.5, ha="center", va="bottom", color=INK)
    ax_g.set_ylim(0, 1.10); ax_g.set_yticks([])
    ax_g.text(-0.158, 0.50, "genes", transform=ax_g.transAxes, fontsize=8.5, color=INK,
              ha="left", va="center")
    ax_g.set_xlabel("%s (bp)" % chrom, fontsize=9, color=INK)

    callrun = "bo09_48"
    ax_chip.set_ylim(0, 1); ax_chip.set_yticks([]); ax_chip.axis("off")
    for (sa, sb), col, nm in ([(x, REF_COLOR, "MacIsaac ABF1") for x in MAC_ABF1]
                              + [(x, REB1_COLOR, "MacIsaac REB1") for x in MAC_REB1]):
        for ax in (ax_bo, ax_bu, ax_m):
            ax.axvspan(sa, sb, color=col, alpha=0.16, lw=0, zorder=0)
        for ax in (ax_bo, ax_bu, ax_m):
            ax.plot([sa, sb], [ax.get_ylim()[1]] * 2, color=col, lw=2.6,
                    solid_capstyle="butt", clip_on=False, zorder=6)
        if nm.endswith("ABF1"):
            mx = {l: site_max(D[l]["tf"], D[l]["cov"], pos, sa, sb) for l, _, _ in RUNS}
            chip = ("MacIsaac ABF1  \u00b7  %s\u2013%s  \u00b7  bo09_48 %.2f %s  \u00b7  bu01_11 %.2f %s"
                    % (f"{sa:,}", f"{sb:,}", mx["bo09_48"], "hit" if mx["bo09_48"] >= 0.10 else "miss",
                       mx["bu01_11"], "hit" if mx["bu01_11"] >= 0.10 else "miss"))
            ax_chip.text((sa + sb) / 2, 0.45, chip, ha="center", va="center", fontsize=9, color=INK,
                         bbox=dict(boxstyle="round,pad=0.35", fc="#fdeef3", ec=REF_COLOR, lw=1.0))
    ax_chip.set_xlim(lo, hi)
    for x in ROSSI_CX:
        if lo <= x <= hi:
            for ax in (ax_bo, ax_bu):
                ax.plot([x], [1.05], marker="v", ms=7, color="#333333", clip_on=False, zorder=7)
    lab_y = {412620: 1.14, 413359: 1.14, 413414: 1.26}
    for x in ROSSI_CX:
        if lo <= x <= hi:
            ax_bo.text(x, lab_y.get(x, 1.14), "Rossi %s" % f"{x:,}", fontsize=7.2,
                       ha="center", color="#333333", clip_on=False)

    CALLS = [(412619, 412632, "A upstream\nACAAGTGATAGAAA\n10/14 informative bp", 412560, 0.72),
             (413417, 413431, "B downstream-a\nTGATTGAAAAATTT\n12/14 informative bp", 413120, 0.88),
             (413461, 413474, "C downstream-b\nTTTTTTTTTTTTTT\n14/14 informative bp", 413690, 0.40)]
    for (ca, cb, txt, tx, ty) in CALLS:
        ax_bo.annotate(txt, xy=((ca + cb) / 2, 0.98), xytext=(tx, ty), fontsize=7.6,
                       ha="center", va="center", color="#0f7a0c", zorder=9,
                       arrowprops=dict(arrowstyle="-|>", color="#0f7a0c", lw=1.2),
                       bbox=dict(boxstyle="round,pad=0.3", fc="#f2fdf1", ec="#0f7a0c", lw=0.7))

    for ax in axes:
        ax.set_xlim(lo, hi)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(RULE); ax.spines["bottom"].set_color(RULE)
        ax.tick_params(labelsize=8, colors=MUTED, length=3)
        if ax is not ax_g:
            ax.set_xticklabels([])
    for ax in (ax_h, ax_g):
        ax.spines["left"].set_visible(False)
    ax_g.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{int(v):,}"))

    fig.suptitle("ABF1 at the RPC19 locus \u2014 %s:%s\u2013%s" % (chrom, f"{lo:,}", f"{hi:,}"),
                 x=0.045, ha="left", y=0.975, fontsize=16, color=INK)
    fig.text(0.045, 0.935,
             "Rows: Abf1_murphy posterior (green) over nucleosome occupancy (grey), 0\u20131; dotted line = the 0.10 call threshold.",
             fontsize=9.5, color=MUTED, ha="left")
    fig.text(0.045, 0.040,
             "Sites: MacIsaac p005_c1 ABF1 412,615\u2013412,627 (pink band) and REB1 412,600\u2013412,607 (teal band); triangles = Rossi Abf1_CX summits.",
             fontsize=7.8, color=MUTED, ha="left")
    fig.text(0.045, 0.019,
             "hit = call-run posterior \u2265 0.10 within \u00b120 bp of the site interval. bo09 chrXIV:413,001\u2013413,408 is masked: segment 307's state posterior sums to 1.027e13 there and no clean overlapping segment covers it.",
             fontsize=7.8, color=MUTED, ha="left")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, facecolor="white")
    plt.close("all")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
