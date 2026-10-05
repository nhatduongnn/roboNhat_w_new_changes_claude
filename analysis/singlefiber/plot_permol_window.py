#!/usr/bin/env python
"""Figures of the single-fiber decode over the prototype window.

The per-molecule tree deliberately has no plotting (439 molecules would emit thousands of
PNGs), so nothing is drawn at decode time. This reads what the decode already wrote and
draws it in the locus-slide style:

  overview   genes / aggregate decode / molecule-average of the single-fiber decodes /
             per-molecule raw m6A raster / per-molecule nucleosome-posterior raster /
             per-molecule ABF1-posterior raster / pooled m6A / axis
  zoom       the same, +-400 bp around one MacIsaac ABF1 site

Sources, all already on disk:
  <arm>/permol.npz                    per-molecule (439, 5000, 7) factor table + avg
  <arm>/molecules.tsv                 per-molecule spans and m6A counts
  agg_baseline/tmpDir/info_0_1.h5     the aggregate (pooled-binomial) decode, state space
  the BAM                             re-read for the raw calls raster (read_calls.py)

Usage:
  python singlefiber/plot_permol_window.py --arm-dir /usr/project/xtmp/nd141/permol_proto/armA_effoff
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, 'pkgvar/permol_seq_maskoff/')
sys.path.insert(0, HERE)
sys.path.insert(0, ANALYSIS)

import h5py                                                       # noqa: E402
import pickle                                                     # noqa: E402
import collapse as C                                              # noqa: E402
import read_calls as RC                                           # noqa: E402
from robocop.utils.getNucleotides import getNucleotideSequence     # noqa: E402

FASTA = os.path.join(ANALYSIS, "inputs", "SacCer3.fa")
MACISAAC = os.path.join(ANALYSIS, "inputs", "MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed")

GREY = "#9a9a9a"
ABF1C = "#1f9e4a"
MET = "#d1442f"
CAN = "#dcdcdc"


# ---------------------------------------------------------------- inputs

def load_permol(arm_dir):
    z = np.load(os.path.join(arm_dir, "permol.npz"))
    return dict(per_mol=z["per_mol"], cov=z["per_mol_cov"], cols=[str(c) for c in z["cols"]],
                names=[str(n) for n in z["names"]], strands=[str(s) for s in z["strands"]],
                depth=z["cov"], avg=z["avg_optable"], win=tuple(int(v) for v in z["win"]))


def load_aggregate(agg_dir, traindir):
    """Collapse the pooled-binomial decode's state posterior to the same factor columns."""
    with open(os.path.join(traindir, "HMMconfig.pkl"), "rb") as fh:
        hmm = pickle.load(fh)
    dshared = hmm["dshared"] if "dshared" in hmm else hmm
    M = C.collapse_matrix(dshared)
    cols = C.column_names(dshared)
    with h5py.File(os.path.join(agg_dir, "tmpDir", "info_0_1.h5"), "r") as f:
        g = f["segment_0/posterior"]
        n_states = dshared["n_states"]
        post = sparse.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                                 shape=(len(g["indptr"]) - 1, n_states))
    out = post @ M
    out = out.toarray() if sparse.issparse(out) else np.asarray(out)
    return out, cols


def load_calls(chrom, win, bam, flank=400):
    ext0 = max(0, win[0] - 1 - flank)
    ext1 = win[1] + flank
    codes = getNucleotideSequence(FASTA, chrom, ext0 + 1, ext1)
    mols, es, ee, stats = RC.read_molecules(bam, chrom, win[0], win[1], codes, flank=flank)
    by_name = {}
    for m in mols:
        by_name[m.name] = (m.calls, es)
    return by_name


def macisaac_sites(chrom, win, factor="ABF1"):
    """Merged MacIsaac intervals overlapping the window, as (start1, end1, centre)."""
    raw = []
    with open(MACISAAC) as fh:
        for line in fh:
            p = line.split()
            if p[0] != chrom or p[3].upper() != factor:
                continue
            s, e = int(p[1]) + 1, int(p[2])
            if e >= win[0] and s <= win[1]:
                raw.append((s, e))
    raw.sort()
    merged = []
    for s, e in raw:
        if merged and s <= merged[-1][1] + 20:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e, (s + e) // 2) for s, e in merged]


def load_genes(chrom, start, end):
    from make_posterior_viewer import load_genes as lg
    return lg(chrom, start, end)


# ---------------------------------------------------------------- drawing

def order_molecules(names, calls_by_name, win):
    """Sort by aligned start, then by span, so the raster reads as a stack of fibers."""
    key = []
    for i, n in enumerate(names):
        c, es = calls_by_name.get(n, (None, None))
        if c is None:
            key.append((10 ** 9, 0, i))
            continue
        obs = np.nonzero(c >= 0)[0]
        s = es + obs[0] if len(obs) else 10 ** 9
        e = es + obs[-1] if len(obs) else 0
        key.append((s, -(e - s), i))
    return [k[2] for k in sorted(key)]


def calls_raster(names, order, calls_by_name, win):
    """(n_mol, L) int8 raster: -1 no observation, 0 canonical A, 1 methylated A."""
    L = win[1] - win[0] + 1
    R = np.full((len(order), L), -1, dtype=np.int8)
    for row, i in enumerate(order):
        got = calls_by_name.get(names[i])
        if got is None:
            continue
        c, es = got
        a = win[0] - 1 - es
        R[row] = c[a:a + L]
    return R


def shade_sites(ax, sites, win, color="#c62828", alpha=0.12, bar=False):
    for s, e, _ in sites:
        ax.axvspan(max(s, win[0]), min(e, win[1]), color=color, alpha=alpha, lw=0, zorder=0)


def draw(arm_dir, agg_dir, traindir, bam, chrom, out_png, zoom=None, title=None):
    P = load_permol(arm_dir)
    win = P["win"]
    agg, agg_cols = load_aggregate(agg_dir, traindir)
    calls = load_calls(chrom, win, bam)
    order = order_molecules(P["names"], calls, win)
    R = calls_raster(P["names"], order, calls, win)

    x = np.arange(win[0], win[1] + 1)
    sel = slice(0, len(x)) if zoom is None else slice(max(0, zoom[0] - win[0]),
                                                     min(len(x), zoom[1] - win[0] + 1))
    xs = x[sel]
    vw = (xs[0], xs[-1])

    ci = {c: i for i, c in enumerate(P["cols"])}
    pm = P["per_mol"][order][:, sel, :]
    cov = P["cov"][order][:, sel]
    Rz = R[:, sel]

    nuc_pm = pm[:, :, ci["nucleosome"]] + pm[:, :, ci["nuc_center"]]
    abf_pm = pm[:, :, ci["Abf1_murphy"]]

    sites = macisaac_sites(chrom, vw)
    genes = load_genes(chrom, vw[0], vw[1])

    n_mol = pm.shape[0]
    heights = [0.5, 1.15, 1.15, 2.6, 2.6, 2.6, 0.9]
    fig, axes = plt.subplots(7, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw=dict(height_ratios=heights, hspace=0.12,
                                              left=0.075, right=0.985, top=0.945, bottom=0.05))
    ax_g, ax_agg, ax_avg, ax_m6a, ax_nuc, ax_abf, ax_pool = axes

    # --- genes -------------------------------------------------------------------
    ax_g.set_ylim(-1, 1); ax_g.set_yticks([]); ax_g.axis("off")
    for k, g in enumerate(genes):
        name, s, e, strand = g["name"], g["start"], g["end"], g.get("strand", "+")
        y = 0.35 if strand == "+" else -0.35
        ax_g.add_patch(Rectangle((max(s, vw[0]), y - 0.16), min(e, vw[1]) - max(s, vw[0]), 0.32,
                                 color="#6fb7d8" if strand == "+" else "#e08d8d", lw=0))
        ax_g.text((max(s, vw[0]) + min(e, vw[1])) / 2, y + 0.22, name, ha="center",
                  va="bottom", fontsize=8)

    # --- aggregate decode --------------------------------------------------------
    an = agg[sel, agg_cols.index("nucleosome")] + agg[sel, agg_cols.index("nuc_center")]
    aa = agg[sel, agg_cols.index("Abf1_murphy")]
    ax_agg.fill_between(xs, an, color=GREY, lw=0)
    ax_agg.plot(xs, aa, color=ABF1C, lw=1.2)
    ax_agg.set_ylabel("aggregate\n(pooled binomial)", fontsize=8)

    # --- molecule average --------------------------------------------------------
    # average over the molecules that actually cover each position, not over all 439
    ncov = cov.sum(0).astype(float)
    vn = np.where(ncov > 0, np.where(cov, nuc_pm, 0).sum(0) / np.maximum(ncov, 1), np.nan)
    va = np.where(ncov > 0, np.where(cov, abf_pm, 0).sum(0) / np.maximum(ncov, 1), np.nan)
    ax_avg.fill_between(xs, vn, color=GREY, lw=0)
    ax_avg.plot(xs, va, color=ABF1C, lw=1.2)
    ax_avg.set_ylabel("mean of the fibers\ncovering each bp", fontsize=8)

    for ax in (ax_agg, ax_avg):
        ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=7)
        shade_sites(ax, sites, vw)

    ext = [vw[0] - 0.5, vw[1] + 0.5, n_mol, 0]

    # --- raw m6A calls -----------------------------------------------------------
    cmap = ListedColormap(["#ffffff", CAN, MET])
    ax_m6a.imshow(Rz, aspect="auto", interpolation="nearest", extent=ext,
                  cmap=cmap, norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3))
    ax_m6a.set_ylabel("raw m6A calls\n(one row = one fiber)", fontsize=8)

    # --- per-molecule nucleosome posterior ---------------------------------------
    nz = np.where(cov, nuc_pm, np.nan)
    ax_nuc.imshow(nz, aspect="auto", interpolation="nearest", extent=ext,
                  cmap="Greys", vmin=0, vmax=1)
    ax_nuc.set_ylabel("decoded nucleosome\nposterior per fiber", fontsize=8)

    # --- per-molecule ABF1 posterior ---------------------------------------------
    az = np.where(cov, abf_pm, np.nan)
    im = ax_abf.imshow(az, aspect="auto", interpolation="nearest", extent=ext,
                       cmap="Greens", vmin=0, vmax=max(0.02, float(np.nanmax(az))))
    ax_abf.set_ylabel("decoded ABF1\nposterior per fiber", fontsize=8)
    cb = fig.colorbar(im, ax=ax_abf, pad=0.005, fraction=0.02)
    cb.ax.tick_params(labelsize=6)

    for ax in (ax_m6a, ax_nuc, ax_abf):
        ax.set_yticks([0, n_mol]); ax.tick_params(labelsize=7)
        for s, e, _ in sites:
            ax.add_patch(Rectangle((s, 0), e - s, n_mol, fill=False, ec="#c62828",
                                   lw=1.0, zorder=5))

    # --- pooled m6A --------------------------------------------------------------
    obs = (Rz >= 0).sum(0).astype(float)
    met = (Rz == 1).sum(0).astype(float)
    rate = np.where(obs > 0, met / np.maximum(obs, 1), np.nan)
    ax_pool.plot(xs, rate, ".", ms=1.6, color="#2f4fd1")
    ax_pool.set_ylabel("pooled\nm6A / A", fontsize=8)
    ax_pool.set_ylim(0, 1); ax_pool.set_yticks([0, 0.5, 1]); ax_pool.tick_params(labelsize=7)
    shade_sites(ax_pool, sites, vw)
    ax_pool.set_xlabel(chrom, fontsize=9)
    ax_pool.set_xlim(vw[0], vw[1])

    for s, e, c in sites:
        ax_agg.text((s + e) / 2, 1.06, "ABF1 %d-%d" % (s, e), ha="center", fontsize=7,
                    color="#c62828")

    fig.suptitle(title or "%s  %s:%d-%d   %d molecules   arm %s"
                 % (os.path.basename(arm_dir), chrom, vw[0], vw[1], n_mol,
                    os.path.basename(arm_dir)), fontsize=11)
    fig.savefig(out_png, dpi=140)
    plt.close("all")
    print("wrote", out_png)
    return dict(n_mol=n_mol, sites=sites, abf_max=float(np.nanmax(az)),
                nuc_mean=float(np.nanmean(nz)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", default="/usr/project/xtmp/nd141/permol_proto/armA_effoff")
    ap.add_argument("--agg-dir", default="/usr/project/xtmp/nd141/permol_proto/agg_baseline")
    ap.add_argument("--traindir", default=os.path.join(ANALYSIS, "robocop_train_fiberonly"))
    ap.add_argument("--bam", default=("/usr/xtmp/nd141/projects/Fiber_seq/"
                                      "process_nanopore_sequencing/combine_sequencing_runs/"
                                      "merged_Mar20_barcode01_Jun25_barcode21-24_"
                                      "May07_barcode03-04.bam"))
    ap.add_argument("--chrom", default="chrI")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--zoom-pad", type=int, default=400)
    a = ap.parse_args()
    outdir = a.outdir or os.path.join(a.arm_dir, "figures")
    os.makedirs(outdir, exist_ok=True)
    tag = os.path.basename(a.arm_dir.rstrip("/"))

    r = draw(a.arm_dir, a.agg_dir, a.traindir, a.bam, a.chrom,
             os.path.join(outdir, "permol_overview_%s.png" % tag))
    for k, (s, e, c) in enumerate(r["sites"], 1):
        draw(a.arm_dir, a.agg_dir, a.traindir, a.bam, a.chrom,
             os.path.join(outdir, "permol_abf1site%d_%s.png" % (k, tag)),
             zoom=(c - a.zoom_pad, c + a.zoom_pad),
             title="%s  ABF1 site %d  %s:%d-%d" % (tag, k, a.chrom, s, e))


if __name__ == "__main__":
    main()
