#!/usr/bin/env python
"""Single-fiber RoboCOP decode drawn in the paper's figure-2 fiber style.

WHY THIS FILE EXISTS
  `plot_permol_window.py` draws the same data as a seven-row stack whose raw-fiber
  raster paints every *canonical* A grey (so the panel is a grey wash, not a dot plot)
  and whose decoded rasters are full-black nucleosome occupancy. The paper's figure 2
  (/usr/xtmp/nd141/projects/Fiber_seq/paper_draft/figure_2/version3) draws fibers the
  other way round: a light backbone line per molecule with a dot only where there IS an
  m6A call, a Blues posterior heatmap, strand + cluster sidebars in their own axes, and
  cluster separators. This module reuses that repo's DRAWERS unchanged and feeds them
  the RoboCOP per-molecule decode.

  Nothing here re-decodes and nothing here writes into `hmm_for_clustering_claude_code`
  or `paper_draft`; both are read-only inputs. `plot_permol_window.py` is untouched.

WHAT IS BORROWED AND WHAT IS NOT
  Imported verbatim from
  /usr/project/xtmp/nd141/projects/Fiber_seq/hmm_for_clustering_claude_code/scripts/fig2_panels.py
      draw_gene_track, draw_raw_fibers, draw_posterior, draw_sidebar,
      draw_locus_overlays, METH_COLOR, STRAND_COLOR, CLUSTER_COLORS, NOISE_COLOR,
      POSTERIOR_CMAP
  Ported (copied, not imported -- they live inside the build script, not the module)
  from paper_draft/figure_2/version3/build_figure2_gal1_steps_v3.py:
      the fixed-geometry frame layout (X0/X1/SB_X0/SB_W/Y0/Y1/LEG_Y), the font sizes,
      `_ruler`, `_common_x`, `_legend`, `_colorbar`, `_sidebar`.
  NOT borrowed: the decode. figure 2's "P(accessible)" is a 4-state per-fiber kmer HSMM
  (n_states=4 in version2/fig2_right_cache.npz; scripts/decode_chrom_save.py) with no
  connection to RoboCOP. What is drawn here is RoboCOP's own 3,485-state per-molecule
  posterior from /usr/project/xtmp/nd141/permol_proto/<arm>/permol.npz.

WHAT THE COLOUR MEANS  (see --help for the one-line version)
  figure 2 plots P(accessible) = posterior(open) + posterior(linker) of a 4-state model
  (scripts/extract_fiber_features.py:205-208). RoboCOP has ONE unbound state, so the
  analogue is the `background` column on its own: P(this bp is free DNA -- no
  nucleosome and none of the 153 motifs). It is taken straight from permol.npz, not as
  1 - protected, because permol.npz keeps only 7 of RoboCOP's 159 collapsed columns:
  over armA_effoff's covered bp the 7 stored columns (dropping the nuc_center column,
  which is a SUBSET of `nucleosome` -- collapse.py:55-56 -- and would double count) sum
  to a mean of 0.9801 and a minimum of 0.2260, i.e. up to 77% of the posterior at a bp
  can sit in the 149 motifs that were not stored. `1 - (nucleosome + unknown + ABF1 +
  REB1)` would therefore over-call accessibility; `background` is exact.

FRAMES (all the same figure size and axes boxes, so they stack/cross-fade)
  step1_raw       gene track + population panel + raw m6A fibers, shuffled order
  step2_decoded   per-fiber P(accessible) = background, SAME order as step1
  step3_clustered P(accessible), clustered order, cluster + strand sidebars
  step4_rawclust  raw m6A in the clustered order, same sidebars  (figure 2's Jraw)
  step5_abf1      per-fiber ABF1 posterior, clustered order, its own colour scale
  composite       gene track + population + raw (shuffled) + decoded (clustered)

CLUSTERING
  Mirrors the reference: feature vector = the per-bp P(accessible) over a window
  centred on an ABF1 site, z-scored per column, PCA to 50 components, UMAP(2D,
  n_neighbors=15, min_dist=0.1, random_state=42), HDBSCAN(min_cluster_size=10)
  -- scripts/extract_fiber_features.py:190-216 + scripts/cluster_fibers.py:61-72
  (UMAP/HDBSCAN) and :270-276 (the PCA(50) pre-step), driven by
  scripts/run_phase4_visual_eval.sbatch:84-106 with the GAL1 row of
  scripts/submit_phase4_visual_eval.sh:29 (anchor 278625, window 125).
  umap/hdbscan/sklearn are NOT in robocop-2024, so that step is run as a subprocess in
  the reference repo's own interpreter (read-only use of `.venv/bin/python`); pass
  --cluster-method ward to use the scipy-only fallback instead.
  RAGGED FIBERS: the reference's cache gives every fiber the full window; ours do not.
  Molecules whose aligned span does not cover the whole clustering window are NOT
  dropped -- they are drawn in their own grey block, labelled and counted in the legend.

Usage:
  python singlefiber/fig_permol_fibers.py --arm-dir /usr/project/xtmp/nd141/permol_proto/armA_effoff
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)

FIGREPO = "/usr/project/xtmp/nd141/projects/Fiber_seq/hmm_for_clustering_claude_code"
FIGVENV = os.path.join(FIGREPO, ".venv", "bin", "python")

# the reference repo is READ-ONLY for us: do not leave .pyc files in its tree
sys.dont_write_bytecode = True
sys.path.insert(0, FIGREPO)
sys.path.insert(0, os.path.join(FIGREPO, "scripts"))
from scripts.fig2_panels import (                                   # noqa: E402
    draw_gene_track, draw_raw_fibers, draw_posterior, draw_sidebar,
    draw_locus_overlays, METH_COLOR, STRAND_COLOR, CLUSTER_COLORS, NOISE_COLOR,
    POSTERIOR_CMAP)

# project paths go in FRONT of the reference repo's, so our modules win
sys.path.insert(0, os.path.join(ANALYSIS, "pkgvar", "permol_seq_maskoff"))
sys.path.insert(0, HERE)
sys.path.insert(0, ANALYSIS)

FASTA = os.path.join(ANALYSIS, "inputs", "SacCer3.fa")
MACISAAC = os.path.join(ANALYSIS, "inputs", "MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed")
BAM = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/"
       "combine_sequencing_runs/"
       "merged_Mar20_barcode01_Jun25_barcode21-24_May07_barcode03-04.bam")

# ---- geometry + fonts, ported from build_figure2_gal1_steps_v3.py --------------
LAB_FS, TICK_FS, TITLE_FS, LEG_FS = 10, 9, 11, 9.5
X0, X1 = 0.085, 0.935          # main panel
SB_X0, SB_W = 0.018, 0.030     # sidebar column
Y0, Y1 = 0.17, 0.935           # full-height panel
LEG_Y = 0.006

ACC_FILL = plt.get_cmap(POSTERIOR_CMAP)(0.80)   # same blue family as the heatmaps
ABF1_COLOR = "#2ca02c"
ABF1_CMAP = "Greens"
MEAN_COLOR = "#8b1a1a"


# =============================================================== data loading
def load_permol(arm_dir):
    z = np.load(os.path.join(arm_dir, "permol.npz"))
    return dict(per_mol=z["per_mol"], cov=z["per_mol_cov"],
                cols=[str(c) for c in z["cols"]],
                names=[str(n) for n in z["names"]],
                strands=[str(s) for s in z["strands"]],
                depth=z["cov"], avg=z["avg_optable"],
                win=(int(z["win"][0]), int(z["win"][1])))


def load_aggregate(agg_dir, traindir):
    """Collapse the pooled-binomial (aggregate) decode to RoboCOP's factor columns."""
    import pickle
    import h5py
    from scipy import sparse
    import collapse as C
    with open(os.path.join(traindir, "HMMconfig.pkl"), "rb") as fh:
        hmm = pickle.load(fh)
    dshared = hmm["dshared"] if "dshared" in hmm else hmm
    M = C.collapse_matrix(dshared)
    cols = C.column_names(dshared)
    with h5py.File(os.path.join(agg_dir, "tmpDir", "info_0_1.h5"), "r") as f:
        g = f["segment_0/posterior"]
        post = sparse.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                                 shape=(len(g["indptr"]) - 1, dshared["n_states"]))
    out = post @ M
    return (out.toarray() if sparse.issparse(out) else np.asarray(out)), cols


def load_calls(chrom, win, bam, flank=400):
    """Per-molecule m6A calls, re-read from the BAM (permol.npz does not store them)."""
    import read_calls as RC
    from robocop.utils.getNucleotides import getNucleotideSequence
    ext0 = max(0, win[0] - 1 - flank)
    ext1 = win[1] + flank
    codes = getNucleotideSequence(FASTA, chrom, ext0 + 1, ext1)
    mols, es, ee, stats = RC.read_molecules(bam, chrom, win[0], win[1], codes,
                                            flank=flank)
    return {m.name: m for m in mols}, es, stats


def macisaac_sites(chrom, win, factor="ABF1", slop=20):
    """Merged MacIsaac intervals overlapping the window, 1-based inclusive."""
    raw = []
    with open(MACISAAC) as fh:
        for line in fh:
            p = line.split()
            if p[0] != chrom or p[3].upper() != factor.upper():
                continue
            s, e = int(p[1]) + 1, int(p[2])
            if e >= win[0] and s <= win[1]:
                raw.append((s, e))
    raw.sort()
    merged = []
    for s, e in raw:
        if merged and s <= merged[-1][1] + slop:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e, (s + e) // 2) for s, e in merged]


# =============================================================== fiber table
def build_table(P, calls_by_name, ext_start0, win):
    """Canonical fiber table: one row per molecule, in permol.npz's own row order.

    starts/ends are the molecule's ALIGNED span (1-based inclusive, clipped to the
    read-calls extent); dot_rows/dot_pos are the methylated A's inside `win`.
    post/abf are (N, W) with NaN where the molecule has no coverage.
    """
    ci = {c: i for i, c in enumerate(P["cols"])}
    cov = P["cov"]
    W = win[1] - win[0] + 1
    post = np.where(cov, P["per_mol"][:, :, ci["background"]], np.nan).astype(np.float32)
    abf = np.where(cov, P["per_mol"][:, :, ci["Abf1_murphy"]], np.nan).astype(np.float32)
    nuc = np.where(cov, P["per_mol"][:, :, ci["nucleosome"]], np.nan).astype(np.float32)

    starts = np.zeros(len(P["names"]), np.int64)
    ends = np.zeros(len(P["names"]), np.int64)
    strands = np.empty(len(P["names"]), dtype=object)
    dot_rows, dot_pos = [], []
    n_missing = 0
    for r, name in enumerate(P["names"]):
        strands[r] = "+" if P["strands"][r] == "watson" else "-"
        m = calls_by_name.get(name)
        if m is None:                      # never seen: fall back to the covered span
            n_missing += 1
            oc = np.nonzero(cov[r])[0]
            starts[r] = win[0] + (oc[0] if len(oc) else 0)
            ends[r] = win[0] + (oc[-1] if len(oc) else 0)
            continue
        starts[r], ends[r] = m.ref_start0 + 1, m.ref_end0
        a = win[0] - 1 - ext_start0
        c = m.calls[a:a + W]
        pos = np.nonzero(c == 1)[0]
        dot_rows.extend([r] * len(pos))
        dot_pos.extend((win[0] + pos).tolist())
    return dict(starts=starts, ends=ends, strands=strands,
                dot_rows=np.asarray(dot_rows, np.int64),
                dot_pos=np.asarray(dot_pos, np.int64),
                post=post, abf=abf, nuc=nuc, cov=cov, n_missing=n_missing)


# =============================================================== clustering
CLUSTER_WORKER = r"""
import sys, numpy as np
d = np.load(sys.argv[1]); X = d['X']
from sklearn.decomposition import PCA
n_comp = min(50, X.shape[1], X.shape[0] - 1)
if X.shape[1] > n_comp:
    X = PCA(n_components=n_comp, random_state=42).fit_transform(X)
import umap, hdbscan
emb = umap.UMAP(n_components=2, n_neighbors=15, random_state=42,
                min_dist=0.1).fit_transform(X)
lab = hdbscan.HDBSCAN(min_cluster_size=int(d['mcs']), min_samples=None).fit_predict(emb)
np.savez(sys.argv[2], labels=np.asarray(lab, np.int64), n_comp=np.int64(n_comp))
"""


def _prepare_matrix(F):
    """z-score columns ignoring NaN, impute the rest with the column mean.

    Same recipe as cluster_fibers.prepare_matrix (scripts/cluster_fibers.py:28-67).
    """
    A = np.asarray(F, dtype=np.float64).copy()
    keep = ~np.all(np.isnan(A), axis=0)
    A = A[:, keep]
    mu = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0) + 1e-9
    idx = np.nonzero(np.isnan(A))
    A[idx] = np.take(mu, idx[1])
    return (A - mu) / sd


def cluster_fibers(post, win, cwin, method="umap_hdbscan", min_cluster_size=10,
                   min_window_cov=1.0, verbose=True):
    """Cluster molecules on their P(accessible) over `cwin` = (start1, end1).

    Returns (labels, info). labels: >=0 a cluster, -1 HDBSCAN noise, -2 the molecule
    does not cover `min_window_cov` of the clustering window (NOT dropped -- drawn in
    its own block).
    """
    lo, hi = cwin[0] - win[0], cwin[1] - win[0] + 1
    F = post[:, lo:hi]
    frac = np.mean(~np.isnan(F), axis=1)
    elig = frac >= min_window_cov
    labels = np.full(post.shape[0], -2, dtype=np.int64)
    info = dict(cluster_window=[int(cwin[0]), int(cwin[1])],
                n_total=int(post.shape[0]), n_eligible=int(elig.sum()),
                n_no_window_coverage=int((~elig).sum()),
                min_window_cov=float(min_window_cov),
                min_cluster_size=int(min_cluster_size))
    if elig.sum() < min_cluster_size * 2:
        info["method"] = "none (too few molecules span the window)"
        return labels, info

    X = _prepare_matrix(F[elig])
    used = method
    if method == "umap_hdbscan":
        try:
            with tempfile.TemporaryDirectory() as td:
                fin = os.path.join(td, "X.npz")
                fout = os.path.join(td, "lab.npz")
                fw = os.path.join(td, "w.py")
                np.savez(fin, X=X, mcs=np.int64(min_cluster_size))
                with open(fw, "w") as fh:
                    fh.write(CLUSTER_WORKER)
                r = subprocess.run([FIGVENV, fw, fin, fout], capture_output=True,
                                   text=True, timeout=1800)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr.strip()[-800:])
                z = np.load(fout)
                lab = z["labels"]
                info["pca_components"] = int(z["n_comp"])
            info["method"] = ("PCA(50) + UMAP(2D, n_neighbors=15, min_dist=0.1, "
                              "random_state=42) + HDBSCAN(min_cluster_size=%d) "
                              "via %s" % (min_cluster_size, FIGVENV))
        except Exception as exc:                                   # noqa: BLE001
            info["umap_hdbscan_error"] = str(exc)
            used = "ward"
            if verbose:
                print("  umap_hdbscan unavailable (%s); falling back to ward" % exc)
    if used == "ward":
        from scipy.cluster.hierarchy import fcluster, linkage
        k = info.get("k", 2)
        lab = fcluster(linkage(X, method="ward"), t=k, criterion="maxclust") - 1
        info["method"] = "scipy ward linkage on the z-scored per-bp track, k=%d" % k

    # relabel so cluster ids are 0..K-1 in descending size, noise stays -1
    lab = np.asarray(lab, dtype=np.int64)
    ids = [c for c in np.unique(lab) if c >= 0]
    ids.sort(key=lambda c: -int((lab == c).sum()))
    remap = {c: i for i, c in enumerate(ids)}
    lab = np.array([remap.get(int(c), -1) for c in lab], dtype=np.int64)
    labels[elig] = lab
    info["sizes"] = {str(int(c)): int((labels == c).sum())
                     for c in np.unique(labels)}
    return labels, info


def subset(T, rows, win, vwin):
    """Restrict the fiber table to `rows` and to the bp range `vwin` (1-based incl.)."""
    lo, hi = vwin[0] - win[0], vwin[1] - win[0] + 1
    rows = np.asarray(rows, dtype=np.int64)
    remap = np.full(len(T["starts"]), -1, dtype=np.int64)
    remap[rows] = np.arange(len(rows))
    m = ((remap[T["dot_rows"]] >= 0) & (T["dot_pos"] >= vwin[0])
         & (T["dot_pos"] <= vwin[1])) if len(T["dot_rows"]) else np.zeros(0, bool)
    return dict(starts=T["starts"][rows], ends=T["ends"][rows],
                strands=T["strands"][rows],
                dot_rows=remap[T["dot_rows"][m]], dot_pos=T["dot_pos"][m],
                post=T["post"][rows][:, lo:hi], abf=T["abf"][rows][:, lo:hi],
                nuc=T["nuc"][rows][:, lo:hi], cov=T["cov"][rows][:, lo:hi],
                n_missing=T["n_missing"])


def order_shuffled(n, seed=0):
    return np.random.default_rng(seed).permutation(n)


def order_clustered(labels, starts):
    """cluster ascending, HDBSCAN noise then no-coverage last, ties by fiber start.

    Same rule as fig2_build_cache.py:158-161 ('cluster asc, noise last, ties by start'),
    extended with the extra -2 block the ragged spans force on us.
    """
    def key(i):
        c = int(labels[i])
        rank = 10_000 if c == -1 else (20_000 if c == -2 else c)
        return (rank, int(starts[i]), i)
    return np.array(sorted(range(len(labels)), key=key), dtype=np.int64)


# =============================================================== frame helpers
# (ported verbatim from build_figure2_gal1_steps_v3.py unless marked)
def _ruler(ax, length=147, right_bp=None):
    from matplotlib.transforms import blended_transform_factory
    x1 = right_bp if right_bp is not None else ax.get_xlim()[1] - 30
    x0 = x1 - length
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    kw = dict(transform=tr, clip_on=False, color="black", lw=1.6)
    ax.plot([x0, x1], [1.022, 1.022], solid_capstyle="butt", **kw)
    ax.plot([x0, x0], [1.008, 1.036], **kw)
    ax.plot([x1, x1], [1.008, 1.036], **kw)
    ax.text((x0 + x1) / 2, 1.045, "%d bp" % length, transform=tr, ha="center",
            va="bottom", fontsize=LAB_FS - 0.5)


def _nice_step(n):
    for s in (10, 25, 50, 100, 200, 500):
        if n / s <= 10:
            return s
    return 1000


def _common_x(ax, chrom, tick_step=None):
    _ruler(ax)
    n = int(round(ax.get_ylim()[1] + 0.5))
    ax.set_yticks(np.arange(0, n, tick_step or _nice_step(n)))
    ax.tick_params(axis="y", left=True, labelleft=True, labelsize=TICK_FS,
                   length=2, pad=1)
    ax.set_xlabel("%s position (bp)" % chrom, fontsize=LAB_FS, labelpad=2)
    ax.tick_params(axis="x", labelsize=TICK_FS, labelbottom=True)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)


def _legend(fig, handles, ncol):
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(X0, LEG_Y),
               ncol=ncol, fontsize=LEG_FS, frameon=False, handlelength=1.4,
               handletextpad=0.5, columnspacing=1.4)


def _colorbar(fig, im, title="P(accessible)", ticklabels=None):
    cax = fig.add_axes([X1 - 0.10, LEG_Y + 0.018, 0.10, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1])
    if ticklabels is not None:
        cb.ax.set_xticklabels(ticklabels)
    cb.ax.tick_params(labelsize=TICK_FS - 1, length=2, pad=1)
    cax.set_title(title, fontsize=LEG_FS - 1, pad=1)


def _sidebar(fig, n, strands, clusters=None, height=None):
    ax = fig.add_axes([SB_X0, Y0, SB_W, height if height is not None
                       else Y1 - Y0 - 0.05])
    draw_sidebar(ax, strands=strands, clusters=clusters, n_rows=n,
                 lab_fs=LAB_FS, tick_fs=TICK_FS)
    ax.set_yticks([])
    ax.set_ylabel("Fibers", fontsize=LAB_FS, labelpad=2)
    return ax


STRAND_H = [Patch(facecolor=STRAND_COLOR["+"], label="Watson (+)"),
            Patch(facecolor=STRAND_COLOR["-"], label="Crick (−)")]


def gene_track(ax, chrom, win):
    """draw_gene_track + a y-limit taken from the patches it actually drew.

    build_figure2_gal1_steps_v3.py hard-codes ylim(-60, 118) for GAL10/GAL1, which
    holds only for a 2-row track; chrI:60001-65000 needs 3 rows (patches span
    -95..98, PTA1's label sits at y=-82) and the hard-coded limit crops PTA1.
    """
    draw_gene_track(ax, region_str="%s:%d-%d" % (chrom, win[0], win[1]))
    for t in ax.texts:
        t.set_fontsize(LAB_FS)
    ys = [b for p in ax.patches
          for b in (p.get_path().get_extents().y0, p.get_path().get_extents().y1)]
    ys += [t.get_position()[1] for t in ax.texts]
    if ys:
        ax.set_ylim(min(ys) - 8, max(ys) + 22)


def locus_handles(sites, cwin):
    h = [Line2D([], [], color="black", ls="--", lw=1, label="MacIsaac ABF1 site")]
    if cwin is not None:
        h.append(Line2D([], [], color="blue", ls="--", lw=1, alpha=0.6,
                        label="clustering window"))
    return h


def overlay(ax, sites, cwin):
    """figure 2's draw_locus_overlays, once per ABF1 site (+ the clustering band)."""
    draw_locus_overlays(ax, tss=None, cluster_window=cwin)
    for (_s, _e, c) in sites:
        draw_locus_overlays(ax, tss=c, cluster_window=None)


def cluster_handles(labels_in_order):
    clu = np.asarray(labels_in_order)
    h = []
    for c in sorted(set(int(v) for v in clu if v >= 0)):
        h.append(Patch(facecolor=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                       label="cluster %d (n=%d)" % (c, int((clu == c).sum()))))
    if (clu == -1).any():
        h.append(Patch(facecolor=NOISE_COLOR,
                       label="unclustered (n=%d)" % int((clu == -1).sum())))
    if (clu == -2).any():
        h.append(Patch(facecolor=NOISE_COLOR,
                       label="no clustering-window coverage (n=%d)"
                             % int((clu == -2).sum())))
    return h


def _separators(ax, clu):
    for b in np.nonzero(np.diff(np.asarray(clu)))[0]:
        ax.axhline(b + 0.5, color="black", lw=0.8, zorder=5)


def _sidebar_clusters(clu):
    """draw_sidebar paints c == -1 grey and anything else from CLUSTER_COLORS; our -2
    block must read grey too, so it is passed as -1 (the legend keeps them apart)."""
    return [(-1 if int(c) < 0 else int(c)) for c in clu]


# =============================================================== frames
def frame_raw(T, a, o, chrom, win, sites, cwin, agg=None, mean_acc=None,
              title=None, clusters=None):
    """figure 2's frame 1 (B+D) / frame 5 (Jraw) depending on `o` and `clusters`."""
    n = len(o)
    fig = plt.figure(figsize=(a.width_in, a.height_in), dpi=a.dpi)
    with_top = agg is not None
    if with_top:
        axG = fig.add_axes([X0, 0.875, X1 - X0, 0.06])
        axB = fig.add_axes([X0, 0.655, X1 - X0, 0.19])
        h = 0.435
        axD = fig.add_axes([X0, Y0, X1 - X0, h])
        gene_track(axG, chrom, win)
        _population(axB, agg, win, mean_acc)
        overlay(axB, sites, cwin)
    else:
        h = Y1 - Y0 - 0.05
        axD = fig.add_axes([X0, Y0, X1 - X0, h])

    rank = np.full(len(T["starts"]), -1, dtype=np.int64)
    rank[o] = np.arange(n)
    keep = rank[T["dot_rows"]] >= 0
    draw_raw_fibers(axD, T["starts"][o], T["ends"][o], rank[T["dot_rows"][keep]],
                    T["dot_pos"][keep], win[0], win[1], n, tick_fs=TICK_FS,
                    line_lw=a.line_lw, dot_s=a.dot_s)
    if clusters is not None:
        _separators(axD, clusters)
    axD.set_title(title, fontsize=TITLE_FS, loc="left", pad=3)
    overlay(axD, sites, cwin)
    _common_x(axD, chrom)
    _sidebar(fig, n, T["strands"][o],
             clusters=_sidebar_clusters(clusters) if clusters is not None else None,
             height=h)
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=METH_COLOR, label="m6A")]
    if clusters is not None:
        handles += cluster_handles(clusters)
    handles += STRAND_H + locus_handles(sites, cwin)
    _legend(fig, handles, 5 if clusters is None else 4)
    return fig


def _population(ax, agg, win, mean_acc=None):
    """The population row. figure 2 puts MNase-seq here -- the ensemble measurement the
    single molecules are to be compared against. Ours is RoboCOP's decode of the POOLED
    modkit pileup over the same window (agg_baseline)."""
    x = np.arange(win[0], win[1] + 1)
    ax.fill_between(x, agg["acc"], color=ACC_FILL, lw=0,
                    label="P(accessible), pooled")
    ax.plot(x, agg["abf"], color=ABF1_COLOR, lw=1.0)
    if mean_acc is not None:
        ax.plot(x, mean_acc, color=MEAN_COLOR, lw=0.9, ls="--")
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("P(accessible)", fontsize=LAB_FS, labelpad=1)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(labelsize=TICK_FS, length=2, pad=1)
    ax.tick_params(axis="x", labelbottom=False)
    ttl = ("RoboCOP on the POOLED Fiber-seq pileup (binomial emission, "
           "one track for the whole population)")
    if mean_acc is not None:
        ttl += "  — dashed: mean over the molecules covering each bp"
    ax.set_title(ttl, fontsize=TITLE_FS, loc="left", pad=3)


def frame_heat(T, a, o, chrom, win, sites, cwin, which="post", clusters=None,
               title=None):
    n = len(o)
    fig = plt.figure(figsize=(a.width_in, a.height_in), dpi=a.dpi)
    ax = fig.add_axes([X0, Y0, X1 - X0, Y1 - Y0 - 0.05])
    if which == "post":
        im = draw_posterior(ax, T["post"][o], win[0], win[1], max_bins=10 ** 6,
                            tick_fs=TICK_FS, hide_xticklabels=False)
        cb_title, cb_ticks = "P(accessible)", None
    else:
        Aim = T["abf"][o]
        vmax = float(np.nanmax(Aim))
        im = draw_posterior(ax, Aim / max(vmax, 1e-30), win[0], win[1],
                            cmap=ABF1_CMAP, max_bins=10 ** 6, tick_fs=TICK_FS,
                            hide_xticklabels=False)
        cb_title = "P(ABF1)"
        cb_ticks = ["0", "%.1e" % vmax]
    if clusters is not None:
        _separators(ax, clusters)
    ax.set_title(title, fontsize=TITLE_FS, loc="left", pad=3)
    overlay(ax, sites, cwin)
    _common_x(ax, chrom)
    _sidebar(fig, n, T["strands"][o],
             clusters=_sidebar_clusters(clusters) if clusters is not None else None)
    handles = (cluster_handles(clusters) if clusters is not None else [])
    handles += STRAND_H + locus_handles(sites, cwin)
    _legend(fig, handles, 4)
    _colorbar(fig, im, cb_title, cb_ticks)
    return fig


def frame_composite(T, a, o_shuf, o_clu, chrom, win, sites, cwin, agg, clusters,
                    mean_acc=None, subtitle=""):
    """figure 2's B/D/J composite: population + raw (shuffled) + decoded (clustered)."""
    fig = plt.figure(figsize=(a.width_in, a.height_in * 1.35), dpi=a.dpi)
    axG = fig.add_axes([X0, 0.895, X1 - X0, 0.070])
    axB = fig.add_axes([X0, 0.768, X1 - X0, 0.110])
    axD = fig.add_axes([X0, 0.452, X1 - X0, 0.278])
    axJ = fig.add_axes([X0, 0.122, X1 - X0, 0.298])
    gene_track(axG, chrom, win)
    _population(axB, agg, win, mean_acc)

    n = len(o_shuf)
    rank = np.full(len(T["starts"]), -1, dtype=np.int64)
    rank[o_shuf] = np.arange(n)
    keep = rank[T["dot_rows"]] >= 0
    draw_raw_fibers(axD, T["starts"][o_shuf], T["ends"][o_shuf],
                    rank[T["dot_rows"][keep]], T["dot_pos"][keep], win[0], win[1], n,
                    tick_fs=TICK_FS, line_lw=a.line_lw, dot_s=a.dot_s)
    axD.set_title("Fiber-seq raw m6A, single molecules (n=%d fibers)" % n,
                  fontsize=TITLE_FS, loc="left", pad=3)
    axD.tick_params(axis="x", labelbottom=False)

    im = draw_posterior(axJ, T["post"][o_clu], win[0], win[1], max_bins=10 ** 6,
                        tick_fs=TICK_FS, hide_xticklabels=False)
    _separators(axJ, clusters)
    axJ.set_title("RoboCOP per-molecule P(accessible) = P(background), fibers "
                  "clustered on the ABF1 window", fontsize=TITLE_FS, loc="left", pad=3)
    axJ.set_xlabel("%s position (bp)" % chrom, fontsize=LAB_FS, labelpad=2)
    axJ.tick_params(axis="x", labelsize=TICK_FS, labelbottom=True)
    axJ.ticklabel_format(axis="x", style="plain", useOffset=False)

    for ax, rows, box in ((axD, T["strands"][o_shuf], [SB_X0, 0.452, SB_W, 0.278]),
                          (axJ, T["strands"][o_clu], [SB_X0, 0.122, SB_W, 0.298])):
        sb = fig.add_axes(box)
        draw_sidebar(sb, strands=rows,
                     clusters=(_sidebar_clusters(clusters) if ax is axJ else None),
                     n_rows=len(rows), lab_fs=LAB_FS, tick_fs=TICK_FS)
        sb.set_yticks([])
        sb.set_ylabel("Fibers", fontsize=LAB_FS, labelpad=2)
    for ax in (axB, axD, axJ):
        overlay(ax, sites, cwin)
    handles = ([Line2D([], [], marker="o", ls="", ms=4, color=METH_COLOR, label="m6A")]
               + cluster_handles(clusters) + STRAND_H + locus_handles(sites, cwin))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(X0, 0.004), ncol=5,
               fontsize=LEG_FS, frameon=False, handlelength=1.4, handletextpad=0.5,
               columnspacing=1.4)
    cax = fig.add_axes([X1 - 0.10, 0.022, 0.10, 0.013])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.tick_params(labelsize=TICK_FS - 1, length=2, pad=1)
    cax.set_title("P(accessible)", fontsize=LEG_FS - 1, pad=1)
    if subtitle:
        fig.text(X0, 0.983, subtitle, fontsize=TITLE_FS, ha="left", va="bottom")
    return fig


# =============================================================== driver
def save(fig, out_prefix, dpi, sizes):
    fig.savefig(out_prefix + ".png", dpi=dpi)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(out_prefix + ".png").size
    sizes[os.path.basename(out_prefix)] = "%dx%d" % (w, h)
    print("wrote %s.png (%dx%d px) + .pdf" % (out_prefix, w, h))


def main():
    ap = argparse.ArgumentParser(
        description=("Fiber figures for a RoboCOP single-molecule decode, drawn with "
                     "paper figure-2's drawers. Colour = P(accessible) := RoboCOP's "
                     "`background` column (free DNA: no nucleosome, no motif)."))
    ap.add_argument("--arm-dir", default="/usr/project/xtmp/nd141/permol_proto/armA_effoff")
    ap.add_argument("--agg-dir", default="/usr/project/xtmp/nd141/permol_proto/agg_baseline")
    ap.add_argument("--traindir", default=os.path.join(ANALYSIS, "robocop_train_fiberonly"))
    ap.add_argument("--bam", default=BAM)
    ap.add_argument("--chrom", default="chrI")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cluster-site", type=int, default=2,
                    help="which MacIsaac ABF1 site in the window to cluster on (1-based)")
    ap.add_argument("--cluster-halfwidth", type=int, default=125,
                    help="reference used anchor +/-125 (run_phase4_visual_eval GAL1)")
    ap.add_argument("--cluster-method", default="umap_hdbscan",
                    choices=["umap_hdbscan", "ward"])
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--min-window-cov", type=float, default=1.0)
    ap.add_argument("--with-molecule-mean", action="store_true",
                    help="overlay the molecule-average P(accessible) on the population row")
    ap.add_argument("--width-in", type=float, default=14.0)
    ap.add_argument("--height-in", type=float, default=8.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--dot-s", type=float, default=3.0)
    ap.add_argument("--line-lw", type=float, default=0.35)
    ap.add_argument("--zoom-halfwidth", type=int, default=1000,
                    help="half-width of the zoom frames around the clustering site; "
                         "0 disables them")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tag = os.path.basename(a.arm_dir.rstrip("/"))
    outdir = a.outdir or os.path.join(a.arm_dir, "figures")
    os.makedirs(outdir, exist_ok=True)

    P = load_permol(a.arm_dir)
    win = P["win"]
    sites = macisaac_sites(a.chrom, win)
    calls, ext0, cstats = load_calls(a.chrom, win, a.bam)
    T = build_table(P, calls, ext0, win)
    aggtab, aggcols = load_aggregate(a.agg_dir, a.traindir)
    agg = dict(acc=aggtab[:, aggcols.index("background")],
               abf=aggtab[:, aggcols.index("Abf1_murphy")],
               nuc=aggtab[:, aggcols.index("nucleosome")])

    site = sites[a.cluster_site - 1]
    cwin = (site[2] - a.cluster_halfwidth, site[2] + a.cluster_halfwidth)
    labels, cinfo = cluster_fibers(T["post"], win, cwin, method=a.cluster_method,
                                   min_cluster_size=a.min_cluster_size,
                                   min_window_cov=a.min_window_cov)
    o_shuf = order_shuffled(len(labels), a.seed)
    o_clu = order_clustered(labels, T["starts"])
    clu = labels[o_clu]

    ncov = T["cov"].sum(0).astype(float)
    mean_acc = (np.nansum(np.where(T["cov"], T["post"], 0.0), axis=0)
                / np.maximum(ncov, 1)) if a.with_molecule_mean else None

    n = len(labels)
    sizes = {}
    save(frame_raw(T, a, o_shuf, a.chrom, win, sites, cwin, agg=agg,
                   mean_acc=mean_acc,
                   title="Fiber-seq raw m6A, single molecules (n=%d fibers, shuffled "
                         "order)" % n),
         os.path.join(outdir, "permol_step1_raw_%s" % tag), a.dpi, sizes)
    save(frame_heat(T, a, o_shuf, a.chrom, win, sites, cwin, which="post",
                    title="RoboCOP per-molecule P(accessible) = P(background), same "
                          "fibers and order as the raw m6A"),
         os.path.join(outdir, "permol_step2_decoded_%s" % tag), a.dpi, sizes)
    save(frame_heat(T, a, o_clu, a.chrom, win, sites, cwin, which="post", clusters=clu,
                    title="RoboCOP per-molecule P(accessible), fibers clustered on "
                          "%s:%d-%d" % (a.chrom, cwin[0], cwin[1])),
         os.path.join(outdir, "permol_step3_clustered_%s" % tag), a.dpi, sizes)
    save(frame_raw(T, a, o_clu, a.chrom, win, sites, cwin, clusters=clu,
                   title="Raw m6A, same fibers in the clustered order (no decode)"),
         os.path.join(outdir, "permol_step4_rawclustered_%s" % tag), a.dpi, sizes)
    save(frame_heat(T, a, o_clu, a.chrom, win, sites, cwin, which="abf", clusters=clu,
                    title="RoboCOP per-molecule ABF1 posterior, clustered order "
                          "(note the colour-bar scale)"),
         os.path.join(outdir, "permol_step5_abf1_%s" % tag), a.dpi, sizes)
    save(frame_composite(T, a, o_shuf, o_clu, a.chrom, win, sites, cwin, agg, clu,
                         mean_acc=mean_acc,
                         subtitle="%s  %s:%d-%d  %d molecules" % (tag, a.chrom, win[0],
                                                                  win[1], n)),
         os.path.join(outdir, "permol_composite_%s" % tag), a.dpi, sizes)

    # ---- zoom frames: only the molecules that span the clustering window, over a
    # ---- +/- zoom-halfwidth view, i.e. the reference's --require-window-coverage
    # ---- geometry (every drawn row is a clustered row, no grey block).
    zoom = {}
    rows = np.asarray([i for i in o_clu if labels[i] > -2], dtype=np.int64)
    if len(rows) >= 4 and a.zoom_halfwidth > 0:
        vwin = (max(win[0], site[2] - a.zoom_halfwidth),
                min(win[1], site[2] + a.zoom_halfwidth))
        Z = subset(T, rows, win, vwin)
        zclu = labels[rows]
        zsites = [s for s in sites if s[1] >= vwin[0] and s[0] <= vwin[1]]
        oz = np.arange(len(rows))
        zagg = dict(acc=agg["acc"][vwin[0] - win[0]:vwin[1] - win[0] + 1],
                    abf=agg["abf"][vwin[0] - win[0]:vwin[1] - win[0] + 1],
                    nuc=agg["nuc"][vwin[0] - win[0]:vwin[1] - win[0] + 1])
        zmean = (mean_acc[vwin[0] - win[0]:vwin[1] - win[0] + 1]
                 if mean_acc is not None else None)
        save(frame_heat(Z, a, oz, a.chrom, vwin, zsites, cwin, which="post",
                        clusters=zclu,
                        title="RoboCOP per-molecule P(accessible), the %d molecules "
                              "that span %s:%d-%d, clustered"
                              % (len(rows), a.chrom, cwin[0], cwin[1])),
             os.path.join(outdir, "permol_zoom1_clustered_%s" % tag), a.dpi, sizes)
        save(frame_raw(Z, a, oz, a.chrom, vwin, zsites, cwin, clusters=zclu,
                       title="Raw m6A, same %d molecules in the clustered order "
                             "(no decode)" % len(rows)),
             os.path.join(outdir, "permol_zoom2_rawclustered_%s" % tag), a.dpi, sizes)
        save(frame_heat(Z, a, oz, a.chrom, vwin, zsites, cwin, which="abf",
                        clusters=zclu,
                        title="RoboCOP per-molecule ABF1 posterior, same molecules "
                              "and order (note the colour-bar scale)"),
             os.path.join(outdir, "permol_zoom3_abf1_%s" % tag), a.dpi, sizes)
        save(frame_raw(Z, a, oz, a.chrom, vwin, zsites, cwin, agg=zagg, mean_acc=zmean,
                       clusters=zclu,
                       title="Raw m6A, clustered order (n=%d molecules)" % len(rows)),
             os.path.join(outdir, "permol_zoom0_raw_pop_%s" % tag), a.dpi, sizes)
        zoom = dict(view=[int(vwin[0]), int(vwin[1])], n_rows=int(len(rows)))

    meta = dict(arm=tag, arm_dir=a.arm_dir, window=[win[0], win[1]], chrom=a.chrom,
                n_molecules=n, macisaac_abf1_sites=[[s, e] for s, e, _ in sites],
                clustering=cinfo, zoom=zoom, shuffle_seed=a.seed,
                accessible_definition="permol.npz column 'background'",
                abf1_max_posterior=float(np.nanmax(T["abf"])),
                molecules_not_matched_in_bam=int(T["n_missing"]),
                figure_px=sizes, drawers_from=os.path.join(FIGREPO, "scripts",
                                                           "fig2_panels.py"))
    with open(os.path.join(outdir, "fig_permol_fibers_%s.json" % tag), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(cinfo, indent=1))
    print("ABF1 max posterior over the window: %.6g" % meta["abf1_max_posterior"])


if __name__ == "__main__":
    main()
