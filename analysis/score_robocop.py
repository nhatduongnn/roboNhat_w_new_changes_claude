"""
Quantitative scoring of a RoboCOP decode against yeast ground truth.

Purpose
-------
Give an OBJECTIVE, repeatable number for "how good is this decode" so that model
changes (emission tweaks, EM on/off, adding the sequence/MNase layers, removing the
ABF1 mask, ...) can be compared instead of eyeballing plots. Run it on any outDir that
has a decoded tmpDir/info*.h5 (produced by run_robocop_without_em, or with_em if the
tmpDir was kept).

What it scores
--------------
1. NUCLEOSOME dyad accuracy         vs inputs/Chereji_2018_+1_-1_nucs.bed
      - median dyad-distance error, % of reference dyads matched within +/-20 / +/-30 bp
      - precision / recall / F1 of predicted dyads
2. NUCLEOSOME phasing periodicity   (internal) -- autocorrelation period of nuc occupancy
      biological expectation ~= 160-170 bp
3. ABF1 footprint accuracy          vs inputs/MacIsaac_..._Abf1_Reb1_match_PWM.bed (ABF1 rows)
      - precision / recall / F1 of predicted ABF1 footprints within tolerance
      - occupancy enrichment at reference sites vs genome-wide, and AUROC
4. ACCESSIBILITY consistency        (internal, no external ref) -- correlation between
      (1 - nucleosome occupancy) and the Fiber-seq meth/A ratio. High = the model is
      putting nucleosomes where the DNA is protected, as it should.

Only (1)-(3) need external references; (4) is a self-contained sanity check on whether
the Fiber-seq signal is being used correctly.

The "have we maxed out Fiber-seq?" workflow
-------------------------------------------
* Run this on a decode -> get numbers.
* Change the model / add a layer -> re-run -> compare the JSON reports.
* COVERAGE SATURATION: subsample the bam to 25/50/75/100%, decode each, score each,
  and plot any metric vs coverage. A plateau => model-limited (tune the model);
  still climbing => data-limited (can't do better without more fibers).

Usage
-----
    python score_robocop.py <outDir> [--regions regions.tsv] [--label NAME]
                            [--tol-nuc 20] [--tol-abf1 20] [--out report.json]

    <outDir>       a RoboCOP output dir (e.g. robocop_all_fiber/). Must contain
                   config.ini, coords.tsv, and tmpDir/info*.h5.
    --regions      optional TSV (chr<TAB>start<TAB>end, with header) of regions to score.
                   Default: the decoded regions in <outDir>/coords.tsv, merged so
                   overlapping windows are not double-counted.
    --label        name for this run in the printed/JSON report (default: outDir basename).

Reference bed files are auto-located under analysis/inputs/ (override with env vars
ROBOCOP_CHEREJI_BED / ROBOCOP_ABF1_BED).
"""
import os
import sys
import glob
import json
import argparse
import configparser

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import h5py
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pkg"))
from robocop import get_posterior_binding_probability_df  # noqa: E402

try:
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover
    find_peaks = None

# ---- column names in the optable (see sum_for_dbf_probs in robocop.py) ----
ABF1_COL = "Abf1_murphy"
NUC_DYAD_COL = "nuc_center"     # posterior mass on the dyad-center states
NUC_OCC_COL = "nucleosome"      # posterior occupancy across the nucleosome body

DEFAULT_CHEREJI = os.environ.get(
    "ROBOCOP_CHEREJI_BED", os.path.join(_HERE, "inputs", "Chereji_2018_+1_-1_nucs.bed"))
DEFAULT_ABF1 = os.environ.get(
    "ROBOCOP_ABF1_BED",
    os.path.join(_HERE, "inputs", "MacIsaac_sacCer3_liftOver_Abf1_Reb1_match_PWM.bed"))
DEFAULT_BROGAARD = os.environ.get(
    "ROBOCOP_BROGAARD_TSV",
    "/usr/xtmp/nd141/projects/Fiber_seq/hmm_for_clustering_claude_code/"
    "output/genome_4st_10iter_flip/overlap/brogaard_top2000_dyads.tsv")


# ----------------------------------------------------------------------------
# Loading the decode
# ----------------------------------------------------------------------------
def _get_sparse_todense(f, k):
    g = f[k]
    v = np.array(sparse.csr_matrix(
        (g['data'][:], g['indices'][:], g['indptr'][:]), g.attrs['shape']).todense())
    return v[0] if v.shape[0] == 1 else v


def _resolve_train_dir(outDir, train_dir):
    """config trainDir is relative to the analysis dir; try sensible bases."""
    cands = [
        train_dir,
        os.path.join(os.path.dirname(outDir.rstrip("/")), train_dir),
        os.path.join(outDir, train_dir),
        os.path.join(_HERE, train_dir),
    ]
    for c in cands:
        if os.path.isfile(os.path.join(c, "HMMconfig.pkl")):
            return c
    raise FileNotFoundError(
        "Could not find HMMconfig.pkl for trainDir=%r near %s" % (train_dir, outDir))


def load_decode(outDir):
    outDir = outDir if outDir.endswith("/") else outDir + "/"
    config = configparser.ConfigParser()
    config.read(outDir + "config.ini")
    train_dir = _resolve_train_dir(outDir, config.get("main", "trainDir"))
    import pickle
    dshared = pickle.load(open(os.path.join(train_dir, "HMMconfig.pkl"), "rb"))
    dshared["info_file"] = None  # get_posterior_binding_probability_df only assigns it
    coords = pd.read_csv(outDir + "coords.tsv", sep="\t")
    infofiles = glob.glob(outDir + "tmpDir/info*.h5")
    if not infofiles:
        raise FileNotFoundError("No tmpDir/info*.h5 under %s (was the decode kept?)" % outDir)
    tech2 = config.get("main", "tech2", fallback=None)
    return dict(outDir=outDir, config=config, dshared=dshared, coords=coords,
                infofiles=infofiles, tech2=tech2)


def _seg_idxs(coords, chrm, start, end):
    c = coords[(coords['chr'] == chrm) & (start <= coords['end']) & (end >= coords['start'])]
    return list(c.index)


def region_optable(dec, chrm, start, end):
    """Assemble per-factor posterior (positions x factors) for a region.

    Returns (optable_df, covered_mask, fiber_ratio) where fiber_ratio is the
    combined Watson+Crick meth/A ratio per position (NaN where no A coverage), or
    None if Fiber-seq datasets are absent.
    """
    dshared, coords = dec["dshared"], dec["coords"]
    n = end - start + 1
    ptable = np.zeros((n, dshared["n_states"]))
    counts = np.zeros(n)
    meth = np.zeros(n)
    acov = np.zeros(n)
    have_fiber = dec["tech2"] is not None
    idxs = _seg_idxs(coords, chrm, start, end)
    for infofile in dec["infofiles"]:
        f = h5py.File(infofile, "r")
        for idx in idxs:
            k = "segment_" + str(idx)
            if k not in f.keys():
                continue
            dp = _get_sparse_todense(f, k + "/posterior")
            seg_start = coords.loc[idx]["start"]
            seg_end = coords.loc[idx]["end"]
            ds = max(0, start - seg_start)
            de = min(end - seg_start + 1, seg_end - seg_start + 1)
            ps = max(0, seg_start - start)
            pe = ps + de - ds
            ptable[ps:pe] += dp[ds:de, :]
            counts[ps:pe] += 1
            if have_fiber:
                try:
                    t = dec["tech2"]
                    mw = _get_sparse_todense(f, "%s/%s_count_meth_watson" % (k, t))
                    mc = _get_sparse_todense(f, "%s/%s_count_meth_crick" % (k, t))
                    aw = _get_sparse_todense(f, "%s/%s_count_A_watson" % (k, t))
                    ac = _get_sparse_todense(f, "%s/%s_count_A_crick" % (k, t))
                    meth[ps:pe] += mw[ds:de] + mc[ds:de]
                    acov[ps:pe] += aw[ds:de] + ac[ds:de]
                except Exception:
                    have_fiber = False
        f.close()

    covered = counts > 0
    ptable[covered] /= counts[covered, np.newaxis]
    import contextlib
    with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
        optable = get_posterior_binding_probability_df(dshared, ptable)  # silence its debug prints

    fiber_ratio = None
    if have_fiber:
        with np.errstate(divide="ignore", invalid="ignore"):
            fiber_ratio = np.where(acov > 0, meth / acov, np.nan)
    return optable, covered, fiber_ratio


def region_fiber_counts(dec, chrm, start, end):
    """Per-strand Fiber-seq counts for a region.

    region_optable() collapses Watson+Crick into one meth/A ratio, which cannot drive a
    per-strand track. This returns the four count arrays separately, stitched and
    averaged over overlapping segments exactly the way region_optable does.

    Returns dict with keys meth_watson, meth_crick, A_watson, A_crick (float arrays of
    length end-start+1), or None if the decode carries no Fiber-seq datasets.
    """
    if dec["tech2"] is None:
        return None
    coords = dec["coords"]
    tech = dec["tech2"]
    n = end - start + 1
    keys = ("meth_watson", "meth_crick", "A_watson", "A_crick")
    out = {k: np.zeros(n) for k in keys}
    counts = np.zeros(n)
    idxs = _seg_idxs(coords, chrm, start, end)
    for infofile in dec["infofiles"]:
        f = h5py.File(infofile, "r")
        for idx in idxs:
            k = "segment_" + str(idx)
            if k not in f.keys():
                continue
            try:
                arrs = [_get_sparse_todense(f, "%s/%s_count_%s" % (k, tech, w))
                        for w in keys]
            except Exception:
                continue
            seg_start = coords.loc[idx]["start"]
            seg_end = coords.loc[idx]["end"]
            ds = max(0, start - seg_start)
            de = min(end - seg_start + 1, seg_end - seg_start + 1)
            ps = max(0, seg_start - start)
            pe = ps + de - ds
            for name, a in zip(keys, arrs):
                out[name][ps:pe] += a[ds:de]
            counts[ps:pe] += 1
        f.close()
    m = counts > 0
    for name in keys:
        out[name][m] /= counts[m]
    return out


# ----------------------------------------------------------------------------
# Reference beds
# ----------------------------------------------------------------------------
def load_chereji(path):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                     names=["chr", "start", "end"])
    df["dyad"] = ((df["start"] + df["end"]) // 2).astype(int)
    return df


def load_brogaard(path):
    """Brogaard top-2000 well-positioned dyads. TSV with header: chrom<TAB>dyad."""
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"chrom": "chr"})
    df["dyad"] = df["dyad"].astype(int)
    return df[["chr", "dyad"]]


def load_abf1(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["chr", "start", "end", "name", "score", "strand"])
    df = df[df["name"].str.upper() == "ABF1"].copy()
    df["center"] = ((df["start"] + df["end"]) // 2).astype(int)
    return df


# ----------------------------------------------------------------------------
# Peak calling + matching
# ----------------------------------------------------------------------------
def call_peaks(track, height, distance):
    track = np.nan_to_num(np.asarray(track, dtype=float))
    if find_peaks is not None:
        pk, _ = find_peaks(track, height=height, distance=distance)
        return list(pk)
    # fallback: naive local maxima with spacing
    pk = []
    for i in range(1, len(track) - 1):
        if track[i] >= track[i - 1] and track[i] > track[i + 1] and track[i] >= height:
            if not pk or i - pk[-1] >= distance:
                pk.append(i)
    return pk


def _above_threshold_runs(track, height, min_len=1):
    """Low-level: index spans (i, j) of every contiguous run where track >= height.
    Single source of truth for the footprint/peak logic used everywhere below --
    footprint_centers, call_abf1 and the scorer all derive from this one scan."""
    track = np.nan_to_num(np.asarray(track, dtype=float))
    above = track >= height
    runs, i, n = [], 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            if (j - i + 1) >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def footprint_centers(track, height, min_len=1):
    """Return the CENTER index of each contiguous run where track >= height.

    A TF footprint is an interval, not a point. When the posterior saturates to a
    flat plateau across the motif, a single-peak call (find_peaks/argmax) is
    degenerate -- the chosen base is decided by float64 round-off and shifts with
    the scoring window. Anchoring on the footprint center instead is stable
    (window/ULP independent) and lands on the motif center, which is what a
    midpoint-to-midpoint comparison against MacIsaac should use."""
    return [(i + j) // 2 for (i, j) in _above_threshold_runs(track, height, min_len)]


def abf1_call_threshold(gmax):
    """Scorer's ABF1 call threshold: 0.30 x (whole-chrI ABF1 max), floored at 0.10.
    `gmax` is the global (per-chromosome) posterior max; on a whole-chrI region this
    equals np.nanmax(track)."""
    return max(0.10, 0.30 * gmax) if gmax and gmax > 0 else 0.10


def call_abf1(track, pos, threshold, min_len=1):
    """THE ABF1 peak-caller -- single source of truth for score() and the locus plot.

    Returns the scorer's ABF1 calls over `track` (genomic coords `pos`): the CENTER of
    each contiguous run where the posterior >= threshold, plus the run extent. Anything
    that wants "what would the scorer call here" invokes this; change it and both the
    recall numbers and the locus plot's arrows move together.
    Returns list of dict(center, start, end) in genomic coordinates."""
    pos = np.asarray(pos)
    return [dict(center=int(pos[(i + j) // 2]), start=int(pos[i]), end=int(pos[j]))
            for (i, j) in _above_threshold_runs(track, threshold, min_len)]


def match_peaks(pred, ref, tol):
    """Greedy nearest-neighbour matching. Returns dict of match stats.

    pred, ref are lists of genomic coordinates. A ref is 'recovered' if some unused
    pred lies within tol. Precision counts preds matched to a ref.
    """
    ref = sorted(ref)
    pred = sorted(pred)
    used = [False] * len(pred)
    dists = []
    tp = 0
    for r in ref:
        best, bd = None, tol + 1
        for pi, p in enumerate(pred):
            if used[pi]:
                continue
            d = abs(p - r)
            if d < bd:
                bd, best = d, pi
        if best is not None and bd <= tol:
            used[best] = True
            tp += 1
            dists.append(bd)
    n_ref, n_pred = len(ref), len(pred)
    fp = n_pred - tp   # predicted peaks not matched to any reference site
    fn = n_ref - tp    # reference sites with no predicted peak nearby
    recall = tp / n_ref if n_ref else float("nan")
    precision = tp / n_pred if n_pred else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) > 0 else 0.0)
    return dict(n_ref=n_ref, n_pred=n_pred, tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1,
                median_dist=float(np.median(dists)) if dists else float("nan"),
                mean_dist=float(np.mean(dists)) if dists else float("nan"))


def auroc(scores, labels):
    """Mann-Whitney AUROC; no sklearn dependency."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = ranks[labels].sum()
    return float((sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg))


def dominant_period(track, lo=120, hi=220):
    x = np.nan_to_num(np.asarray(track, dtype=float))
    x = x - x.mean()
    if np.allclose(x, 0) or len(x) <= hi:
        return None
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] == 0:
        return None
    ac = ac / ac[0]
    seg = ac[lo:hi + 1]
    if len(seg) == 0 or seg.max() <= 0:
        return None
    return int(np.argmax(seg) + lo)


# ----------------------------------------------------------------------------
# Region merging
# ----------------------------------------------------------------------------
def merge_regions(coords):
    out = []
    for chrm, g in coords.sort_values(["chr", "start"]).groupby("chr"):
        cs, ce = None, None
        for _, r in g.iterrows():
            if cs is None:
                cs, ce = r["start"], r["end"]
            elif r["start"] <= ce + 1:
                ce = max(ce, r["end"])
            else:
                out.append((chrm, int(cs), int(ce)))
                cs, ce = r["start"], r["end"]
        if cs is not None:
            out.append((chrm, int(cs), int(ce)))
    return out


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score(outDir, regions=None, tol_nuc=20, tol_abf1=20, label=None,
          chereji_path=DEFAULT_CHEREJI, abf1_path=DEFAULT_ABF1,
          brogaard_path=DEFAULT_BROGAARD,
          abf1_global_max=None, return_abf1_tracks=False):
    """Score a decode. `abf1_global_max`: use this (per-run whole-chrI) posterior max
    for the ABF1 call threshold instead of the scored region's max -- lets a caller
    score a small window while keeping the scorer's real (global) threshold. When None
    and regions is the default whole-chrI merge, the region max IS the global max, so
    behaviour is unchanged. `return_abf1_tracks`: attach the ABF1 track + threshold +
    calls per region to result["_per_region"] so a plot can draw exactly what was
    called."""
    dec = load_decode(outDir)
    label = label or os.path.basename(outDir.rstrip("/"))
    if regions is None:
        regions = merge_regions(dec["coords"])
    chereji = load_chereji(chereji_path) if os.path.isfile(chereji_path) else None
    abf1 = load_abf1(abf1_path) if os.path.isfile(abf1_path) else None
    brogaard = load_brogaard(brogaard_path) if os.path.isfile(brogaard_path) else None

    # accumulators
    nuc_pred, nuc_ref = [], []
    brog_ref = []
    abf1_pred, abf1_ref = [], []
    periods = []
    abf1_scores_all, abf1_labels_all = [], []
    access_corrs = []
    per_region = []

    for (chrm, start, end) in regions:
        optable, covered, fiber_ratio = region_optable(dec, chrm, start, end)
        if covered.sum() == 0:
            continue
        pos = np.arange(start, end + 1)

        # --- nucleosome dyads ---
        dyad_track = optable[NUC_DYAD_COL].values
        occ_track = optable[NUC_OCC_COL].values
        dh = max(0.02, 0.20 * np.nanmax(dyad_track)) if np.nanmax(dyad_track) > 0 else 0.02
        pk = call_peaks(dyad_track, height=dh, distance=120)
        pred_dyads = [int(pos[i]) for i in pk]
        nuc_pred += pred_dyads

        reg_nuc = None
        if chereji is not None:
            ref = chereji[(chereji["chr"] == chrm) &
                          (chereji["dyad"] >= start) & (chereji["dyad"] <= end)]["dyad"].tolist()
            nuc_ref += ref
            if ref:
                reg_nuc = match_peaks(pred_dyads, ref, tol_nuc)

        # --- nucleosome dyads vs Brogaard top-2000 (same predictions, 2nd reference) ---
        if brogaard is not None:
            bref = brogaard[(brogaard["chr"] == chrm) &
                            (brogaard["dyad"] >= start) & (brogaard["dyad"] <= end)]["dyad"].tolist()
            brog_ref += bref

        # --- phasing ---
        p = dominant_period(occ_track)
        if p is not None:
            periods.append(p)

        # --- ABF1 ---
        reg_abf1 = None
        reg_abf1_detail = None
        if ABF1_COL in optable.columns:
            abf1_track = optable[ABF1_COL].values
            gmax = abf1_global_max if abf1_global_max is not None else float(np.nanmax(abf1_track))
            ah = abf1_call_threshold(gmax)
            # Option A: anchor on the CENTER of each above-threshold footprint (call_abf1),
            # not a single find_peaks base -- robust to the flat-plateau/ULP degeneracy.
            calls = call_abf1(abf1_track, pos, ah)
            pred_a = [c["center"] for c in calls]
            abf1_pred += pred_a
            if return_abf1_tracks:
                reg_abf1_detail = dict(threshold=ah, global_max=gmax, calls=calls,
                                       pos=[int(p) for p in pos],
                                       track=[float(v) for v in np.nan_to_num(abf1_track)])
            if abf1 is not None:
                ref_a = abf1[(abf1["chr"] == chrm) &
                             (abf1["center"] >= start) & (abf1["center"] <= end)]["center"].tolist()
                abf1_ref += ref_a
                if ref_a:
                    reg_abf1 = match_peaks(pred_a, ref_a, tol_abf1)
                # AUROC: label positions within +/-tol of a ref ABF1 site as positive
                lab = np.zeros(len(pos), dtype=bool)
                for c in ref_a:
                    lo = max(0, c - tol_abf1 - start)
                    hi = min(len(pos), c + tol_abf1 - start + 1)
                    lab[lo:hi] = True
                m = covered
                if lab[m].any() and (~lab[m]).any():
                    abf1_scores_all.append(abf1_track[m])
                    abf1_labels_all.append(lab[m])

        # --- accessibility consistency (internal) ---
        if fiber_ratio is not None:
            access = 1.0 - occ_track
            m = covered & np.isfinite(fiber_ratio)
            if m.sum() > 50 and np.std(access[m]) > 1e-9 and np.nanstd(fiber_ratio[m]) > 1e-9:
                c = float(np.corrcoef(access[m], fiber_ratio[m])[0, 1])
                if np.isfinite(c):
                    access_corrs.append(c)

        per_region.append(dict(region="%s:%d-%d" % (chrm, start, end),
                               n_pred_dyads=len(pred_dyads),
                               nuc=reg_nuc, abf1=reg_abf1, period=p,
                               abf1_detail=reg_abf1_detail))

    # --- aggregate ---
    result = dict(label=label, outDir=os.path.abspath(outDir),
                  n_regions=len(per_region))

    def _nuc_entry(ref_list, reference_name):
        agg = match_peaks(nuc_pred, ref_list, tol_nuc)
        agg20 = match_peaks(nuc_pred, ref_list, 20)
        agg30 = match_peaks(nuc_pred, ref_list, 30)
        return dict(
            reference=reference_name,
            n_ref=agg["n_ref"], n_pred=agg["n_pred"],
            tp=agg["tp"], fp=agg["fp"], fn=agg["fn"],
            precision=agg["precision"], recall=agg["recall"], f1=agg["f1"],
            median_dyad_err=agg["median_dist"], mean_dyad_err=agg["mean_dist"],
            pct_within_20bp=agg20["recall"], pct_within_30bp=agg30["recall"],
            tol_bp=tol_nuc)

    if chereji is not None and nuc_ref:
        result["nucleosome"] = _nuc_entry(nuc_ref, "Chereji 2018 +1/-1")

    if brogaard is not None and brog_ref:
        result["nucleosome_brogaard"] = _nuc_entry(brog_ref, "Brogaard top-2000")

    if periods:
        result["phasing"] = dict(median_period_bp=float(np.median(periods)),
                                 n_regions_with_period=len(periods),
                                 expected_bp="~160-170")

    if abf1 is not None and abf1_ref:
        agg = match_peaks(abf1_pred, abf1_ref, tol_abf1)
        entry = dict(n_ref=agg["n_ref"], n_pred=agg["n_pred"],
                     tp=agg["tp"], fp=agg["fp"], fn=agg["fn"],
                     precision=agg["precision"], recall=agg["recall"], f1=agg["f1"],
                     median_dist=agg["median_dist"], tol_bp=tol_abf1)
        if abf1_scores_all:
            s = np.concatenate(abf1_scores_all)
            l = np.concatenate(abf1_labels_all)
            entry["auroc"] = auroc(s, l)
            entry["mean_post_at_sites"] = float(np.mean(s[l]))
            entry["mean_post_background"] = float(np.mean(s[~l]))
            entry["enrichment"] = (entry["mean_post_at_sites"] /
                                   entry["mean_post_background"]
                                   if entry["mean_post_background"] > 0 else float("inf"))
        result["abf1"] = entry

    if access_corrs:
        result["accessibility_consistency"] = dict(
            mean_corr_access_vs_methylation=float(np.mean(access_corrs)),
            n_regions=len(access_corrs),
            note="corr( 1-nuc_occupancy , Fiber meth/A ). Higher = model puts "
                 "nucleosomes where DNA is protected.")

    result["_per_region"] = per_region
    return result


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _fmt(x, nd=3):
    if x is None:
        return "  n/a"
    if isinstance(x, float):
        return "  nan" if np.isnan(x) else ("%.*f" % (nd, x))
    return str(x)


def print_report(r):
    print("=" * 68)
    print("RoboCOP decode score:  %s" % r["label"])
    print("regions scored: %d   outDir: %s" % (r["n_regions"], r["outDir"]))
    print("=" * 68)

    def _print_nuc(entry, tag, subset_note):
        print("\n[1] NUCLEOSOME dyads  (vs %s, tol=%d bp)" % (entry["reference"], entry["tol_bp"]))
        print("    reference dyads     : %d  (%s)" % (entry["n_ref"], subset_note))
        print("    predicted dyads     : %d" % entry["n_pred"])
        print("    TP / FN             : %d / %d" % (entry["tp"], entry["fn"]))
        print("    recall (KEY)        : %s   <- frac of ref dyads recovered" % _fmt(entry["recall"]))
        print("    median dyad error   : %s bp (KEY)" % _fmt(entry["median_dyad_err"], 1))
        print("    %% within +/-20 bp   : %s" % _fmt(entry["pct_within_20bp"]))
        print("    %% within +/-30 bp   : %s" % _fmt(entry["pct_within_30bp"]))
        print("    FP=%d, precision=%s, F1=%s"
              % (entry["fp"], _fmt(entry["precision"]), _fmt(entry["f1"])))

    n = r.get("nucleosome")
    if n:
        _print_nuc(n, "chereji",
                   "Chereji lists ONLY +1/-1 nucs -> precision/F1 NOT meaningful")
    nb = r.get("nucleosome_brogaard")
    if nb:
        _print_nuc(nb, "brogaard",
                   "Brogaard top-2000 well-positioned nucs, genome-wide")
    if not n and not nb:
        print("\n[1] NUCLEOSOME dyads  : no reference dyads in scored regions")

    p = r.get("phasing")
    if p:
        print("\n[2] PHASING periodicity (internal)")
        print("    median period       : %s bp   (expected %s)"
              % (_fmt(p["median_period_bp"], 1), p["expected_bp"]))
        print("    regions with period : %d" % p["n_regions_with_period"])

    a = r.get("abf1")
    if a:
        print("\n[3] ABF1 footprints  (vs MacIsaac match_PWM, tol=%d bp)" % a["tol_bp"])
        print("    reference sites     : %d" % a["n_ref"])
        print("    predicted footprints: %d" % a["n_pred"])
        print("    TP / FP / FN        : %d / %d / %d" % (a["tp"], a["fp"], a["fn"]))
        print("      TP = predicted ABF1 that matches a real ABF1 site")
        print("      FP = predicted ABF1 with NO ABF1 site nearby (spurious / other TF)")
        print("      FN = real ABF1 site the model missed")
        print("    precision / recall  : %s / %s" % (_fmt(a["precision"]), _fmt(a["recall"])))
        print("    F1                  : %s" % _fmt(a["f1"]))
        if "auroc" in a:
            print("    AUROC               : %s" % _fmt(a["auroc"]))
            print("    mean post @ sites   : %s" % _fmt(a["mean_post_at_sites"]))
            print("    mean post @ bg      : %s" % _fmt(a["mean_post_background"]))
            print("    enrichment          : %sx" % _fmt(a["enrichment"], 1))
    else:
        print("\n[3] ABF1 footprints  : no reference ABF1 sites in scored regions")

    ac = r.get("accessibility_consistency")
    if ac:
        print("\n[4] ACCESSIBILITY consistency (internal, Fiber-seq)")
        print("    corr(1-nuc, meth/A) : %s   (over %d regions)"
              % (_fmt(ac["mean_corr_access_vs_methylation"]), ac["n_regions"]))
        print("    -> higher = nucleosomes placed where DNA is protected")
    print("")


def main():
    ap = argparse.ArgumentParser(description="Score a RoboCOP decode against yeast ground truth.")
    ap.add_argument("outDir")
    ap.add_argument("--regions", default=None,
                    help="TSV (chr start end, with header) of regions to score. "
                         "Default: merged coords.tsv of the decode.")
    ap.add_argument("--label", default=None)
    ap.add_argument("--tol-nuc", type=int, default=20)
    ap.add_argument("--tol-abf1", type=int, default=20)
    ap.add_argument("--out", default=None, help="Write JSON report here "
                    "(default: <outDir>/score_report.json).")
    ap.add_argument("--brogaard", default=DEFAULT_BROGAARD,
                    help="Brogaard top-2000 dyads TSV (chrom<TAB>dyad).")
    args = ap.parse_args()

    regions = None
    if args.regions:
        rdf = pd.read_csv(args.regions, sep="\t")
        regions = [(row["chr"], int(row["start"]), int(row["end"]))
                   for _, row in rdf.iterrows()]

    r = score(args.outDir, regions=regions, tol_nuc=args.tol_nuc,
              tol_abf1=args.tol_abf1, label=args.label, brogaard_path=args.brogaard)
    print_report(r)

    out = args.out or (args.outDir.rstrip("/") + "/score_report.json")
    with open(out, "w") as fh:
        json.dump(r, fh, indent=2, default=lambda o: None)
    print("JSON report written:", out)


if __name__ == "__main__":
    main()
