#!/usr/bin/env python
"""The four pre-registered checks for the single-fiber prototype.

Thresholds are written down in the plan (2026-09-28) BEFORE any run; nothing here is
tuned to the result.

  1 Correctness.  Average the per-molecule posteriors, collapse, and correlate against an
    AGGREGATE decode of the same window under the same tree and trainDir, for nucleosome,
    nuc_center, background and Abf1_murphy.  PASS r >= 0.90 on nucleosome occupancy;
    FAIL r < 0.70 (the emission is mis-specified).  Arm A, efficiency off.
  2 Nucleosomes.  Per-molecule dyad calls, pooled, vs Chereji +1/-1 on this window, with
    score_robocop's own peak rule.  PASS recall >= 0.70 at <= 20 bp AND molecule-to-molecule
    dyad sd > 10 bp.  FAIL if recall collapses, or every molecule returns the same path.
  3 TF.  Per-molecule ABF1 posterior at the two MacIsaac sites vs the same molecules at
    within-NDR shifted controls.  AUROC >= 0.70 would overturn the scoping verdict;
    AUROC <= 0.60 is what scoping predicts.
  4 Context survives the HMM.  Fraction of molecules putting < 0.1 posterior on nucleosome
    states across the two motifs.  Expected ~0.84, matching the isolated LLR; much less
    means the transition priors override per-molecule evidence.

    python singlefiber/checks.py --runs-root /usr/project/xtmp/nd141/permol_proto \\
        --agg agg_baseline --out /usr/project/xtmp/nd141/permol_proto/checks.json

Run from analysis/.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

import score_robocop as S   # noqa: E402

TREE = os.path.join(ANALYSIS, "pkgvar", "permol_seq_maskoff")
NUC_TOL = 20
ABF1_BED = S.DEFAULT_ABF1
CHEREJI_BED = S.DEFAULT_CHEREJI


def load_permol(d):
    z = np.load(os.path.join(d, "permol.npz"), allow_pickle=False)
    cols = [str(x) for x in z["cols"]]
    win = z["win"]
    return dict(per_mol=z["per_mol"], cov=z["per_mol_cov"], cols=cols,
                depth=z["cov"], names=[str(x) for x in z["names"]],
                strands=[str(x) for x in z["strands"]],
                start=int(win[0]), end=int(win[1]),
                avg=z["avg_optable"], dir=d)


def agg_optable(agg_dir, chrom, start, end):
    S.use_pkg(TREE)
    dec = S.load_decode(agg_dir if agg_dir.endswith("/") else agg_dir + "/")
    op, covered, _fr = S.region_optable(dec, chrom, start, end)
    return op, covered


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ---------------------------------------------------------------------------
def check1(pm, agg_op, chrom):
    """Molecule-average vs the aggregate decode."""
    res = {"name": "correctness (molecule-average vs aggregate)",
           "pass_if": "r >= 0.90 on nucleosome", "fail_if": "r < 0.70"}
    per_col = {}
    for c in ("nucleosome", "nuc_center", "background", "Abf1_murphy"):
        if c not in pm["cols"] or c not in agg_op.columns:
            continue
        mine = pm["avg"][:, pm["cols"].index(c)]
        theirs = agg_op[c].values
        n = min(len(mine), len(theirs))
        per_col[c] = dict(r=pearson(mine[:n], theirs[:n]),
                          mean_permol=float(np.nanmean(mine[:n])),
                          mean_agg=float(np.nanmean(theirs[:n])))
    res["r"] = per_col
    r_nuc = per_col.get("nucleosome", {}).get("r", float("nan"))
    res["r_nucleosome"] = r_nuc
    res["verdict"] = ("PASS" if r_nuc >= 0.90 else
                      ("FAIL" if r_nuc < 0.70 else "MARGINAL"))

    # Diagnostics. A low r here does NOT by itself mean the emission is wrong: the
    # AVERAGE of N weak single-molecule posteriors is not the posterior of the POOLED
    # data. singlefiber/pooled_emission_check.py settles that question separately by
    # multiplying all molecules' Bernoulli factors into one chain. These numbers say how
    # the two tracks differ instead.
    def _smooth(x, w):
        k = np.ones(w) / w
        return np.convolve(np.asarray(x, float), k, mode="same")

    diag = {}
    if "nucleosome" in pm["cols"] and "nucleosome" in agg_op.columns:
        mine = pm["avg"][:, pm["cols"].index("nucleosome")]
        theirs = agg_op["nucleosome"].values
        n = min(len(mine), len(theirs))
        mine, theirs = mine[:n], theirs[:n]
        from scipy.stats import spearmanr
        diag["spearman_nucleosome"] = float(spearmanr(mine, theirs).correlation)
        for w in (25, 75, 147):
            diag["r_nucleosome_smooth%d" % w] = pearson(_smooth(mine, w), _smooth(theirs, w))
        d = pm["depth"][:n]
        for cut in (50, 100, 150):
            m = d >= cut
            diag["r_nucleosome_depth_ge_%d" % cut] = (
                pearson(mine[m], theirs[m]) if m.sum() > 10 else float("nan"))
            diag["n_positions_depth_ge_%d" % cut] = int(m.sum())
        diag["frac_positions_agg_occ_gt_0p9"] = float((theirs > 0.9).mean())
        diag["mean_permol_where_agg_gt_0p9"] = float(mine[theirs > 0.9].mean())
        diag["mean_permol_where_agg_lt_0p1"] = float(mine[theirs < 0.1].mean())
        diag["median_molecule_depth"] = float(np.median(pm["depth"]))
    res["diagnostics"] = diag
    return res


def check2(pm, chrom):
    """Per-molecule dyads, pooled, vs Chereji; plus molecule-to-molecule dyad spread."""
    res = {"name": "nucleosomes (pooled per-molecule dyads vs Chereji)",
           "pass_if": "recall >= 0.70 at <= %d bp AND dyad sd > 10 bp" % NUC_TOL}
    ch = S.load_chereji(CHEREJI_BED)
    ref = sorted(ch[(ch["chr"] == chrom) & (ch["dyad"] >= pm["start"]) &
                    (ch["dyad"] <= pm["end"])]["dyad"].tolist())
    res["n_reference_dyads"] = len(ref)
    j = pm["cols"].index("nuc_center")
    pos = np.arange(pm["start"], pm["end"] + 1)

    all_calls, per_mol_calls, paths = [], [], []
    for i in range(pm["per_mol"].shape[0]):
        cvg = pm["cov"][i]
        if not cvg.any():
            per_mol_calls.append([])
            continue
        tr = pm["per_mol"][i, :, j].astype(float)
        tr[~cvg] = 0.0
        mx = float(np.nanmax(tr))
        h = max(0.02, 0.20 * mx) if mx > 0 else 0.02
        pk = S.call_peaks(tr, height=h, distance=120)
        calls = [int(pos[k]) for k in pk]
        per_mol_calls.append(calls)
        all_calls += calls
        paths.append(np.round(tr, 6).tobytes())

    res["n_molecules_with_calls"] = int(sum(1 for c in per_mol_calls if c))
    res["n_pooled_calls"] = len(all_calls)
    res["n_distinct_paths"] = len(set(paths))
    res["n_paths_total"] = len(paths)

    if ref and all_calls:
        m = S.match_peaks(all_calls, ref, NUC_TOL)
        res["match"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in m.items() if not isinstance(v, (list, tuple))}
        res["recall"] = float(m.get("recall", m.get("sensitivity", np.nan)))
    else:
        res["recall"] = float("nan")

    # molecule-to-molecule spread: per reference dyad, the sd over molecules of that
    # molecule's nearest dyad call within +/-73 bp (half a nucleosome).
    sds, ns = [], []
    for r in ref:
        d = []
        for calls in per_mol_calls:
            near = [c for c in calls if abs(c - r) <= 73]
            if near:
                d.append(min(near, key=lambda c: abs(c - r)))
        if len(d) >= 5:
            sds.append(float(np.std(d)))
            ns.append(len(d))
    res["per_site_dyad_sd_bp"] = sds
    res["per_site_n_molecules"] = ns
    res["median_dyad_sd_bp"] = float(np.median(sds)) if sds else float("nan")
    rec, sd = res["recall"], res["median_dyad_sd_bp"]
    res["verdict"] = ("PASS" if (rec >= 0.70 and sd > 10) else "FAIL")
    return res


def ndr_controls(agg_op, pm, sites, min_shift=30, max_shift=300,
                 ndr_cut=0.2, max_control_occ=0.10):
    """For each motif, a same-length control span inside the same NDR.

    The control must be as nucleosome-free as the motif is, or the AUROC in check 3 ends
    up measuring "NDR vs nucleosome" rather than "motif vs no motif". So: NDR = the
    contiguous run of positions around the motif where the AGGREGATE nucleosome occupancy
    is below ndr_cut; candidates are same-length spans wholly inside that run, at least
    min_shift and at most max_shift from the motif center, whose MEAN aggregate occupancy
    is at most max_control_occ; the farthest such candidate wins. If none qualifies, the
    candidate with the lowest mean occupancy is used and `basis` records the relaxation.
    """
    occ = agg_op["nucleosome"].values
    below = occ < ndr_cut
    n = len(below)
    out = []
    for (s1, e1) in sites:
        L = e1 - s1
        c = (s1 + e1) // 2
        ci = c - pm["start"]
        if 0 <= ci < n and below[ci]:
            lo = ci
            while lo > 0 and below[lo - 1]:
                lo -= 1
            hi = ci
            while hi + 1 < n and below[hi + 1]:
                hi += 1
        else:
            lo = hi = ci
        cands = []
        for k in range(lo, hi + 1):
            a, b = k - L // 2, k - L // 2 + L
            if a < lo or b > hi or a < 0 or b >= n:
                continue
            sh = abs(k - ci)
            if sh < min_shift or sh > max_shift:
                continue
            cands.append((k, float(np.mean(occ[a:b + 1])), sh))
        qual = [x for x in cands if x[1] <= max_control_occ]
        if qual:
            k, mo, sh = max(qual, key=lambda x: x[2])
            basis = "ndr_run"
        elif cands:
            k, mo, sh = min(cands, key=lambda x: x[1])
            basis = "ndr_run_relaxed_occ"
        else:
            k = ci + 150 if ci + 150 + L < n else ci - 150
            a, b = k - L // 2, k - L // 2 + L
            mo = float(np.mean(occ[max(0, a):min(n, b + 1)]))
            basis = "fallback_shift150"
        cs = k + pm["start"]
        a1 = cs - L // 2
        out.append(dict(motif=(s1, e1), control=(a1, a1 + L),
                        ndr_run=(int(lo + pm["start"]), int(hi + pm["start"])),
                        shift=int(cs - c), basis=basis,
                        n_candidates=len(cands), n_qualifying=len(qual),
                        occ_motif=float(np.mean(occ[max(0, s1 - pm["start"]):e1 - pm["start"] + 1])),
                        occ_control=float(mo)))
    return out


def _span_stat(pm, i, col, s1, e1, how="max"):
    j = pm["cols"].index(col)
    lo, hi = s1 - pm["start"], e1 - pm["start"] + 1
    lo, hi = max(0, lo), min(pm["per_mol"].shape[1], hi)
    if hi <= lo:
        return np.nan
    if not pm["cov"][i, lo:hi].all():
        return np.nan
    v = pm["per_mol"][i, lo:hi, j].astype(float)
    return float(v.max() if how == "max" else v.mean())


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    m = np.isfinite(scores)
    scores, labels = scores[m], labels[m]
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan"), npos, nneg
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)), npos, nneg


def motif_methylation(pm, chrom, sites, bam=None, flank=400):
    """Per-molecule methylated fraction over each motif span, straight from the BAM.

    Needed to ask whether the per-molecule ABF1 posterior carries any per-molecule FIBER
    information at all, or whether its ordering is just the (molecule-independent) PWM.
    """
    import read_calls as RC
    sys.path.insert(0, os.path.join(ANALYSIS, "pkgvar", "permol_seq_maskoff"))
    from robocop.utils.getNucleotides import getNucleotideSequence
    if bam is None:
        import run_permol_window as RPW
        bam = RPW.BAM
        fasta = RPW.FASTA
    else:
        fasta = os.path.join(ANALYSIS, "inputs", "SacCer3.fa")
    es = max(0, pm["start"] - 1 - flank)
    ee = pm["end"] + flank
    codes = getNucleotideSequence(fasta, chrom, es + 1, ee)
    mols, es2, _ee2, _st = RC.read_molecules(bam, chrom, pm["start"], pm["end"],
                                             codes, flank=flank)
    by_name = {m.name: m for m in mols}
    out = []
    for (s1, e1) in sites:
        d = {}
        for nm in pm["names"]:
            m = by_name.get(nm)
            if m is None:
                continue
            c = m.calls[(s1 - es2):(e1 - es2 + 1)]
            inf = int((c != RC.NO_OBS).sum())
            d[nm] = (int((c == RC.METH).sum()), inf)
        out.append(d)
    return out


def check3(pm, agg_op, chrom, sites, controls):
    res = {"name": "TF (per-molecule ABF1, real sites vs within-NDR controls)",
           "pass_if": "AUROC >= 0.70 overturns the scoping verdict",
           "predicted": "AUROC <= 0.60"}
    try:
        meth = motif_methylation(pm, chrom, sites)
    except Exception as e:                                # pragma: no cover
        print("motif_methylation unavailable:", repr(e))
        meth = [None] * len(sites)
    scores, labels, detail = [], [], []
    for k, (site, ctl) in enumerate(zip(sites, controls)):
        s1, e1 = site
        c1, c2 = ctl["control"]
        pos_s, neg_s, pos_names = [], [], []
        for i in range(pm["per_mol"].shape[0]):
            a = _span_stat(pm, i, "Abf1_murphy", s1, e1, "max")
            b = _span_stat(pm, i, "Abf1_murphy", c1, c2, "max")
            if np.isfinite(a) and np.isfinite(b):
                pos_s.append(a)
                neg_s.append(b)
                pos_names.append(pm["names"][i])
        scores += pos_s + neg_s
        labels += [1] * len(pos_s) + [0] * len(neg_s)
        a_, npos, nneg = auroc(pos_s + neg_s, [1] * len(pos_s) + [0] * len(neg_s))
        detail.append(dict(site="%s:%d-%d" % (chrom, s1, e1),
                           control="%s:%d-%d" % (chrom, c1, c2),
                           control_basis=ctl["basis"], shift=ctl["shift"],
                           occ_motif=ctl["occ_motif"], occ_control=ctl["occ_control"],
                           n_molecules=len(pos_s), auroc=a_,
                           median_posterior_site=float(np.median(pos_s)) if pos_s else np.nan,
                           median_posterior_control=float(np.median(neg_s)) if neg_s else np.nan,
                           max_posterior_site=float(np.max(pos_s)) if pos_s else np.nan,
                           n_molecules_over_0p10=int(np.sum(np.asarray(pos_s) > 0.10)),
                           n_molecules_over_0p50=int(np.sum(np.asarray(pos_s) > 0.50))))
        # How much does the ABF1 posterior actually VARY between molecules at one site,
        # and does that variation track the molecule's own protection? If the ordering is
        # just the PWM, the spread is negligible and the correlation is ~0.
        ps_arr = np.asarray(pos_s, float)
        if ps_arr.size:
            detail[-1].update(
                min_posterior_site=float(ps_arr.min()),
                spread_max_over_min=float(ps_arr.max() / ps_arr.min()) if ps_arr.min() > 0 else np.inf,
                cv_posterior_site=float(ps_arr.std() / ps_arr.mean()) if ps_arr.mean() else np.nan)
        if meth[k]:
            frac, sc = [], []
            for nm, v in zip(pos_names, ps_arr):
                km, nn = meth[k].get(nm, (0, 0))
                if nn >= 3:
                    frac.append(km / nn)
                    sc.append(v)
            if len(frac) >= 20:
                from scipy.stats import spearmanr
                rho = spearmanr(frac, sc)
                detail[-1].update(
                    n_molecules_with_ge3_calls_in_motif=len(frac),
                    mean_motif_meth_fraction=float(np.mean(frac)),
                    spearman_posterior_vs_own_methylation=float(rho.correlation),
                    spearman_p=float(rho.pvalue))
            else:
                detail[-1]["n_molecules_with_ge3_calls_in_motif"] = len(frac)
    res["per_site"] = detail
    a_all, npos, nneg = auroc(scores, labels)
    res["auroc_pooled"] = a_all
    res["n_pos"], res["n_neg"] = npos, nneg
    res["verdict"] = ("OVERTURNS (>=0.70)" if a_all >= 0.70 else
                      ("AS PREDICTED (<=0.60)" if a_all <= 0.60 else "BETWEEN 0.60 AND 0.70"))
    return res


def check4(pm, chrom, sites, controls, cut=0.1):
    res = {"name": "context survives the HMM (per-molecule nucleosome rejection)",
           "expected": "~0.84 of molecules put < %.2f on nucleosome states" % cut}
    out = []
    frac_all, frac_ctl = [], []
    for site, ctl in zip(sites, controls):
        s1, e1 = site
        c1, c2 = ctl["control"]
        mean_site, mean_ctl, max_site = [], [], []
        for i in range(pm["per_mol"].shape[0]):
            a = _span_stat(pm, i, "nucleosome", s1, e1, "mean")
            am = _span_stat(pm, i, "nucleosome", s1, e1, "max")
            b = _span_stat(pm, i, "nucleosome", c1, c2, "mean")
            if np.isfinite(a):
                mean_site.append(a)
                max_site.append(am)
            if np.isfinite(b):
                mean_ctl.append(b)
        f = float(np.mean(np.asarray(mean_site) < cut)) if mean_site else np.nan
        fm = float(np.mean(np.asarray(max_site) < cut)) if max_site else np.nan
        fc = float(np.mean(np.asarray(mean_ctl) < cut)) if mean_ctl else np.nan
        out.append(dict(site="%s:%d-%d" % (chrom, s1, e1), n_molecules=len(mean_site),
                        frac_below_cut_mean=f, frac_below_cut_max=fm,
                        frac_below_cut_control=fc,
                        median_nuc_occ_site=float(np.median(mean_site)) if mean_site else np.nan,
                        median_nuc_occ_control=float(np.median(mean_ctl)) if mean_ctl else np.nan))
        frac_all.append(f)
        frac_ctl.append(fc)
    res["per_site"] = out
    res["frac_below_cut"] = float(np.nanmean(frac_all))
    res["frac_below_cut_control"] = float(np.nanmean(frac_ctl))
    res["verdict"] = ("MATCHES ~0.84" if abs(res["frac_below_cut"] - 0.84) <= 0.15
                      else "DIVERGES FROM 0.84")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/usr/project/xtmp/nd141/permol_proto")
    ap.add_argument("--agg", default="agg_baseline")
    ap.add_argument("--primary", default="armA_effoff")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    root = a.runs_root
    runs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isfile(os.path.join(d, "permol.npz")))
    print("per-molecule runs found:", [os.path.basename(d) for d in runs])
    prim = os.path.join(root, a.primary)
    pm = load_permol(prim)
    chrom = pd.read_csv(os.path.join(prim, "coords.tsv"), sep="\t")["chr"].iloc[0]
    agg_dir = os.path.join(root, a.agg)
    agg_op, agg_cov = agg_optable(agg_dir, chrom, pm["start"], pm["end"])

    ab = S.load_abf1(ABF1_BED)
    sub = ab[(ab["chr"] == chrom) & (ab["center"] >= pm["start"]) & (ab["center"] <= pm["end"])]
    sites = [(int(r["start"]), int(r["end"])) for _, r in sub.iterrows()]
    print("MacIsaac ABF1 sites in window:", sites)
    controls = ndr_controls(agg_op, pm, sites)
    for c in controls:
        print("  control:", json.dumps(c, default=float))

    out = {"window": "%s:%d-%d" % (chrom, pm["start"], pm["end"]),
           "primary_run": a.primary, "agg_baseline": a.agg,
           "abf1_sites": ["%s:%d-%d" % (chrom, s, e) for s, e in sites],
           "controls": controls,
           "check1": check1(pm, agg_op, chrom),
           "check2": check2(pm, chrom),
           "check3": check3(pm, agg_op, chrom, sites, controls),
           "check4": check4(pm, chrom, sites, controls)}
    pc = os.path.join(root, "pooled_check", "pooled_check.json")
    if os.path.isfile(pc):
        out["pooled_emission_check"] = json.load(open(pc))

    # --- every arm: check 1 correlation, check 3 AUROC, check 4 fraction --------------
    per_arm = {}
    for d in runs:
        nm = os.path.basename(d)
        try:
            q = load_permol(d)
        except Exception as e:
            per_arm[nm] = {"error": repr(e)}
            continue
        c1 = check1(q, agg_op, chrom)
        c3 = check3(q, agg_op, chrom, sites, controls)
        c4 = check4(q, chrom, sites, controls)
        st = {}
        if os.path.isfile(os.path.join(d, "run_stats.json")):
            st = json.load(open(os.path.join(d, "run_stats.json")))
        per_arm[nm] = dict(
            r_nucleosome=c1["r_nucleosome"],
            r_abf1=c1["r"].get("Abf1_murphy", {}).get("r"),
            auroc_abf1=c3["auroc_pooled"],
            abf1_median_site=[s["median_posterior_site"] for s in c3["per_site"]],
            abf1_max_site=[s["max_posterior_site"] for s in c3["per_site"]],
            n_over_0p10=[s["n_molecules_over_0p10"] for s in c3["per_site"]],
            frac_nuc_below_0p1=c4["frac_below_cut"],
            secs_decode=st.get("secs_decode"), peak_rss_gib=st.get("peak_rss_gib"),
            n_molecules=st.get("n_molecules_decoded"),
            secs_per_molecule=st.get("secs_per_molecule"))
    out["per_arm"] = per_arm

    print("\n=== checks ===")
    for k in ("check1", "check2", "check3", "check4"):
        c = out[k]
        print("%s  %-55s  %s" % (k, c["name"], c["verdict"]))
    print("\nper arm:")
    hdr = ("run", "r_nuc", "r_abf1", "AUROC", "nuc<0.1", "s/mol", "GiB")
    print("%-22s %7s %7s %7s %8s %7s %6s" % hdr)
    for nm in sorted(per_arm):
        v = per_arm[nm]
        if "error" in v:
            print("%-22s  %s" % (nm, v["error"]))
            continue
        print("%-22s %7.4f %7.4f %7.4f %8.3f %7.2f %6.2f" % (
            nm, v["r_nucleosome"] or np.nan, (v["r_abf1"] or np.nan),
            v["auroc_abf1"] or np.nan, v["frac_nuc_below_0p1"] or np.nan,
            v["secs_per_molecule"] or np.nan, v["peak_rss_gib"] or np.nan))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
