#!/usr/bin/env python
"""Tests for run_permol_window_logfull_5to3.py at ERV46 (chrI:60,001-65,000), first N molecules.

Runs (each a subprocess; skipped if <out>/<run>/run_stats.json exists, unless --force):
  old            run_permol_window_logfull.py (unchanged)          --max-molecules N
  ref_m1         new driver --direction ref  --start motif1st      (default columns)
  {ref,5to3}_{m1,any}_all        real model, every collapsed column saved
  sym_{ref,5to3}_{m1,any}_all    TEST model made strand-symmetric: --seq-layer off --ps-symmetric

Checks:
  1. ref_m1 == old, bit for bit (every permol.npz array, every info_0_1.h5 dataset + attrs,
     molecules.tsv except timing).
  2. symmetric model, start=any: 5to3 == ref to ~1e-10 on every collapsed column, read ends
     included, no padding (the user's consistency criterion). start=motif1st shown for contrast.
  3. real model: max |5to3 - ref| per collapsed column, interior vs ends (< EDGE bp from a
     decoded segment end), and ABF1 / nucleosome at the two ABF1 sites, for both start rules.
  4. consensus flag counts (flag only, sequence never changed); posterior validity (row sums within 1e-9, no NaN) from run_stats.json.
Columns nuc_start / nuc_end are excluded from the 5to3-vs-ref comparisons: nuc_start is the
single state at nucleosome position 0, whose mirror is position 146 (a different column), and
nuc_end is the central silent state (always 0).

Usage (from analysis/): ROBOCOP_LOG_EMISSION=exact python singlefiber/test_5to3.py --out <dir>
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
OLD = os.path.join(HERE, "run_permol_window_logfull.py")
NEW = os.path.join(HERE, "run_permol_window_logfull_5to3.py")
SITES = {"abf1_site1": (61163, 61177), "abf1_site2": (62657, 62671)}   # 1-based inclusive
EDGE = 200
EXCLUDE = ("nuc_start", "nuc_end")

# The driver's defaults moved to the Brogaard / all-7-barcode model on 2026-10-30. These tests are
# about the OLD model (check 1 is bit-for-bit against the old driver), so every new-driver run
# passes the old model explicitly.
OLD_MODEL = ["--traindir", "robocop_train_fiberonly", "--pkg-tree", "pkgvar/permol_seq_maskoff_logfull",
             "--nuc-params", "nucleosome_params.pkl", "--bg-params", "bg_params.pkl",
             # 2026-11-02: the driver defaults now also set the TF file and read filters; pin the old ones.
             "--tf-params", "all_TFs_1000pealVal_params_pseudo.pkl", "--min-mapq", "0",
             "--min-aln-len", "0", "--min-qs", "-1", "--keep-reads", "none"]

RUNS = {
    "old": (OLD, []),
    "ref_m1": (NEW, ["--direction", "ref", "--start", "motif1st"]),
    "ref_m1_all": (NEW, ["--direction", "ref", "--start", "motif1st", "--cols", "all"]),
    "5to3_m1_all": (NEW, ["--direction", "5to3", "--start", "motif1st", "--cols", "all"]),
    "ref_any_all": (NEW, ["--direction", "ref", "--start", "any", "--cols", "all"]),
    "5to3_any_all": (NEW, ["--direction", "5to3", "--start", "any", "--cols", "all"]),
}
for _d in ("ref", "5to3"):
    for _s, _t in (("m1", "motif1st"), ("any", "any")):
        RUNS["sym_%s_%s_all" % (_d, _s)] = (NEW, ["--direction", _d, "--start", _t, "--cols", "all",
                                                  "--seq-layer", "off", "--ps-symmetric"])


def run(out, name, n, force):
    d = os.path.join(out, name)
    if os.path.exists(os.path.join(d, "run_stats.json")) and not force:
        print("[skip] %s exists" % name)
        return
    script, extra = RUNS[name]
    if script == NEW:
        extra = OLD_MODEL + extra
    cmd = [sys.executable, script, "--arm", "A", "--max-molecules", str(n), "--out", d] + extra
    print("[run]", " ".join(cmd))
    sys.stdout.flush()
    with open(os.path.join(out, name + ".log"), "w") as log:
        subprocess.run(cmd, cwd=ANALYSIS, stdout=log, stderr=subprocess.STDOUT, check=True)


def load(out, name):
    d = os.path.join(out, name)
    z = dict(np.load(os.path.join(d, "permol.npz")))
    mol = pd.read_csv(os.path.join(d, "molecules.tsv"), sep="\t")
    st = json.load(open(os.path.join(d, "run_stats.json")))
    return z, mol, st


def h5_items(path):
    out = {}
    with h5py.File(path, "r") as f:
        def visit(k, obj):
            if isinstance(obj, h5py.Dataset):
                out["D:" + k] = obj[()]
            for ak, av in obj.attrs.items():
                out["A:%s:%s" % (k, ak)] = av
        f.visititems(visit)
        for ak, av in f.attrs.items():
            out["A::%s" % ak] = av
    return out


def bit_identical(out):
    za, ma, _ = load(out, "old")
    zb, mb, _ = load(out, "ref_m1")
    res = {}
    for k in za:
        a, b = za[k], zb[k]
        same = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        md = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) \
            if a.dtype.kind in "fiub" and a.shape == b.shape else None
        res["npz:" + k] = dict(identical=bool(same), max_abs=md)
    ha = h5_items(os.path.join(out, "old", "tmpDir", "info_0_1.h5"))
    hb = h5_items(os.path.join(out, "ref_m1", "tmpDir", "info_0_1.h5"))
    assert set(ha) == set(hb), set(ha) ^ set(hb)
    for k in ha:
        a, b = np.asarray(ha[k]), np.asarray(hb[k])
        res["h5:" + k] = dict(identical=bool(a.shape == b.shape and np.array_equal(a, b)))
    cols = [c for c in ma.columns if c != "secs"]
    res["molecules.tsv"] = dict(identical=bool(ma[cols].equals(mb[cols])))
    return res


def compare(out, a, b):
    """max |b - a| per collapsed column over covered positions, interior vs ends; per strand;
    sites; loglik."""
    za, ma, _ = load(out, a)
    zb, mb, _ = load(out, b)
    cols = list(za["cols"])
    assert cols == list(zb["cols"])
    ci = [i for i, c in enumerate(cols) if c not in EXCLUDE]
    win0 = int(za["win"][0])
    pa, pb, cov = za["per_mol"].astype(np.float64), zb["per_mol"].astype(np.float64), za["per_mol_cov"]
    assert (cov == zb["per_mol_cov"]).all()
    n_mol, n_win, _ = pa.shape
    endmask = np.zeros_like(cov)
    for i in range(n_mol):
        w = np.where(cov[i])[0]
        if w.size:
            lo, hi = w[0], w[-1] + 1
            endmask[i, lo:min(hi, lo + EDGE)] = True
            endmask[i, max(lo, hi - EDGE):hi] = True
    inner = cov & ~endmask
    dl = np.abs(pb - pa)
    strands = za["strands"][:n_mol]
    per_col = {}
    for j in ci:
        c = cols[j]
        e = dl[:, :, j][cov & endmask]
        it = dl[:, :, j][inner]
        per_col[c] = dict(ends=float(e.max()) if e.size else 0.0,
                          interior=float(it.max()) if it.size else 0.0)
    by_strand = {}
    for s in ("watson", "crick"):
        m = (strands == s)[:, None] & cov
        by_strand[s] = float(dl[:, :, ci][m].max()) if m.any() else None
    ia, inuc = cols.index("Abf1_murphy"), cols.index("nucleosome")
    sites = {}
    for k, (s1, e1) in SITES.items():
        i0, i1 = s1 - win0, e1 - win0 + 1
        mid = (i0 + i1) // 2
        rr = [i for i in range(n_mol) if cov[i, i0:i1].all()]
        A_a = np.array([pa[i, i0:i1, ia].max() for i in rr])
        A_b = np.array([pb[i, i0:i1, ia].max() for i in rr])
        N_a = np.array([pa[i, mid, inuc] for i in rr])
        N_b = np.array([pb[i, mid, inuc] for i in rr])
        rc = [i for i in rr if strands[i] == "crick"]
        sites[k] = dict(n_reads=len(rr), n_crick=len(rc),
                        abf1_max_abs=float(np.abs(A_b - A_a).max()) if rr else None,
                        nuc_max_abs=float(np.abs(N_b - N_a).max()) if rr else None,
                        abf1_mean=[float(A_a.mean()), float(A_b.mean())] if rr else None,
                        nuc_mean=[float(N_a.mean()), float(N_b.mean())] if rr else None)
    ok = ma["skipped"].isna() | (ma["skipped"] == "")
    dll = np.abs(mb["loglik"][ok].values - ma["loglik"][ok].values)
    worst = max(per_col.items(), key=lambda kv: max(kv[1].values()))
    return dict(a=a, b=b, n_mol=int(n_mol), n_crick=int((strands == "crick").sum()),
                max_abs_all=float(dl[:, :, ci][cov].max()),
                max_abs_ends=float(dl[:, :, ci][cov & endmask].max()),
                max_abs_interior=float(dl[:, :, ci][inner].max()) if inner.any() else 0.0,
                worst_column=worst[0], by_strand=by_strand,
                loglik_max_abs=float(dll.max()), per_column=per_col, sites=sites)


def top_cols(pc, k=6):
    return [(c, "%.3g/%.3g" % (v["ends"], v["interior"]))
            for c, v in sorted(pc.items(), key=lambda kv: -max(kv[1].values()))[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="", help="comma list of runs to (re)make")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = a.only.split(",") if a.only else list(RUNS)
    for nm in names:
        run(a.out, nm, a.n, a.force)

    R = {}
    # 1. bit-for-bit
    bi = bit_identical(a.out)
    R["ref_vs_old"] = bi
    n_bad = sum(1 for v in bi.values() if not v["identical"])
    print("\n== 1. --direction ref --start motif1st vs old driver: %d of %d items differ ==" % (n_bad, len(bi)))
    for k, v in bi.items():
        if not v["identical"]:
            print("   DIFFERS:", k, v)
    md = max((v["max_abs"] or 0.0) for k, v in bi.items() if k.startswith("npz:"))
    print("   max |delta| over permol.npz arrays: %g" % md)

    # 2./3. 5to3 vs ref
    for tag, pre in (("symmetric model (seq off, symmetric ps)", "sym_"), ("real model", "")):
        for st in ("any", "m1"):
            c = compare(a.out, "%sref_%s_all" % (pre, st), "%s5to3_%s_all" % (pre, st))
            R["%s5to3_vs_ref_%s" % (pre, st)] = c
            print("\n== %s, start=%s: 5to3 vs ref (%d reads, %d Crick) ==" % (
                tag, "any" if st == "any" else "motif1st", c["n_mol"], c["n_crick"]))
            print("   max |delta| any column: all %.3g | ends (<%d bp) %.3g | interior %.3g | worst column %s"
                  % (c["max_abs_all"], EDGE, c["max_abs_ends"], c["max_abs_interior"], c["worst_column"]))
            print("   by strand: watson %s  crick %s ; |d loglik| max %.3g" % (
                c["by_strand"]["watson"], "%.3g" % c["by_strand"]["crick"], c["loglik_max_abs"]))
            print("   top columns (ends/interior):", top_cols(c["per_column"]))
            for k, v in c["sites"].items():
                print("   %s: %d reads (%d Crick)  ABF1 max|d| %.3g  mean %.4f->%.4f ; nuc max|d| %.3g mean %.4f->%.4f"
                      % (k, v["n_reads"], v["n_crick"], v["abf1_max_abs"], v["abf1_mean"][0],
                         v["abf1_mean"][1], v["nuc_max_abs"], v["nuc_mean"][0], v["nuc_mean"][1]))

    # 4. consensus + validity
    print("\n== 4. consensus and posterior validity ==")
    st_any = json.load(open(os.path.join(a.out, "5to3_any_all", "run_stats.json")))
    cs = st_any["consensus_summary"]
    tsv = pd.read_csv(os.path.join(a.out, "5to3_any_all", "consensus_sites.tsv"), sep="\t")
    R["consensus"] = dict(summary=cs, tsv_rows=int(len(tsv)))
    print("   consensus:", {k: cs[k] for k in ("n_reads", "median_coverage", "min_coverage", "n_flagged_alt",
                                                 "n_flagged_low_cov", "n_flagged_indel",
                                                 "max_top_alt_frac_cov_ge_min", "max_del_frac_cov_ge_min",
                                                 "max_ins_frac_cov_ge_min")}, "; tsv rows", len(tsv))
    val = {}
    for nm in RUNS:
        st = json.load(open(os.path.join(a.out, nm, "run_stats.json")))
        v = st["posterior_validity"]
        val[nm] = dict(n_nan=v["n_nan"], max_row_sum_dev=v["max_row_sum_dev"], max_silent=v["max_silent"],
                       n_decoded=st["n_molecules_decoded"], n_flipped=st.get("n_flipped"))
    R["validity"] = val
    print("   worst row-sum deviation over all runs: %.3g ; NaN total %d ; max silent %.3g" % (
        max(v["max_row_sum_dev"] for v in val.values()), sum(v["n_nan"] for v in val.values()),
        max(v["max_silent"] for v in val.values())))
    print("   start=any stationary:", st_any["start_info"])
    print("   |mirror(ps_crick) - ps_watson| max:", st_any["ps_mirror_vs_watson_max_abs"])
    json.dump(R, open(os.path.join(a.out, "test_5to3.json"), "w"), indent=1, default=float)
    print("\nwrote", os.path.join(a.out, "test_5to3.json"))


if __name__ == "__main__":
    main()
