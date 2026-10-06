"""Compare posterior-fix arms on one window: per-factor optable AND the raw stored p_table.

Original three-way ablation (still the default):
A  read-side normalisation in score_robocop.region_optable; decode is the UNCHANGED binary,
   so the fix is applied when the stored posterior is read.
B  linear normalisation inside posterior_decoding (librobocop_norm.so).
C  log-space with log-sum-exp inside posterior_decoding (librobocop_logspace.so).

Any other arms can be named with --arms; each arm's decode dir is --template formatted with
prefix / arm / layer (default `<prefix>_<arm>_<layer>`), e.g. the Phase A / D arms
    Bprod     production fix B (librobocop.so, c6dfe41c)
    logfull   full log-space recursion + exact logpmf emission (librobocop_logfull.so)
    logfullD0 log-space recursion + today's floored emission (gate-only)
    B         pre-cutover ablation arm B (librobocop_norm.so)

All arms of one layer must read the same trainDir and coords; the only difference is the
binary / emission path. That is checked (coords.tsv equal, h5 state counts equal).

TWO COMPARISONS
  optable  score_robocop.region_optable per arm (it re-normalises each row and sums states into
           factor columns), then pairwise max abs / max rel and calls at --thr. Per-state errors
           can cancel inside a factor's sum, so this alone is not proof (POSTERIOR_FIX_PLAN.md A4).
  raw      per segment, the stored tmpDir/info*.h5 segment_N/posterior (CSR) for each arm pair:
             max |p-q| over every cell;
             max rel |p-q|/max(p,q) over cells where BOTH > 1e-4 (robocop.py:33 stores values
               < 1e-4 as 0, so a cell that is 0 in one arm and 1e-4.. in the other is a
               save-floor crossing, counted separately as `floor_x`, not a relative error);
           and per arm: row sums over the emitting states [:silent_states_begin] (min/max),
           non-zero cells in the silent block [silent_states_begin:], cells > 1, and the segment
           attr log_likelihood, side by side.

    python compare_posterior_fixes.py --prefix robocop_w30k_bv75_49 --chrom chrXIV     # A/B/C
    python compare_posterior_fixes.py --prefix robocop_w30k_bv75_49 --arms Bprod,logfull,logfullD0,B \
        --window chrXIV:396001-433000
    python compare_posterior_fixes.py --prefix robocop_rdna_chrXII --arms Bprod,logfull,logfullD0 \
        --window chrXII:450001-470000 --ref Bprod --no-optable

What to expect for the original A/B/C, written down before running:
  * B vs C near-exact (~1e-12 relative) -- same formula, different arithmetic path.
  * A vs B/C may differ in the TAIL only: A normalises AFTER robocop.py:33 zeroes stored values
    below 1e-4, B and C before it. On a healthy row the floor removes true values < 1e-4; on a
    blown-up row it corresponds to a far smaller true value and removes almost nothing.
A disagreement beyond the tail means the damage was never a clean scale factor, and A would be
unsafe as a retroactive fix.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_robocop as S                                              # noqa: E402

LAYERS = [("both", "both layers"), ("fib", "fiber only"), ("seq", "sequence only")]
ARMS = ["A", "B", "C"]
SAVE_FLOOR = 1e-4                                                      # robocop.py:33


# ------------------------------------------------------------------ dirs / trainDir
def arm_dir(template, prefix, arm, layer):
    d = template.format(prefix=prefix, arm=arm, layer=layer)
    if not os.path.isdir(d) and not os.path.isabs(d) and os.path.isdir(os.path.join(HERE, d)):
        d = os.path.join(HERE, d)
    return d


def real_traindir(outdir):
    """(path, how). RUN_INFO.json's traindir is what the decode really used; config.ini is copied
    from whichever trainDir was cloned and names the pristine one (score_robocop.load_decode)."""
    ri = None
    try:
        ri = json.load(open(os.path.join(outdir, "RUN_INFO.json")))
    except Exception:
        pass
    if ri and ri.get("traindir"):
        td = ri["traindir"] if os.path.isabs(ri["traindir"]) else os.path.join(HERE, ri["traindir"])
        if os.path.isfile(os.path.join(td, "HMMconfig.pkl")):
            return td, "RUN_INFO.json"
    dec = S.load_decode(outdir)
    return None, "config.ini (%s)" % dec["config"].get("main", "trainDir")


def load_dec(outdir):
    """score_robocop.load_decode, with dshared swapped for the RUN_INFO trainDir's when known."""
    dec = S.load_decode(outdir)
    td, how = real_traindir(outdir)
    if td:
        ds = pickle.load(open(os.path.join(td, "HMMconfig.pkl"), "rb"))
        ds["info_file"] = None
        dec["dshared"] = ds
    dec["traindir_how"] = how
    return dec


# ------------------------------------------------------------------ optable comparison
def optable(dec, chrom, start, end):
    tab, covered, _ = S.region_optable(dec, chrom, start, end)
    return tab, covered


def compare(x, y, cols):
    """max abs / max rel difference, and which column carries the worst relative one."""
    a, b = x[cols].to_numpy(float), y[cols].to_numpy(float)
    d = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    rel = np.divide(d, denom, out=np.zeros_like(d), where=denom > 1e-12)
    k = int(np.unravel_index(np.argmax(rel), rel.shape)[1]) if rel.size else 0
    return d.max(), rel.max(), cols[k]


def calls(tab, col, thr):
    v = tab[col].to_numpy(float)
    return set(np.nonzero(v >= thr)[0].tolist())


def pairs_of(arms, ref):
    if ref:
        return [(ref, q) for q in arms if q != ref]
    return [(arms[i], arms[j]) for i in range(len(arms)) for j in range(i + 1, len(arms))]


def optable_section(decs, arms, pairs, chrom, start, end, thr, factor):
    tabs = {}
    for arm in arms:
        if arm not in decs:
            continue
        try:
            tabs[arm], _ = optable(decs[arm], chrom, start, end)
        except Exception as e:                                          # noqa: BLE001
            print("  %s  optable FAILED: %s" % (arm, e))
    if len(tabs) < 2:
        print("  optable: fewer than two arms readable")
        return
    cols = [c for c in list(tabs.values())[0].columns if all(c in t.columns for t in tabs.values())]
    print("\n  [optable] pairwise differences over %d factor columns x %d bp" % (len(cols), end - start + 1))
    print("  %-22s %14s %14s   %s" % ("pair", "max abs", "max rel", "worst column"))
    for p, q in pairs:
        if p in tabs and q in tabs:
            da, dr, w = compare(tabs[p], tabs[q], cols)
            print("  %-22s %14.3e %14.3e   %s" % ("%s vs %s" % (p, q), da, dr, w))

    if factor in cols:
        print("\n  [optable] %s calls at posterior >= %.2f" % (factor, thr))
        cs = {k: calls(t, factor, thr) for k, t in tabs.items()}
        for k in arms:
            if k in cs:
                print("    %-10s : %d positions" % (k, len(cs[k])))
        for p, q in pairs:
            if p in cs and q in cs:
                only_p, only_q = cs[p] - cs[q], cs[q] - cs[p]
                verdict = "identical" if not (only_p or only_q) else \
                          "DIFFER: %d only in %s, %d only in %s" % (len(only_p), p, len(only_q), q)
                print("    %s vs %s : %s" % (p, q, verdict))

    print("\n  [optable] summed posterior per factor (the `occ` numerator), top 6 by magnitude")
    tot = {k: t[cols].to_numpy(float).sum(axis=0) for k, t in tabs.items()}
    order = np.argsort(-list(tot.values())[0])[:6]
    print("  %-16s" % "factor" + "".join("%14s" % k for k in arms if k in tot))
    for idx in order:
        row = "  %-16s" % cols[idx][:16]
        for k in arms:
            if k in tot:
                row += "%14.4f" % tot[k][idx]
        print(row)


# ------------------------------------------------------------------ raw p_table comparison
def seg_index(outdir):
    """segment number -> (info file, log_likelihood attr), over every tmpDir/info*.h5."""
    out = {}
    for p in sorted(glob.glob(os.path.join(outdir, "tmpDir", "info*.h5"))):
        try:
            f = h5py.File(p, "r")
        except OSError as e:                     # locked / half-written: a decode is still running
            raise RuntimeError("%s unreadable (decode still running?): %s" % (p, e))
        with f:
            for k in f.keys():
                m = re.match(r"^segment_(\d+)$", k)
                if m and "posterior" in f[k]:
                    out[int(m.group(1))] = (p, f[k].attrs.get("log_likelihood", np.nan))
    return out


def read_post(path, i):
    with h5py.File(path, "r") as f:
        return np.asarray(S._get_sparse_todense(f, "segment_%d/posterior" % i), dtype=float)


def arm_stats(P, sb):
    rs = P[:, :sb].sum(axis=1)
    return dict(rs_min=rs.min(), rs_max=rs.max(), nz_silent=int(np.count_nonzero(P[:, sb:])),
                gt1=int(np.count_nonzero(P > 1.0)), neg=int(np.count_nonzero(P < 0)))


def pair_stats(P, Q):
    d = np.abs(P - Q)
    both = (P > SAVE_FLOOR) & (Q > SAVE_FLOOR)
    den = np.maximum(P, Q)
    rel = np.divide(d, den, out=np.zeros_like(d), where=both)
    floor_x = int(np.count_nonzero((P == 0) != (Q == 0)))
    k = np.unravel_index(np.argmax(d), d.shape)
    return dict(abs=d.max(), rel=rel.max() if both.any() else np.nan, floor_x=floor_x,
                worst=(int(k[0]), int(k[1])), n_diff=int(np.count_nonzero(d)))


def raw_section(dirs, decs, arms, pairs, chrom, start, end):
    """dirs: arm -> outdir (only arms that exist)."""
    idx = {}
    for a in list(dirs):
        try:
            idx[a] = seg_index(dirs[a])
        except RuntimeError as e:
            print("  [raw] %s skipped: %s" % (a, e))
            dirs = {k: v for k, v in dirs.items() if k != a}
    arms = [x for x in arms if x in dirs]
    pairs = [(p, q) for p, q in pairs if p in dirs and q in dirs]
    if len(dirs) < 2:
        print("  [raw] fewer than two readable arms")
        return
    coords = {a: pd.read_csv(os.path.join(d, "coords.tsv"), sep="\t") for a, d in dirs.items()}
    ref = arms[0] if arms[0] in coords else list(coords)[0]
    for a, c in coords.items():
        if not c[["chr", "start", "end"]].equals(coords[ref][["chr", "start", "end"]]):
            print("  [raw] WARNING: %s coords.tsv differs from %s's -- segment numbers are not comparable" % (a, ref))
    segs = S._seg_idxs(coords[ref], chrom, start, end)
    sbs = {a: int(decs[a]["dshared"]["silent_states_begin"]) for a in dirs}
    ns = {a: int(decs[a]["dshared"]["n_states"]) for a in dirs}
    print("\n  [raw] stored posterior, %d segment(s) overlapping %s:%d-%d; silent_states_begin %s; "
          "trainDir from %s" % (len(segs), chrom, start, end,
                               ",".join("%s=%d" % kv for kv in sbs.items()),
                               ",".join("%s=%s" % (a, decs[a]["traindir_how"]) for a in dirs)))
    if len(set(sbs.values())) > 1 or len(set(ns.values())) > 1:
        print("  [raw] WARNING: arms disagree on silent_states_begin / n_states -- not the same model")

    print("\n  per arm: row sums over [:ssb] (min..max), non-zero silent cells, cells >1 / <0, log_likelihood")
    print("  %5s %-13s %-12s %12s %12s %10s %8s %24s" % ("seg", "region", "arm", "rowsum min", "rowsum max",
                                                        "nz silent", ">1/<0", "log_likelihood"))
    cache = {}
    tot_pair = {pq: dict(abs=0.0, rel=0.0, floor_x=0, n_diff=0, segs=0) for pq in pairs}
    for i in segs:
        r = coords[ref].loc[i]
        reg = "%d-%d" % (r["start"], r["end"])
        for a in dirs:
            if i not in idx[a]:
                print("  %5d %-13s %-12s MISSING segment_%d" % (i, reg, a, i))
                continue
            P = read_post(idx[a][i][0], i)
            if P.shape[1] != ns[a]:
                print("  [raw] WARNING: %s segment_%d has %d state columns, trainDir says %d" % (a, i, P.shape[1], ns[a]))
            cache[a] = P
            s = arm_stats(P, sbs[a])
            print("  %5d %-13s %-12s %12.6f %12.6f %10d %8s %24.10f"
                  % (i, reg, a, s["rs_min"], s["rs_max"], s["nz_silent"], "%d/%d" % (s["gt1"], s["neg"]),
                     idx[a][i][1]))
        for p, q in pairs:
            if p in cache and q in cache and i in idx[p] and i in idx[q]:
                if cache[p].shape != cache[q].shape:
                    print("        %s vs %s: SHAPE DIFFERS %s vs %s" % (p, q, cache[p].shape, cache[q].shape))
                    continue
                ps = pair_stats(cache[p], cache[q])
                dll = idx[q][i][1] - idx[p][i][1]
                print("        %-22s max abs %.3e  max rel(both>1e-4) %.3e  floor_x %d  cells differing %d  "
                      "worst (row %d, state %d)  dLL(%s-%s) %+.6e"
                      % ("%s vs %s" % (p, q), ps["abs"], ps["rel"], ps["floor_x"], ps["n_diff"],
                         ps["worst"][0], ps["worst"][1], q, p, dll))
                t = tot_pair[(p, q)]
                t["abs"] = max(t["abs"], ps["abs"])
                t["rel"] = np.nanmax([t["rel"], ps["rel"]])
                t["floor_x"] += ps["floor_x"]
                t["n_diff"] += ps["n_diff"]
                t["segs"] += 1
        cache.clear()

    print("\n  [raw] over all segments")
    print("  %-22s %6s %14s %20s %10s %14s" % ("pair", "segs", "max abs", "max rel (both>1e-4)", "floor_x",
                                               "cells differ"))
    for (p, q), t in tot_pair.items():
        if t["segs"]:
            print("  %-22s %6d %14.3e %20.3e %10d %14d" % ("%s vs %s" % (p, q), t["segs"], t["abs"], t["rel"],
                                                           t["floor_x"], t["n_diff"]))
    lls = {a: sum(idx[a][i][1] for i in segs if i in idx[a]) for a in dirs}
    print("  summed log_likelihood: " + "  ".join("%s %.10f" % kv for kv in lls.items()))


# ------------------------------------------------------------------ main
def parse_window(w):
    m = re.match(r"^(\w+):([\d,]+)-([\d,]+)$", w)
    if not m:
        raise SystemExit("--window must be chrom:start-end, got %r" % w)
    return m.group(1), int(m.group(2).replace(",", "")), int(m.group(3).replace(",", ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="robocop_w30k_bv75_49")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list, default A,B,C")
    ap.add_argument("--layers", default=",".join(l for l, _ in LAYERS), help="comma list, default both,fib,seq")
    ap.add_argument("--template", default="{prefix}_{arm}_{layer}",
                    help="decode dir per arm/layer; fields {prefix} {arm} {layer}")
    ap.add_argument("--ref", default=None, help="compare every arm against this one only (default: all pairs)")
    ap.add_argument("--chrom", default="chrXIV")
    ap.add_argument("--start", type=int, default=400001)
    ap.add_argument("--end", type=int, default=430000)
    ap.add_argument("--window", default=None, help="chrom:start-end, overrides --chrom/--start/--end")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--factor", default="Abf1_murphy")
    ap.add_argument("--no-optable", action="store_true", help="skip the region_optable comparison")
    ap.add_argument("--no-raw", action="store_true", help="skip the raw stored-posterior comparison")
    a = ap.parse_args()
    if a.window:
        a.chrom, a.start, a.end = parse_window(a.window)
    arms = [x for x in a.arms.split(",") if x]
    if a.ref and a.ref not in arms:
        arms = [a.ref] + arms
    pairs = pairs_of(arms, a.ref)
    labels = dict(LAYERS)

    for lay in [x for x in a.layers.split(",") if x]:
        print("\n" + "=" * 78)
        print("LAYER: %s   (%s:%d-%d)" % (labels.get(lay, lay), a.chrom, a.start, a.end))
        print("=" * 78)
        dirs, decs = {}, {}
        for arm in arms:
            d = arm_dir(a.template, a.prefix, arm, lay)
            if not os.path.isdir(d):
                print("  %-10s MISSING (%s)" % (arm, d))
                continue
            try:
                decs[arm] = load_dec(d)
                dirs[arm] = d
                print("  %-10s %s" % (arm, d))
            except Exception as e:                                      # noqa: BLE001
                print("  %-10s FAILED to load: %s" % (arm, e))
        if len(dirs) < 2:
            continue
        if not a.no_optable:
            optable_section(decs, arms, pairs, a.chrom, a.start, a.end, a.thr, a.factor)
        if not a.no_raw:
            raw_section(dirs, decs, [x for x in arms if x in dirs], pairs, a.chrom, a.start, a.end)


if __name__ == "__main__":
    main()
