"""Tuner v2 validation: P/R/F1 vs MacIsaac and vs Rossi ChExMix _CX, per round, per set.

For a tw campaign round, reads counts_tw_<run>_NN/calls/<chrom>.tsv (count_calls.py --calls:
posterior >= 0.10 fixed-threshold calls) and scores the 58 MacIsaac-and-Rossi groups on
  set = tune     (chrXIV + chrII)     or     holdout (chrIV)
  ref = macisaac | rossi_cx           (rossi_validate.load_references, used as shipped)
  fitted = all | fitted | nonfitted   (fitted = a group containing one of the 12 motifs whose fiber
                                       footprint was fitted on Rossi peaks; flattering for Rossi)
with rossi_validate.fast_match (== score_robocop.match_peaks, 30 bp, greedy one-to-one). Calls of a
group's motifs are pooled. P = matched/calls, R = matched/sites, F1 pooled over the groups.

Before scoring, the MacIsaac match re-derived from the calls must equal the count step's own
macisaac/<chrom>.tsv sidecar exactly (as rossi_validate does), or the round is refused.

Outputs
  conc_tuning/<run>/validation.tsv          one row per (round, set, ref, fitted) + tuning metrics
  conc_tuning/<run>/validation_groups.tsv   one row per (round, set, ref, group)
  conc_tuning/tw_legacy_validation{,_groups}.tsv   u001/m001 rounds on the same groups/chromosomes

Usage
  python tw_validate.py --run fw01 sw01 bw01            # (re)score every available round
  python tw_validate.py --legacy u001=0,7 m001=0,7      # legacy baseline rows
"""
import argparse
import collections
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rossi_validate as RV     # noqa: E402

REFS = ("macisaac", "rossi_cx")
AGG_COLS = ["run", "round", "set", "chroms", "ref", "fitted", "n_groups", "sites", "calls", "matched",
            "P", "R", "F1", "within2x", "n_T5", "n_capped_under", "n_falling_over",
            "nuc_copies", "nuc_rel_r0", "unknown_occ", "invalid_positions"]
GRP_COLS = ["run", "round", "set", "ref", "group", "fitted", "sites", "calls", "matched", "P", "R", "F1",
            "T_tune", "E"]
_REFS_CACHE = {}


def references(groups):
    key = tuple(sorted(groups))
    if key not in _REFS_CACHE:
        refs, _ = RV.load_references(collections.OrderedDict((g, groups[g]) for g in sorted(groups)))
        _REFS_CACHE[key] = refs
    return _REFS_CACHE[key]


def read_calls(calls_dir, chroms):
    calls = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in chroms:
        p = os.path.join(calls_dir, "calls", c + ".tsv")
        if not os.path.exists(p):
            raise RuntimeError("missing %s" % p)
        with open(p) as fh:
            fh.readline()
            for line in fh:
                f, ch, center, _, _ = line.rstrip("\n").split("\t")
                calls[f][ch].append(int(center))
    return calls


def read_occ(counts_dir, chroms):
    import count_calls as CC
    occ, bad = collections.Counter(), {}
    for c in chroms:
        p = os.path.join(counts_dir, c + ".tsv")
        if not os.path.exists(p):
            return None, None
        rows = CC.read(p)
        bad[c] = max((r["n_excluded"] for r in rows), default=0)
        for r in rows:
            occ[r["factor"]] += r["occ"]
    return occ, bad


def prf(s, c, m):
    P = m / c if c else float("nan")
    R = m / s if s else float("nan")
    F = 2 * P * R / (P + R) if (c and s and P + R > 0) else (0.0 if (c and s) else float("nan"))
    return P, R, F


def score(groups, fitted, calls_dir, counts_dir, chroms):
    """-> per-group rows {(ref, group): (sites, calls, matched)} after the sidecar cross-check."""
    refs = references(groups)
    calls = read_calls(calls_dir, chroms)
    for c in chroms:
        side = os.path.join(counts_dir, "macisaac", c + ".tsv")
        if not os.path.exists(side):
            raise RuntimeError("missing MacIsaac sidecar %s" % side)
        srow = {r["group"]: r for r in csv.DictReader(open(side), delimiter="\t")}
        for g, ms in groups.items():
            pc = [x for m in ms for x in calls.get(m, {}).get(c, [])]
            rc = refs[g]["macisaac"].get(c, [])
            got = (len(pc), len(rc), RV.fast_match(pc, rc))
            s = srow.get(g)
            if s is None:                # no MacIsaac target -> no sidecar row (added 2026-09-16)
                if rc:
                    raise RuntimeError("%s %s has %d MacIsaac sites but no sidecar row in %s"
                                       % (g, c, len(rc), side))
                continue
            want = (int(s["n_calls"]), int(s["n_macisaac"]), int(s["n_matched"]))
            if got != want:
                raise RuntimeError("MISMATCH %s %s %s: calls/sites/tp %s vs sidecar %s"
                                   % (calls_dir, g, c, got, want))
    out = {}
    for g, ms in groups.items():
        for ref in REFS:
            R = refs[g][ref]
            if R is None:
                continue
            s = n = tp = 0
            for c in chroms:
                pc = [x for m in ms for x in calls.get(m, {}).get(c, [])]
                rc = R.get(c, [])
                s, n, tp = s + len(rc), n + len(pc), tp + RV.fast_match(pc, rc)
            out[(ref, g)] = (s, n, tp)
    return out


def aggregate(run, rnd, which, chroms, groups, fitted, per, extra):
    agg, grp = [], []
    for ref in REFS:
        for cat, keep in (("all", lambda g: True), ("fitted", lambda g: fitted[g]),
                          ("nonfitted", lambda g: not fitted[g])):
            gs = [g for g in groups if keep(g) and (ref, g) in per]
            s = sum(per[(ref, g)][0] for g in gs)
            n = sum(per[(ref, g)][1] for g in gs)
            m = sum(per[(ref, g)][2] for g in gs)
            P, R, F = prf(s, n, m)
            row = dict(run=run, round=rnd, set=which, chroms="+".join(chroms), ref=ref, fitted=cat,
                       n_groups=len(gs), sites=s, calls=n, matched=m, P=P, R=R, F1=F)
            row.update(extra)
            agg.append(row)
        for g in groups:
            if (ref, g) in per:
                s, n, m = per[(ref, g)]
                P, R, F = prf(s, n, m)
                grp.append(dict(run=run, round=rnd, set=which, ref=ref, group=g, fitted=fitted[g],
                                sites=s, calls=n, matched=m, P=P, R=R, F1=F,
                                T_tune=extra.get("_T", {}).get(g, ""), E=extra.get("_E", {}).get(g, "")))
    return agg, grp


def fmt(v):
    if isinstance(v, float):
        return "nan" if v != v else "%.6g" % v
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True).replace("\t", " ")
    return str(v)


def upsert(path, cols, rows, key):
    old = []
    if os.path.exists(path):
        old = list(csv.DictReader(open(path), delimiter="\t"))
    newkeys = {tuple(str(r[k]) for k in key) for r in rows}
    old = [r for r in old if tuple(r[k] for k in key) not in newkeys]
    with open(path + ".tmp", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in old:
            fh.write("\t".join(r.get(c, "") for c in cols) + "\n")
        for r in rows:
            fh.write("\t".join(fmt(r.get(c, "")) for c in cols) + "\n")
    os.replace(path + ".tmp", path)


def validate_round(st, t, which, counts_root, calls_from=None):
    run = st["run"]
    chroms = st["tune_chroms"] if which == "tune" else [st["holdout_chrom"]]
    cdir = os.path.join(counts_root, "counts_tw_%s_%02d" % (run, t))
    calls_dir = calls_from or cdir
    counts_dir = calls_from or cdir
    groups = st["groups"]
    per = score(groups, st["fitted"], calls_dir, counts_dir, chroms)
    occ, bad = read_occ(counts_dir, chroms)
    extra = dict(invalid_positions=bad, unknown_occ=occ.get("unknown", float("nan")),
                 nuc_copies=occ.get("nucleosome", float("nan")))
    h = next((x for x in st["history"] if x["iter"] == t), None)
    if which == "tune" and h:
        extra.update(within2x=h["within2x"], n_T5=h["n_T5"], n_capped_under=len(h["capped_under"]),
                     n_falling_over=len(h["falling_over"]), nuc_rel_r0=h["nucleosome_rel_r0"],
                     _T=st["target"], _E=h["E"])
    elif which == "holdout":
        r0, _ = read_occ(os.path.join(counts_root, "counts_tw_%s_00" % run), chroms)
        if r0:
            extra["nuc_rel_r0"] = extra["nuc_copies"] / r0["nucleosome"] - 1.0
    agg, grp = aggregate(run, t, which, chroms, groups, st["fitted"], per, extra)
    d = os.path.join(counts_root, run)
    upsert(os.path.join(d, "validation.tsv"), AGG_COLS, agg, ("round", "set", "ref", "fitted"))
    upsert(os.path.join(d, "validation_groups.tsv"), GRP_COLS, grp, ("round", "set", "ref", "group"))
    for r in agg:
        if r["fitted"] == "all":
            print("  validate %s r%02d %-7s %-8s sites %5d calls %6d matched %4d  P %.3f R %.3f F1 %.3f"
                  % (run, t, which, r["ref"], r["sites"], r["calls"], r["matched"], r["P"], r["R"], r["F1"]))
    return agg


def legacy(specs):
    import tune_w as TW
    groups, T, fitted = TW.factor_set()
    agg_all, grp_all = [], []
    for spec in specs:
        run, rounds = spec.split("=")
        rounds = [int(x) for x in rounds.split(",")]
        nuc0 = {}
        for t in rounds:
            calls_dir = os.path.join(HERE, "conc_tuning", "calls_ct_%s_%02d" % (run, t))
            counts_dir = os.path.join(HERE, "conc_tuning", "counts_ct_%s_%02d" % (run, t))
            for which, chroms in (("tune", TW.TUNE_CHROMS), ("holdout", [TW.HOLDOUT_CHROM])):
                per = score(groups, fitted, calls_dir, counts_dir, chroms)
                occ, bad = read_occ(counts_dir, chroms)
                E = {g: sum(occ[m] for m in groups[g]) for g in groups}
                extra = dict(invalid_positions=bad, unknown_occ=occ.get("unknown", float("nan")),
                             nuc_copies=occ["nucleosome"])
                nuc0.setdefault(which, occ["nucleosome"])
                extra["nuc_rel_r0"] = occ["nucleosome"] / nuc0[which] - 1.0
                if which == "tune":
                    t5 = [g for g in groups if T[g] >= TW.WITHIN2X_MIN_T]
                    extra.update(n_T5=len(t5), _T=T, _E=E, within2x=sum(
                        1 for g in t5 if E[g] > 0 and abs(math.log(E[g] / T[g])) <= math.log(2)))
                agg, grp = aggregate(run, t, which, chroms, groups, fitted, per, extra)
                agg_all += agg
                grp_all += grp
                for r in agg:
                    if r["fitted"] == "all":
                        print("legacy %s r%02d %-7s %-8s sites %5d calls %6d matched %4d  P %.3f R %.3f F1 %.3f%s"
                              % (run, t, which, r["ref"], r["sites"], r["calls"], r["matched"], r["P"], r["R"],
                                 r["F1"], ("  within2x %d/%d" % (r["within2x"], r["n_T5"])) if which == "tune" else ""))
    base = os.path.join(HERE, "conc_tuning", "tw_legacy_validation")
    upsert(base + ".tsv", AGG_COLS, agg_all, ("run", "round", "set", "ref", "fitted"))
    upsert(base + "_groups.tsv", GRP_COLS, grp_all, ("run", "round", "set", "ref", "group"))
    print("wrote %s{,_groups}.tsv" % os.path.relpath(base, HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--legacy", nargs="*", default=[])
    ap.add_argument("--root", default=os.path.join(HERE, "conc_tuning"))
    a = ap.parse_args()
    for run in a.run:
        st = json.load(open(os.path.join(a.root, run, "state.json")))
        for h in st["history"]:
            validate_round(st, h["iter"], "tune", a.root)
        for t in range(st["iter"] + 1):
            if os.path.exists(os.path.join(a.root, "counts_tw_%s_%02d" % (run, t), "calls",
                                           st["holdout_chrom"] + ".tsv")):
                validate_round(st, t, "holdout", a.root)
    if a.legacy:
        legacy(a.legacy)


if __name__ == "__main__":
    main()
