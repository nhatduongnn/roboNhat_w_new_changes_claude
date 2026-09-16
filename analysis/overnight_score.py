"""Overnight 2026-09-15 scoring for jobs A and C (launched by overnight_launch.py).

Both use tw_validate.py's own code: tw_validate.score (the MacIsaac sidecar cross-check, then
rossi_validate.fast_match, 30 bp, greedy one-to-one, calls of a group's motifs pooled) on the
58 MacIsaac-and-Rossi groups from tune_w.factor_set, split all / fitted / nonfitted.

A  fixed-prior fiber veto test, sw01 round 05 weights (robocop_train_tw_sw01_05) decoded as
     seqonly  = sw01's own round-05 decode       conc_tuning/counts_tw_sw01_05/
     both     = seq + fiber layers (seq_mr58)     overnight/counts_{chrXIV_chrII,chrIV}_twS05_both/
     fib      = fiber layers only (fiber_mr58)    overnight/counts_{chrXIV_chrII,chrIV}_twS05_fib/
   sets: tune = chrXIV+chrII, holdout = chrIV.
   Rule 7: only the layer configuration differs from sw01_05.
   Veto/added (not one-to-one, 30 bp, per group, pooled motifs):
     veto  = a seq-only call with no call of the same group within 30 bp in the compared config
     added = a compared-config call with no seq-only call of the same group within 30 bp
   each split by MacIsaac / Rossi _CX support (any site within 30 bp).
   -> overnight/A_validation.tsv, A_groups.tsv, A_veto_groups.tsv, A_veto_calls.tsv

C  genome-wide validation of the three final tuned models + legacy u001/m001 round 7:
     tw_sw01_05 (seq-only), tw_fw01_07 (fiber), tw_bw01_07 (both)   overnight/counts_genome_tw_*/
     u001_07, m001_07   conc_tuning/calls_ct_*_07 (calls) + counts_ct_*_07 (counts, sidecar)
   scopes: genome (16 chromosomes), tuned (chrXIV+chrII), untuned (the other 14), and every
   chromosome. n_excluded = invalid-posterior positions count_calls dropped (per chromosome max
   over factors, exactly as tw_validate.read_occ reports them).
   Rule 7: only the decode scope differs from each campaign's final round.
   -> overnight/C_validation.tsv, C_chroms.tsv, C_groups.tsv

Usage:  python overnight_score.py A      python overnight_score.py C
A missing input is recorded as status=missing rows, never a crash of the whole part.
"""
import bisect
import collections
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tw_validate as TV     # noqa: E402
import tune_w as TW          # noqa: E402

OUT = os.path.join(HERE, "overnight")
CT = os.path.join(HERE, "conc_tuning")
TOL = 30
REFS = TV.REFS
GENOME = ["chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI", "chrVII", "chrVIII", "chrIX", "chrX",
          "chrXI", "chrXII", "chrXIII", "chrXIV", "chrXV", "chrXVI"]
TUNED = ["chrXIV", "chrII"]

A_DIRS = {
    "seqonly": {"tune": os.path.join(CT, "counts_tw_sw01_05"), "holdout": os.path.join(CT, "counts_tw_sw01_05")},
    "both": {"tune": os.path.join(OUT, "counts_chrXIV_chrII_twS05_both"), "holdout": os.path.join(OUT, "counts_chrIV_twS05_both")},
    "fib": {"tune": os.path.join(OUT, "counts_chrXIV_chrII_twS05_fib"), "holdout": os.path.join(OUT, "counts_chrIV_twS05_fib")},
}
A_DECODE = {"seqonly": {"tune": "robocop_chrXIV_chrII_tw_sw01_05", "holdout": "robocop_chrIV_tw_sw01_05"},
            "both": {"tune": "robocop_chrXIV_chrII_twS05_both", "holdout": "robocop_chrIV_twS05_both"},
            "fib": {"tune": "robocop_chrXIV_chrII_twS05_fib", "holdout": "robocop_chrIV_twS05_fib"}}
C_MODELS = collections.OrderedDict([
    ("tw_sw01_05", dict(calls=os.path.join(OUT, "counts_genome_tw_sw01_05"), counts=os.path.join(OUT, "counts_genome_tw_sw01_05"),
                        config="seqonly", decode="robocop_genome_tw_sw01_05")),
    ("tw_fw01_07", dict(calls=os.path.join(OUT, "counts_genome_tw_fw01_07"), counts=os.path.join(OUT, "counts_genome_tw_fw01_07"),
                        config="fiber", decode="robocop_genome_tw_fw01_07")),
    ("tw_bw01_07", dict(calls=os.path.join(OUT, "counts_genome_tw_bw01_07"), counts=os.path.join(OUT, "counts_genome_tw_bw01_07"),
                        config="both", decode="robocop_genome_tw_bw01_07")),
    ("u001_07", dict(calls=os.path.join(CT, "calls_ct_u001_07"), counts=os.path.join(CT, "counts_ct_u001_07"),
                     config="fib+seq legacy (84 live + 69 unmasked)", decode="legacy")),
    ("m001_07", dict(calls=os.path.join(CT, "calls_ct_m001_07"), counts=os.path.join(CT, "counts_ct_m001_07"),
                     config="fib+seq legacy (69 non-MacIsaac masked)", decode="legacy")),
])


def write_tsv(path, cols, rows):
    with open(path + ".tmp", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(TV.fmt(r.get(c, "")) for c in cols) + "\n")
    os.replace(path + ".tmp", path)


def have(d, chroms):
    return all(os.path.exists(os.path.join(d, "calls", c + ".tsv")) and os.path.exists(os.path.join(d, c + ".tsv"))
               and os.path.exists(os.path.join(d, "macisaac", c + ".tsv")) for c in chroms)


def pool(per_list):
    out = collections.Counter()
    for per in per_list:
        for k, (s, n, m) in per.items():
            a = out.get(k, (0, 0, 0))
            out[k] = (a[0] + s, a[1] + n, a[2] + m)
    return dict(out)


def agg_rows(per, groups, fitted, base):
    rows = []
    for ref in REFS:
        for cat, keep in (("all", lambda g: True), ("fitted", lambda g: fitted[g]),
                          ("nonfitted", lambda g: not fitted[g])):
            gs = [g for g in groups if keep(g) and (ref, g) in per]
            s = sum(per[(ref, g)][0] for g in gs)
            n = sum(per[(ref, g)][1] for g in gs)
            m = sum(per[(ref, g)][2] for g in gs)
            P, R, F = TV.prf(s, n, m)
            r = dict(base)
            r.update(ref=ref, fitted=cat, n_groups=len(gs), sites=s, calls=n, matched=m, P=P, R=R, F1=F)
            rows.append(r)
    return rows


def group_rows(per, groups, fitted, base):
    rows = []
    for ref in REFS:
        for g in groups:
            if (ref, g) in per:
                s, n, m = per[(ref, g)]
                P, R, F = TV.prf(s, n, m)
                r = dict(base)
                r.update(ref=ref, group=g, fitted=fitted[g], sites=s, calls=n, matched=m, P=P, R=R, F1=F)
                rows.append(r)
    return rows


def near_any(x, sorted_other, tol=TOL):
    k = bisect.bisect_left(sorted_other, x - tol)
    return k < len(sorted_other) and sorted_other[k] <= x + tol


# ================================================================ A
def part_a():
    groups, T, fitted = TW.factor_set()
    refs = TV.references(groups)
    agg, grp, vg, vc = [], [], [], []
    calls = {}
    for cfg, dirs in A_DIRS.items():
        for which, chroms in (("tune", TUNED), ("holdout", ["chrIV"])):
            d = dirs[which]
            base = dict(config=cfg, set=which, chroms="+".join(chroms), weights="tw_sw01_05",
                        decode=A_DECODE[cfg][which])
            if not have(d, chroms):
                agg.append(dict(base, status="missing", ref="", fitted=""))
                print("A %s %s: MISSING %s" % (cfg, which, d))
                continue
            try:
                per = TV.score(groups, fitted, d, d, chroms)
                occ, bad = TV.read_occ(d, chroms)
            except Exception as e:
                agg.append(dict(base, status="error: %r" % e, ref="", fitted=""))
                traceback.print_exc()
                continue
            base.update(status="ok", n_excluded=json.dumps(bad, sort_keys=True),
                        nuc_copies=occ.get("nucleosome", float("nan")))
            agg += agg_rows(per, groups, fitted, base)
            grp += group_rows(per, groups, fitted, base)
            calls[(cfg, which)] = TV.read_calls(d, chroms)
            for r in agg[-6:]:
                if r["fitted"] == "all":
                    print("A %-7s %-7s %-8s sites %5d calls %6d matched %4d  P %.3f R %.3f F1 %.3f"
                          % (cfg, which, r["ref"], r["sites"], r["calls"], r["matched"], r["P"], r["R"], r["F1"]))
    # veto / added vs seq-only
    for cmp_cfg in ("both", "fib"):
        for which, chroms in (("tune", TUNED), ("holdout", ["chrIV"])):
            if (cmp_cfg, which) not in calls or ("seqonly", which) not in calls:
                vg.append(dict(comparison="%s_vs_seqonly" % cmp_cfg, set=which, group="", status="missing"))
                continue
            S, X = calls[("seqonly", which)], calls[(cmp_cfg, which)]
            for g, ms in groups.items():
                row = dict(comparison="%s_vs_seqonly" % cmp_cfg, set=which, group=g, fitted=fitted[g], status="ok",
                           calls_seqonly=0, calls_cmp=0, veto=0, veto_mac=0, veto_rossi=0,
                           added=0, added_mac=0, added_rossi=0, kept=0, kept_mac=0, kept_rossi=0)
                for c in chroms:
                    s = sorted(x for m in ms for x in S.get(m, {}).get(c, []))
                    x = sorted(y for m in ms for y in X.get(m, {}).get(c, []))
                    mac = sorted(refs[g]["macisaac"].get(c, []))
                    cx = sorted(refs[g]["rossi_cx"].get(c, [])) if refs[g]["rossi_cx"] is not None else []
                    row["calls_seqonly"] += len(s)
                    row["calls_cmp"] += len(x)
                    for p in s:
                        supp_m, supp_r = near_any(p, mac), near_any(p, cx)
                        kind = "kept" if near_any(p, x) else "veto"
                        row[kind] += 1
                        row[kind + "_mac"] += int(supp_m)
                        row[kind + "_rossi"] += int(supp_r)
                        if kind == "veto":
                            vc.append(dict(comparison=row["comparison"], set=which, kind="veto", group=g, chrom=c,
                                           center=p, macisaac_support=int(supp_m), rossi_support=int(supp_r)))
                    for p in x:
                        if not near_any(p, s):
                            supp_m, supp_r = near_any(p, mac), near_any(p, cx)
                            row["added"] += 1
                            row["added_mac"] += int(supp_m)
                            row["added_rossi"] += int(supp_r)
                            vc.append(dict(comparison=row["comparison"], set=which, kind="added", group=g, chrom=c,
                                           center=p, macisaac_support=int(supp_m), rossi_support=int(supp_r)))
                vg.append(row)
            tot = collections.Counter()
            for r in vg:
                if r.get("status") == "ok" and r["comparison"] == "%s_vs_seqonly" % cmp_cfg and r["set"] == which:
                    for k in ("calls_seqonly", "calls_cmp", "veto", "veto_mac", "veto_rossi", "added", "added_mac", "added_rossi"):
                        tot[k] += r[k]
            print("A %s vs seqonly %s: %s" % (cmp_cfg, which, dict(tot)))
    os.makedirs(OUT, exist_ok=True)
    write_tsv(os.path.join(OUT, "A_validation.tsv"),
              ["config", "set", "chroms", "weights", "decode", "status", "ref", "fitted", "n_groups", "sites", "calls",
               "matched", "P", "R", "F1", "n_excluded", "nuc_copies"], agg)
    write_tsv(os.path.join(OUT, "A_groups.tsv"),
              ["config", "set", "ref", "group", "fitted", "sites", "calls", "matched", "P", "R", "F1"], grp)
    write_tsv(os.path.join(OUT, "A_veto_groups.tsv"),
              ["comparison", "set", "group", "fitted", "status", "calls_seqonly", "calls_cmp", "kept", "kept_mac",
               "kept_rossi", "veto", "veto_mac", "veto_rossi", "added", "added_mac", "added_rossi"], vg)
    write_tsv(os.path.join(OUT, "A_veto_calls.tsv"),
              ["comparison", "set", "kind", "group", "chrom", "center", "macisaac_support", "rossi_support"], vc)
    print("wrote overnight/A_*.tsv")


# ================================================================ C
def part_c():
    groups, T, fitted = TW.factor_set()
    TV.references(groups)
    agg, chrom_rows, grp = [], [], []
    for model, spec in C_MODELS.items():
        per_chrom, excluded, nuc, missing = {}, {}, {}, []
        for c in GENOME:
            if not (os.path.exists(os.path.join(spec["calls"], "calls", c + ".tsv"))
                    and os.path.exists(os.path.join(spec["counts"], c + ".tsv"))
                    and os.path.exists(os.path.join(spec["counts"], "macisaac", c + ".tsv"))):
                missing.append(c)
                continue
            try:
                per_chrom[c] = TV.score(groups, fitted, spec["calls"], spec["counts"], [c])
                occ, bad = TV.read_occ(spec["counts"], [c])
                excluded[c] = bad[c]
                nuc[c] = occ.get("nucleosome", float("nan"))
            except Exception as e:
                missing.append(c)
                print("C %s %s: ERROR %r" % (model, c, e))
        base0 = dict(model=model, config=spec["config"], decode=spec["decode"])
        for c in GENOME:
            if c in per_chrom:
                b = dict(base0, scope=c, chroms=c, status="ok", n_excluded=excluded[c], nuc_copies=nuc[c],
                         tuned=int(c in TUNED))
                chrom_rows += agg_rows(per_chrom[c], groups, fitted, b)
            else:
                chrom_rows.append(dict(base0, scope=c, chroms=c, status="missing"))
        for scope, chroms in (("genome", GENOME), ("tuned_chrXIV+chrII", TUNED),
                              ("untuned_14", [c for c in GENOME if c not in TUNED])):
            got = [c for c in chroms if c in per_chrom]
            miss = [c for c in chroms if c not in per_chrom]
            status = "ok" if not miss else ("missing" if not got else "incomplete (missing %s)" % ",".join(miss))
            b = dict(base0, scope=scope, chroms="+".join(got), status=status,
                     n_excluded=sum(excluded[c] for c in got),
                     n_excluded_by_chrom=json.dumps({c: excluded[c] for c in got if excluded[c]}, sort_keys=True),
                     nuc_copies=sum(nuc[c] for c in got) if got else float("nan"))
            if not got:
                agg.append(b)
                continue
            per = pool([per_chrom[c] for c in got])
            rows = agg_rows(per, groups, fitted, b)
            agg += rows
            if scope == "genome":
                grp += group_rows(per, groups, fitted, dict(model=model, scope=scope))
            for r in rows:
                if r["fitted"] == "all":
                    print("C %-10s %-18s %-8s sites %5d calls %6d matched %4d  P %.3f R %.3f F1 %.3f  excl %d  %s"
                          % (model, scope, r["ref"], r["sites"], r["calls"], r["matched"], r["P"], r["R"], r["F1"],
                             b["n_excluded"], status))
    os.makedirs(OUT, exist_ok=True)
    cols = ["model", "config", "decode", "scope", "chroms", "status", "ref", "fitted", "n_groups", "sites", "calls",
            "matched", "P", "R", "F1", "n_excluded", "n_excluded_by_chrom", "nuc_copies"]
    write_tsv(os.path.join(OUT, "C_validation.tsv"), cols, agg)
    write_tsv(os.path.join(OUT, "C_chroms.tsv"), cols[:5] + ["tuned"] + cols[5:15] + ["n_excluded", "nuc_copies"],
              chrom_rows)
    write_tsv(os.path.join(OUT, "C_groups.tsv"),
              ["model", "scope", "ref", "group", "fitted", "sites", "calls", "matched", "P", "R", "F1"], grp)
    print("wrote overnight/C_*.tsv")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else ""
    if part == "A":
        part_a()
    elif part == "C":
        part_c()
    else:
        sys.exit("usage: overnight_score.py A|C")
