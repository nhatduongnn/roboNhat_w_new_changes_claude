"""Compare the unknown-concentration sweep against the lambda=1 control.

Question the sweep asks
-----------------------
At their own MacIsaac sites, factors like Fkh2/Mbp1/Stb5/Hap1 hold posterior 0.0000-0.0004
while `unknown` and `background` hold 0.28-0.87. `unknown` is one motif-less state with flat
emissions carrying 3.3x the prior of all 153 real motifs combined -- a wildcard that fits
anywhere. Does taking prior mass away from it let the real factors win their own sites?

Why the nucleosome is held fixed
--------------------------------
Lowering `unknown` alone drives the shared unbound root up, and the 147 bp nucleosome
amplifies that as p^147: measured at +1085% to +2036% for lambda_unknown 0.3 down to 0.01.
Every trainDir here therefore carries a compensating lambda_nucleosome that pins
nucleosome_prob back to baseline (drift +/-0.001%), so the only thing moving is `unknown`.

Scope
-----
chrIV + chrVII + chrXV = 3.71 Mb, 30.8% of the genome. The control is the SAME three
chromosomes cut out of the genome-wide lambda=1 decode, so the comparison is like-for-like;
these numbers are NOT comparable to the genome-wide baseline in conc_tuning/iter00.tsv.

Usage
-----
    python compare_unk_sweep.py
"""
import collections
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CHROMS = ("chrIV", "chrVII", "chrXV")
LAMS = [("1", "counts_conctune_00"), ("0.3", "counts_unk_0p3"), ("0.1", "counts_unk_0p1"),
        ("0.03", "counts_unk_0p03"), ("0.01", "counts_unk_0p01")]
NON_TF = {"background", "nucleosome", "unknown", "nuc_center", "nuc_start", "nuc_end"}


def load(dirname):
    """Sum per-chromosome occupancy over the three sweep chromosomes."""
    occ = collections.Counter()
    for c in CHROMS:
        p = os.path.join(HERE, "conc_tuning", dirname, c + ".tsv")
        if not os.path.isfile(p):
            return None
        for r in csv.DictReader(open(p), delimiter="\t"):
            occ[r["factor"]] += float(r["occ"])
    return occ


def targets():
    """MacIsaac target per group, restricted to the three sweep chromosomes."""
    tg, per = {}, collections.Counter()
    for r in csv.DictReader(open(os.path.join(HERE, "inputs",
                                              "conc_targets_macisaac_c1.tsv")), delimiter="\t"):
        tg[r["motif"]] = r
        if r["has_target"] == "1":
            per[r["group"]] = sum(int(r[c]) for c in CHROMS)
    return tg, per


def main():
    tg, targ = targets()
    groups = collections.defaultdict(list)
    for m, r in tg.items():
        if r["has_target"] == "1":
            groups[r["group"]].append(m)

    cols, missing = [], []
    for lam, d in LAMS:
        o = load(d)
        (cols if o else missing).append((lam, o) if o else lam)
    if missing:
        print("still counting: lambda %s\n" % ", ".join(missing))
        if not cols:
            return

    print("unknown-concentration sweep -- chrIV+chrVII+chrXV (30.8% of genome)")
    print("nucleosome pinned to baseline throughout; only `unknown` moves.\n")

    # --- global state occupancy ---
    print("%-16s %s" % ("state", "".join("%12s" % ("lam=" + l) for l, _ in cols)))
    print("-" * (16 + 12 * len(cols)))
    for k in ("unknown", "nucleosome", "background"):
        print("%-16s %s" % (k, "".join("%12.0f" % o[k] for _, o in cols)))
    tf_tot = [sum(v for f, v in o.items() if f not in NON_TF) for _, o in cols]
    print("%-16s %s" % ("all 153 TFs", "".join("%12.0f" % t for t in tf_tot)))
    print("%-16s %s" % ("unknown/TFs", "".join("%12.2f" % (o["unknown"] / t)
                                               for (_, o), t in zip(cols, tf_tot))))

    # --- how close are the targeted groups? ---
    print("\n%-16s %s" % ("fit to MacIsaac", "".join("%12s" % ("lam=" + l) for l, _ in cols)))
    print("-" * (16 + 12 * len(cols)))
    stats = []
    for lam, o in cols:
        gaps = []
        for g, ms in groups.items():
            T = targ.get(g, 0)
            E = sum(o.get(m, 0.0) for m in ms)
            if T > 0 and E > 0:
                gaps.append(abs(math.log(T) - math.log(E)))
        gaps.sort()
        stats.append((len(gaps), gaps[len(gaps) // 2], sum(gaps) / len(gaps),
                      sum(1 for x in gaps if x < math.log(2))))
    print("%-16s %s" % ("groups scored", "".join("%12d" % s[0] for s in stats)))
    print("%-16s %s" % ("median |log gap|", "".join("%12.3f" % s[1] for s in stats)))
    print("%-16s %s" % ("mean |log gap|", "".join("%12.3f" % s[2] for s in stats)))
    print("%-16s %s" % ("within 2x", "".join("%12d" % s[3] for s in stats)))

    # --- the four factors that motivated the sweep ---
    print("\nthe factors that lose at their own sites (occupancy, target in brackets)")
    print("%-14s %8s %s" % ("group", "target", "".join("%12s" % ("lam=" + l) for l, _ in cols)))
    print("-" * (14 + 8 + 12 * len(cols) + 1))
    for g in ("FKH2", "MBP1", "STB5", "HAP1", "ABF1", "REB1", "PHO2", "STE12"):
        if g not in groups:
            continue
        vals = [sum(o.get(m, 0.0) for m in groups[g]) for _, o in cols]
        print("%-14s %8d %s" % (g, targ.get(g, 0), "".join("%12.1f" % v for v in vals)))


if __name__ == "__main__":
    main()
