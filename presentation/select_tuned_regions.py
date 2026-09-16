#!/usr/bin/env python3
"""Choose the windows shown in the Tuned Occupancy Browser (tuner v2: fw01 / sw01 / bw01).

Rule, per chromosome (chrXIV, chrII = tuning; chrIV = holdout), 4 kb windows on a 1 kb grid:
  A  "sites"     top 8 by (MacIsaac c1 sites + Rossi _CX summits) over the 58 tuned groups,
                 greedy, non-overlapping.
  B  "disagree"  top 4 by PRIVATE calls at the final round: a call (posterior >= 0.10, one of
                 the 61 tuned motifs, from the tuning loop's own counts_tw_<run>_NN/calls/)
                 made by one layer config (fib / seq / both) with no call of the same group
                 within 30 bp in either other config. Non-overlapping with A.
  C  "random"    2 windows drawn with seed 20260915, non-overlapping with A and B.
References are rossi_validate.load_references (MacIsaac merged-site centers; _CX (s+e)//2).
Writes tuned_regions.tsv, tuned_runs.tsv and tuned_regions_selection.json next to this file.
"""
import bisect, csv, json, os, random, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(HERE, "..", "analysis")
sys.path.insert(0, AN)
os.chdir(AN)
import tune_w, rossi_validate as RV  # noqa: E402

WIN, STEP, TOL = 4000, 1000, 30
N_A, N_B, N_C = 8, 4, 2
CHROMS = [("chrXIV", "tuning"), ("chrII", "tuning"), ("chrIV", "holdout")]
FINAL = {"fib": ("fw01", 7), "seq": ("sw01", 5), "both": ("bw01", 7)}

RUNS = [  # label, run, round  (label is the join key; identical for every region)
    ("fib r0", "fw01", 0), ("fib final", "fw01", 7),
    ("seq r0", "sw01", 0), ("seq final", "sw01", 5),
    ("both r0", "bw01", 0), ("both final", "bw01", 7),
]
LEGACY = ("u001 r7 legacy", "robocop_genome_ct_u001_07")


def sizes():
    return {l.split()[0]: int(l.split()[1]) for l in open("inputs/sacCer3.chrom.sizes")}


def load_calls(run, rnd, chrom, motif2group):
    p = "conc_tuning/counts_tw_%s_%02d/calls/%s.tsv" % (run, rnd, chrom)
    out = defaultdict(list)
    for r in csv.DictReader(open(p), delimiter="\t"):
        g = motif2group.get(r["factor"])
        if g:
            out[g].append(int(r["center"]))
    return {g: sorted(v) for g, v in out.items()}


def count_in(sorted_pos, a, b):
    return bisect.bisect_right(sorted_pos, b) - bisect.bisect_left(sorted_pos, a)


def greedy(scores, n, taken):
    """scores: [(score, start)] -> up to n starts with score > 0, no overlap with taken."""
    out = []
    for sc, s in sorted(scores, key=lambda t: (-t[0], t[1])):
        if len(out) == n or sc <= 0:
            break
        if all(abs(s - t) >= WIN for t in taken + out):
            out.append(s)
    return out


def main():
    G, _, _ = tune_w.factor_set()
    refs, _ = RV.load_references(G)
    m2g = {m: g for g, ms in G.items() for m in ms}
    L = sizes()
    rows, sel = [], {"rule": __doc__.strip(), "chroms": {}}
    rng = random.Random(20260915)
    for chrom, role in CHROMS:
        sites = sorted(p for g in G for p in refs[g]["macisaac"].get(chrom, []))
        rossi = sorted(p for g in G for p in (refs[g]["rossi_cx"] or {}).get(chrom, []))
        starts = list(range(1, L[chrom] - WIN + 2, STEP))
        a_sc = [(count_in(sites, s, s + WIN - 1) + count_in(rossi, s, s + WIN - 1), s)
                for s in starts]
        A = greedy(a_sc, N_A, [])

        calls = {k: load_calls(r, t, chrom, m2g) for k, (r, t) in FINAL.items()}
        private = []
        for k in calls:
            for g, pos in calls[k].items():
                others = [calls[o].get(g, []) for o in calls if o != k]
                for p in pos:
                    if not any(count_in(o, p - TOL, p + TOL) for o in others):
                        private.append(p)
        private.sort()
        b_sc = [(count_in(private, s, s + WIN - 1), s) for s in starts]
        B = greedy(b_sc, N_B, A)

        C = []
        while len(C) < N_C:
            s = rng.choice(starts)
            if all(abs(s - t) >= WIN for t in A + B + C):
                C.append(s)
        sc_a, sc_b = dict((s, v) for v, s in a_sc), dict((s, v) for v, s in b_sc)
        picked = [("sites", s) for s in A] + [("disagree", s) for s in B] + [("random", s) for s in C]
        sel["chroms"][chrom] = []
        for kind, s in sorted(picked, key=lambda t: t[1]):
            e = s + WIN - 1
            key = "%s_%d_%s" % (chrom, s // 1000, kind)
            lab = "%s %s–%s kb · %s%s" % (chrom, f"{(s - 1) / 1000:.0f}", f"{e / 1000:.0f}",
                                         kind, " (holdout)" if role == "holdout" else "")
            rows.append((key, lab, "%s:%d-%d" % (chrom, s, e), "%s:%d-%d" % (chrom, s, e),
                         "tuned_runs.tsv"))
            sel["chroms"][chrom].append(dict(key=key, kind=kind, start=s, end=e,
                                             ref_sites=sc_a[s], private_calls=sc_b[s]))
        print(chrom, "A", A, "B", B, "C", C)
    with open(os.path.join(HERE, "tuned_regions.tsv"), "w") as fh:
        fh.write("# key\tlabel\tregion\tview\truns_file   (built by select_tuned_regions.py)\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    with open(os.path.join(HERE, "tuned_runs.tsv"), "w") as fh:
        fh.write("# label\ttuning_outDir (chrXIV, chrII)\tholdout_outDir (chrIV)\n")
        for lab, r, t in RUNS:
            fh.write("%s\trobocop_chrXIV_chrII_tw_%s_%02d\trobocop_chrIV_tw_%s_%02d\n"
                     % (lab, r, t, r, t))
        fh.write("%s\t%s\t%s\n" % (LEGACY[0], LEGACY[1], LEGACY[1]))
    json.dump(sel, open(os.path.join(HERE, "tuned_regions_selection.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
