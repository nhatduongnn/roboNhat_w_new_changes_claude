"""Score the concentration-tuning rounds against Rossi ChIP-exo, next to MacIsaac.

Why this exists
---------------
Both tuning campaigns (u001, m001) moved every MacIsaac-targeted factor's lambda toward its
MacIsaac site count. Rossi was held back the whole time as validation (HANDOFF §5.2). This
answers: against Rossi, did the same rounds get better, worse, or stay flat?

Inputs (nothing here re-reads a posterior)
------------------------------------------
calls      conc_tuning/calls_ct_<run>_NN/calls/<chrom>.tsv, written by `count_calls.py --calls`
           (sbatch_count_calls_positions.sh). The same fixed-threshold (posterior >= 0.10)
           calls the tuning loop counted and matched to MacIsaac, invalid posteriors dropped the
           same way. Before any of it is used, every round's MacIsaac match is re-derived from
           these calls and must equal counts_ct_<run>_NN/macisaac/ exactly.

references, per tuning group (inputs/conc_targets_macisaac_c1.tsv, the 81 MacIsaac groups)
  macisaac     make_conc_targets.load_macisaac_c1_sites(): interval union, 20 bp slop, midpoint.
  rossi_cx     <TF>_CX.bed, Rossi's merged ChExMix summits, no motif requirement. Used as
               shipped: they are already merged across replicates. Center = (start+end)//2,
               score_factors.py's convention for Rossi beds.
  NOT used:    inputs/rossi_peak_w_strand_all_TFs.bed (the motif-anchored subset) -- by user
               decision the merged ChExMix calls are the vetted Rossi set; and
               rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed -- 74 TFs,
               peakVal-filtered, and the set the fitted fiber footprints were trained on.

Factor join: a group joins a Rossi TF when the names agree case-insensitively, or through the
gene's systematic name in inputs/sacCer3.gtf (YML081W = TDA9, YDR520C = URC2); the second kind
is reported as an alias join.

Match rule -- identical to count_calls.match_macisaac
-----------------------------------------------------
Per group per chromosome, the group's motifs' calls are pooled and matched to reference centers
with score_robocop.match_peaks at 30 bp: greedy, one-to-one, references in coordinate order,
each taking its nearest unused call. So `matched` is at once "sites with a call" and "calls at a
site":  precision = matched / calls,  recall = matched / sites.  `fast_match` below is an exact
re-implementation (checked against match_peaks on every MacIsaac table before use).

Genic fraction
--------------
A call is genic when it lies between some gene's ATG and stop codon (rossi_genic.py's ORF reader
and Genome, one rule, no windows). Calls are 1-based decode positions, so position-1 goes to the
0-based ORF test; Rossi's own genic% comes from rossi_genic/rossi_genic_all_TFs.tsv.

Outputs: conc_tuning/rossi_validation/
  references.tsv          per group: site counts per reference, join, fitted flag, Rossi _cx genic%,
                          MacIsaac sites with a Rossi site within 30 bp
  match_<run>.tsv         run round group ref n_ref n_calls n_matched
  genic_<run>.tsv         run round group n_calls n_genic

Usage:  python rossi_validate.py u001=7 m001=7
"""
import bisect
import collections
import csv
import glob
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import count_calls as CC                # noqa: E402
import make_conc_targets as T           # noqa: E402
import rossi_genic as RG                 # noqa: E402
import score_robocop as S               # noqa: E402

CXDIR = "/usr/project/xtmp/nd141/projects/data/rossi_strand"
GENIC_TSV = os.path.join(HERE, "rossi_genic", "rossi_genic_all_TFs.tsv")
OUT = os.path.join(HERE, "conc_tuning", "rossi_validation")
TOL = CC.MATCH_TOL
SLOP = T.MERGE_SLOP
FITTED = {"Abf1_murphy", "Cin5_murphy", "Fhl1_zhu", "Fkh1_zhu", "Mcm1_zhu", "Nhp6a_zhu",
          "Rap1_telomeric", "Reb1_badis", "Sko1_murphy", "Spt15_zhu", "Tbf1_zhu", "Ume6_zhu"}
REFS = ("macisaac", "rossi_cx")


def fast_match(pred, ref, tol=TOL):
    """Exactly score_robocop.match_peaks' tp, in O((n+m) log n).

    match_peaks walks refs in sorted order and gives each the nearest unused pred (ties to the
    lower coordinate, because its scan is ascending with a strict <). The nearest unused pred on
    each side of a ref is found with two path-compressed 'next unused' pointer arrays.
    """
    pred, ref = sorted(pred), sorted(ref)
    n = len(pred)
    if not n or not ref:
        return 0
    left = list(range(n))       # left[i]: largest unused index <= i  (-1 if none)
    right = list(range(n + 1))  # right[i]: smallest unused index >= i (n if none)

    def find_l(i):
        root = i
        while root >= 0 and left[root] != root:
            root = left[root]
        while i >= 0 and left[i] != i:
            left[i], i = root, left[i]
        return root

    def find_r(i):
        root = i
        while root < n and right[root] != root:
            root = right[root]
        while i < n and right[i] != i:
            right[i], i = root, right[i]
        return root

    tp = 0
    for r in ref:
        k = bisect.bisect_left(pred, r)
        L, R = find_l(k - 1) if k > 0 else -1, find_r(k)
        dl = r - pred[L] if L >= 0 else tol + 1
        dr = pred[R] - r if R < n else tol + 1
        if min(dl, dr) > tol:
            continue
        j = L if dl <= dr else R
        left[j] = j - 1
        right[j] = j + 1
        tp += 1
    return tp


def chroms():
    return T.chroms()


def merge_sites(spans):
    """MacIsaac's rule (make_conc_targets.load_macisaac_c1_sites): union with SLOP, midpoint.
    Used here only to report how many _cx summits that rule would fuse; it is not applied."""
    out, cur = [], None
    for a, b in sorted(spans):
        if cur is None or a - SLOP > cur[1]:
            if cur:
                out.append((cur[0] + cur[1]) // 2)
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        out.append((cur[0] + cur[1]) // 2)
    return out


def load_groups():
    g = collections.OrderedDict()
    for r in csv.DictReader(open(CC.TARGETS), delimiter="\t"):
        if r["has_target"] == "1":
            g.setdefault(r["group"], []).append(r["motif"])
    return g


def alias_names():
    """systematic -> standard gene name, lowercased, from the gtf."""
    out = {}
    for line in open(RG.GTF):
        f = line.split("\t")
        if len(f) > 8 and f[2] == "gene" and 'gene_name "' in f[8]:
            sid = f[8].split('gene_id "', 1)[1].split('"', 1)[0]
            out[sid.lower()] = f[8].split('gene_name "', 1)[1].split('"', 1)[0].lower()
    return out


def load_references(groups):
    valid = set(chroms())
    # MacIsaac
    mac = collections.defaultdict(lambda: collections.defaultdict(list))
    for c, tf, center in T.load_macisaac_c1_sites():
        mac[tf][c].append(center)

    # Rossi _cx
    cx_files = {os.path.basename(p)[:-len("_CX.bed")].lower(): p
                for p in glob.glob(os.path.join(CXDIR, "*_CX.bed"))}

    alias = alias_names()
    refs, meta = {}, []
    for g, motifs in groups.items():
        key = g.lower()
        cx_key = key if key in cx_files else (alias.get(key) if alias.get(key) in cx_files else None)
        cx = collections.defaultdict(list)
        n_cx_raw = 0
        if cx_key:
            d = pd.read_csv(cx_files[cx_key], sep="\t", header=None, usecols=[0, 1, 2])
            d.columns = ["chr", "start", "end"]
            d["chr"] = d["chr"].map(RG.ARABIC2ROMAN).fillna(d["chr"])
            d = d[d["chr"].isin(valid)]
            n_cx_raw = len(d)
            for c, s, e in d.itertuples(index=False):
                cx[c].append((s + e) // 2)
        refs[g] = dict(macisaac=mac.get(g, {}), rossi_cx=cx if cx_key else None)
        # cx summits that the 20 bp merge rule would fuse (reported, not applied)
        cx_fusable = sum(len(v) - len(merge_sites([(p, p) for p in v])) for v in cx.values())
        meta.append(dict(
            group=g, motifs=",".join(motifs), fitted=int(any(m in FITTED for m in motifs)),
            n_macisaac=sum(len(v) for v in mac.get(g, {}).values()),
            cx_tf=cx_key or "", cx_join=("" if not cx_key else ("name" if cx_key == key else "alias")),
            n_cx=n_cx_raw, cx_fusable_20bp=cx_fusable))
    return refs, meta


def within_any(sites, other, tol=TOL):
    """How many of `sites` have ANY `other` center within tol (not one-to-one)."""
    o = sorted(other)
    hit = 0
    for s in sites:
        k = bisect.bisect_left(o, s - tol)
        hit += int(k < len(o) and o[k] <= s + tol)
    return hit


def read_calls(run, it):
    d = os.path.join(HERE, "conc_tuning", "calls_ct_%s_%02d" % (run, it))
    calls = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in chroms():
        p = os.path.join(d, "calls", c + ".tsv")
        if not os.path.exists(p):
            sys.exit("missing %s" % p)
        with open(p) as fh:
            fh.readline()
            for line in fh:
                f, ch, center, _, _ = line.rstrip("\n").split("\t")
                calls[f][ch].append(int(center))
    return calls


def main():
    runs = dict(kv.split("=") for kv in sys.argv[1:]) or {"u001": "7", "m001": "7"}
    os.makedirs(OUT, exist_ok=True)
    groups = load_groups()
    assert len(groups) == 81, len(groups)
    refs, meta = load_references(groups)

    # Rossi genic% and MacIsaac-in-Rossi overlap, per group
    gen = {r["TF"].lower(): r for r in csv.DictReader(open(GENIC_TSV), delimiter="\t")}
    for m in meta:
        g = m["group"]
        m["genic_pct_cx"] = gen.get(m["cx_tf"], {}).get("genic_pct_cx", "NA") if m["cx_tf"] else "NA"
        m["n_cx_genic_tsv"] = gen.get(m["cx_tf"], {}).get("n_cx", "NA") if m["cx_tf"] else "NA"
        R = refs[g]["rossi_cx"]
        m["macisaac_in_rossi_cx"] = ("" if R is None else
                                     sum(within_any(refs[g]["macisaac"].get(c, []), R.get(c, []))
                                         for c in chroms()))
    cols = list(meta[0].keys())
    with open(os.path.join(OUT, "references.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for m in meta:
            fh.write("\t".join(str(m[c]) for c in cols) + "\n")

    orfs = RG.read_orfs()
    genome = RG.Genome(orfs)

    for run, last in runs.items():
        if not os.path.exists(os.path.join(HERE, "conc_tuning", run, "STOPPED")):
            sys.exit("%s has no STOPPED marker: still running, refusing to score it" % run)
        mrows, grows = [], []
        for it in range(int(last) + 1):
            calls = read_calls(run, it)
            # 1. the calls must reproduce the tuning loop's own MacIsaac match, exactly.
            side = {(r["group"], r["chrom"]): r for c in chroms() for r in csv.DictReader(
                open(os.path.join(HERE, "conc_tuning", "counts_ct_%s_%02d" % (run, it), "macisaac", c + ".tsv")),
                delimiter="\t")}
            for g, motifs in groups.items():
                for c in chroms():
                    pc = [x for m in motifs for x in calls.get(m, {}).get(c, [])]
                    ref_c = refs[g]["macisaac"].get(c, [])
                    tp = fast_match(pc, ref_c)
                    s = side[(g, c)]
                    if (len(pc), len(ref_c), tp) != (int(s["n_calls"]), int(s["n_macisaac"]), int(s["n_matched"])):
                        sys.exit("MISMATCH %s %02d %s %s: calls/sites/tp %s vs sidecar %s"
                                 % (run, it, g, c, (len(pc), len(ref_c), tp),
                                    (s["n_calls"], s["n_macisaac"], s["n_matched"])))
                    if it == 0 and c == "chrI":          # fast_match == match_peaks, spot-checked
                        for ref in REFS:
                            R = refs[g][ref]
                            if R is not None:
                                assert fast_match(pc, R.get(c, [])) == S.match_peaks(pc, R.get(c, []), TOL)["tp"]
            # 2. score against each reference, and the genic fraction of the calls
            for g, motifs in groups.items():
                for ref in REFS:
                    R = refs[g][ref]
                    if R is None:
                        continue
                    nr = nc = tp = 0
                    for c in chroms():
                        pc = [x for m in motifs for x in calls.get(m, {}).get(c, [])]
                        rc = R.get(c, [])
                        nr, nc, tp = nr + len(rc), nc + len(pc), tp + fast_match(pc, rc)
                    mrows.append((run, it, g, ref, nr, nc, tp))
                ch, pos = [], []
                for m in motifs:
                    for c, v in calls.get(m, {}).items():
                        ch += [c] * len(v)
                        pos += v
                ng = int(genome.genic(np.array(ch, dtype=object), np.array(pos, dtype=int) - 1).sum()) if pos else 0
                grows.append((run, it, g, len(pos), ng))
            print("%s round %02d: MacIsaac sidecar reproduced for 81 groups x 16 chromosomes; scored" % (run, it))
        with open(os.path.join(OUT, "match_%s.tsv" % run), "w") as fh:
            fh.write("run\tround\tgroup\tref\tn_ref\tn_calls\tn_matched\n")
            for r in mrows:
                fh.write("\t".join(map(str, r)) + "\n")
        with open(os.path.join(OUT, "genic_%s.tsv" % run), "w") as fh:
            fh.write("run\tround\tgroup\tn_calls\tn_genic\n")
            for r in grows:
                fh.write("\t".join(map(str, r)) + "\n")
        print("wrote %s/{match,genic}_%s.tsv" % (os.path.relpath(OUT, HERE), run))


if __name__ == "__main__":
    main()
