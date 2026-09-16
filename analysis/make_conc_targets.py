"""Per-motif target site counts, for concentration calibration.

Why this exists
---------------
RoboCOP's TF concentrations are never fitted: `tf_prob` is whatever `calculateKD` returns
from the PWM alone (parameterize.py:37-42), a pure function of motif length and information
content with no data in it, and `robocop_em.py:116` pins `iterations = 0`. Nhp6a sits 4019x
above Abf1 because it is shorter, not because anything measured it. The model consequently
over-calls by 1-2 orders of magnitude.

To correct that we need, per TF, a number saying how many sites it actually has. This script
produces that table. It is the only place a target set is parsed, so switching targets is a
`--source` flag rather than an edit.

The sources
-----------
`macisaac_c1`  (default) -- MacIsaac 2006 conserved binding sites, p<0.005, category c1.
    /usr/project/xtmp/nd141/projects/replicate_prob_dyad_plot/data/ref-data/MacIsaac_p005_c1_V64_SGD.gff3
    27,870 sites, 119 TFs, sacCer3/V64. NOT the 2-TF file in inputs/ -- that one
    (MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed) is a hand-filtered slice of this, keeping
    151 of 315 ABF1 and 156 of 293 REB1. Verified: all 307 of its sites match this file at
    exactly 0 bp offset, so the two agree on coordinates and differ only in membership.
    Chosen as the default because it is a conserved-MOTIF-anchored set -- the same kind of
    object RoboCOP emits -- unlike ChExMix summits, which include indirect binding a
    PWM-driven HMM cannot represent.

`rossi_cx`     -- Rossi merged ChExMix summits, 182,582 peaks / 378 TFs, from
    rossi_genic/rossi_peaks_genic_all_cx.tsv (roman chrom, capitalised TF).
`rossi_motif`  -- the motif-filtered subset, 29,105 peaks / 358 TFs, from
    rossi_genic/rossi_peaks_genic_all_motif.tsv (roman chrom, lowercase TF).

Rossi is normally held back as VALIDATION, so `macisaac_c1` is what the tuning loop uses.
The Rossi sources exist so the same loop can be re-run against a different target later
without touching code.

The join, and its one ambiguity
-------------------------------
RoboCOP motif -> TF name is `motif.split('_')[0].upper()`, the same rule
`rossi_genic_all.py` uses. 153 motifs collapse to 150 distinct prefixes because **Rap1 has
four motifs** (Rap1_zhu, Rap1_motif1, Rap1_motif2, Rap1_telomeric) against a single target.

That is a genuine many-to-one, and it is handled by emitting a `group` column: the four Rap1
rows share one group, and the tuning loop must compare the SUM of their predicted counts
against the one target and apply the SAME lambda to all four. Splitting the target four ways
would be wrong -- the four PWMs are alternative descriptions of one factor, not four factors.

Usage
-----
    python make_conc_targets.py                          # -> inputs/conc_targets_macisaac_c1.tsv
    python make_conc_targets.py --source rossi_cx
    python make_conc_targets.py --verify                 # run the self-checks and exit
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

MACISAAC_C1 = ("/usr/project/xtmp/nd141/projects/replicate_prob_dyad_plot/data/ref-data/"
               "MacIsaac_p005_c1_V64_SGD.gff3")
ROSSI_CX = os.path.join(HERE, "rossi_genic", "rossi_peaks_genic_all_cx.tsv")
ROSSI_MOTIF = os.path.join(HERE, "rossi_genic", "rossi_peaks_genic_all_motif.tsv")
MEME = os.path.join(HERE, "inputs", "motifs_meme.txt")
REPO_MACISAAC = os.path.join(HERE, "inputs", "MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed")
CHROM_SIZES = os.path.join(HERE, "inputs", "sacCer3.chrom.sizes")


def motif_names(meme=MEME):
    """The 153 motif names, in MEME file order."""
    out = []
    with open(meme) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                out.append(line.split()[1])
    if not out:
        sys.exit("no MOTIF records in %s" % meme)
    return out


def tf_of(motif):
    """RoboCOP motif name -> target-set TF key. Rap1_telomeric -> RAP1."""
    return motif.split("_")[0].upper()


MERGE_SLOP = 20


def load_macisaac_c1(path=MACISAAC_C1, merge=True):
    """(chrom, TF) rows from the MacIsaac GFF, one row per DISTINCT SITE.

    The file carries a HEADER row whose first field is the literal 'chr'; GFF coordinates
    are 1-based inclusive. The header must be skipped or it becomes a phantom TF named 'TF'.

    MERGING IS NOT OPTIONAL FOR A COUNT. The gff3 emits redundant overlapping motif calls
    for the same physical site, so raw row counts overstate the number of sites -- ABF1 has
    315 rows but 300 sites, RAP1 282 rows but 233. Since the whole point of this table is
    "how many sites does this TF have", counting rows would inflate every target, and worst
    for the factors with the most redundancy (FHL1 1.20x, PHO2 1.19x, YAP5 1.17x).

    The rule is INTERVAL UNION WITH 20 bp SLOP, per a user instruction of 2026-06-24 that
    also fixes the midpoint as the anchor. It is pinned by reproducing the three recorded
    counts exactly: ABF1 300, REB1 279, RAP1 233. Midpoint-gap merging does NOT reproduce
    them (it gives 309/284/247), so the rule is not interchangeable and `verify()` checks it.
    """
    if not merge:
        return [k for k, v in _macisaac_intervals(path).items() for _ in v]
    return [(c, tf) for c, tf, _ in load_macisaac_c1_sites(path)]


def _macisaac_intervals(path):
    """(chrom, TF) -> list of raw (start, end) motif intervals, GFF 1-based inclusive."""
    iv = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[0] == "chr" or "Name=" not in f[8]:
                continue
            key = (f[0], f[8].split("Name=")[1].strip().upper())
            iv.setdefault(key, []).append((int(f[3]), int(f[4])))
    return iv


def load_macisaac_c1_sites(path=MACISAAC_C1):
    """(chrom, TF, center) per DISTINCT SITE -- the same merge `load_macisaac_c1` counts.

    The center is the midpoint of the merged interval, which is what decoded calls are matched
    against. `load_macisaac_c1` is derived from this, so a site count and a site list can never
    disagree about what a site is.
    """
    sites = []
    for (c, tf), spans in _macisaac_intervals(path).items():
        cur = None
        for a, b in sorted(spans):
            if cur is None or a - MERGE_SLOP > cur[1]:
                if cur:
                    sites.append((c, tf, (cur[0] + cur[1]) // 2))
                cur = [a, b]                  # a new distinct site
            else:
                cur[1] = max(cur[1], b)       # same site, another overlapping call
        sites.append((c, tf, (cur[0] + cur[1]) // 2))
    return sites


def load_rossi(path):
    """(chrom, TF) rows from a rossi_genic per-peak TSV. Chrom is already roman."""
    rows = []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ci, ti = header.index("chr"), header.index("TF")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(ci, ti):
                continue
            rows.append((f[ci], f[ti].strip().upper()))
    return rows


SOURCES = {
    "macisaac_c1": (lambda: load_macisaac_c1(), MACISAAC_C1),
    "rossi_cx": (lambda: load_rossi(ROSSI_CX), ROSSI_CX),
    "rossi_motif": (lambda: load_rossi(ROSSI_MOTIF), ROSSI_MOTIF),
}


def chroms():
    """Nuclear chromosomes in sacCer3.chrom.sizes order, chrM excluded.

    MacIsaac has no chrM and Rossi has none either, so including it would give every TF a
    zero column that means 'not assayed' rather than 'no sites'.
    """
    out = []
    with open(CHROM_SIZES) as fh:
        for line in fh:
            c = line.split()[0]
            if c != "chrM":
                out.append(c)
    return out


def build(source):
    rows = SOURCES[source][0]()
    motifs = motif_names()
    per = collections.Counter(rows)
    tot = collections.Counter(t for _, t in rows)
    cs = chroms()

    # Rap1's four motifs share one target; group id is the TF key itself.
    by_tf = collections.defaultdict(list)
    for m in motifs:
        by_tf[tf_of(m)].append(m)

    recs = []
    for m in motifs:
        tf = tf_of(m)
        recs.append(dict(
            motif=m, tf=tf, group=tf, n_motifs_in_group=len(by_tf[tf]),
            has_target=int(tf in tot), n_total=int(tot.get(tf, 0)),
            **{c: int(per.get((c, tf), 0)) for c in cs}))
    return recs, cs, rows


def write(recs, cs, out):
    cols = ["motif", "tf", "group", "n_motifs_in_group", "has_target", "n_total"] + cs
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in recs:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")


def verify():
    """Self-checks. These are the reason to trust the join, not a note claiming it works."""
    ok = True
    raw = load_macisaac_c1(merge=False)
    rows = load_macisaac_c1(merge=True)
    rawtot = collections.Counter(t for _, t in raw)
    tot = collections.Counter(t for _, t in rows)

    # 1. raw row counts, and the merged SITE counts the 2026-06-24 rule fixes
    for tf, want in (("ABF1", 315), ("REB1", 293)):
        got = rawtot[tf]
        print("  %-28s %5d  (expect %d)  %s" % ("raw rows " + tf, got, want,
                                                "OK" if got == want else "MISMATCH"))
        ok &= got == want
    for tf, want in (("ABF1", 300), ("REB1", 279), ("RAP1", 233)):
        got = tot[tf]
        print("  %-28s %5d  (expect %d)  %s" % ("merged sites " + tf, got, want,
                                                "OK" if got == want else "MISMATCH"))
        ok &= got == want
    print("  %-28s %5d -> %5d  (%.1f%% redundant)" % (
        "total rows -> sites", len(raw), len(rows), 100 * (1 - len(rows) / len(raw))))
    ok &= len(raw) == 27870
    print("  %-28s %5d  (expect 119)   %s" % ("distinct TFs", len(tot),
                                              "OK" if len(tot) == 119 else "MISMATCH"))
    ok &= len(tot) == 119

    # 2. the repo's 2-TF bed must be a coordinate-exact SUBSET of c1
    mac = collections.defaultdict(list)
    with open(MACISAAC_C1) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[0] == "chr":
                continue
            mac[(f[0], f[8].split("Name=")[1].strip().upper())].append(int(f[3]))
    off = []
    with open(REPO_MACISAAC) as fh:
        for line in fh:
            f = line.split("\t")
            cand = mac.get((f[0], f[3].strip().upper()), [])
            if cand:
                off.append(min(abs(int(f[1]) + 1 - x) for x in cand))  # bed 0-based -> gff 1-based
    exact = sum(1 for d in off if d == 0)
    print("  %-28s %d/%d at 0 bp offset  %s" % ("repo bed inside c1", exact, len(off),
                                                "OK" if exact == len(off) == 307 else "MISMATCH"))
    ok &= exact == len(off) == 307

    # 3. the join, and the one many-to-one case
    motifs = motif_names()
    prefixes = {tf_of(m) for m in motifs}
    joined = sorted(prefixes & set(tot))
    print("  %-28s %d motifs -> %d prefixes, %d join c1" % (
        "join", len(motifs), len(prefixes), len(joined)))
    ok &= len(motifs) == 153 and len(joined) == 81
    multi = {p: [m for m in motifs if tf_of(m) == p] for p in prefixes}
    multi = {k: v for k, v in multi.items() if len(v) > 1}
    print("  %-28s %s" % ("many-to-one groups", multi if multi else "none"))
    ok &= list(multi) == ["RAP1"] and len(multi["RAP1"]) == 4
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="macisaac_c1", choices=sorted(SOURCES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true", help="run self-checks and exit")
    args = ap.parse_args()

    if args.verify:
        print("verifying make_conc_targets.py")
        sys.exit(0 if verify() else 1)

    path = SOURCES[args.source][1]
    if not os.path.exists(path):
        sys.exit("source file missing: %s" % path)
    recs, cs, rows = build(args.source)
    out = args.out or os.path.join(HERE, "inputs", "conc_targets_%s.tsv" % args.source)
    write(recs, cs, out)

    n_t = sum(1 for r in recs if r["has_target"])
    n30 = sum(1 for r in recs if r["n_total"] >= 30)
    print("source     %s\n           %s" % (args.source, path))
    print("sites      %d over %d TFs" % (len(rows), len({t for _, t in rows})))
    print("motifs     %d, of which %d have a target (%d with >=30 sites)"
          % (len(recs), n_t, n30))
    print("wrote      %s" % out)


if __name__ == "__main__":
    main()
