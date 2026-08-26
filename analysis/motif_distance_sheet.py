#!/usr/bin/env python
"""Three-way motif distance sheet: native vs JASPAR vs Rossi, ED and KL.

THREE DATASETS, NO PRIVILEGED ONE.

  native  -- what RoboCOP ships and decodes with (inputs/motifs_meme.txt);
             internally a mix of murphy / zhu / badis sub-collections
  jaspar  -- JASPAR CORE fungi, latest
  rossi   -- Rossi et al. ChExMix, the pre-computed MEME motifs

Rossi is not a reference and neither is JASPAR; they are simply two more
measurements of the same factor.  So the question is asked symmetrically: of the
three, is one of them the odd one out?  With three points there is always a
closest pair, and the odd one out is the member opposite it -- equivalently, the
member with the largest mean distance to the other two.  The margin between the
two smallest pairwise distances says whether that verdict is meaningful or
whether all three are simply equidistant.

Distances are TOMTOM's own column functions, applied directly and reported RAW
as a mean per aligned column -- no p-value.  The p-value asks 'closer than two
random matrices', which saturates for any pair sharing a strong core, and that
is exactly what hid the ABF1 spacer defect.

    ED(X,Y)  = sqrt( sum_a (X_a - Y_a)^2 )
    KLD(X,Y) = 1/2 ( sum_a X_a ln(X_a/Y_a) + sum_a Y_a ln(Y_a/X_a) )

after -motif-pseudo 0.1 spread by the yeast genome background.

ONE COMMON WINDOW PER ROW.  The three matrices are rarely the same width --
RAP1 is native 20 / JASPAR 12 / Rossi 16 -- and if each pair is aligned on its
own the three numbers in a row rest on different column counts (12 / 16 / 12
for that RAP1 row).  They are then not commensurable, and "which pair is
closest" can be decided by how many columns each pair happened to share rather
than by the matrices.

So the window is fixed once per row, in two stages (see common_window).

  FRAME.  Every offset and orientation of the non-shortest motifs is searched
  against a fixed W-column window (W = width of the shortest motif), columns a
  motif cannot reach priced against the genome background, and the placement
  minimising the SUM of the pairwise mean KLs is taken -- a sum-of-pairs
  multiple alignment, symmetric in the datasets, no anchor privileged beyond
  being the shortest.  Scoring against a FIXED-width window is what stops a
  variable window being minimised by shrinking onto the columns that happen to
  match.

  NUMBERS.  All three pairwise distances are then reported over the columns
  every motif in that frame actually reaches -- the intersection of the three
  placements.  One column set, all three pairs, and nothing is ever scored
  against a matrix that never measured it.

Aligning each pair the way TOMTOM would and intersecting the results is the
obvious approach, and this reduces to it wherever it is well defined.  It is
not always: alignment is not transitive, and the three pairwise optima are
mutually realisable in a single frame in only 59% of rows
(pairwise_frame_consistency.py).  TOMTOM itself cannot arbitrate -- it compares
one query motif to target motifs and has no three-motif mode.

Two consequences worth stating.  (1) Columns outside the window are not scored,
so a matrix longer than the others is compared only on the part the others also
measured; the extra width shows in w_native / w_jaspar / w_rossi and in the
logo, not in the distance.  (2) The joint frame can still cost a pair its own
best partial alignment; window_cost_kl records, per row, how much worse the
worst pair is under the common window than under its free alignment, so rows
where the constraint bites are visible.

Because all three pairs now come from ONE alignment, the three numbers and the
drawn logo are guaranteed consistent -- the old "J-R measured off-frame" caveat
cannot arise.

ONE ROW PER ROSSI REPLICATE.  JASPAR ships exactly one matrix per gene, but
Rossi runs each factor as several independent ChExMix samples (up to 6), each
yielding up to 3 motifs.  Collapsing those to a single "best" matrix threw away
the most valuable thing in the data -- the replication.  So a native motif gets
one row per Rossi replicate, and the same verdict reached independently in
several replicates is far stronger evidence than any single pick.

Within a replicate the top motif is taken when it beats the runner-up by at
least AMBIG_LOG orders of magnitude in E-value (87% of samples clear 2 orders).
When it does not, the runner-up gets its own row rather than being silently
dropped -- see rossi_evalue_structure.py for where these numbers come from.

E-VALUES ARE NOT COMPARABLE BETWEEN FACTORS.  MEME's E-value is the expected
number of motifs scoring as well in a shuffled dataset of the same size, so it
scales with how much data the run had (log10 E correlates with log10 nsites at
r = -0.54).  The best motif of one factor sits at 1e-500 and of another at
1e-1.4.  No absolute cutoff is meaningful across factors, which is why the
tie-break here is RELATIVE -- a gap within one run.

CONDITION.  The <id>_YEP directory names are the Yeast Epigenome Project tag,
NOT a growth medium, and the bundles carry no metadata.  The real condition
comes from GEO GSE147927 via fetch_rossi_conditions.py: Rossi heat-shocked a
subset of samples (37 C for 3 or 6 min), and 21 of those are on this disk.
Heat-shock replicates are kept as rows and labelled, but EXCLUDED from the
replicate consensus -- a motif measured under heat shock differs from a normal
one for biological reasons, not because anybody's matrix is wrong.

STATISTICAL POWER.  Each row carries the number of sites MEME fitted the motif
from.  It matters: the median surviving Rossi motif rests on just 28 sites and
10% on fewer than 10, so a large distance from a 9-site motif is noise while the
same distance from a 150-site motif is a finding.

For the logos, all three are drawn in the one common frame, oriented so that
native reads as shipped, and the W compared columns are shaded.  The matrix
drawn is the matrix that was compared, at the alignment that produced the
numbers.
"""
import os
import re
import csv
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "motifdb")
OUT_TSV = os.path.join(HERE, "motif_distance_sheet.tsv")
OUT_JSON = os.path.join(HERE, "motif_distance_sheet.json")

BG = np.array([0.30980641, 0.19088229, 0.19059636, 0.30871494])
PSEUDO = 0.1

# Two sets "agree" when their KL/column is at or below this.  Not a guess: it is
# the median of the smallest-of-three pairwise distances over the genes where
# all three datasets have a motif, i.e. the typical distance between the two
# datasets that agree best.  Printed by the report so it can be re-checked.
CLOSE_KL = 0.060
MARGIN_KL = 0.030          # the odd one out must be this much further out

# Within one Rossi replicate, the top motif is taken outright when it leads the
# runner-up by this many orders of magnitude in E.  87% of multi-motif samples
# clear 2; the rest are genuinely ambiguous and get a second row.
AMBIG_LOG = 2.0
LOGE_FLOOR = -500.0        # MEME reports 0.0 for the strongest motifs

SETS = ("native", "jaspar", "rossi")
PAIRS = (("native", "jaspar"), ("native", "rossi"), ("jaspar", "rossi"))

FITTED = {"Abf1_murphy", "Cin5_murphy", "Fhl1_zhu", "Fkh1_zhu", "Mcm1_zhu",
          "Nhp6a_zhu", "Rap1_telomeric", "Reb1_badis", "Sko1_murphy",
          "Spt15_zhu", "Tbf1_zhu", "Ume6_zhu"}


# --------------------------------------------------------------------- parsing
def parse_meme(path):
    out, mid, gene, rows = {}, None, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                if mid and rows:
                    out[mid] = (gene, np.array(rows))
                p = line.split()
                mid = p[1]
                gene = p[2] if len(p) > 2 else mid
                if gene == mid:
                    gene = mid.rsplit("_", 1)[0]
                gene = gene.upper()
                rows = []
                continue
            if mid is None or line.startswith("letter-probability"):
                continue
            p = line.split()
            if len(p) == 4:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    pass
    if mid and rows:
        out[mid] = (gene, np.array(rows))
    return out


# ------------------------------------------------------------------- distances
def add_pseudo(pwm):
    p = (pwm + PSEUDO * BG[None, :]) / (1.0 + PSEUDO)
    return p / p.sum(1, keepdims=True)


def ed_col(x, y):
    return float(np.sqrt(((x - y) ** 2).sum()))


def kld_col(x, y):
    return float(0.5 * ((x * np.log(x / y)).sum() + (y * np.log(y / x)).sum()))


def rc(pwm):
    return pwm[::-1, ::-1]


def best_align(a, b):
    """Every offset, both orientations of b. Scored by mean KLD per aligned
    column; the shorter motif must be >=80% covered (at least 6 columns).
    Returns (kl, ed, offset_of_b_in_a, orient, n_aligned)."""
    pa = add_pseudo(a)
    need = min(max(6, int(np.ceil(0.8 * min(len(a), len(b))))), len(a), len(b))
    best = None
    for orient in "+-":
        pb = add_pseudo(b if orient == "+" else rc(b))
        for off in range(-len(pb) + 1, len(pa)):
            lo, hi = max(0, off), min(len(pa), off + len(pb))
            n = hi - lo
            if n < need:
                continue
            kl = sum(kld_col(pa[k], pb[k - off]) for k in range(lo, hi)) / n
            ed = sum(ed_col(pa[k], pb[k - off]) for k in range(lo, hi)) / n
            if best is None or kl < best[0]:
                best = (kl, ed, off, orient, n)
    return best


def _pair_kl_ed(x, y):
    """Mean KL and mean ED over the W columns of two window slices."""
    kl = float((0.5 * ((x * np.log(x / y)).sum(1)
                       + (y * np.log(y / x)).sum(1))).mean())
    ed = float(np.sqrt(((x - y) ** 2).sum(1)).mean())
    return kl, ed


MIN_REAL = 0.8             # of the window, a motif must supply this many real
                           # columns; the rest is background


def _place(p, off, W):
    """Lay a motif into the W-column search window at offset `off`, filling
    columns it does not reach with the genome background.

    This is used to CHOOSE the frame, not to report the numbers.  Two things
    have to be true at once and one fixed window cannot do both:

      * a motif must be free to sit where it really belongs.  ABF1 replicate
        19997 is registered two columns over from the other two; pinned inside
        a 14-column window a 15-column motif can shift by one, and forcing it
        gave KL 1.03 in place of 0.17.
      * the search must not be able to buy a better score by shrinking the
        window onto the columns that happen to match.

    Holding the search window fixed at W and pricing the columns a motif cannot
    reach -- each contributes its own distance from background -- gives freedom
    without the shrink incentive.  The reported distances are then taken over
    the columns every motif actually reaches; see common_window.
    -> (W x 4 window matrix, number of real (unpadded) columns)"""
    idx = np.arange(W) - off
    ok = (idx >= 0) & (idx < len(p))
    out = np.repeat(BG[None, :], W, axis=0).copy()
    out[ok] = p[idx[ok]]
    return out, int(ok.sum())


def common_window(mots):
    """One window, one alignment, for all the motifs in a row.

    mots: {set_name: pwm} for the 2 or 3 sets that have a motif.

    Two stages, for the reason given in _place.

      FRAME.  Search every offset and orientation of the non-shortest motifs
      against a fixed W-column window (W = width of the shortest motif), with
      unreached columns priced against background, and take the placement
      minimising the SUM of the pairwise mean KLs -- a sum-of-pairs multiple
      alignment, symmetric in the datasets.

      NUMBERS.  Report all three pairwise distances over the columns every
      motif in that frame actually reaches, i.e. the intersection of the three
      placements.  One column set, all three pairs, nothing scored against a
      matrix that never measured it.

    Doing the second stage on the pairwise optima directly -- align each pair
    the way TOMTOM would, then intersect -- is the obvious approach and it is
    what this reduces to whenever it is well defined.  It is not always: the
    three pairwise optima are mutually realisable in a single frame in only
    59% of rows (pairwise_frame_consistency.py).  Alignment is not transitive,
    and TOMTOM has no opinion on three motifs at once -- it compares one query
    motif to target motifs, nothing more.

    The result is finally oriented globally so native reads as shipped.

    -> (window [lo, hi) in shortest-motif coordinates, W of the shortest motif,
        anchor, {set: (off, orient)}, {frozenset(pair): (kl, ed)})
    """
    present = [s for s in SETS if s in mots]
    if len(present) < 2:
        return None
    W = min(len(mots[s]) for s in present)
    anchor = min(present, key=lambda s: (len(mots[s]), SETS.index(s)))
    others = [s for s in present if s != anchor]
    need = max(4, int(np.ceil(MIN_REAL * W)))

    cand = {}
    for s in others:
        L = len(mots[s])
        cs = []
        for orient in "+-":
            p = add_pseudo(mots[s] if orient == "+" else rc(mots[s]))
            for off in range(-L + need, W - need + 1):
                m, nreal = _place(p, off, W)
                if nreal >= need:
                    cs.append((off, orient, m, nreal))
        cand[s] = cs
    A = add_pseudo(mots[anchor])[:W]

    best = None
    if len(others) == 1:
        s = others[0]
        for off, orient, m, nr in cand[s]:
            k = _pair_kl_ed(A, m)[0]
            if best is None or k < best[0]:
                best = (k, {s: (off, orient)})
    else:
        s1, s2 = others
        c1 = [(o, r, m, _pair_kl_ed(A, m)[0]) for o, r, m, _ in cand[s1]]
        c2 = [(o, r, m, _pair_kl_ed(A, m)[0]) for o, r, m, _ in cand[s2]]
        for o1, r1, m1, k1 in c1:
            for o2, r2, m2, k2 in c2:
                tot = k1 + k2 + _pair_kl_ed(m1, m2)[0]
                if best is None or tot < best[0]:
                    best = (tot, {s1: (o1, r1), s2: (o2, r2)})
    if best is None:
        return None

    place = dict(best[1])
    place[anchor] = (0, "+")
    # KL and ED are per-column, so reverse-complementing the whole frame leaves
    # every distance untouched.  Use that freedom to keep native as shipped.
    if place.get("native", (0, "+"))[1] == "-":
        flip = {"+": "-", "-": "+"}
        place = {s: (W - len(mots[s]) - off, flip[o])
                 for s, (off, o) in place.items()}

    # ---- stage two: score on the columns every motif reaches ---------------
    lo, hi = 0, W
    for s in present:
        off = place[s][0]
        lo, hi = max(lo, off), min(hi, off + len(mots[s]))
    if hi - lo < 4:                       # degenerate; fall back to the window
        lo, hi = 0, W

    def cut(s):
        off, o = place[s]
        p = add_pseudo(mots[s] if o == "+" else rc(mots[s]))
        m, _ = _place(p, off, W)
        return m[lo:hi]

    dist = {}
    for a, b in PAIRS:
        if a in mots and b in mots:
            dist[frozenset((a, b))] = _pair_kl_ed(cut(a), cut(b))
    return (lo, hi), W, anchor, place, dist


def parse_meme_meta(path):
    """Like parse_meme but also keeps each motif's E-value and nsites.
    -> {motif_id: (gene, pwm, evalue, nsites)}"""
    out, mid, gene, ev, ns, rows = {}, None, None, None, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                if mid and rows:
                    out[mid] = (gene, np.array(rows), ev, ns)
                p = line.split()
                mid = p[1]
                gene = (p[2] if len(p) > 2 else mid)
                if gene == mid:
                    gene = mid.rsplit("_", 1)[0]
                gene = gene.upper()
                ev, ns, rows = None, None, []
                continue
            if mid is None:
                continue
            if line.startswith("letter-probability"):
                if "E=" in line:
                    try:
                        ev = float(line.split("E=")[1].split()[0])
                    except ValueError:
                        ev = None
                if "nsites=" in line:
                    try:
                        ns = int(float(line.split("nsites=")[1].split()[0]))
                    except ValueError:
                        ns = None
                continue
            p = line.split()
            if len(p) == 4:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    pass
    if mid and rows:
        out[mid] = (gene, np.array(rows), ev, ns)
    return out


def closest(query, cands):
    """Nearest candidate by KL. Used for JASPAR, which ships one matrix per
    gene, so 'nearest' and 'only' are the same thing and no bias can arise."""
    scored = []
    for cid, pwm in cands:
        r = best_align(query, pwm)
        if r is not None:
            scored.append((r[0], cid, r, pwm))
    if not scored:
        return None, None, None, len(cands)
    scored.sort(key=lambda t: t[0])
    return scored[0][1], scored[0][2], scored[0][3], len(cands)


def source_of(mid):
    tail = mid.rsplit("_", 1)[-1].lower()
    return tail if tail in ("murphy", "zhu", "badis") else "other"


# --------------------------------------------------------------------- verdict
def call_odd_one_out(d):
    """d maps frozenset({a,b}) -> KL distance, for whichever pairs exist."""
    have = [s for s in SETS
            if any(s in k and d.get(k) is not None for k in d)]
    present = [s for s in SETS if s in have]
    full = [p for p in PAIRS if d.get(frozenset(p)) is not None]
    if len(full) < 3:
        return None, None, ("two_datasets_only" if len(full) == 1
                            else ("no_comparison" if not full else "partial"))
    vals = {frozenset(p): d[frozenset(p)] for p in PAIRS}
    ordered = sorted(vals.items(), key=lambda kv: kv[1])
    (best_pair, best_v), (_, second_v) = ordered[0], ordered[1]
    odd = [s for s in SETS if s not in best_pair][0]
    margin = second_v - best_v
    if max(vals.values()) <= CLOSE_KL:
        return odd, margin, "all_three_agree"
    if best_v > CLOSE_KL:
        return odd, margin, "no_consensus"
    if margin < MARGIN_KL:
        return odd, margin, "ambiguous"
    return odd, margin, "%s_is_odd" % odd


# ------------------------------------------------------------------------ main
def load_replicates():
    """sample_id -> (replicate, condition).

    Condition is NOT in the downloaded bundles -- the <id>_YEP directory name is
    the Yeast Epigenome Project tag, not a growth medium.  It comes from GEO via
    fetch_rossi_conditions.py.  If that table is missing we fall back to the
    replicate label alone and say so loudly, rather than silently calling
    everything normal."""
    m = {}
    cond_tsv = os.path.join(HERE, "inputs", "rossi_sample_conditions.tsv")
    if os.path.exists(cond_tsv):
        with open(cond_tsv) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                m[r["sample_id"]] = (r["replicate"], r["condition"])
        return m
    print("WARNING: inputs/rossi_sample_conditions.tsv missing -- run "
          "fetch_rossi_conditions.py. Conditions will read as 'unknown'.")
    bed = os.path.join(HERE, "inputs", "rossi_peak_w_strand_all_TFs.bed")
    with open(bed) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            m[r["sample_id"]] = (r["replicate"], "unknown")
    return m


def power_tier(nsites):
    """MEME caps nsites at 200; the median surviving motif has 28."""
    if nsites is None:
        return ""
    if nsites >= 100:
        return "strong"
    if nsites >= 30:
        return "moderate"
    return "weak"


def sample_of(motif_id):
    mm = re.match(r".+_rossi_(\d+)_m\d+$", motif_id)
    return mm.group(1) if mm else None


def main():
    native = parse_meme(os.path.join(DB, "shipped.meme"))
    jaspar = parse_meme(os.path.join(DB, "jaspar.meme"))
    rossi = parse_meme_meta(os.path.join(DB, "rossi.meme"))
    repmap = load_replicates()

    jas_by_gene = {}
    for mid, (gene, pwm) in jaspar.items():
        jas_by_gene.setdefault(gene, []).append((mid, pwm))

    # gene -> sample -> [candidates sorted by E]
    ros_by_gene = {}
    for mid, (gene, pwm, ev, ns) in rossi.items():
        s = sample_of(mid)
        le = np.log10(ev) if (ev is not None and ev > 0) else LOGE_FLOOR
        rep, cond = repmap.get(s, ("?", "unknown"))
        ros_by_gene.setdefault(gene, {}).setdefault(s, []).append(
            dict(id=mid, pwm=pwm, E=ev, logE=le, nsites=ns,
                 rep=rep, cond=cond))
    for g in ros_by_gene:
        for s in ros_by_gene[g]:
            ros_by_gene[g][s].sort(key=lambda c: c["logE"])

    rows, records = [], []
    for mid, (gene, npwm) in sorted(native.items()):
        jl = jas_by_gene.get(gene, [])
        jid, jpwm = jl[0] if jl else (None, None)

        samples = ros_by_gene.get(gene, {})
        # every (replicate, pick) this native motif should be scored against
        jobs = []
        for s, cands in sorted(samples.items(), key=lambda kv: kv[1][0]["logE"]):
            gap = (cands[1]["logE"] - cands[0]["logE"]) if len(cands) > 1 else None
            jobs.append((s, cands[0], 1, gap, len(cands)))
            if gap is not None and gap < AMBIG_LOG:
                jobs.append((s, cands[1], 2, gap, len(cands)))
        if not jobs:
            jobs = [(None, None, None, None, 0)]

        for s, pick, rank, gap, ncand in jobs:
            mots = {"native": npwm}
            ids = {"native": mid}
            if jid:
                mots["jaspar"], ids["jaspar"] = jpwm, jid
            if pick:
                mots["rossi"], ids["rossi"] = pick["pwm"], pick["id"]

            cw = common_window(mots)
            (wlo, whi), wshort, anchor, place, dist = (
                cw if cw else ((0, 0), None, "", {"native": (0, "+")}, {}))
            W = whi - wlo if cw else None
            d = {frozenset(p): (dist[frozenset(p)][0] if frozenset(p) in dist
                                else None) for p in PAIRS}
            e = {frozenset(p): (dist[frozenset(p)][1] if frozenset(p) in dist
                                else None) for p in PAIRS}
            odd, margin, verdict = call_odd_one_out(d)

            # what the common window costs: the worst pair's excess KL over the
            # alignment that pair would have chosen for itself
            cost = None
            for a, b in PAIRS:
                if d[frozenset((a, b))] is None:
                    continue
                free = best_align(mots[a], mots[b])
                if free is not None:
                    x = d[frozenset((a, b))] - free[0]
                    cost = x if cost is None else max(cost, x)

            layers = []
            for k in SETS:
                if k not in mots:
                    continue
                off, orient = place[k]
                layers.append(dict(set=k, off=off, motif=ids[k], orient=orient,
                                   pwm=np.round(rc(mots[k]) if orient == "-"
                                                else mots[k], 3).tolist()))
            f0 = min([l["off"] for l in layers] or [0])
            f1 = max([l["off"] + len(l["pwm"]) for l in layers] or [0])

            rec = dict(
                gene=gene, native_motif=mid, source=source_of(mid),
                fitted="yes" if mid in FITTED else "", w_native=len(npwm),
                jaspar_motif=jid or "",
                rossi_sample=s or "", replicate=pick["rep"] if pick else "",
                condition=pick["cond"] if pick else "",
                heat_shock=("yes" if (pick and pick["cond"] not in
                                      ("normal", "unknown")) else
                            ("no" if pick else "")),
                rossi_motif=pick["id"] if pick else "",
                rossi_E=pick["E"] if pick else None,
                rossi_logE=pick["logE"] if pick else None,
                rossi_nsites=pick["nsites"] if pick else None,
                power=power_tier(pick["nsites"]) if pick else "",
                e_rank=rank, e_gap_log10=gap, n_motifs_in_sample=ncand,
                e_ambiguous=("yes" if (gap is not None and gap < AMBIG_LOG) else
                             ("no" if gap is not None else "")),
                ed_native_jaspar=e[frozenset(("native", "jaspar"))],
                kl_native_jaspar=d[frozenset(("native", "jaspar"))],
                ed_native_rossi=e[frozenset(("native", "rossi"))],
                kl_native_rossi=d[frozenset(("native", "rossi"))],
                ed_jaspar_rossi=e[frozenset(("jaspar", "rossi"))],
                kl_jaspar_rossi=d[frozenset(("jaspar", "rossi"))],
                w_jaspar=len(jpwm) if jid else None,
                w_rossi=len(pick["pwm"]) if pick else None,
                w_window=W, w_shortest=wshort, window_anchor=anchor,
                n_dropped=(wshort - W if cw else None),
                window_cost_kl=cost,
                off_native=place.get("native", (None, ""))[0],
                orient_native=place.get("native", (None, ""))[1],
                off_jaspar=place.get("jaspar", (None, ""))[0],
                orient_jaspar=place.get("jaspar", (None, ""))[1],
                off_rossi=place.get("rossi", (None, ""))[0],
                orient_rossi=place.get("rossi", (None, ""))[1],
                odd_one_out=odd or "", margin_kl=margin, verdict=verdict,
            )
            rows.append(rec)
            records.append(dict(rec, frame=[f0, f1], window=[wlo, whi],
                                layers=layers))

    # ---- roll the replicates up per native motif ---------------------------
    from collections import defaultdict, Counter
    prim = defaultdict(list)
    for r in rows:
        if r["e_rank"] in (1, None):
            prim[r["native_motif"]].append(r)
    roll = {}
    for m, rs in prim.items():
        # Heat-shock replicates are shown but do NOT vote: a motif measured at
        # 37 C can differ from a 25 C one for real biological reasons, which is
        # a different claim from "somebody's matrix is wrong".
        usable = [r for r in rs if r["rossi_sample"] and r["heat_shock"] != "yes"]
        vs = [r["verdict"] for r in usable]
        c = Counter(vs)
        top, n = (c.most_common(1)[0] if c else ("", 0))
        n_hs = sum(1 for r in rs if r["heat_shock"] == "yes")
        roll[m] = dict(n_reps=len(vs), consensus=top, n_consensus=n,
                       n_native_odd=c.get("native_is_odd", 0),
                       n_rossi_odd=c.get("rossi_is_odd", 0),
                       n_heat_shock=n_hs,
                       max_nsites=max([r["rossi_nsites"] for r in usable
                                       if r["rossi_nsites"] is not None] or [0]),
                       unanimous=("yes" if (vs and n == len(vs)) else
                                  ("no" if vs else "")))
    for r in rows + records:
        r.update({("rep_" + k): v for k, v in roll[r["native_motif"]].items()})

    order = {"native_is_odd": 0, "jaspar_is_odd": 1, "rossi_is_odd": 1,
             "no_consensus": 2, "ambiguous": 3, "all_three_agree": 4,
             "two_datasets_only": 5, "partial": 5, "no_comparison": 6}

    def key(r):
        return (order.get(r["rep_consensus"], 9),
                -(r["rep_n_native_odd"] or 0),
                r["gene"], r["native_motif"],
                r["rossi_logE"] if r["rossi_logE"] is not None else 1e9,
                r["e_rank"] or 0)
    rows.sort(key=key)
    records.sort(key=key)

    keys = list(rows[0].keys())
    with open(OUT_TSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else
                            ("%.4f" % r[k] if isinstance(r[k], float) else r[k]))
                        for k in keys})
    with open(OUT_JSON, "w") as fh:
        json.dump(records, fh, separators=(",", ":"))
    print("wrote %s (%d rows)\nwrote %s" % (OUT_TSV, len(rows), OUT_JSON))

    # ------------------------------------------------------------------ report
    nmot = len({r["native_motif"] for r in rows})
    withros = [r for r in rows if r["rossi_sample"]]
    print("\n%d rows = %d native motifs x their Rossi replicates" % (len(rows), nmot))
    print("   rows with a Rossi replicate: %d over %d distinct samples"
          % (len(withros), len({r["rossi_sample"] for r in withros})))
    amb = [r for r in withros if r["e_rank"] == 2]
    print("   extra rows added for an ambiguous E-value (gap < %.0f orders): %d"
          % (AMBIG_LOG, len(amb)))

    three = [r for r in rows if r["e_rank"] == 1
             and r["kl_native_jaspar"] is not None
             and r["kl_native_rossi"] is not None]
    if three:
        mins = np.array([min(r["kl_native_jaspar"], r["kl_native_rossi"],
                             r["kl_jaspar_rossi"]) for r in three])
        print("\nCLOSEST-PAIR distance, per replicate with all three datasets "
              "(n=%d)" % len(three))
        print("  median %.3f  p25 %.3f  p75 %.3f  p90 %.3f  max %.3f  "
              "(CLOSE_KL = %.3f)"
              % (np.median(mins), np.percentile(mins, 25),
                 np.percentile(mins, 75), np.percentile(mins, 90), mins.max(),
                 CLOSE_KL))

        print("\nmean KL to the other two, per replicate -- symmetric, so this")
        print("says which dataset is the most out of step overall:")
        for s in SETS:
            v = []
            for r in three:
                o = [r["kl_%s_%s" % tuple(sorted((s, t), key=SETS.index))]
                     for t in SETS if t != s]
                v.append(sum(o) / 2)
            v = np.array(v)
            print("   %-8s median %.3f   mean %.3f" % (s, np.median(v), v.mean()))

    from collections import Counter
    print("\nper-replicate verdicts (%d rows with a Rossi motif):" % len(withros))
    for v, n in sorted(Counter(r["verdict"] for r in withros).items(),
                       key=lambda kv: order.get(kv[0], 9)):
        print("   %-20s %3d" % (v, n))

    # rolled up to one call per native motif
    seen, roll_rows = set(), []
    for r in rows:
        if r["native_motif"] not in seen:
            seen.add(r["native_motif"])
            roll_rows.append(r)
    print("\nrolled up to %d native motifs (majority verdict over replicates):"
          % len(roll_rows))
    for v, n in sorted(Counter(r["rep_consensus"] for r in roll_rows).items(),
                       key=lambda kv: order.get(kv[0], 9)):
        print("   %-20s %3d" % (v or "(no rossi)", n))

    hs = [r for r in rows if r["heat_shock"] == "yes"]
    print("\nHEAT SHOCK: %d rows use a heat-shocked Rossi replicate (37 C, "
          "3-6 min).\nThey are shown but excluded from the consensus:" % len(hs))
    for r in hs:
        print("   %-16s %-6s sample %-7s %-18s nsites %-4s  %s"
              % (r["native_motif"], r["replicate"], r["rossi_sample"],
                 r["condition"], r["rossi_nsites"], r["verdict"]))

    ns = [r["rossi_nsites"] for r in withros if r["rossi_nsites"] is not None]
    if ns:
        ns = np.array(ns)
        print("\nSTATISTICAL POWER: sites behind each Rossi motif in the sheet")
        print("   median %d   p25 %d   p75 %d   max %d"
              % (np.median(ns), np.percentile(ns, 25), np.percentile(ns, 75),
                 ns.max()))
        print("   weak (<30 sites) %d   moderate (30-99) %d   strong (>=100) %d"
              % (sum(1 for r in withros if r["power"] == "weak"),
                 sum(1 for r in withros if r["power"] == "moderate"),
                 sum(1 for r in withros if r["power"] == "strong")))

    print("\nNATIVE IS ODD, with replicate support (heat shock excluded):")
    print("   %-16s %-7s %-4s %-8s %8s %8s %8s %6s  %s"
          % ("motif", "source", "fit", "reps", "KL n-j", "KL n-r", "KL j-r",
             "sites", "unanimous"))
    for r in roll_rows:
        if r["rep_consensus"] != "native_is_odd":
            continue
        print("   %-16s %-7s %-4s %d of %-4d %8.3f %8.3f %8.3f %6d  %s"
              % (r["native_motif"], r["source"], r["fitted"] or "-",
                 r["rep_n_native_odd"], r["rep_n_reps"],
                 r["kl_native_jaspar"], r["kl_native_rossi"],
                 r["kl_jaspar_rossi"], r["rep_max_nsites"], r["rep_unanimous"]))

    ww = [r for r in rows if r["w_window"]]
    if ww:
        w = np.array([r["w_window"] for r in ww])
        print("\nCOMMON WINDOW: all three distances in a row are measured on the "
              "same columns -- the ones every motif in the row reaches.")
        print("   scored columns: median %d  min %d  max %d  |  under 8: %d of %d"
              % (np.median(w), w.min(), w.max(), int((w < 8).sum()), len(w)))
        anc = Counter(r["window_anchor"] for r in ww)
        print("   the shortest motif is: %s"
              % "  ".join("%s %d" % (k, v) for k, v in anc.most_common()))
        dr = np.array([r["n_dropped"] for r in ww])
        print("   columns of the shortest motif dropped because another motif "
              "does not reach them:")
        print("      none %d   one %d   two %d   (max %d)"
              % (int((dr == 0).sum()), int((dr == 1).sum()),
                 int((dr == 2).sum()), dr.max()))
        cost = np.array([r["window_cost_kl"] for r in ww
                         if r["window_cost_kl"] is not None])
        print("   cost of the common window (worst pair's excess KL vs the "
              "alignment that pair would pick for itself):")
        print("      median %.4f  p75 %.4f  p90 %.4f  max %.4f  |  "
              "rows above 0.05: %d of %d"
              % (np.median(cost), np.percentile(cost, 75),
                 np.percentile(cost, 90), cost.max(),
                 int((cost > 0.05).sum()), len(cost)))

    print("\nthe 12 Fiber-fitted TFs (rolled up):")
    print("   %-16s %-7s %-8s %8s %8s %8s  %s"
          % ("motif", "source", "reps", "KL n-j", "KL n-r", "KL j-r", "consensus"))
    f = lambda v: "    --  " if v is None else "%8.3f" % v
    for r in roll_rows:
        if not r["fitted"]:
            continue
        print("   %-16s %-7s %d of %-3d %s %s %s  %s"
              % (r["native_motif"], r["source"], r["rep_n_consensus"],
                 r["rep_n_reps"], f(r["kl_native_jaspar"]),
                 f(r["kl_native_rossi"]), f(r["kl_jaspar_rossi"]),
                 r["rep_consensus"] or r["verdict"]))


if __name__ == "__main__":
    main()
