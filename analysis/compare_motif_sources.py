"""Three-way motif comparison: shipped collection vs JASPAR vs Rossi.

Answers "how different are our motifs from everyone else's, and how bad is the problem"
using TOMTOM (Gupta, Stamatoyannopoulos, Noble, Bailey 2007), which is the established
method because it searches every offset AND both orientations before scoring -- motifs
from different sources routinely differ by a shift or a flip, which a naive column-by-
column correlation would report as total disagreement.

TWO METRICS, ON PURPOSE.
  ed        Euclidean distance. TOMTOM's default and the winner of the paper's own
            retrieval benchmark (mean ROC 0.9889).
  kullback  KL divergence. Runner-up in that benchmark, and the metric most sensitive to
            a column that is FLAT in one motif and BIASED in the other.
The disagreement between them is the diagnostic. ABF1's defect is a near-identical core
(ATCAC...ACGA) plus a divergent low-information spacer (ATGGC vs TATAT); that is exactly
the shape where ED stays small and KL blows up. Ranking by "KL rank minus ED rank" finds
the other cases without anyone having to eyeball 153 matrices.

BACKGROUND. -bfile is REJECTED by tomtom for ed/kullback/pearson/sandelin -- those are
pure column comparisons that never see a background. It is accepted only for allr and the
Bayesian metrics (blic*, llr*). Background still enters via -motif-pseudo, which is spread
according to each FILE's header, so build_motif_dbs.py writes all three databases with the
identical yeast header (A 0.3098 C 0.1909 G 0.1906 T 0.3087, RoboCOP's own genome count).
That makes the pseudocount treatment identical across sources rather than a confound.

A THIRD NUMBER THAT IS NOT A MOTIF DISTANCE. Two matrices can be formally similar and
still place mass differently. `r_track` scores real chrI with both PWMs and correlates the
resulting log-odds profiles, which is the only quantity here tied to what RoboCOP will
actually do.

    conda activate pyranges_env3
    python compare_motif_sources.py
"""
import argparse
import collections
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_motif_dbs import parse_meme, gene_of, compute_background

MEME_BIN = "/home/users/nd141/miniconda3/envs/meme/bin"
DB = "motifdb"
GENOME = "inputs/SacCer3.fa"
OUT = "motif_comparison.tsv"
PAIRS = [("shipped", "jaspar"), ("shipped", "rossi"), ("jaspar", "rossi")]
DISTS = ["ed", "kullback"]
FITTED = ["ABF1", "CIN5", "FHL1", "FKH1", "MCM1", "NHP6A",
          "RAP1", "REB1", "SKO1", "SPT15", "TBF1", "UME6"]


# ---------------------------------------------------------------- tomtom

def run_tomtom(q, t, dist, outdir, force=False):
    """Cached tomtom -text run. thresh is deliberately huge: we need the same-name pair
    even when it scores badly, which is the interesting case."""
    cache = os.path.join(outdir, "tomtom_%s_vs_%s_%s.tsv" % (q, t, dist))
    if os.path.isfile(cache) and not force and os.path.getsize(cache) > 0:
        return cache
    cmd = [os.path.join(MEME_BIN, "tomtom"), "-text", "-dist", dist,
           "-thresh", "1e6", "-evalue",
           os.path.join(outdir, q + ".meme"), os.path.join(outdir, t + ".meme")]
    print("    %s" % " ".join(cmd[1:]), flush=True)
    with open(cache, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.exit("tomtom failed (%s vs %s, %s): %s" % (q, t, dist, r.stderr[:400]))
    return cache


def load_tomtom(path):
    """(query, target) -> record. Keeps the BEST (lowest p) row per pair."""
    best = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(hdr):
                continue
            try:
                p = float(f[ix["p-value"]])
            except ValueError:
                continue
            k = (f[ix["Query_ID"]], f[ix["Target_ID"]])
            if k in best and best[k]["p"] <= p:
                continue
            best[k] = {"p": p, "E": float(f[ix["E-value"]]), "q": float(f[ix["q-value"]]),
                       "offset": int(f[ix["Optimal_offset"]]),
                       "overlap": int(f[ix["Overlap"]]),
                       "orient": f[ix["Orientation"]],
                       "qcons": f[ix["Query_consensus"]],
                       "tcons": f[ix["Target_consensus"]]}
    return best


# ---------------------------------------------------------------- functional track

def read_chrom(fasta, want="chrI"):
    seq, keep = [], False
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                if keep:
                    break
                keep = line[1:].split()[0] == want
                continue
            if keep:
                seq.append(line.strip().upper())
    return "".join(seq)


def encode(seq):
    lut = np.full(256, -1, np.int8)
    for i, b in enumerate("ACGT"):
        lut[ord(b)] = i
    return lut[np.frombuffer(seq.encode(), np.uint8)]


def matrix_of(motif):
    return np.array([[float(x) for x in r.split()] for r in motif["rows"]], float)


def log_odds(pwm, bg, pseudo=0.1):
    """Log-odds with the pseudocount SPREAD BY BACKGROUND, matching what FIMO and TOMTOM
    do with --motif-pseudo (default 0.1).

    This is not cosmetic. Rossi's matrices come straight from MEME and contain hard 0.0 and
    1.0 entries; a small flat pseudocount turns those columns into scores of order -10 bits,
    so one mismatched position dominates the whole chromosome track and the correlation
    collapses to ~0 between matrices that are visibly the same motif. Spreading by
    background bounds the penalty at log2(bg/(bg + ...)) and makes sources with different
    nsites comparable."""
    p = (pwm + pseudo * bg[None, :]) / (1.0 + pseudo)
    p = p / p.sum(1, keepdims=True)
    return np.log2(p / bg[None, :])


def scan(code, lo):
    """Forward-strand log-odds, anchored at the motif's own column 0.

    Deliberately NOT max-over-strands. Taking the max mixes two different anchor
    conventions: a forward hit puts motif column 0 at the window start, a reverse hit puts
    it at the window end, so two motifs of different width align on their forward hits and
    are displaced by (w_a - w_b) on their reverse hits. Rossi ABF1 is JASPAR ABF1 plus one
    trailing column, and under max-of-strands that alone dropped their correlation from
    0.98 to 0.11 -- an artifact, not a difference between the matrices. Orientation is
    handled explicitly in best_track_corr instead, the same way TOMTOM handles it.
    """
    w = lo.shape[0]
    n = len(code)
    if n <= w:
        return None
    m = n - w + 1
    valid = code >= 0
    f = np.zeros(m)
    ok = np.ones(m, bool)
    for j in range(w):
        col = code[j:j + m]
        ok &= valid[j:j + m]
        f += lo[j][np.where(col >= 0, col, 0)]
    f[~ok] = np.nan
    track = np.full(n, np.nan)
    track[:m] = f
    return track


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 1000:
        return float("nan")
    sa, sb = a[m], b[m]
    if sa.std() == 0 or sb.std() == 0:
        return float("nan")
    return float(np.corrcoef(sa, sb)[0, 1])


def best_track_corr(code, lo_a, lo_b, maxlag=25):
    """Correlation of two motifs' score profiles at their best alignment.

    Mirrors what TOMTOM does -- search every offset and both orientations, keep the best --
    but in score space rather than matrix space, so it answers "would these two matrices
    light up the same places on real DNA" rather than "are these matrices similar".
    Returns (r, lag, orientation).
    """
    def shift(t, lag):
        if lag == 0:
            return t
        out = np.full_like(t, np.nan)
        if lag > 0:
            out[lag:] = t[:-lag]
        else:
            out[:lag] = t[-lag:]
        return out

    ta = scan(code, lo_a)
    best = (float("-inf"), 0, "+")
    for orient, lb in (("+", lo_b), ("-", lo_b[::-1, ::-1])):
        tb = scan(code, lb)
        if tb is None:
            continue
        for lag in range(-maxlag, maxlag + 1):
            r = _corr(ta, shift(tb, lag))
            if np.isfinite(r) and r > best[0]:
                best = (r, lag, orient)
    return best if np.isfinite(best[0]) else (float("nan"), 0, "")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--chrom", default="chrI")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-track", action="store_true", help="skip the chrI score correlation")
    args = ap.parse_args()

    print("== loading databases ==")
    dbs = {n: parse_meme(open(os.path.join(args.db, n + ".meme")).read())
           for n in ("shipped", "jaspar", "rossi")}
    by_name = {n: {m["name"]: m for m in ms} for n, ms in dbs.items()}
    for n, ms in dbs.items():
        print("  %-8s %d motifs" % (n, len(ms)))

    gene_of_motif = {}
    for m in dbs["shipped"]:
        gene_of_motif[m["name"]] = gene_of(m["name"])
    for n in ("jaspar", "rossi"):
        for m in dbs[n]:
            gene_of_motif[m["name"]] = m["alt"].upper()

    print("\n== tomtom ==")
    T = {}
    for q, t in PAIRS:
        for d in DISTS:
            T[(q, t, d)] = load_tomtom(run_tomtom(q, t, d, args.db, args.force))
            print("    %-8s vs %-7s %-9s %d scored pairs"
                  % (q, t, d, len(T[(q, t, d)])), flush=True)

    targets_by_gene = {n: collections.defaultdict(list) for n in ("jaspar", "rossi")}
    for n in ("jaspar", "rossi"):
        for m in dbs[n]:
            targets_by_gene[n][m["alt"].upper()].append(m["name"])

    # ------------------------------------------------ functional tracks
    code, LO = None, {}
    if not args.no_track:
        print("\n== functional scan (%s) ==" % args.chrom)
        bgd = compute_background(GENOME)
        bg = np.array([bgd[b] for b in "ACGT"])
        seq = read_chrom(GENOME, args.chrom)
        code = encode(seq)
        print("  %s: %d bp, %d non-ACGT" % (args.chrom, len(seq), int((code < 0).sum())))
        for n, ms in dbs.items():
            for m in ms:
                LO[m["name"]] = log_odds(matrix_of(m), bg)
        print("  %d log-odds matrices ready" % len(LO))

    # ------------------------------------------------ per shipped motif
    print("\n== building table ==")
    rows = []
    for m in dbs["shipped"]:
        qn = m["name"]
        g = gene_of(qn)
        row = {"shipped_motif": qn, "gene": g,
               "source": qn.rsplit("_", 1)[1],
               "w_shipped": len(m["rows"])}
        for n in ("jaspar", "rossi"):
            cands = targets_by_gene[n].get(g, [])
            row["n_" + n] = len(cands)
            best, bestd = None, None
            for d in DISTS:
                tt = T[("shipped", n, d)]
                hits = [(tt[(qn, c)]["p"], c) for c in cands if (qn, c) in tt]
                if not hits:
                    row["p_%s_%s" % (n, d)] = ""
                    row["q_%s_%s" % (n, d)] = ""
                    continue
                p, c = min(hits)
                row["p_%s_%s" % (n, d)] = "%.4g" % p
                row["q_%s_%s" % (n, d)] = "%.4g" % tt[(qn, c)]["q"]
                if d == "ed":
                    best, bestd = c, tt[(qn, c)]
            if best is not None:
                row["match_" + n] = best
                row["offset_" + n] = bestd["offset"]
                row["orient_" + n] = bestd["orient"]
                row["overlap_" + n] = bestd["overlap"]
                row["w_" + n] = len(by_name[n][best]["rows"])
                row["cons_" + n] = bestd["tcons"]
                if LO:
                    rr, lag, orient = best_track_corr(code, LO[qn], LO[best])
                    row["r_track_" + n] = "%.4f" % rr
                    row["r_lag_" + n] = lag
                    row["r_orient_" + n] = orient
            else:
                for k in ("match", "offset", "orient", "overlap", "w", "cons",
                          "r_track", "r_lag", "r_orient"):
                    row[k + "_" + n] = ""
        row["cons_shipped"] = ""
        # best hit ignoring the name -- catches a motif that matches some OTHER TF better
        tt = T[("shipped", "jaspar", "ed")]
        mine = [(v["p"], tgt) for (qq, tgt), v in tt.items() if qq == qn]
        if mine:
            p, tgt = min(mine)
            row["best_any_jaspar"] = tgt
            row["best_any_p"] = "%.4g" % p
            row["mislabel_flag"] = ("YES" if gene_of_motif.get(tgt, "") != g
                                    and row.get("n_jaspar") else "")
            row["cons_shipped"] = tt[(qn, tgt)]["qcons"]
        rows.append(row)

    cols = ["gene", "shipped_motif", "source", "bucket",
            "w_shipped", "w_jaspar", "w_rossi",
            "n_jaspar", "n_rossi",
            "p_jaspar_ed", "p_jaspar_kullback", "q_jaspar_ed", "q_jaspar_kullback",
            "p_rossi_ed", "p_rossi_kullback", "q_rossi_ed", "q_rossi_kullback",
            "offset_jaspar", "orient_jaspar", "overlap_jaspar",
            "offset_rossi", "orient_rossi", "overlap_rossi",
            "r_track_jaspar", "r_lag_jaspar", "r_orient_jaspar",
            "r_track_rossi", "r_lag_rossi", "r_orient_rossi",
            "match_jaspar", "match_rossi",
            "cons_shipped", "cons_jaspar", "cons_rossi",
            "best_any_jaspar", "best_any_p", "mislabel_flag"]
    for r in rows:
        nj, nr = r.get("n_jaspar", 0), r.get("n_rossi", 0)
        r["bucket"] = ("3way" if nj and nr else "2way_jaspar" if nj
                       else "2way_rossi" if nr else "nothing_to_compare")
    with open(args.out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print("  wrote %s  (%d rows)" % (args.out, len(rows)))

    # ------------------------------------------------ summary
    b = collections.Counter(r["bucket"] for r in rows)
    print("\n== coverage (per shipped motif) ==")
    for k in ("3way", "2way_jaspar", "2way_rossi", "nothing_to_compare"):
        print("  %-20s %d" % (k, b[k]))

    have = [r for r in rows if r["p_jaspar_ed"] != ""]
    print("\n== agreement with JASPAR, by source ==")
    print("  %-10s %5s %8s %8s %8s" % ("source", "n", "q<1e-3", "q<0.05", "q>=0.05"))
    for s in sorted({r["source"] for r in have}):
        sub = [r for r in have if r["source"] == s]
        qs = [float(r["q_jaspar_ed"]) for r in sub]
        print("  %-10s %5d %8d %8d %8d"
              % (s, len(sub), sum(q < 1e-3 for q in qs),
                 sum(1e-3 <= q < 0.05 for q in qs), sum(q >= 0.05 for q in qs)))

    print("\n== worst shipped-vs-JASPAR (ED), top 15 ==")
    print("  %-16s %-9s %10s %10s %8s %7s %8s" %
          ("motif", "source", "q_ed", "q_kl", "orient", "offset", "r_track"))
    for r in sorted(have, key=lambda r: -float(r["q_jaspar_ed"]))[:15]:
        print("  %-16s %-9s %10.3g %10.3g %8s %7s %8s"
              % (r["shipped_motif"], r["source"], float(r["q_jaspar_ed"]),
                 float(r["q_jaspar_kullback"]), r["orient_jaspar"],
                 r["offset_jaspar"], r.get("r_track_jaspar", "")))

    # THE ABF1 CLASS: TOMTOM calls it the same motif, but the two matrices do not light up
    # the same DNA. ABF1 is the worked example -- q_ed 2.5e-07 (a confident match) yet
    # r_track only 0.84, against 0.98 for Rossi-vs-JASPAR on the same TF. Neither ED nor KL
    # separates ABF1 from a clean match; only the track correlation does. That is the point
    # of carrying a third number.
    conf = [r for r in have
            if float(r["q_jaspar_ed"]) < 0.01 and str(r.get("r_track_jaspar", "")) != ""]
    conf.sort(key=lambda r: float(r["r_track_jaspar"]))
    print("\n== confident TOMTOM match but DIVERGENT behaviour (q_ed<0.01, lowest r_track) ==")
    print("  %-16s %-9s %10s %10s %9s %6s %6s" %
          ("motif", "source", "q_ed", "q_kl", "r_track", "lag", "orient"))
    for r in conf[:15]:
        print("  %-16s %-9s %10.3g %10.3g %9.4f %6s %6s"
              % (r["shipped_motif"], r["source"], float(r["q_jaspar_ed"]),
                 float(r["q_jaspar_kullback"]), float(r["r_track_jaspar"]),
                 r.get("r_lag_jaspar", ""), r.get("r_orient_jaspar", "")))

    ed_rank = {r["shipped_motif"]: i for i, r in
               enumerate(sorted(have, key=lambda r: float(r["q_jaspar_ed"])))}
    kl_rank = {r["shipped_motif"]: i for i, r in
               enumerate(sorted(have, key=lambda r: float(r["q_jaspar_kullback"])))}
    ab = "Abf1_murphy"
    if ab in ed_rank:
        print("\n  [ED vs KL cross-check] Abf1_murphy ranks %d/%d by ED and %d/%d by KL --"
              % (ed_rank[ab], len(have), kl_rank[ab], len(have)))
        print("  both call it a confident match, so the ED/KL disagreement does NOT isolate")
        print("  the spacer defect. r_track is what separates it.")

    print("\n== the 12 fitted TFs ==")
    print("  %-16s %-9s %10s %10s %10s %8s %8s" %
          ("motif", "source", "q_jaspar", "q_rossi", "r_trk_jas", "orient", "bucket"))
    for r in rows:
        if r["gene"] in FITTED:
            print("  %-16s %-9s %10s %10s %10s %8s %8s"
                  % (r["shipped_motif"], r["source"],
                     r["q_jaspar_ed"] or "-", r["q_rossi_ed"] or "-",
                     r.get("r_track_jaspar", "") or "-",
                     r.get("orient_jaspar", "") or "-", r["bucket"]))

    mis = [r for r in rows if r.get("mislabel_flag") == "YES"]
    print("\n== possible mislabels: best JASPAR hit is a DIFFERENT gene (%d) ==" % len(mis))
    for r in sorted(mis, key=lambda r: float(r["best_any_p"]))[:20]:
        print("  %-16s %-9s own q_ed %-10s best hit %-22s p %s"
              % (r["shipped_motif"], r["source"], r["q_jaspar_ed"],
                 r["best_any_jaspar"], r["best_any_p"]))


if __name__ == "__main__":
    main()
