#!/usr/bin/env python
"""Why a 'confident TOMTOM match' can still lose 3 of 5 real sites.

TOMTOM asks 'are these the same motif' by comparing PROBABILITY columns and
scoring the answer against a null of random matrices.  FIMO asks 'does this
14-mer beat a threshold' by SUMMING LOG-ODDS over all 14 columns.  Those are
different questions, and this script measures the gap between them on the
5 MacIsaac ABF1 sites of chrI.

For each site we align both matrices at the same register, decompose the
log-odds score column by column, split it into the shared core (cols 0-4,
10-13) and the divergent spacer (cols 5-9), and convert each total into the
exact FIMO p-value by convolving the per-column null distributions.

Outputs: analysis/abf1_score_split.tsv and a printed report.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FASTA = os.path.join(HERE, "inputs", "SacCer3.fa")
MACISAAC = os.path.join(HERE, "inputs", "MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed")
MURPHY = os.path.join(HERE, "inputs", "motifs_meme.txt")
JASPAR = os.path.join(HERE, "inputs", "jaspar_abf1_motifs_meme.txt")
OUT = os.path.join(HERE, "abf1_score_split.tsv")

# RoboCOP's own genome background (parameterize.computeBackground)
BG = np.array([0.30980641, 0.19088229, 0.19059636, 0.30871494])
CORE = list(range(0, 5)) + list(range(10, 14))
SPACER = list(range(5, 10))
FIMO_THRESH = 1e-4


def read_chrom(path, name="chrI"):
    seq, on = [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if on:
                    break
                on = line[1:].split()[0] == name
                continue
            if on:
                seq.append(line.strip())
    return "".join(seq).upper()


def read_meme_motif(path, motif="Abf1_murphy"):
    rows, grab = [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                grab = line.split()[1] == motif
                continue
            if grab:
                if line.startswith("letter-probability"):
                    continue
                parts = line.split()
                if len(parts) == 4:
                    try:
                        rows.append([float(x) for x in parts])
                    except ValueError:
                        break
                elif rows:
                    break
    return np.array(rows)


def log_odds(pwm, pseudo=0.1):
    """FIMO's transform: pseudocount spread by background, then log2(p/bg)."""
    p = (pwm + pseudo * BG[None, :]) / (1.0 + pseudo)
    p = p / p.sum(1, keepdims=True)
    return np.log2(p / BG[None, :])


CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode(s):
    return np.array([CODE.get(c, -1) for c in s])


def rc(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def pvalue_table(lo, nbins=10000):
    """Exact FIMO p-values: discretise each column's log-odds, convolve the
    per-column null distributions (base drawn from BG), return (grid, sf)."""
    lo_min, lo_max = lo.min(axis=1).sum(), lo.max(axis=1).sum()
    step = (lo_max - lo_min) / nbins
    dist = np.zeros(1)
    dist[0] = 1.0
    offset = 0
    for j in range(lo.shape[0]):
        idx = np.round((lo[j] - lo[j].min()) / step).astype(int)
        col = np.zeros(idx.max() + 1)
        for b in range(4):
            col[idx[b]] += BG[b]
        dist = np.convolve(dist, col)
        offset += lo[j].min()
    sf = np.cumsum(dist[::-1])[::-1]          # P(score >= bin)
    grid = offset + np.arange(len(dist)) * step
    return grid, sf, step


def pval(score, grid, sf, step):
    i = int(np.floor((score - grid[0]) / step))
    i = max(0, min(i, len(sf) - 1))
    return float(sf[i])


def best_window(code, lo, lo_len, lo_start, lo_stop):
    """Best-scoring window of `lo` over [lo_start, lo_stop), both strands.
    Returns (score, start, strand)."""
    best = (-np.inf, None, None)
    for strand in "+-":
        m = lo if strand == "+" else lo[::-1, ::-1]
        for s in range(lo_start, lo_stop - lo_len + 1):
            w = code[s:s + lo_len]
            if (w < 0).any():
                continue
            sc = m[np.arange(lo_len), w].sum()
            if sc > best[0]:
                best = (sc, s, strand)
    return best


def contributions(code, lo, start, strand):
    """Per-column log-odds at a fixed placement, in MOTIF column order."""
    w = lo.shape[0]
    sub = code[start:start + w]
    if strand == "-":
        sub = 3 - sub[::-1]                   # reverse-complement the window
    return lo[np.arange(w), sub], sub


def main():
    chrom = read_chrom(FASTA, "chrI")
    mur = read_meme_motif(MURPHY)
    jas = read_meme_motif(JASPAR)
    assert mur.shape == jas.shape == (14, 4), (mur.shape, jas.shape)
    lo_m, lo_j = log_odds(mur), log_odds(jas)
    gm, sm, stm = pvalue_table(lo_m)
    gj, sj, stj = pvalue_table(lo_j)

    sites = []
    with open(MACISAAC) as fh:
        for line in fh:
            f = line.split()
            if f[0] == "chrI" and f[3] == "ABF1":
                sites.append((int(f[1]), int(f[2]), f[5]))

    code = encode(chrom)
    rows = []
    print("\nchrI has %d MacIsaac ABF1 sites.  FIMO threshold p < %g\n" %
          (len(sites), FIMO_THRESH))

    for k, (bs, be, strand) in enumerate(sites, 1):
        # search a +/-10 bp window around the annotated site
        lo_start, lo_stop = max(0, bs - 10), min(len(code), be + 10)
        sc_m, st_m, sd_m = best_window(code, lo_m, 14, lo_start, lo_stop)
        # score JASPAR at the SAME placement Murphy chose is not fair; let each
        # matrix pick its own best window, as FIMO does.
        sc_j, st_j, sd_j = best_window(code, lo_j, 14, lo_start, lo_stop)

        cm, subm = contributions(code, lo_m, st_m, sd_m)
        cj, subj = contributions(code, lo_j, st_j, sd_j)
        pm = pval(sc_m, gm, sm, stm)
        pj = pval(sc_j, gj, sj, stj)
        seq_m = "".join("ACGT"[b] for b in subm)
        seq_j = "".join("ACGT"[b] for b in subj)

        rows.append(dict(
            site=k, chrom_start=bs, chrom_end=be, macisaac_strand=strand,
            seq_murphy=seq_m, seq_jaspar=seq_j,
            score_murphy=sc_m, score_jaspar=sc_j,
            core_murphy=cm[CORE].sum(), core_jaspar=cj[CORE].sum(),
            spacer_murphy=cm[SPACER].sum(), spacer_jaspar=cj[SPACER].sum(),
            p_murphy=pm, p_jaspar=pj,
            pass_murphy=pm < FIMO_THRESH, pass_jaspar=pj < FIMO_THRESH,
        ))

        print("site %d  chrI:%d-%d (%s)" % (k, bs, be, strand))
        print("   Murphy  %s  total %6.2f = core %6.2f + spacer %6.2f   p=%.2e  %s"
              % (seq_m, sc_m, cm[CORE].sum(), cm[SPACER].sum(), pm,
                 "PASS" if pm < FIMO_THRESH else "fail"))
        print("   JASPAR  %s  total %6.2f = core %6.2f + spacer %6.2f   p=%.2e  %s"
              % (seq_j, sc_j, cj[CORE].sum(), cj[SPACER].sum(), pj,
                 "PASS" if pj < FIMO_THRESH else "fail"))
        print("   spacer costs Murphy %.2f bits relative to JASPAR\n"
              % (cj[SPACER].sum() - cm[SPACER].sum()))

    # ---- column-level: what TOMTOM sees vs what the scan sees ----------------
    print("per-column view  (ED = Euclidean distance TOMTOM compares with;")
    print("                  gap = mean log-odds difference over the 5 sites)\n")
    print("  col   consensus M/J    ED(prob)   mean log-odds gap (J - M)")
    percol_gap = np.zeros(14)
    for r in rows:
        _, sm_ = contributions(code, lo_m, 0, "+"), None
    gaps = []
    for k, (bs, be, strand) in enumerate(sites, 1):
        lo_start, lo_stop = max(0, bs - 10), min(len(code), be + 10)
        _, st_m, sd_m = best_window(code, lo_m, 14, lo_start, lo_stop)
        _, st_j, sd_j = best_window(code, lo_j, 14, lo_start, lo_stop)
        cm, _ = contributions(code, lo_m, st_m, sd_m)
        cj, _ = contributions(code, lo_j, st_j, sd_j)
        gaps.append(cj - cm)
    percol_gap = np.mean(gaps, axis=0)
    ed = np.sqrt(((mur - jas) ** 2).sum(axis=1))
    for j in range(14):
        tag = "spacer" if j in SPACER else "core"
        print("   %2d    %s / %s        %.3f      %+7.2f   %s"
              % (j, "ACGT"[mur[j].argmax()], "ACGT"[jas[j].argmax()],
                 ed[j], percol_gap[j], tag))
    print("\n  core ED sum   %.3f   core log-odds gap   %+.2f bits"
          % (ed[CORE].sum(), percol_gap[CORE].sum()))
    print("  spacer ED sum %.3f   spacer log-odds gap %+.2f bits"
          % (ed[SPACER].sum(), percol_gap[SPACER].sum()))

    # ---- how far do the failing sites miss by? -----------------------------
    thr_m = gm[np.searchsorted(-sm, -FIMO_THRESH)]
    thr_j = gj[np.searchsorted(-sj, -FIMO_THRESH)]
    print("\nscore threshold for p<%g:  Murphy %.2f bits   JASPAR %.2f bits"
          % (FIMO_THRESH, thr_m, thr_j))
    for r in rows:
        if not r["pass_murphy"]:
            miss = thr_m - r["score_murphy"]
            cost = r["spacer_jaspar"] - r["spacer_murphy"]
            print("  site %d misses Murphy's threshold by %.2f bits; "
                  "the spacer alone costs %.2f bits (%.0f%% of the shortfall)"
                  % (r["site"], miss, cost, 100 * cost / miss))

    keys = list(rows[0].keys())
    with open(OUT, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[k]) for k in keys) + "\n")
    print("\nwrote %s" % OUT)

    nm = sum(r["pass_murphy"] for r in rows)
    nj = sum(r["pass_jaspar"] for r in rows)
    print("recovered at p<%g:  Murphy %d/%d   JASPAR %d/%d"
          % (FIMO_THRESH, nm, len(rows), nj, len(rows)))


if __name__ == "__main__":
    main()
