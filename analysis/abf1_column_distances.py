#!/usr/bin/env python
"""Raw per-column distance between Murphy ABF1 and JASPAR ABF1 -- ED and KLD.

This deliberately reports the RAW summed distances, not TOMTOM p-values.  The
p-value asks 'is this pair closer than two random matrices', which saturates for
any pair sharing a strong core.  The raw distance is the actual answer to 'how
far apart are they'.

TOMTOM's definitions (Gupta et al. 2007, Genome Biology 8:R24), applied here
exactly, per column, then summed over the alignment:

    ED(X,Y)  = sqrt( sum_a (X_a - Y_a)^2 )            [sqrt confirmed by
                                                       experiment, see notes]
    KLD(X,Y) = 1/2 * ( sum_a X_a ln(X_a/Y_a)
                     + sum_a Y_a ln(Y_a/X_a) )        [symmetrised, natural log]

Both are preceded by TOMTOM's -motif-pseudo 0.1, spread by the background.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MURPHY = os.path.join(HERE, "inputs", "motifs_meme.txt")
JASPAR = os.path.join(HERE, "inputs", "jaspar_abf1_motifs_meme.txt")
OUT = os.path.join(HERE, "abf1_column_distances.tsv")

BG = np.array([0.30980641, 0.19088229, 0.19059636, 0.30871494])  # RoboCOP genome
PSEUDO = 0.1
CORE = list(range(0, 5)) + list(range(10, 14))
SPACER = list(range(5, 10))
B = "ACGT"


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
                p = line.split()
                if len(p) == 4:
                    try:
                        rows.append([float(x) for x in p])
                    except ValueError:
                        break
                elif rows:
                    break
    return np.array(rows)


def add_pseudo(pwm, pseudo=PSEUDO, bg=BG):
    """TOMTOM -motif-pseudo: add `pseudo` counts spread by the background."""
    p = (pwm + pseudo * bg[None, :]) / (1.0 + pseudo)
    return p / p.sum(1, keepdims=True)


def ed_col(x, y):
    return float(np.sqrt(((x - y) ** 2).sum()))


def kld_col(x, y):
    return float(0.5 * ((x * np.log(x / y)).sum() + (y * np.log(y / x)).sum()))


def show_worked(j, x, y, xr, yr, label):
    print("\n" + "=" * 74)
    print("WORKED EXAMPLE -- column %d  (%s)" % (j, label))
    print("=" * 74)
    print("  raw from the MEME files       Murphy      JASPAR")
    for b in range(4):
        print("     P(%s)                     %.6f    %.6f" % (B[b], xr[b], yr[b]))
    print("\n  after -motif-pseudo 0.1 spread by background")
    print("     P(a) = (raw + 0.1*bg) / 1.1,  bg = %s" %
          np.array2string(BG, precision=4))
    for b in range(4):
        print("     P(%s)   M (%.6f + 0.1*%.4f)/1.1 = %.6f   |   "
              "J (%.6f + 0.1*%.4f)/1.1 = %.6f"
              % (B[b], xr[b], BG[b], x[b], yr[b], BG[b], y[b]))

    print("\n  -- Euclidean distance -------------------------------------------")
    terms = []
    for b in range(4):
        d = x[b] - y[b]
        terms.append(d * d)
        print("     (%.6f - %.6f)^2 = (%+.6f)^2 = %.6f"
              % (x[b], y[b], d, d * d))
    print("     sum = %.6f      ED = sqrt(%.6f) = %.4f"
          % (sum(terms), sum(terms), np.sqrt(sum(terms))))
    print("     (a column can be at most sqrt(2) = 1.4142 apart)")

    print("\n  -- Kullback-Leibler divergence, symmetrised ---------------------")
    f, r = 0.0, 0.0
    for b in range(4):
        t = x[b] * np.log(x[b] / y[b])
        f += t
        print("     M->J: %.6f * ln(%.6f / %.6f) = %.6f * (%+.4f) = %+.6f"
              % (x[b], x[b], y[b], x[b], np.log(x[b] / y[b]), t))
    print("     sum M->J = %+.6f" % f)
    for b in range(4):
        t = y[b] * np.log(y[b] / x[b])
        r += t
        print("     J->M: %.6f * ln(%.6f / %.6f) = %.6f * (%+.4f) = %+.6f"
              % (y[b], y[b], x[b], y[b], np.log(y[b] / x[b]), t))
    print("     sum J->M = %+.6f" % r)
    print("     KLD = 0.5 * (%.6f + %.6f) = %.4f nats" % (f, r, 0.5 * (f + r)))
    print("     (unbounded above -- no ceiling the way ED has one)")


def main():
    mr = read_meme_motif(MURPHY)
    jr = read_meme_motif(JASPAR)
    m, j = add_pseudo(mr), add_pseudo(jr)

    ed = np.array([ed_col(m[k], j[k]) for k in range(14)])
    kl = np.array([kld_col(m[k], j[k]) for k in range(14)])

    print("\nABF1: Murphy (inputs/motifs_meme.txt) vs JASPAR MA0265.3-rc")
    print("consensus  Murphy  ATCAC ATGGC ACGA")
    print("           JASPAR  ATCAC TATAT ACGA")
    print("\n%-4s %-6s %-38s %-38s %8s %8s %7s %7s"
          % ("col", "region", "Murphy  A/C/G/T", "JASPAR  A/C/G/T",
             "ED", "KLD", "%ED", "%KLD"))
    print("-" * 118)
    for k in range(14):
        reg = "spacer" if k in SPACER else "core"
        print("%-4d %-6s %-38s %-38s %8.4f %8.4f %6.1f%% %6.1f%%"
              % (k, reg,
                 " ".join("%.3f" % v for v in mr[k]),
                 " ".join("%.3f" % v for v in jr[k]),
                 ed[k], kl[k], 100 * ed[k] / ed.sum(), 100 * kl[k] / kl.sum()))
    print("-" * 118)
    print("%-52s %-38s %8.4f %8.4f" % ("TOTAL over 14 columns", "", ed.sum(), kl.sum()))
    print("%-52s %-38s %8.4f %8.4f %6.1f%% %6.1f%%"
          % ("  9 core columns", "", ed[CORE].sum(), kl[CORE].sum(),
             100 * ed[CORE].sum() / ed.sum(), 100 * kl[CORE].sum() / kl.sum()))
    print("%-52s %-38s %8.4f %8.4f %6.1f%% %6.1f%%"
          % ("  5 spacer columns", "", ed[SPACER].sum(), kl[SPACER].sum(),
             100 * ed[SPACER].sum() / ed.sum(), 100 * kl[SPACER].sum() / kl.sum()))
    print("\nmean per core column    ED %.4f   KLD %.4f"
          % (ed[CORE].mean(), kl[CORE].mean()))
    print("mean per spacer column  ED %.4f   KLD %.4f"
          % (ed[SPACER].mean(), kl[SPACER].mean()))
    print("spacer / core ratio     ED %.1fx      KLD %.1fx"
          % (ed[SPACER].mean() / ed[CORE].mean(),
             kl[SPACER].mean() / kl[CORE].mean()))

    show_worked(0, m[0], j[0], mr[0], jr[0], "core -- the two agree")
    show_worked(7, m[7], j[7], mr[7], jr[7], "spacer -- the worst column")

    # ---- calibration: what do these totals mean? ---------------------------
    # No random null here -- just real, interpretable anchors from our own data,
    # every pair scored at offset 0 over the common 14 columns.
    rossi = read_meme_motif(os.path.join(HERE, "motifdb", "rossi.meme"),
                            "ABF1_rossi_15254_m1")[:14]
    reb1 = read_meme_motif(MURPHY, "Reb1_badis")
    anchors = [
        ("Murphy ABF1 vs itself", mr, mr),
        ("JASPAR ABF1 vs Rossi ABF1", jr, rossi),
        ("Murphy ABF1 vs Rossi ABF1", mr, rossi),
        ("Murphy ABF1 vs JASPAR ABF1", mr, jr),
    ]
    anchors.append(("Murphy ABF1 vs Reb1 (unrelated)", mr, reb1))

    def best_align(a, b, min_overlap=6):
        """Search every offset and both orientations, as TOMTOM does; score by
        MEAN KLD per aligned column so unequal overlaps stay comparable."""
        pa = add_pseudo(a)
        best = None
        for orient in "+-":
            pb = add_pseudo(b if orient == "+" else b[::-1, ::-1])
            for off in range(-len(pb) + 1, len(pa)):
                lo, hi = max(0, off), min(len(pa), off + len(pb))
                if hi - lo < min_overlap:
                    continue
                idx = range(lo, hi)
                n = hi - lo
                d = sum(kld_col(pa[k], pb[k - off]) for k in idx) / n
                e = sum(ed_col(pa[k], pb[k - off]) for k in idx) / n
                if best is None or d < best[0]:
                    best = (d, e, off, orient, n)
        return best

    print("\n" + "=" * 78)
    print("CALIBRATION -- raw distances at the best alignment. No null, no p-value.")
    print("Reported per aligned column so different overlaps are comparable.")
    print("=" * 78)
    print("  %-32s %9s %9s %7s %6s %5s"
          % ("", "ED/col", "KLD/col", "offset", "orient", "cols"))
    for name, a, b in anchors:
        d, e, off, orient, n = best_align(a, b)
        print("  %-32s %9.4f %9.4f %7d %6s %5d"
              % (name, e, d, off, orient, n))

    with open(OUT, "w") as fh:
        fh.write("col\tregion\t" + "\t".join("murphy_%s" % b for b in B) + "\t"
                 + "\t".join("jaspar_%s" % b for b in B) + "\ted\tkld\n")
        for k in range(14):
            fh.write("%d\t%s\t%s\t%s\t%.6f\t%.6f\n"
                     % (k, "spacer" if k in SPACER else "core",
                        "\t".join("%.6f" % v for v in mr[k]),
                        "\t".join("%.6f" % v for v in jr[k]), ed[k], kl[k]))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
