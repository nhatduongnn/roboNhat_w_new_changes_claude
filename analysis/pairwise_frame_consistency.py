#!/usr/bin/env python
"""Does the obvious thing work: align each pair optimally, then intersect?

The three motifs in a row are different widths, so a fair comparison needs one
window.  The natural proposal is: let TOMTOM's own pairwise optimum place each
pair, then take the window all three placements share.  That is only well posed
if the three pairwise optima are MUTUALLY CONSISTENT -- if native->jaspar and
native->rossi fix jaspar and rossi in one frame, that frame already fixes
jaspar relative to rossi, and there is no guarantee it fixes it where the
jaspar<->rossi pairwise optimum wants it.  Alignment is not transitive.

This script measures how often that happens, and how large the resulting
intersection window is, over every row of the sheet.  Two questions:

  1. consistency -- is the frame-implied jaspar<->rossi alignment the same one
     the free jaspar<->rossi search picks?
  2. size -- how many columns does the triple intersection leave?

TOMTOM itself cannot answer this: it is strictly one query motif against target
motifs (see tomtom.html, "Each of these motifs will be searched against the
target files").  There is no multiple-motif alignment anywhere in the suite --
the only alignment tools in the distribution operate on sequence alignments.
"""
import csv
import os
import numpy as np

import motif_distance_sheet as M

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "motifdb")
# 1e-6, not 1e-9: the yeast background is not exactly strand-symmetric
# (A 0.30981 vs T 0.30871), so a pseudocount added before vs after
# reverse-complementing differs in the 8th decimal.  That is a rounding
# artefact, not a genuinely different alignment.
EPS = 1e-6


def oriented(pwm, orient):
    return M.add_pseudo(pwm if orient == "+" else M.rc(pwm))


def score_at(pa, pb, off):
    """mean KL of two oriented matrices at a given relative offset, over their
    overlap. -> (kl, n_overlap) or (None, 0)"""
    lo, hi = max(0, off), min(len(pa), off + len(pb))
    n = hi - lo
    if n <= 0:
        return None, 0
    kl = sum(M.kld_col(pa[k], pb[k - off]) for k in range(lo, hi)) / n
    return kl, n


def main():
    native = M.parse_meme(os.path.join(DB, "shipped.meme"))
    jaspar = M.parse_meme(os.path.join(DB, "jaspar.meme"))
    rossi = M.parse_meme_meta(os.path.join(DB, "rossi.meme"))

    rows = list(csv.DictReader(open(os.path.join(HERE,
                                                 "motif_distance_sheet.tsv")),
                               delimiter="\t"))
    out = []
    for r in rows:
        if not (r["jaspar_motif"] and r["rossi_motif"]):
            continue
        n = native[r["native_motif"]][1]
        j = jaspar[r["jaspar_motif"]][1]
        s = rossi[r["rossi_motif"]][1]

        nj = M.best_align(n, j)          # (kl, ed, off, orient, novl)
        nr = M.best_align(n, s)
        jr = M.best_align(j, s)
        if not (nj and nr and jr):
            continue

        # place jaspar and rossi in native's frame from their own optima
        pj = oriented(j, nj[3])
        ps = oriented(s, nr[3])
        # jaspar<->rossi as that frame implies it
        impl_kl, impl_n = score_at(pj, ps, nr[2] - nj[2])
        consistent = (impl_kl is not None
                      and impl_n == jr[4]
                      and abs(impl_kl - jr[0]) < EPS)

        # triple intersection in native coordinates
        lo = max(0, nj[2], nr[2])
        hi = min(len(n), nj[2] + len(j), nr[2] + len(s))
        w3 = max(0, hi - lo)

        out.append(dict(
            motif=r["native_motif"], rep=r["replicate"],
            wn=len(n), wj=len(j), wr=len(s), wmin=min(len(n), len(j), len(s)),
            w3=w3,
            ovl_nj=nj[4], ovl_nr=nr[4], ovl_jr=jr[4],
            consistent=consistent,
            jr_free=jr[0],
            jr_implied=impl_kl if impl_kl is not None else float("nan"),
            excess=(impl_kl - jr[0]) if impl_kl is not None else float("nan"),
        ))

    n_all = len(out)
    ok = [o for o in out if o["consistent"]]
    print("rows with all three datasets: %d" % n_all)
    print("frame-implied J<->R equals the free J<->R optimum: %d (%.0f%%)"
          % (len(ok), 100.0 * len(ok) / n_all))
    bad = [o for o in out if not o["consistent"]]
    if bad:
        ex = np.array([o["excess"] for o in bad])
        print("   where it does not: the frame costs J<->R a median of %.3f KL "
              "(p90 %.3f, max %.3f)"
              % (np.median(ex), np.percentile(ex, 90), ex.max()))

    w3 = np.array([o["w3"] for o in out])
    wm = np.array([o["wmin"] for o in out])
    print("\ntriple-intersection window vs the shortest motif's width:")
    print("   W3    median %d  p25 %d  min %d  max %d"
          % (np.median(w3), np.percentile(w3, 25), w3.min(), w3.max()))
    print("   Wmin  median %d  p25 %d  min %d  max %d"
          % (np.median(wm), np.percentile(wm, 25), wm.min(), wm.max()))
    print("   W3 < Wmin in %d of %d rows; W3 < 5 in %d; W3 < 4 in %d"
          % (int((w3 < wm).sum()), n_all, int((w3 < 5).sum()),
             int((w3 < 4).sum())))
    print("   rows where BOTH hold (consistent and W3 >= 6): %d (%.0f%%)"
          % (sum(1 for o in out if o["consistent"] and o["w3"] >= 6),
             100.0 * sum(1 for o in out if o["consistent"] and o["w3"] >= 6)
             / n_all))

    print("\nthe ABF1 replicates -- the case that forced background padding:")
    for o in out:
        if o["motif"] == "Abf1_murphy":
            print("   %-5s widths %2d/%2d/%2d  pairwise overlaps N-J %d  N-R %d"
                  "  J-R %d  ->  W3 %d  consistent %s"
                  % (o["rep"], o["wn"], o["wj"], o["wr"], o["ovl_nj"],
                     o["ovl_nr"], o["ovl_jr"], o["w3"], o["consistent"]))

    # ---- does the choice of window actually change any verdict? ------------
    # For the rows where the naive proposal IS well defined, score all three
    # pairs on the triple intersection and re-call the verdict, then compare
    # with the sheet's (shortest-motif window, background-padded).
    byrow = {(r["native_motif"], r["rossi_motif"]): r for r in rows}
    agree = diff = 0
    changes = []
    for o in out:
        if not (o["consistent"] and o["w3"] >= 5):
            continue
        r = None
        for (m, rm), rr in byrow.items():
            if m == o["motif"] and rr["replicate"] == o["rep"]:
                r = rr
                break
        if r is None:
            continue
        n = native[r["native_motif"]][1]
        j = jaspar[r["jaspar_motif"]][1]
        s = rossi[r["rossi_motif"]][1]
        nj = M.best_align(n, j)
        nr = M.best_align(n, s)
        pn, pj, ps = M.add_pseudo(n), oriented(j, nj[3]), oriented(s, nr[3])
        lo = max(0, nj[2], nr[2])
        hi = min(len(n), nj[2] + len(j), nr[2] + len(s))
        cut = {"native": pn[lo:hi],
               "jaspar": pj[lo - nj[2]:hi - nj[2]],
               "rossi": ps[lo - nr[2]:hi - nr[2]]}
        d = {frozenset(p): M._pair_kl_ed(cut[p[0]], cut[p[1]])[0]
             for p in M.PAIRS}
        _, _, v = M.call_odd_one_out(d)
        if v == r["verdict"]:
            agree += 1
        else:
            diff += 1
            changes.append((o["motif"], o["rep"], o["w3"], r["w_window"],
                            r["verdict"], v))
    print("\nWINDOW CHOICE, on the %d rows where the naive proposal is well "
          "defined:" % (agree + diff))
    print("   same verdict as the sheet's padded shortest-motif window: %d"
          % agree)
    print("   different: %d" % diff)
    for c in sorted(changes):
        print("      %-16s %-5s  intersection W=%-2d -> %-18s | "
              "sheet W=%-2s -> %s" % (c[0], c[1], c[2], c[5], c[3], c[4]))

    print("\nworst inconsistencies (frame cannot realise the J<->R optimum):")
    print("   %-16s %-5s %8s %8s %8s   %s"
          % ("motif", "rep", "J-R free", "implied", "excess", "W3"))
    for o in sorted(bad, key=lambda x: -x["excess"])[:12]:
        print("   %-16s %-5s %8.3f %8.3f %8.3f   %d"
              % (o["motif"], o["rep"], o["jr_free"], o["jr_implied"],
                 o["excess"], o["w3"]))


if __name__ == "__main__":
    main()
