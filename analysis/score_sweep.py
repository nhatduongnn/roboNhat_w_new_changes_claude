"""Read a concentration sweep's score reports and print the F1-vs-lambda curve.

The deliverable of a sweep round is the CURVE, not an argmax. Three things are read off it
that a single best-lambda would hide:

  * where the peak is -- and whether it is INTERIOR. A peak pinned at the top of the grid
    means the bracket was too narrow, which is exactly the mistake the constrained-EM
    ceiling makes (robocop_em.py:113 pins 28 of 154 TFs to mean+2sd of the initial priors).
  * how FLAT it is. If F1 barely moves across four decades of lambda, concentration is not
    the lever for that factor and no amount of calibration will help.
  * where precision and recall cross. Raising lambda always buys recall and always costs
    precision; F1's peak is just where that trade turns over. Seeing both columns says
    whether the peak is a real optimum or a knife edge.

    conda activate robocop-2024
    python score_sweep.py                          # conc_sweep_runs.tsv + conc_scores/
    python score_sweep.py --runs r2.tsv --dir s2   # a later round
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read_runs(path):
    """-> [(label, outDir, lambda)] from the sweep table."""
    out = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if line:
            f = line.split()
            if len(f) != 3:
                sys.exit("%s: expected '<label> <outDir> <lambda>', got %r" % (path, line))
            out.append((f[0], f[1], float(f[2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "conc_sweep_runs.tsv"))
    ap.add_argument("--dir", default=os.path.join(HERE, "conc_scores"))
    ap.add_argument("--tf", default="abf1", help="report key holding the site-level scores")
    a = ap.parse_args()

    runs = sorted(read_runs(a.runs), key=lambda r: r[2])
    rows, missing = [], []
    for label, outdir, lam in runs:
        p = os.path.join(a.dir, "report_%s.json" % label)
        if not os.path.isfile(p):
            missing.append(label)
            continue
        r = json.load(open(p))
        e = r.get(a.tf)
        if not e:
            missing.append(label + "(no %s entry)" % a.tf)
            continue
        nuc = r.get("nucleosome") or {}
        ph = r.get("phasing") or {}
        rows.append(dict(label=label, lam=lam, f1=e.get("f1"), prec=e.get("precision"),
                         rec=e.get("recall"), tp=e.get("tp"), fp=e.get("fp"), fn=e.get("fn"),
                         auroc=e.get("auroc"), enr=e.get("enrichment"),
                         at=e.get("mean_post_at_sites"), bg=e.get("mean_post_background"),
                         nucrec=nuc.get("recall"), period=ph.get("median_period_bp")))
    if missing:
        print("not scored yet: %s\n" % ", ".join(missing))
    if not rows:
        sys.exit("nothing to report")

    n_ref = json.load(open(os.path.join(a.dir, "report_%s.json" % rows[0]["label"])))\
        [a.tf].get("n_ref")
    print("%s: %d reference sites, %d lambda points\n" % (a.tf.upper(), n_ref, len(rows)))
    hdr = ("%-9s %10s %7s %7s %7s %6s %6s %6s %7s %9s %9s %7s %7s"
           % ("label", "lambda", "F1", "prec", "recall", "TP", "FP", "FN",
              "AUROC", "post@site", "post@bg", "nucRec", "phase"))
    print(hdr)
    print("-" * len(hdr))
    best = max(rows, key=lambda r: (r["f1"] if r["f1"] is not None else -1))
    for r in rows:
        f = lambda v, s="%7.3f": ("%7s" % "-") if v is None else s % v
        mark = "  <- best F1" if r is best else ""
        print("%-9s %10.4g %s %s %s %6s %6s %6s %s %9.5f %9.5f %s %s%s"
              % (r["label"], r["lam"], f(r["f1"]), f(r["prec"]), f(r["rec"]),
                 r["tp"], r["fp"], r["fn"], f(r["auroc"]),
                 r["at"] or 0, r["bg"] or 0, f(r["nucrec"]),
                 f(r["period"], "%7.0f"), mark))

    print()
    # --- the three things the curve is for -------------------------------------
    lo, hi = rows[0], rows[-1]
    if best is lo or best is hi:
        print("WARNING: best F1 sits at the %s edge of the grid (lambda=%.4g). The bracket "
              "is too narrow -- extend it before trusting this peak. Pinning at an edge is "
              "the same failure mode as the constrained-EM ceiling."
              % ("low" if best is lo else "high", best["lam"]))
    else:
        print("peak is INTERIOR at lambda=%.4g (F1 %.3f), between %.4g and %.4g -- "
              "bracket is wide enough." % (best["lam"], best["f1"],
                                           rows[max(0, rows.index(best) - 1)]["lam"],
                                           rows[min(len(rows) - 1, rows.index(best) + 1)]["lam"]))

    f1s = [r["f1"] for r in rows if r["f1"] is not None]
    if f1s:
        span = max(f1s) - min(f1s)
        print("F1 range across the sweep: %.3f to %.3f (span %.3f)%s"
              % (min(f1s), max(f1s), span,
                 "  -- FLAT: concentration is not the lever here" if span < 0.05 else ""))

    ats = [r["at"] for r in rows if r["at"] is not None]
    if len(ats) > 1:
        mono = all(b >= a_ - 1e-12 for a_, b in zip(ats, ats[1:]))
        print("posterior at true sites is monotone in lambda: %s%s"
              % (mono, "" if mono else "  -- REVERSAL: something other than the prior moved"))


if __name__ == "__main__":
    main()
