"""Render the per-factor scoring grid from score_factors.py as a standalone HTML page.

score_factors.py writes a long-format TSV: one row per (run, reference, factor). That
answers the question but does not let anyone SEE it -- the thing worth seeing is whether
a configuration got better at ABF1 specifically or just started firing every factor
everywhere, and that is a shape, not a number.

Layout is small multiples, one panel per factor, seven horizontal bars per panel in a
fixed run order. Bars are coloured by model FAMILY (fiber-only / sequence-layer /
alternative Fiber-seq background) rather than by run, because the fiber-vs-sequence split
is the actual finding and three hues survive colour-vision checks where seven would not.
Run identity comes from the axis label, which is also the relief for the aqua family's
light-mode contrast.

Usage
-----
    python make_factor_chart.py --scores chrXIV_factor_scores.tsv \
        --out chrXIV_factor_chart.html
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Which family each run belongs to. The order here is the order bars appear in every
# panel, so the eye can compare across panels without re-reading labels.
FAMILY = [
    ("fib",       "fiber",  "Fiber layer only, untrained"),
    ("fibem10",   "fiber",  "Fiber layer only, EM 10 iterations"),
    ("seq",       "seq",    "Fiber + sequence, untrained"),
    ("seqem10",   "seq",    "Fiber + sequence, EM 10 iterations"),
    ("em10nocap", "seq",    "Fiber + sequence, EM 10 iterations, cap off"),
    ("capA",      "bg",     "Fiber + sequence, merged-pileup background"),
    ("capB",      "bg",     "Fiber + sequence, TSS-top-10% background"),
    # Widened-footprint chain (chrI). Same three-hue scheme: the baseline is a plain
    # sequence-layer run, and the two widened runs are grouped apart from it because the
    # comparison that matters is wideAnull -> wideA, not baseline -> wideA.
    ("baseline",  "seq",    "ABF1 = 14 motif states (unwidened control)"),
    ("wideAnull", "bg",     "ABF1 = 23 states, pads emit the background rate"),
    ("wideA",     "fiber",  "ABF1 = 23 states, pads emit the fitted footprint"),
]
FAMILY_LABEL = {"fiber": "Fiber layer only",
                "seq": "Fiber + sequence layer",
                "bg": "Alternative Fiber-seq background"}

METRICS = [
    ("precision",  "Precision",  "linear", "Of the footprints called, the share landing on a reference site."),
    ("recall",     "Recall",     "linear", "Of the reference sites, the share recovered by a call."),
    ("f1",         "F1",         "linear", "Harmonic mean of precision and recall."),
    ("enrichment", "Enrichment", "log",    "Mean posterior at reference sites divided by mean posterior elsewhere. Independent of the call threshold, so this is the statistic that survives a threshold argument."),
    ("auroc",      "AUROC",      "linear", "Ranking quality: can the posterior separate reference positions from the rest without any threshold at all?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="TSV from score_factors.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=os.path.join(HERE, "factor_chart_template.html"))
    ap.add_argument("--chrom", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.scores, sep="\t")
    chrom = args.chrom or (df["chrom"].iloc[0] if "chrom" in df.columns else "?")

    present = [r for r in FAMILY if r[0] in set(df["run"])]
    missing = [r[0] for r in FAMILY if r[0] not in set(df["run"])]
    if missing:
        print("WARNING: no rows for %s -- omitted from the chart" % ", ".join(missing))

    rows = []
    for _, r in df.iterrows():
        rec = {"run": r["run"], "reference": r["reference"], "factor": r["factor"]}
        for k in ("n_ref", "n_pred", "tp", "fp", "fn", "precision", "recall", "f1",
                  "enrichment", "auroc", "call_threshold", "median_dist", "global_max"):
            v = r.get(k)
            rec[k] = None if v is None or (isinstance(v, float) and pd.isna(v)) else (
                float(v) if k in ("precision", "recall", "f1", "enrichment", "auroc",
                                  "call_threshold", "median_dist", "global_max")
                else int(v))
        # A state EM drove to zero has no posterior mass anywhere on the chromosome, so
        # its precision is 0/0 and its enrichment 0/0. Older score_factors.py runs wrote
        # that as inf; recompute the flag from global_max so those tables render right
        # too, and drop the bogus infinity.
        gm = rec.get("global_max")
        rec["extinct"] = bool(r.get("state_extinct", False)) or (
            gm is not None and gm < 1e-6)
        if rec["extinct"] or (rec.get("enrichment") is not None
                              and not (rec["enrichment"] < float("inf"))):
            rec["enrichment"] = None
            if rec["extinct"]:
                rec["auroc"] = None
        rows.append(rec)

    n_ext = sum(r["extinct"] for r in rows)
    if n_ext:
        print("%d of %d rows are extinct states (no posterior mass on the chromosome); "
              "they render as 'no mass', not as a zero or an infinity" % (n_ext, len(rows)))

    # Factor display order: by reference-site count, descending, with the pooled tail last.
    order = {}
    for r in rows:
        if r["reference"] == "rossi":
            order[r["factor"]] = max(order.get(r["factor"], 0), r["n_ref"] or 0)
    factors = sorted(order, key=lambda f: (f.startswith("_other"), -order[f]))

    payload = dict(
        chrom=chrom,
        runs=[dict(id=a, family=b, desc=c) for a, b, c in present],
        familyLabels=FAMILY_LABEL,
        metrics=[dict(key=a, label=b, scale=c, note=d) for a, b, c, d in METRICS],
        factors=factors,
        rows=rows,
    )

    tpl = open(args.template).read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    with open(args.out, "w") as fh:
        fh.write(html)
    size = os.path.getsize(args.out)
    print("wrote %s (%.2f MB, %d rows, %d factors, %d runs)"
          % (args.out, size / 1e6, len(rows), len(factors), len(present)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
