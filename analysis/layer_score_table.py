"""Comparison table over the layer_scores/ reports, with the block-width correction.

Why this exists
---------------
`score_robocop.py` writes one JSON per run. Reading fifteen of them side by side is the
actual question ("did widening help?"), and for the widened runs that question cannot be
answered from the raw numbers.

THE BLOCK-WIDTH CORRECTION. Every `widememe`-method run keeps the shipped
`sum_for_dbf_probs`, so the posterior collapse sums the WHOLE state block into the TF
column -- pads included. A widened ABF1 call therefore renders as a plateau as wide as its
block (314 bp at wide150) instead of a 14 bp peak. That inflates `mean_post_background`
roughly in proportion to the block width, and `enrichment` is
`mean_post_at_sites / mean_post_background`, so enrichment DEFLATES by the same factor for
purely arithmetic reasons. A raw enrichment drop across a widening is expected and says
nothing about the model.

`enr_corr` divides that out: enrichment x (block_bp / 14). It is a first-order correction,
not an identity -- the site mean is also spread over the block -- so treat it as "the same
statistic on a common footprint scale", not as a measured quantity. The metrics that need
NO correction, and are therefore the ones to judge on, are recall, median_dist, AUROC and
every nucleosome number: those are threshold/rank statistics over positions, blind to how
wide the block is.

Block widths are read from each run's OWN trainDir HMMconfig (`tf_lens[Abf1_murphy]`),
not hardcoded, so a new variant needs no edit here.

Usage
-----
    python layer_score_table.py                       # both chromosomes
    python layer_score_table.py --scores layer_scores --runs layer_runs_chrI.tsv
"""
import argparse
import configparser
import glob
import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_ABF1_BP = 14          # the unwidened Murphy ABF1 motif; the common scale


def _run_order(runs_tsv):
    order = []
    with open(runs_tsv) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            order.append(line.split()[0])
    return order


def abf1_block_bp(outDir):
    """Width of the ABF1 state block for this decode, from its own trainDir."""
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(outDir, "config.ini"))
    train = cfg.get("main", "trainDir")
    for cand in (train, os.path.join(HERE, train), os.path.join(outDir, train)):
        p = os.path.join(cand, "HMMconfig.pkl")
        if os.path.isfile(p):
            d = pickle.load(open(p, "rb"))
            tfs = list(d["tfs"])
            return int(np.asarray(d["tf_lens"])[tfs.index("Abf1_murphy")])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", default=[])
    ap.add_argument("--runs", action="append", default=[])
    args = ap.parse_args()
    if not args.scores:
        args.scores = ["layer_scores", "layerXIV_scores"]
        args.runs = ["layer_runs_chrI.tsv", "layer_runs_chrXIV.tsv"]

    for scores, runs_tsv in zip(args.scores, args.runs):
        order = _run_order(os.path.join(HERE, runs_tsv))
        rows = []
        for label in order:
            p = os.path.join(HERE, scores, "report_%s.json" % label)
            if not os.path.isfile(p):
                rows.append((label, None, None))
                continue
            r = json.load(open(p))
            rows.append((label, r, abf1_block_bp(r["outDir"])))

        print("\n=== %s  (%s) ===" % (scores, runs_tsv))
        print("%-18s %5s | %6s %6s %6s %6s %9s %9s | %6s %6s %5s %6s" % (
            "run", "blk", "recall", "prec", "F1", "medD", "enr_raw", "enr_corr",
            "AUROC", "nucRec", "dyad", "period"))
        print("-" * 108)
        for label, r, blk in rows:
            if r is None:
                print("%-18s %s" % (label, "MISSING report -- did its array task run?"))
                continue
            a, n, ph = r["abf1"], r["nucleosome"], r["phasing"]
            scale = (blk / BASE_ABF1_BP) if blk else float("nan")
            print("%-18s %5s | %6.3f %6.3f %6.3f %6.1f %9.1f %9.1f | %6.3f %6.3f %5.1f %6.1f" % (
                label, blk if blk else "?",
                a["recall"], a["precision"], a["f1"],
                a["median_dist"] if a["median_dist"] is not None else float("nan"),
                a["enrichment"], a["enrichment"] * scale,
                a["auroc"], n["recall"], n["median_dyad_err"],
                ph["median_period_bp"] if ph["median_period_bp"] else float("nan")))
        print("\nblk = ABF1 state-block width in bp (14 = unwidened motif).")
        print("enr_corr = enr_raw x blk/14, undoing the block-wide posterior collapse.")
        print("recall / medD / AUROC / nucRec / dyad / period need no correction.")


if __name__ == "__main__":
    main()
