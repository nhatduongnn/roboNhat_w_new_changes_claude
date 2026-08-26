"""Whole-chrI scoring across the ABF1 concentration sweep -> conc_sweep_metrics.{json,tsv}

Calls score_robocop.score() -- the single source of truth (HANDOFF.md section 1) -- once per
run and flattens the ABF1 + nucleosome metrics into one table.

Why every column matters: raising lambda raises ABF1's posterior EVERYWHERE, so recall alone
is not evidence of improvement. n_pred is the precision cost, and AUROC is the only
threshold-free discrimination metric here. The hypothesis under test says
    recall rises, n_pred rises, AUROC stays FLAT
i.e. lambda buys level, not information -- the same signature the JASPAR PWM swap showed
(AUROC 0.5866 vs 0.5862).

Whole-chrI scoring is ~9 min/run, so run one lambda per slurm task:
    python score_conc_sweep.py --only robocop_chrI_seq_maskon_conc30
    python score_conc_sweep.py --merge          # combine the per-run JSONs into the table
"""
import argparse
import glob
import json
import os

import score_robocop as S

LAMBDAS = [1, 3, 10, 30, 100, 300, 1000]
VARIANTS = ["seq_maskon", "seq_maskoff"]
SHARD_DIR = "conc_sweep_scores"

# lambda=1 == the existing *_revfix runs (the lambda=1 patch is bit-identical to
# robocop_train_fiberonly, the trainDir those runs used). JASPAR lambda=1 is the reference
# point the sweep is measured against.
def run_table():
    runs = []
    for v in VARIANTS:
        for lam in LAMBDAS:
            out = ("robocop_chrI_%s_revfix" % v if lam == 1
                   else "robocop_chrI_%s_conc%d" % (v, lam))
            runs.append(dict(variant=v, lam=lam, pwm="murphy", outDir=out))
        runs.append(dict(variant=v, lam=1, pwm="jaspar",
                         outDir="robocop_chrI_%s_JASPAR" % v))
    return runs


def label_of(r):
    return "%s_%s_lam%g" % (r["pwm"], r["variant"], r["lam"])


def score_one(r):
    res = S.score(r["outDir"], label=label_of(r))
    a, n = res.get("abf1", {}), res.get("nucleosome", {})
    return dict(
        label=label_of(r), variant=r["variant"], lam=r["lam"], pwm=r["pwm"],
        outDir=r["outDir"],
        abf1_recall=a.get("recall"), abf1_tp=a.get("tp"), abf1_n_pred=a.get("n_pred"),
        abf1_auroc=a.get("auroc"),
        abf1_post_sites=a.get("mean_post_at_sites"),
        abf1_post_bg=a.get("mean_post_background"),
        abf1_enrichment=a.get("enrichment"),
        nuc_recall=n.get("recall"), nuc_median_err=n.get("median_dyad_err"),
        phasing_bp=res.get("phasing", {}).get("median_period_bp"),
        access_corr=res.get("accessibility_consistency", {}).get(
            "mean_corr_access_vs_methylation"),
    )


COLS = ["label", "variant", "lam", "pwm", "abf1_tp", "abf1_recall", "abf1_n_pred",
        "abf1_auroc", "abf1_post_sites", "abf1_post_bg", "abf1_enrichment",
        "nuc_recall", "nuc_median_err", "phasing_bp", "access_corr", "outDir"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="score just this outDir and shard the result")
    ap.add_argument("--merge", action="store_true", help="merge shards into the final table")
    args = ap.parse_args()
    os.makedirs(SHARD_DIR, exist_ok=True)
    runs = run_table()

    if args.only:
        r = next((x for x in runs if x["outDir"].rstrip("/") == args.only.rstrip("/")), None)
        if r is None:
            raise SystemExit("unknown outDir %s" % args.only)
        row = score_one(r)
        with open(os.path.join(SHARD_DIR, label_of(r) + ".json"), "w") as f:
            json.dump(row, f, indent=2)
        print(json.dumps(row, indent=2))
        return

    if args.merge:
        rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(SHARD_DIR, "*.json")))]
        rows.sort(key=lambda r: (r["pwm"], r["variant"], r["lam"]))
        with open("conc_sweep_metrics.json", "w") as f:
            json.dump(rows, f, indent=2)
        with open("conc_sweep_metrics.tsv", "w") as f:
            f.write("\t".join(COLS) + "\n")
            for r in rows:
                f.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in COLS) + "\n")
        print("%-26s %5s %6s %7s %8s %9s %9s %8s" %
              ("label", "tp/5", "npred", "auroc", "post@s", "post_bg", "enrich", "nucrec"))
        for r in rows:
            print("%-26s %5s %6s %7.4f %8.4f %9.5f %9.2f %8.3f" % (
                r["label"], "%d/5" % (r["abf1_tp"] or 0), r["abf1_n_pred"],
                r["abf1_auroc"] or 0, r["abf1_post_sites"] or 0, r["abf1_post_bg"] or 0,
                r["abf1_enrichment"] or 0, r["nuc_recall"] or 0))
        print("\nwrote conc_sweep_metrics.json + .tsv (%d runs)" % len(rows))
        return

    for r in runs:
        print("%-38s %s" % (label_of(r), "on disk" if os.path.isdir(r["outDir"]) else "MISSING"))


if __name__ == "__main__":
    main()
