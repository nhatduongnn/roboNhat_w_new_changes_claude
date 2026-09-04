"""Per-factor detection accuracy for a set of RoboCOP decodes.

Why this exists
---------------
`score_robocop.py` scores exactly one transcription factor -- ABF1 -- because that is
the factor the Fiber-seq work has been chasing. But "ABF1 precision fell from 0.024 to
0.006 when EM was turned on" is unreadable without knowing what happened to everything
else: did the model get worse at ABF1 specifically, or did it start firing every factor
everywhere? Those have opposite fixes.

So this runs the SAME peak-caller over every factor that has ground truth, for every
run, and puts the answers in one table.

It deliberately reuses score_robocop's machinery rather than reimplementing it --
`region_optable`, `abf1_call_threshold`, `call_abf1`, `match_peaks`, `auroc` and
`merge_regions` are all already factor-generic (only `load_abf1` is not, and it is the
one thing replaced here). If the peak-calling convention changes there, these numbers
move with it, which is the point: the per-factor table and the ABF1 headline number must
never be able to disagree because two different callers were used.

Ground truth
------------
Two independent references, both scored when available:

  rossi    inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed
           ChIP-exo peaks conformed to the PWM, 74 factors, already in RoboCOP's naming
           ("Abf1_murphy", "Reb1_badis"), so it joins straight to optable columns. This
           is the same source the shipped all_TFs_1000pealVal_params_pseudo.pkl was
           built against.
  macisaac inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1_match_PWM.bed
           ABF1 and REB1 only, but derived independently. The cross-check: where the two
           disagree for those factors, neither number is quotable on its own.

Because Rossi peaks are ChIP-derived, a "false positive" here may be a real site that was
never ChIPped. Precision is therefore a comparative statistic across runs, not an absolute
one. Enrichment (mean posterior at reference sites / mean elsewhere) does not depend on
the call threshold and is the more stable comparison.

Usage
-----
    python score_factors.py --chrom chrXIV \
        --run fib=robocop_chrXIV_maskoff_fib \
        --run seq=robocop_chrXIV_seq_maskoff_revfix \
        --out chrXIV_factor_scores

    python score_factors.py --chrom chrXIV --runs-from chrXIV_runs.tsv --min-sites 5
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_robocop as S

ROSSI = os.path.join(HERE, "inputs",
                     "rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed")
MACISAAC = os.path.join(HERE, "inputs",
                        "MacIsaac_sacCer3_liftOver_Abf1_Reb1_match_PWM.bed")

# MacIsaac names factors in caps and without the PWM-source suffix the model uses.
MACISAAC_TO_MODEL = {"ABF1": "Abf1_murphy", "REB1": "Reb1_badis"}

# Read the optable in slices rather than one 784 kb x 160 DataFrame. Values at a shared
# position are identical across slices (region_optable stitches the decode's own windows),
# so tiling and concatenating the narrow per-factor tracks is exact; calls are made on the
# full-length stitched track afterwards, never per slice.
CHUNK = 200_000


# ----------------------------------------------------------------------------
# References
# ----------------------------------------------------------------------------
def load_rossi(chrom, path=ROSSI):
    """-> {model_factor_name: [center, ...]} on `chrom`. Header row, factor in column 6."""
    df = pd.read_csv(path, sep="\t")
    df = df[df["chr"] == chrom]
    df = df.assign(center=((df["start"] + df["end"]) // 2).astype(int))
    return {tf: sorted(g["center"].tolist()) for tf, g in df.groupby("TF")}


def load_macisaac(chrom, path=MACISAAC):
    """-> {model_factor_name: [center, ...]} on `chrom`. No header; name in column 4."""
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["chr", "start", "end", "name", "score", "strand"])
    df = df[df["chr"] == chrom]
    df = df.assign(center=((df["start"] + df["end"]) // 2).astype(int),
                   model=df["name"].str.upper().map(MACISAAC_TO_MODEL))
    df = df[df["model"].notna()]
    return {tf: sorted(g["center"].tolist()) for tf, g in df.groupby("model")}


# ----------------------------------------------------------------------------
# Track assembly
# ----------------------------------------------------------------------------
def chrom_tracks(outDir, chrom, factors):
    """Stitch whole-chromosome posterior tracks for `factors` from a decode.

    Returns (positions, {factor: float32 track}, covered_bool). Factors absent from the
    optable are returned as all-zero, which is the honest reading: the state exists in
    the model but never carried mass here.
    """
    dec = S.load_decode(outDir)
    regions = [r for r in S.merge_regions(dec["coords"]) if r[0] == chrom]
    if not regions:
        raise ValueError("%s decoded no %s" % (outDir, chrom))

    pos_parts, cov_parts = [], []
    track_parts = {f: [] for f in factors}
    for (c, rs, re_) in regions:
        for cs in range(rs, re_ + 1, CHUNK):
            ce = min(cs + CHUNK - 1, re_)
            optable, covered, _ = S.region_optable(dec, c, cs, ce)
            n = ce - cs + 1
            pos_parts.append(np.arange(cs, ce + 1))
            cov_parts.append(np.asarray(covered, dtype=bool))
            for f in factors:
                v = (optable[f].values if f in optable.columns
                     else np.zeros(n, dtype=np.float32))
                track_parts[f].append(np.nan_to_num(v).astype(np.float32))
    return (np.concatenate(pos_parts),
            {f: np.concatenate(v) for f, v in track_parts.items()},
            np.concatenate(cov_parts))


# ----------------------------------------------------------------------------
# Scoring one (run, factor, reference)
# ----------------------------------------------------------------------------
def score_factor(track, pos, covered, ref_centers, tol):
    """Precision/recall/F1/AUROC/enrichment for one factor track against one reference.

    The call threshold is score_robocop's: 0.30 x the whole-chromosome max for THIS
    factor in THIS run, floored at 0.10. Per-factor and per-run by construction -- a
    factor that never rises above 0.05 anywhere should not be judged against a threshold
    calibrated on ABF1.
    """
    gmax = float(np.nanmax(track)) if track.size else 0.0
    thr = S.abf1_call_threshold(gmax)
    calls = S.call_abf1(track, pos, thr)
    pred = [c["center"] for c in calls]
    agg = S.match_peaks(pred, ref_centers, tol)

    out = dict(n_ref=agg["n_ref"], n_pred=agg["n_pred"],
               tp=agg["tp"], fp=agg["fp"], fn=agg["fn"],
               precision=agg["precision"], recall=agg["recall"], f1=agg["f1"],
               median_dist=agg["median_dist"],
               call_threshold=thr, global_max=gmax, tol_bp=tol)

    # A state EM drove to zero carries no posterior anywhere on the chromosome. Its
    # precision is 0/0 and its enrichment is 0/0 -- neither is a score, and reporting
    # the latter as "infinite enrichment" would rank an extinct state as the best
    # detector on the page. Flag it instead and leave the metrics unset.
    out["state_extinct"] = bool(gmax < 1e-6)

    lab = np.zeros(len(pos), dtype=bool)
    p0 = int(pos[0])
    for c in ref_centers:
        lo = max(0, c - tol - p0)
        hi = min(len(pos), c + tol - p0 + 1)
        lab[lo:hi] = True
    m = covered
    if lab[m].any() and (~lab[m]).any():
        s, l = track[m], lab[m]
        out["mean_post_at_sites"] = float(np.mean(s[l]))
        out["mean_post_background"] = float(np.mean(s[~l]))
        if not out["state_extinct"]:
            out["auroc"] = S.auroc(s, l)
            if out["mean_post_background"] > 0:
                out["enrichment"] = (out["mean_post_at_sites"] /
                                     out["mean_post_background"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--runs-from", default=None,
                    help="TSV of label<TAB>outDir, appended to --run")
    ap.add_argument("--tol", type=int, default=20)
    ap.add_argument("--min-sites", type=int, default=5,
                    help="factors with fewer reference sites than this are pooled into "
                         "'_other' instead of reported individually")
    ap.add_argument("--out", default=None, help="basename for .tsv/.json (no extension)")
    ap.add_argument("--pkg", default=None,
                    help="RoboCOP package copy used to collapse posteriors "
                         "(default $ROBOCOP_PKG, else ../pkg). Point at a decode's own "
                         "pkgvar for widened-footprint runs.")
    args = ap.parse_args()
    if args.pkg:
        print("collapsing posteriors with %s" % S.use_pkg(args.pkg))

    runs = [tuple(r.split("=", 1)) for r in args.run]
    if args.runs_from:
        for line in open(args.runs_from):
            line = line.strip()
            if line and not line.startswith("#"):
                a, b = line.split("\t")[:2]
                runs.append((a, b))
    if not runs:
        sys.exit("no runs given")

    rossi = load_rossi(args.chrom)
    macisaac = load_macisaac(args.chrom)

    reported = sorted([f for f, c in rossi.items() if len(c) >= args.min_sites],
                      key=lambda f: -len(rossi[f]))
    pooled = sorted(f for f, c in rossi.items() if len(c) < args.min_sites)
    factors = sorted(set(rossi) | set(macisaac))

    print("%s: %d Rossi factors, %d with >= %d sites (%s), %d pooled into _other"
          % (args.chrom, len(rossi), len(reported), args.min_sites,
             ", ".join("%s n=%d" % (f, len(rossi[f])) for f in reported), len(pooled)))
    print("MacIsaac cross-check: %s"
          % ", ".join("%s n=%d" % (f, len(c)) for f, c in sorted(macisaac.items())))

    rows = []
    for label, outDir in runs:
        print("\n== %s (%s)" % (label, outDir), flush=True)
        pos, tracks, covered = chrom_tracks(outDir, args.chrom, factors)
        print("   %d positions, %d covered (%.1f%%)"
              % (len(pos), covered.sum(), 100.0 * covered.mean()))

        for ref_name, ref in (("rossi", rossi), ("macisaac", macisaac)):
            for f in sorted(ref):
                if ref_name == "rossi" and f in pooled:
                    continue
                r = score_factor(tracks[f], pos, covered, ref[f], args.tol)
                r.update(run=label, factor=f, reference=ref_name, chrom=args.chrom)
                rows.append(r)
                print("   %-9s %-18s n=%-3d P=%.3f R=%.3f F1=%.3f enr=%.1fx"
                      % (ref_name, f, r["n_ref"], r["precision"], r["recall"],
                         r["f1"], r.get("enrichment", float("nan"))))

        # Pooled tail: every low-n factor scored as one aggregate, so the long tail is
        # visible without pretending a 2-site factor has a meaningful precision.
        if pooled:
            pooled_pred, pooled_ref = [], []
            for f in pooled:
                t = tracks[f]
                thr = S.abf1_call_threshold(float(np.nanmax(t)) if t.size else 0.0)
                pooled_pred += [c["center"] for c in S.call_abf1(t, pos, thr)]
                pooled_ref += rossi[f]
            agg = S.match_peaks(pooled_pred, sorted(pooled_ref), args.tol)
            rows.append(dict(run=label, factor="_other (%d factors)" % len(pooled),
                             reference="rossi", chrom=args.chrom, tol_bp=args.tol,
                             n_ref=agg["n_ref"], n_pred=agg["n_pred"], tp=agg["tp"],
                             fp=agg["fp"], fn=agg["fn"], precision=agg["precision"],
                             recall=agg["recall"], f1=agg["f1"],
                             median_dist=agg["median_dist"]))
            print("   %-9s %-18s n=%-3d P=%.3f R=%.3f F1=%.3f"
                  % ("rossi", "_other", agg["n_ref"], agg["precision"],
                     agg["recall"], agg["f1"]))
        del tracks

    df = pd.DataFrame(rows)
    lead = ["chrom", "run", "reference", "factor", "n_ref", "n_pred", "tp", "fp", "fn",
            "precision", "recall", "f1", "enrichment", "auroc"]
    df = df[[c for c in lead if c in df.columns]
            + [c for c in df.columns if c not in lead]]

    out = args.out or "%s_factor_scores" % args.chrom
    df.to_csv(out + ".tsv", sep="\t", index=False, float_format="%.6g")
    with open(out + ".json", "w") as fh:
        json.dump(dict(chrom=args.chrom, tol_bp=args.tol, min_sites=args.min_sites,
                       reported_factors=reported, pooled_factors=pooled,
                       runs=[dict(label=a, outDir=os.path.abspath(b)) for a, b in runs],
                       rows=rows), fh, indent=2, default=float)
    print("\nwrote %s.tsv and %s.json  (%d rows)" % (out, out, len(df)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
