"""VARIANT 1b (bg_tss_quantile): re-estimate the Fiber-seq background from the UPPER TAIL
of the TSS-proximal open window, not its average.

make_bg_tss.py already asked "what is the mean methylation of TSS-200..-100?" and got
0.1606 genome-wide -- above the shipped 0.1383 but still far below combined_low_count
(0.2489 watson / 0.2647 crick), so the fallback TF state still outscores background
wherever DNA is most accessible.

This script keeps make_bg_tss.py's TSS source, strand orientation and window definition
verbatim (it imports computeTSSOpenRegions) and only changes the STATISTIC:

    mean    pooled sum(k)/sum(n) over every covered position -- what make_bg_tss did
    top25%  pooled over the most-methylated quarter of positions
    top10%  pooled over the most-methylated tenth  (the user's preferred variant)
    top5%   pooled over the most-methylated twentieth
    max     the single most-methylated position

Two selection modes, because "top 10% of the window" is ambiguous:
    within   rank positions inside EACH TSS window, keep that window's top decile,
             then pool the kept positions genome-wide
    pooled   pool every position from every window first, then keep the global top decile

Per-position rate is k/n, so a position with n=1 and k=1 reads as 1.0. Every statistic is
therefore reported over a sweep of minimum-coverage thresholds; min_n=1 is noise, not
signal. Aggregation of the kept positions is pooled sum(k)/sum(n) -- the same estimator
fit_binomial_parameters() uses -- with the unweighted mean of per-position rates shown
alongside.

Writes analysis/bg_tss_quantile.json (full table) and, for the preferred variant,
inputs/bg_params_tss_top10.pkl in exactly the shape robocop.py expects. NEVER touches
inputs/bg_params.pkl or inputs/all_TFs_1000pealVal_params_pseudo.pkl.

    conda activate robocop-2024
    export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
    python make_bg_tss_quantile.py
"""
import argparse
import json
import pickle

import numpy as np

from make_bg_tss import computeTSSOpenRegions, TSS_FILE, PILEUP
from make_params_pm50 import Pileup

OUT_JSON = "bg_tss_quantile.json"
OUT_PKL = "inputs/bg_params_tss_top10.pkl"
QUANTILES = [("mean", 1.00), ("top25%", 0.25), ("top10%", 0.10),
             ("top5%", 0.05), ("max", 0.0)]
MIN_NS = [1, 5, 10, 20]


def collect(pile, segments, min_n):
    """Per-window arrays of (k, n) for each strand channel, coverage-filtered."""
    out = {"watson": [], "crick": []}
    for s in segments:
        sl = pile.slice(s["chrm"], s["start"], s["stop"])
        if sl is None:
            continue
        ok = (sl["trials"] >= min_n) & (sl["base"] == "A")
        for ch, sym in (("watson", "+"), ("crick", "-")):
            m = ok & (sl["strand"] == sym)
            if m.sum() == 0:
                continue
            out[ch].append((sl["succ"][m].astype(np.int64), sl["trials"][m].astype(np.int64)))
    return out


def stat(windows, frac, mode):
    """Return (pooled p, unweighted mean of per-position rates, n_positions_kept)."""
    if not windows:
        return None
    if mode == "within":
        ks, ns = [], []
        for k, n in windows:
            r = k / n
            if frac >= 1.0:
                sel = np.ones(len(r), bool)
            else:
                keep = max(1, int(np.ceil(len(r) * frac))) if frac > 0 else 1
                idx = np.argsort(-r, kind="stable")[:keep]
                sel = np.zeros(len(r), bool)
                sel[idx] = True
            ks.append(k[sel])
            ns.append(n[sel])
        k, n = np.concatenate(ks), np.concatenate(ns)
    else:  # pooled
        k = np.concatenate([a for a, _ in windows])
        n = np.concatenate([b for _, b in windows])
        r = k / n
        if frac < 1.0:
            keep = max(1, int(np.ceil(len(r) * frac))) if frac > 0 else 1
            idx = np.argsort(-r, kind="stable")[:keep]
            k, n = k[idx], n[idx]
    return float(k.sum() / n.sum()), float(np.mean(k / n)), int(len(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=int, default=200)
    ap.add_argument("--down", type=int, default=100)
    ap.add_argument("--write-pkl", action="store_true",
                    help="also write inputs/bg_params_tss_top10.pkl (within, min_n=10)")
    args = ap.parse_args()

    segments = computeTSSOpenRegions(TSS_FILE, args.up, args.down)
    pile = Pileup(PILEUP)

    old = pickle.load(open("inputs/bg_params.pkl", "rb"))
    ow = float(np.ravel(old["p"]["watson_signal"]["A"])[0])
    oc = float(np.ravel(old["p"]["crick_signal"]["A"])[0])
    lp = pickle.load(open("inputs/all_TFs_1000pealVal_params_pseudo.pkl", "rb"))
    lw = float(np.ravel(lp["p"]["combined_low_count"]["watson_signal"]["A"])[0])
    lc = float(np.ravel(lp["p"]["combined_low_count"]["crick_signal"]["A"])[0])

    print("\nshipped bg          watson %.4f  crick %.4f" % (ow, oc))
    print("combined_low_count  watson %.4f  crick %.4f  <- the bar to clear\n" % (lw, lc))

    table = {}
    for min_n in MIN_NS:
        data = collect(pile, segments, min_n)
        npos = {c: sum(len(k) for k, _ in v) for c, v in data.items()}
        print("=== min coverage n >= %d   (%d watson / %d crick covered positions) ==="
              % (min_n, npos["watson"], npos["crick"]))
        print("%-8s %-8s %10s %10s %10s %10s %8s %8s"
              % ("stat", "mode", "watson_p", "crick_p", "w_meanrate", "c_meanrate",
                 "w>low", "c>low"))
        for mode in ("within", "pooled"):
            for name, frac in QUANTILES:
                w = stat(data["watson"], frac, mode)
                c = stat(data["crick"], frac, mode)
                if w is None or c is None:
                    continue
                table["min_n=%d|%s|%s" % (min_n, mode, name)] = {
                    "watson_pooled_p": w[0], "crick_pooled_p": c[0],
                    "watson_mean_rate": w[1], "crick_mean_rate": c[1],
                    "watson_positions": w[2], "crick_positions": c[2]}
                print("%-8s %-8s %10.4f %10.4f %10.4f %10.4f %8s %8s"
                      % (name, mode, w[0], c[0], w[1], c[1],
                         "PASS" if w[0] > lw else "fail",
                         "PASS" if c[0] > lc else "fail"))
        print("")

    meta = {"shipped_bg": {"watson": ow, "crick": oc},
            "combined_low_count": {"watson": lw, "crick": lc},
            "window": {"up": args.up, "down": args.down},
            "n_segments": len(segments),
            "table": table}
    with open(OUT_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote %s" % OUT_JSON)

    if args.write_pkl:
        key = "min_n=10|within|top10%"
        e = table[key]
        bg = {"p": {"watson_signal": {b: np.zeros(1) for b in "ACGT"},
                    "crick_signal": {b: np.zeros(1) for b in "ACGT"}}}
        bg["p"]["watson_signal"]["A"] = np.array([e["watson_pooled_p"]])
        bg["p"]["crick_signal"]["A"] = np.array([e["crick_pooled_p"]])
        with open(OUT_PKL, "wb") as f:
            pickle.dump(bg, f)
        print("wrote %s  (%s -> watson %.4f crick %.4f)"
              % (OUT_PKL, key, e["watson_pooled_p"], e["crick_pooled_p"]))


if __name__ == "__main__":
    main()
