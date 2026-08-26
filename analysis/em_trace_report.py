"""Read an EM training run's per-iteration prior trace and report where it moved.

Consumes <trainDir>/em_trace/iter{i}.npz (written by pkgvar/seq_maskoff_em10/robocop_em.py)
plus <trainDir>/likelihood.txt, and emits:

    em_trace.tsv   every state x every iteration
    em_trace.png   log-y trajectories for the states that matter, + the likelihood curve
    stdout         start -> end fold changes, which states hit the constrained-EM cap,
                   and the Nhp6a/Abf1 ratio per iteration

The question this exists to answer: the shipped prior is `calculateKD` output, which gives
Nhp6a 7.2239e-4 (6th of 154) and Abf1 1.7974e-7 (148th) -- a 4019x gap set by motif length
alone -- while the model's own posterior on chrI implies only ~6.9x. Does Baum-Welch close
that, and does the freed mass go to Abf1 or get absorbed by the uncapped `unknown` state?

Usage
-----
    python em_trace_report.py robocop_train_em10_chrII
    python em_trace_report.py robocop_train_em10_chrII --baseline robocop_train_fiberonly
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The states worth naming in the figure: the two protagonists, the short-AT-rich cluster
# that shares Nhp6a's failure mode, a well-behaved long motif as a control, and the three
# non-TF states that absorb whatever the TFs give up.
HIGHLIGHT = [
    ("Abf1_murphy", "#17ef13", 2.4),
    ("Nhp6a_zhu", "#3134eb", 2.4),
    ("Nhp6b_zhu", "#7d7fe8", 1.2),
    ("Rox1_badis", "#c86ad0", 1.2),
    ("Sig1_badis", "#e8823a", 1.2),
    ("Sum1_zhu", "#b0a24a", 1.2),
    ("Reb1_badis", "#0b8c8f", 1.6),
    ("unknown", "#8a8a8a", 1.8),
]


def load_trace(trainDir):
    files = glob.glob(os.path.join(trainDir, "em_trace", "iter*.npz"))
    if not files:
        sys.exit("no em_trace/iter*.npz in %s -- was this trained with the em10 variant?"
                 % trainDir)
    files.sort(key=lambda p: int(re.search(r"iter(\d+)\.npz$", p).group(1)))
    iters, tfs, mat, bg, nuc, ll = [], None, [], [], [], []
    for p in files:
        z = np.load(p)
        if tfs is None:
            tfs = [str(x) for x in z["tfs"]]
        elif [str(x) for x in z["tfs"]] != tfs:
            sys.exit("state ordering changed between iterations in %s" % p)
        iters.append(int(z["iteration"]))
        mat.append(np.asarray(z["tf_prob"], dtype=float))
        bg.append(float(z["background_prob"]))
        nuc.append(float(z["nucleosome_prob"]))
        ll.append(float(z["log_likelihood"]))
    return iters, tfs, np.array(mat), np.array(bg), np.array(nuc), np.array(ll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trainDir")
    ap.add_argument("--baseline", default="robocop_train_fiberonly",
                    help="unfitted trainDir to check iteration 0 against")
    ap.add_argument("--top", type=int, default=15, help="rows in the movers table")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    pre = args.out_prefix or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "em_trace_" + os.path.basename(args.trainDir.rstrip("/")))

    iters, tfs, mat, bg, nuc, ll = load_trace(args.trainDir)
    n_it = len(iters)
    idx = {t: i for i, t in enumerate(tfs)}
    print("%s: %d recorded states, iterations %d..%d" % (args.trainDir, len(tfs),
                                                         iters[0], iters[-1]))

    # ---- gate 0/1: iteration 0 must be the unfitted calculateKD prior ----
    if os.path.isdir(args.baseline):
        base = pickle.load(open(os.path.join(args.baseline, "HMMconfig.pkl"), "rb"),
                           encoding="latin1")
        btfs = [str(x) for x in base["tfs"]]
        bp = np.ravel(base["tf_prob"]).astype(float)
        if btfs != tfs:
            print("  WARNING: state list differs from %s -- skipping the iter0 gate"
                  % args.baseline)
        else:
            same = np.array_equal(mat[0], bp)
            print("  iter0 vs %s: %s (max abs diff %.3g)"
                  % (args.baseline, "BIT-IDENTICAL" if same else "*** DIFFERS ***",
                     np.abs(mat[0] - bp).max()))
            a = idx.get("Abf1_murphy")
            if a is not None:
                print("  iter0 Abf1_murphy = %.10g  (unpatched calculateKD value is "
                      "1.79736716e-07)" % mat[0][a])
            if not same:
                print("  *** iteration 0 is NOT the stock prior. Do not interpret this "
                      "trace until that is explained. ***")

    # ---- likelihood ----
    lltxt = os.path.join(args.trainDir, "likelihood.txt")
    lls = [float(x) for x in open(lltxt)] if os.path.isfile(lltxt) else list(ll)
    d = np.diff(lls)
    print("\nlog-likelihood: %d entries, %.6f -> %.6f (delta %+.6f)"
          % (len(lls), lls[0], lls[-1], lls[-1] - lls[0]))
    if len(d) and d.min() < 0:
        print("  *** NOT monotonic: %d of %d steps decreased, worst %+.6f. Baum-Welch "
              "guarantees non-decreasing likelihood -- treat the fit as suspect. ***"
              % (int((d < 0).sum()), len(d), d.min()))
    elif len(d):
        print("  monotonically non-decreasing over all %d steps" % len(d))

    # ---- constrained-EM cap ----
    thr_pool = [mat[0][i] for i, t in enumerate(tfs) if t != "unknown"]
    cap = float(np.mean(thr_pool) + 2 * np.std(thr_pool))
    print("\nconstrained-EM cap (mean+2sd of the initial TF priors) = %.6g" % cap)
    at_cap = []
    for i, t in enumerate(tfs):
        if t == "unknown":
            continue
        hit = np.where(np.isclose(mat[:, i], cap, rtol=1e-9))[0]
        if hit.size:
            at_cap.append((t, iters[hit[0]], mat[-1][i]))
    if at_cap:
        print("  clamped states: " + ", ".join("%s (from iter %d)" % (t, it)
                                               for t, it, _ in at_cap))
    else:
        print("  no state ended up pinned at the cap")

    # ---- the ratio that motivated the run ----
    if "Nhp6a_zhu" in idx and "Abf1_murphy" in idx:
        na, ab = idx["Nhp6a_zhu"], idx["Abf1_murphy"]
        print("\n  iter   Nhp6a        Abf1         Nhp6a/Abf1   background   nucleosome")
        for k in range(n_it):
            r = mat[k][na] / mat[k][ab] if mat[k][ab] > 0 else float("inf")
            print("  %4d   %.4e   %.4e   %10.1fx   %.6f     %.6f"
                  % (iters[k], mat[k][na], mat[k][ab], r, bg[k], nuc[k]))
        print("  (chrI posteriors imply ~6.9x; the unfitted prior says 4019x)")

    # ---- biggest movers ----
    with np.errstate(divide="ignore", invalid="ignore"):
        fold = np.where(mat[0] > 0, mat[-1] / mat[0], np.inf)
    lg = np.abs(np.log10(np.where(np.isfinite(fold) & (fold > 0), fold, 1.0)))
    moved = np.where(lg > 1e-12)[0]          # states that actually changed at all
    order = moved[np.argsort(-lg[moved])]
    print("\n%d of %d states moved; top %d (iter %d -> %d):"
          % (len(moved), len(tfs), min(args.top, len(order)), iters[0], iters[-1]))
    print("  %-16s %12s %12s %10s" % ("state", "start", "end", "fold"))
    for i in order[:args.top]:
        print("  %-16s %12.4e %12.4e %10.3g" % (tfs[i], mat[0][i], mat[-1][i], fold[i]))
    if len(moved) == 0:
        print("  (none -- the prior is unchanged, i.e. EM did not run)")

    # ---- tsv ----
    tsv = pre + ".tsv"
    with open(tsv, "w") as f:
        f.write("state\t" + "\t".join("iter%d" % i for i in iters) + "\n")
        for i, t in enumerate(tfs):
            f.write(t + "\t" + "\t".join("%.10g" % v for v in mat[:, i]) + "\n")
        f.write("background\t" + "\t".join("%.10g" % v for v in bg) + "\n")
        f.write("nucleosome\t" + "\t".join("%.10g" % v for v in nuc) + "\n")
        f.write("log_likelihood\t" + "\t".join("%.10g" % v for v in ll) + "\n")
    print("\nwrote %s" % tsv)

    # ---- figure ----
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    for i, t in enumerate(tfs):
        if t in dict((h[0], 1) for h in HIGHLIGHT):
            continue
        ax.plot(iters, mat[:, i], color="#cfd6dd", lw=0.6, zorder=1)
    for name, col, lw in HIGHLIGHT:
        if name not in idx:
            continue
        ax.plot(iters, mat[:, idx[name]], color=col, lw=lw, zorder=3,
                label=name.split("_")[0].upper(), marker="o", ms=3)
    ax.plot(iters, bg, color="#39424b", lw=1.8, ls="--", zorder=3, label="background")
    ax.plot(iters, nuc, color="#000000", lw=1.4, ls=":", zorder=3, label="nucleosome")
    ax.axhline(cap, color="#d0322f", lw=1, ls="-.", zorder=2)
    ax.text(iters[-1], cap, "  constrained-EM cap %.2e" % cap, color="#d0322f",
            fontsize=8, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("transition prior (concentration)")
    ax.set_title("EM trajectory of the RoboCOP transition prior\n%s"
                 % os.path.basename(args.trainDir.rstrip("/")), fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="center right")
    ax.grid(alpha=0.25, which="both", lw=0.4)

    ax2.plot(range(len(lls)), lls, color="#1f4e79", lw=1.8, marker="o", ms=3)
    ax2.set_xlabel("EM iteration")
    ax2.set_ylabel("log-likelihood")
    ax2.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(pre + ".png", dpi=160)
    plt.close("all")
    print("wrote %s" % (pre + ".png"))


if __name__ == "__main__":
    main()
