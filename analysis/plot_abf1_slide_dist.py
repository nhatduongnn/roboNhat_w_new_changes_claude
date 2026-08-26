"""Score distribution of the sliding ABF1 matched filter over every chrI base pair.

Answers, from the saved tracks (abf1_slide_tracks.npz) rather than a re-scan:
  * what range S actually takes, and what the null looks like
  * how a score maps to the "percentile" column in abf1_slide.tsv
  * where the 5 MacIsaac chrI sites sit in that distribution, per width

Percentile in the sweep is defined exactly as slide_abf1_profile.py computes it:
    pct(site) = 100 * mean( S[covered] >= max(S over site_center +/- tol) )
i.e. the fraction of SCORED positions (n>0 A coverage) that beat the site. Lower is
better; it is an upper-tail rank, not a quantile of the site's own score.

Coverage mask comes from abf1_slide_cov.npz (dense_counts on the modkit pileup); rebuild
with the two-liner at the bottom of this docstring if it is missing.

    python plot_abf1_slide_dist.py --widths 15,25
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"; MUTED = "#6b6a66"; GRID = "#e7e6e2"
COLORS = {15: "#2a78d6", 21: "#7b5ea7", 25: "#eb6834", 51: "#1a7a44", 101: "#9a9a9a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="15,25")
    ap.add_argument("--out", default="abf1_slide_dist")
    args = ap.parse_args()
    want = [int(x) for x in args.widths.split(",")]

    d = json.load(open("abf1_slide.json"))
    rows = {r["width"]: r for r in d["rows"]}
    tracks = np.load("abf1_slide_tracks.npz")
    cov = np.load("abf1_slide_cov.npz")
    valid = np.flatnonzero(cov["ntot"] > 0)
    ref = d["ref"]; tol = d["tol"]

    sel = {}
    for w in want:
        r = rows[w]
        s = tracks["s_half%d" % r["half"]]
        sel[w] = (r, s, s[valid])

    # ---------------- console stats ----------------
    print("chrI: %d bp, %d scored (n>0 A coverage) = %.1f%%"
          % (len(cov["ntot"]), valid.size, 100.0 * valid.size / len(cov["ntot"])))
    print("\n%-6s %8s %8s %8s %8s %8s %8s %8s %8s" % (
        "width", "mean", "sd", "min", "p50", "p99", "p99.9", "max", ">=0 %"))
    for w in want:
        r, s, sv = sel[w]
        print("%-6s %8.3f %8.3f %8.2f %8.3f %8.3f %8.3f %8.2f %8.1f" % (
            "%d bp" % w, sv.mean(), sv.std(), sv.min(), np.percentile(sv, 50),
            np.percentile(sv, 99), np.percentile(sv, 99.9), sv.max(),
            100.0 * (sv >= 0).mean()))

    print("\nscore -> percentile at the 5 MacIsaac sites (best S within +/-%d bp):" % tol)
    print("%-12s %s" % ("site", "  ".join("%18s" % ("%d bp: S / pct" % w) for w in want)))
    for i, c in enumerate(ref):
        cells = []
        for w in want:
            r, s, sv = sel[w]
            b = float(np.max(s[max(0, c - tol): c + tol + 1]))
            cells.append("%18s" % ("%7.3f / %6.3f%%" % (b, 100.0 * np.mean(sv >= b))))
        print("#%d %-9d %s" % (i + 1, c, "  ".join(cells)))

    # a few reference points on the score<->percentile curve
    print("\ncalibration: S needed to reach a given upper-tail percentile")
    print("%-8s %s" % ("pct", "  ".join("%10s" % ("%d bp" % w) for w in want)))
    for p in (10, 1, 0.5, 0.1, 0.05, 0.01):
        print("%-8s %s" % ("%.2f%%" % p, "  ".join(
            "%10.3f" % np.percentile(sel[w][2], 100 - p) for w in want)))

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # A: full distribution, log-y
    a = ax[0, 0]
    lo = min(sel[w][2].min() for w in want); hi = max(sel[w][2].max() for w in want)
    bins = np.linspace(lo, hi, 400)
    for w in want:
        r, s, sv = sel[w]
        a.hist(sv, bins=bins, histtype="step", lw=1.4, color=COLORS.get(w, INK),
               label="%d bp  (mean %.2f, sd %.2f)" % (w, sv.mean(), sv.std()))
    g = np.exp(-0.5 * bins ** 2) / np.sqrt(2 * np.pi) * valid.size * (bins[1] - bins[0])
    m = g >= 0.5                      # don't let the gaussian's tail blow up the y-axis
    a.plot(bins[m], g[m], "--", color=MUTED, lw=1, label="N(0,1) reference")
    a.set_yscale("log"); a.set_ylim(0.5, None)
    a.set_xlabel("matched-filter score S"); a.set_ylabel("chrI positions (log)")
    a.set_title("A  S at every scored chrI base pair (n=%d)" % valid.size)
    a.grid(alpha=.3, color=GRID); a.legend(fontsize=8)

    # B: upper-tail survival = THE percentile definition
    a = ax[0, 1]
    for w in want:
        r, s, sv = sel[w]
        xs = np.sort(sv)[::-1]
        pct = 100.0 * (np.arange(1, xs.size + 1) / xs.size)
        a.plot(xs, pct, color=COLORS.get(w, INK), lw=1.4, label="%d bp" % w)
        for i, c in enumerate(ref):
            b = float(np.max(s[max(0, c - tol): c + tol + 1]))
            p = 100.0 * np.mean(sv >= b)
            a.plot([b], [p], "o", ms=6, mfc="none", mec=COLORS.get(w, INK))
            a.annotate("#%d" % (i + 1), (b, p), fontsize=7, xytext=(4, -3),
                       textcoords="offset points", color=COLORS.get(w, INK))
    a.axhline(100.0 * 5 / valid.size, color=MUTED, ls=":", lw=1)
    a.annotate("5 sites / %d scored bp = %.4f%%" % (valid.size, 100.0 * 5 / valid.size),
               (0.02, 100.0 * 5 / valid.size), xycoords=("axes fraction", "data"),
               fontsize=7, color=MUTED, va="bottom")
    a.set_yscale("log"); a.set_xlim(-2, hi)
    a.set_xlabel("score S"); a.set_ylabel("% of scored positions with S >= x")
    a.set_title("B  score -> percentile (this IS the sweep's percentile column)")
    a.grid(alpha=.3, color=GRID); a.legend(fontsize=8)

    # C: the tail, with the true sites and the 5/5 thresholds
    a = ax[1, 0]
    tb = np.linspace(2, hi, 220)
    for w in want:
        r, s, sv = sel[w]
        a.hist(sv[sv >= 2], bins=tb, histtype="step", lw=1.4, color=COLORS.get(w, INK),
               label="%d bp" % w)
        a.axvline(r["thr_for_5of5"], color=COLORS.get(w, INK), ls="--", lw=1)
    a.set_yscale("log"); a.set_ylim(0.4, None)
    for j, w in enumerate(want):
        r, s, sv = sel[w]
        yv = 0.55 if j == 0 else 0.72          # one row of markers per width
        a.annotate("%d bp: 5/5 threshold S >= %.2f  (%d calls)"
                   % (w, r["thr_for_5of5"], r["n_calls_at_5of5"]),
                   (0.98, 0.95 - 0.07 * j), xycoords="axes fraction", ha="right",
                   fontsize=8, color=COLORS.get(w, INK))
        for i, c in enumerate(ref):
            b = float(np.max(s[max(0, c - tol): c + tol + 1]))
            a.plot([b], [yv], "v", ms=8, color=COLORS.get(w, INK), clip_on=False)
            a.annotate("#%d" % (i + 1), (b, yv * 1.35), fontsize=7, ha="center",
                       color=COLORS.get(w, INK))
    a.set_xlabel("score S"); a.set_ylabel("chrI positions (log)")
    a.set_title("C  the tail that matters (S >= 2); triangles = the 5 MacIsaac sites")
    a.grid(alpha=.3, color=GRID); a.legend(fontsize=8)

    # D: how many positions outrank each site
    a = ax[1, 1]
    xs = np.arange(len(ref)); wdt = 0.8 / len(want)
    for j, w in enumerate(want):
        r, s, sv = sel[w]
        n_better = []
        for c in ref:
            b = float(np.max(s[max(0, c - tol): c + tol + 1]))
            n_better.append(int(np.sum(sv >= b)))
        a.bar(xs + j * wdt - 0.4 + wdt / 2, n_better, wdt, color=COLORS.get(w, INK),
              label="%d bp" % w)
        for x, v in zip(xs, n_better):
            a.annotate("%d" % v, (x + j * wdt - 0.4 + wdt / 2, v), ha="center",
                       fontsize=7, xytext=(0, 2), textcoords="offset points")
    a.set_yscale("log")
    a.set_xticks(xs); a.set_xticklabels(["#%d\n%d" % (i + 1, c) for i, c in enumerate(ref)])
    a.set_ylabel("scored chrI positions scoring >= the site (log)")
    a.set_title("D  how deep you must dig to reach each site\n(percentile x %d scored bp)"
                % valid.size)
    a.grid(alpha=.3, axis="y", color=GRID); a.legend(fontsize=8)

    fig.suptitle("Distribution of the sliding-ABF1 matched-filter score across chrI — "
                 "%s\nLOCO template refit, local +/-500 bp background, "
                 "%d scored bp of %d" % (" vs ".join("%d bp" % w for w in want),
                                         valid.size, len(cov["ntot"])), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out + ".png", dpi=150)
    print("\nwrote %s.png" % args.out)


if __name__ == "__main__":
    main()
