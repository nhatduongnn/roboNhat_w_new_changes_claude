"""Chart the sliding-ABF1-footprint scan -> abf1_slide.png

Reads abf1_slide.json (slide_abf1_profile.py). One row per template width, so the +/-7
(14 bp) width the emission layer actually consumes can be compared against the wider
windows -- including +/-10 (21 bp), which HANDOFF.md section 6.6 asked for and which the
original exploration never ran.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"; MUTED = "#6b6a66"; GRID = "#e7e6e2"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; GREEN = "#1a7a44"


def main():
    d = json.load(open("abf1_slide.json"))
    rows = sorted(d["rows"], key=lambda r: r["half"])
    widths = [r["width"] for r in rows]
    xs = np.arange(len(rows))
    tol, nref = d["tol"], d["n_ref"]

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    # --- A: worst / median true-site percentile (lower = better) ---
    a = ax[0, 0]
    a.plot(xs, [r["worst_site_pct"] for r in rows], "o-", color=ORANGE,
           label="worst of the %d sites" % nref)
    a.plot(xs, [r["median_site_pct"] for r in rows], "s-", color=BLUE, label="median site")
    for x, r in zip(xs, rows):
        a.annotate("%.3f%%" % r["worst_site_pct"], (x, r["worst_site_pct"]),
                   fontsize=7, xytext=(4, 5), textcoords="offset points", color=ORANGE)
    a.set_yscale("log")
    a.set_xticks(xs); a.set_xticklabels(["%d bp\n(±%d)" % (r["width"], r["half"]) for r in rows])
    a.set_ylabel("percentile rank among scored chrI positions")
    a.set_title("A  where the true sites rank (lower is better)\nwidths judged on WORST case, "
                "not median")
    a.grid(alpha=.3, color=GRID); a.legend(fontsize=8)

    # --- B: calls needed to reach full recall ---
    a = ax[0, 1]
    n5 = [r["n_calls_at_5of5"] for r in rows]
    a.bar(xs, n5, color=[GREEN if v == min(n5) else MUTED for v in n5])
    for x, v in zip(xs, n5):
        a.annotate("%d" % v, (x, v), ha="center", fontsize=8,
                   xytext=(0, 3), textcoords="offset points")
    a.set_xticks(xs); a.set_xticklabels(["%d bp" % w for w in widths])
    a.set_ylabel("calls admitted to recover all %d sites" % nref)
    a.set_title("B  precision cost of full recall\n(threshold set to the weakest true site)")
    a.grid(alpha=.3, axis="y", color=GRID)

    # --- C: recall vs number of calls, threshold sweep ---
    a = ax[1, 0]
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(rows):
        sw = [p for p in r["sweep"] if p["n_calls"] > 0]
        if not sw:
            continue
        a.plot([p["n_calls"] for p in sw], [p["recovered"] for p in sw], "o-",
               color=cmap(i / max(1, len(rows) - 1)), label="%d bp" % r["width"], ms=4)
    a.set_xscale("log")
    a.set_xlabel("number of calls on chrI"); a.set_ylabel("sites recovered (of %d)" % nref)
    a.set_ylim(-0.2, nref + 0.2)
    a.set_title("C  recall vs calls, threshold swept\n(call = local max of S, min "
                "separation = template width)")
    a.grid(alpha=.3, color=GRID); a.legend(fontsize=8, title="width", title_fontsize=8)

    # --- D: per-site percentile, one line per site ---
    a = ax[1, 1]
    sites = sorted(rows[0]["site_pcts"].keys(), key=int)
    for j, sc in enumerate(sites):
        a.plot(xs, [r["site_pcts"][sc] for r in rows], "o-",
               label="site #%d (%s)" % (j + 1, sc))
    a.set_yscale("log")
    a.set_xticks(xs); a.set_xticklabels(["%d bp" % w for w in widths])
    a.set_ylabel("percentile rank"); a.set_title("D  each MacIsaac site individually")
    a.grid(alpha=.3, color=GRID); a.legend(fontsize=7)

    fig.suptitle("Sliding the fitted ABF1 Fiber-seq footprint across chrI as a matched "
                 "filter\nleave-one-chromosome-out template refit; match = call centre "
                 "within %d bp of the motif midpoint" % tol, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("abf1_slide.png", dpi=150)
    print("wrote abf1_slide.png")

    # console table -- the "row of how many it matches" summary
    print("\n%-9s %-7s %10s %10s %9s %8s %10s %10s" % (
        "half", "width", "median%", "worst%", "S thr", "calls", "recovered", "precision"))
    for r in rows:
        print("%-9s %-7s %10.4f %10.4f %9.2f %8d %10s %10.5f" % (
            "+/-%d" % r["half"], "%d bp" % r["width"], r["median_site_pct"],
            r["worst_site_pct"], r["thr_for_5of5"], r["n_calls_at_5of5"],
            "%d/%d" % (r["recovered_at_5of5"], nref), r["precision_at_5of5"]))


if __name__ == "__main__":
    main()
