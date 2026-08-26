"""Print + plot the calibration of the sliding-ABF1 matched filter.

Reads abf1_slide_calib.json / abf1_slide_calib_half*.npz (abf1_slide_calibrate.py).

FDR is reported as the RAW ratio expected_null / observed at each threshold. No step-up
monotonisation: with only `nperm` null draws the far tail hits zero null counts by
sampling limitation, and a monotone minimum-from-the-right then propagates that spurious
zero back down to every lower threshold. Resolution floor is 1/nperm of a position.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#0b0b0b"; MUTED = "#6b6a66"; GRID = "#e7e6e2"
C = {15: "#2a78d6", 25: "#eb6834"}


def ratio(obs, null):
    obs = np.array(obs, float); null = np.array(null, float)
    return np.where(obs > 0, null / obs, np.nan)


def main():
    d = json.load(open("abf1_slide_calib.json"))
    K = d["nperm"]
    print("chrI: %d scored positions, %d null draws per statistic" % (d["n_scored"], K))
    print("FDR resolution floor: %.4f expected null positions\n" % (1.0 / K))

    for row in d["rows"]:
        w = row["width"]
        print("=" * 92); print("WIDTH %d bp" % w)

        print("\n  1. Is the null depth-dependent?")
        print("     %-14s %8s | %13s | %13s" % ("depth quintile", "n", "OBSERVED S", "NULL S"))
        for s in row["strata"]:
            print("     %-14s %8d | %6.2f %6.2f | %6.2f %6.2f" % (
                "%.0f-%.0f" % (s["depth_lo"], s["depth_hi"]), s["n"],
                s["obs_mean"], s["obs_sd"], s["null_mean"], s["null_sd"]))

        g = np.array(row["grid"]); fr = ratio(row["n_obs"], row["n_null"])
        print("\n  2. RAW S vs scrambled template")
        print("     %-6s %10s %11s %8s" % ("S >=", "observed", "null (exp)", "FDR"))
        for t in (4, 6, 8, 10, 12, 14):
            i = int(np.argmin(abs(g - t)))
            print("     %-6.1f %10d %11.1f %8.2f" % (
                g[i], row["n_obs"][i], row["n_null"][i], fr[i]))

        zg = np.array(row["zgrid"])
        for kind, title in (("perm", "3. z vs COLUMN-PERMUTED null (shape + smoothness)"),
                            ("phase", "4. z vs PHASE-RANDOMISED null (shape only)")):
            zz = row["z_by_null"][kind]; zr = ratio(zz["obs"], zz["null"])
            print("\n  %s" % title)
            print("     %-6s %10s %11s %8s" % ("z >=", "observed", "null (exp)", "FDR"))
            for t in (1, 2, 3, 4, 5):
                i = int(np.argmin(abs(zg - t)))
                v = zr[i]
                print("     %-6.1f %10d %11.1f %8s" % (
                    zg[i], zz["obs"][i], zz["null"][i],
                    "%.3f" % v if np.isfinite(v) else "-"))

        print("\n  5. The 5 MacIsaac sites          S = r x evidence")
        print("     %-12s %7s %7s %9s %7s %9s %9s %8s" % (
            "site", "S", "r", "evidence", "depth", "z(phase)", "S pct", "z pct"))
        for i, (c, v) in enumerate(row["sites"].items()):
            print("     #%d %-9s %7.2f %7.3f %9.2f %7.0f %9.2f %8.3f%% %7.3f%%" % (
                i + 1, c, v["S"], v["r"], v["evidence"], v["depth"], v["z"],
                v["pct"], v["z_pct"]))
        print()

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    a = ax[0, 0]
    for row in d["rows"]:
        w = row["width"]; xs = np.arange(5)
        lab = ["%.0f-%.0f" % (s["depth_lo"], s["depth_hi"]) for s in row["strata"]]
        a.plot(xs, [s["obs_sd"] for s in row["strata"]], "o-", color=C[w],
               label="%d bp  observed" % w)
        a.plot(xs, [s["null_sd"] for s in row["strata"]], "s--", color=C[w], alpha=.55,
               label="%d bp  null" % w)
    a.set_xticks(xs); a.set_xticklabels(lab, fontsize=8)
    a.axhline(1.0, color=MUTED, ls=":", lw=1)
    a.annotate("sd=1, what the z-scale assumes", (0.02, 1.0), xycoords=("axes fraction", "data"),
               fontsize=7, color=MUTED, va="bottom")
    a.set_xlabel("read depth quintile (mean A-trials per column)")
    a.set_ylabel("sd of S")
    a.set_title("A  the NULL is flat in depth -> S is comparable across coverage\n"
                "(but its sd is 1.2-1.5, not 1, so S is not a z-score)")
    a.grid(alpha=.3, color=GRID); a.legend(fontsize=7, ncol=2)

    a = ax[0, 1]
    for row in d["rows"]:
        w = row["width"]
        z = np.load("abf1_slide_calib_half%d.npz" % row["half"])
        m = np.random.default_rng(0).choice(len(z["r"]), 40000, replace=False)
        a.scatter(z["r"][m], z["evidence"][m], s=1, alpha=.05, color=C[w], rasterized=True)
        for i, (c, v) in enumerate(row["sites"].items()):
            a.plot([v["r"]], [v["evidence"]], "*", ms=13, color=C[w], mec="k", mew=.5)
            a.annotate("#%d" % (i + 1), (v["r"], v["evidence"]), fontsize=7,
                       xytext=(5, 2), textcoords="offset points")
    for sval, st in ((6, ":"), (10, "--"), (14, "-")):
        rr = np.linspace(.05, 1, 100)
        a.plot(rr, sval / rr, st, color=MUTED, lw=1)
        a.annotate("S=%d" % sval, (0.97, sval / 0.97), fontsize=7, color=MUTED)
    a.set_xlim(-.2, 1); a.set_ylim(0, 60)
    a.set_xlabel("r  =  shape agreement with ABF1  (bounded, depth-free)")
    a.set_ylabel("evidence  =  ||y - ybar||  (grows like sqrt(depth))")
    a.set_title("B  S = r x evidence\nthe same S is reachable by a good shape or by deep reads")
    a.grid(alpha=.3, color=GRID)

    a = ax[1, 0]
    for row in d["rows"]:
        w = row["width"]
        g = np.array(row["grid"])
        a.plot(g, ratio(row["n_obs"], row["n_null"]), "-", color=C[w], lw=1.8,
               label="%d bp  raw S" % w)
        zg = np.array(row["zgrid"]); zz = row["z_by_null"]["phase"]
        a.plot(zg, ratio(zz["obs"], zz["null"]), "--", color=C[w], lw=1.8,
               label="%d bp  z (phase null)" % w)
    a.axhline(1.0, color=MUTED, ls=":", lw=1)
    a.annotate("FDR = 1: template no better than a scramble", (0.03, 1.02),
               xycoords=("axes fraction", "data"), fontsize=7, color=MUTED)
    a.set_yscale("log"); a.set_ylim(1e-2, 5); a.set_xlim(0, 16)
    a.set_xlabel("threshold (S on the solid curves, z on the dashed)")
    a.set_ylabel("empirical FDR  =  expected null / observed")
    a.set_title("C  raw S never beats chance at either width;\nz at 25 bp does")
    a.grid(alpha=.3, color=GRID); a.legend(fontsize=8)

    a = ax[1, 1]
    xs = np.arange(5); wd = 0.38
    for j, row in enumerate(d["rows"]):
        w = row["width"]
        a.bar(xs + j * wd - wd / 2, [v["z"] for v in row["sites"].values()], wd,
              color=C[w], label="%d bp" % w)
        for x, v in zip(xs, row["sites"].values()):
            a.annotate("%.2f" % v["z"], (x + j * wd - wd / 2, v["z"]), ha="center",
                       fontsize=7, xytext=(0, 2), textcoords="offset points")
    for row in d["rows"]:
        w = row["width"]; zz = row["z_by_null"]["phase"]
        zr = ratio(zz["obs"], zz["null"]); zg = np.array(row["zgrid"])
        ok = np.flatnonzero(np.isfinite(zr) & (zr <= 0.05))
        if ok.size:
            a.axhline(zg[ok[0]], color=C[w], ls="--", lw=1)
            a.annotate("%d bp: FDR 5%% at z>=%.2f" % (w, zg[ok[0]]),
                       (0.98, zg[ok[0]]), xycoords=("axes fraction", "data"),
                       ha="right", va="bottom", fontsize=7, color=C[w])
    a.set_xticks(xs)
    a.set_xticklabels(["#%d\n%s" % (i + 1, c)
                       for i, c in enumerate(d["rows"][0]["sites"].keys())])
    a.set_ylabel("z against the phase-randomised null")
    a.set_title("D  the 5 MacIsaac sites on the calibrated scale")
    a.grid(alpha=.3, axis="y", color=GRID); a.legend(fontsize=8)

    fig.suptitle("What the sliding-ABF1 score means, and where a cutoff can honestly be "
                 "drawn — chrI, %d scored bp, %d null draws" % (d["n_scored"], K),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("abf1_slide_calib.png", dpi=150)
    print("wrote abf1_slide_calib.png")


if __name__ == "__main__":
    main()
