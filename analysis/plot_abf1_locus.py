"""Reusable per-locus ABF1 panel across any set of models.

The plot does NOT re-implement peak calling. It INVOKES the scorer -- score_robocop.score()
with return_abf1_tracks=True -- and draws exactly what the scorer returns: the ABF1 track,
the call threshold, and the ABF1 calls (footprint centers). The arrows point at score()'s
calls. So if you change the peak-calling inside score_robocop (footprint center -> argmax,
different threshold, etc.), this plot changes with it automatically -- single source of truth.

A locus is RECOVERED if one of score()'s calls lands within `tol` bp of the motif midpoint.

Reusable:
    python plot_abf1_locus.py                                   # MacIsaac site #4, 4 chrI runs
    python plot_abf1_locus.py --motif chrI:45318-45332          # any locus
    python plot_abf1_locus.py --motif chrI:61163-61177 --window 180 --out site3.png

The scorer's ABF1 threshold is 0.30 x whole-chrI max. To score a small window while keeping
that global threshold, we pass score(abf1_global_max=...). The per-run global max is cached in
abf1_thresholds.json; a run missing from the cache is scored once whole-chrI to fill it, so
later loci render instantly.
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import score_robocop as S

THRESH_CACHE = "abf1_thresholds.json"
LEGACY_CACHE = "site4_thresholds.json"   # same schema, older name

BLUE = "#2a78d6"; ORANGE = "#eb6834"
INK = "#0b0b0b"; MUTED = "#6b6a66"; GRID = "#e7e6e2"; BAND = "#cfcec9"
GREEN = "#1a7a44"; GBOX = ("#eafaf1", "#1baf7a"); RED = "#a2322b"; RBOX = ("#fdECEC", "#eb6834")

# default model set: the 4 full-chrI runs. (outDir, key, title, color)
DEFAULT_MODELS = [
    ("robocop_chrI_maskon",      "fiber_abf1", "Fiber / Abf1-only",     BLUE),
    ("robocop_chrI_maskoff",     "fiber_all",  "Fiber / all-TFs",       BLUE),
    ("robocop_chrI_seq_maskon",  "seq_abf1",   "Fiber+seq / Abf1-only", ORANGE),
    ("robocop_chrI_seq_maskoff", "seq_all",    "Fiber+seq / all-TFs",   ORANGE),
]


# ---------------------------------------------------------------------------
def load_cache():
    for p in (THRESH_CACHE, LEGACY_CACHE):
        if os.path.isfile(p):
            with open(p) as fh:
                return json.load(fh)
    return {}


def save_cache(cache):
    with open(THRESH_CACHE, "w") as fh:
        json.dump(cache, fh, indent=2)


def global_max(outDir, key, cache):
    """The scorer's whole-chrI ABF1 max for this run (drives the call threshold).
    Cached; on a miss, run the scorer whole-chrI once to read its own global_max."""
    if key in cache and cache[key].get("global_max"):
        return float(cache[key]["global_max"])
    print("  [cache miss] scoring %s whole-chrI once for global ABF1 max..." % key, flush=True)
    res = S.score(outDir, return_abf1_tracks=True, label=key)
    gm = 0.0
    for reg in res["_per_region"]:
        d = reg.get("abf1_detail")
        if d:
            gm = max(gm, float(d["global_max"]))
    cache[key] = {"global_max": gm, "call_threshold": S.abf1_call_threshold(gm)}
    save_cache(cache)
    return gm


def score_locus(outDir, key, chrm, comp_lo, comp_hi, gmax):
    """Invoke the scorer on the compute window with the run's global threshold, and
    return the ABF1 detail it produced (threshold, calls, track, pos)."""
    res = S.score(outDir, regions=[(chrm, comp_lo, comp_hi)],
                  abf1_global_max=gmax, return_abf1_tracks=True, label=key)
    for reg in res["_per_region"]:
        d = reg.get("abf1_detail")
        if d:
            return d
    return None


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif", default="chrI:62657-62671",
                    help="motif locus chrN:start-end (midpoint = (start+end)//2)")
    ap.add_argument("--window", type=int, default=135, help="display half-width (bp) around midpoint")
    ap.add_argument("--compute-window", type=int, default=400,
                    help="half-width (bp) the scorer sees; must exceed any footprint "
                         "half-width so above-threshold runs aren't clipped at the edge")
    ap.add_argument("--tol", type=int, default=20, help="recovery tolerance (bp)")
    ap.add_argument("--out", default=None, help="output png (default derived from locus)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    chrm, span = args.motif.split(":")
    m_start, m_end = (int(x) for x in span.split("-"))
    mid = (m_start + m_end) // 2
    disp_lo, disp_hi = mid - args.window, mid + args.window
    comp_lo, comp_hi = mid - args.compute_window, mid + args.compute_window
    out = args.out or ("abf1_locus_%s_%d_%d.png" % (chrm, m_start, m_end))

    cache = load_cache()
    models = DEFAULT_MODELS
    n = len(models)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 4.5 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    for ax, (outDir, key, title, color) in zip(axes, models):
        gmax = global_max(outDir, key, cache)
        det = score_locus(outDir, key, chrm, comp_lo, comp_hi, gmax)

        ax.set_facecolor("white")
        ax.axvspan(m_start, m_end, color=BAND, alpha=0.55, zorder=0)
        for xv in (m_start, m_end):
            ax.axvline(xv, color="#a8a7a2", ls=":", lw=1.0, zorder=1)
        ax.axvline(mid, color=INK, ls="--", lw=1.5, zorder=2)

        if det is None:
            ax.text(0.5, 0.5, "no ABF1 column", transform=ax.transAxes, ha="center", color=RED)
        else:
            pos = np.asarray(det["pos"]); tr = np.asarray(det["track"]); thr = det["threshold"]
            calls = det["calls"]  # score()'s ABF1 calls: dicts(center,start,end)

            ax.fill_between(pos, tr, color=color, alpha=0.85, lw=0, zorder=3)
            ax.plot(pos, tr, color=color, lw=1.3, zorder=4)
            ax.axhline(thr, color=RED, ls=":", lw=1.1, zorder=5)
            ax.text(0.015, thr + 0.012, "call threshold %.2f" % thr,
                    transform=ax.get_yaxis_transform(), fontsize=7.8, color=RED,
                    ha="left", va="bottom")

            # the call the scorer would match to this locus = nearest call center to midpoint
            near = min(calls, key=lambda c: abs(c["center"] - mid)) if calls else None
            dist = abs(near["center"] - mid) if near else None
            recovered = near is not None and dist <= args.tol

            if near is not None:
                c, a, b = near["center"], near["start"], near["end"]
                ax.axvspan(a, b, color=color, alpha=0.16, zorder=1)
                yv = float(tr[int(c - pos[0])]) if pos[0] <= c <= pos[-1] else 1.0
                acol = color if recovered else RED
                ax.axvline(c, color=acol, lw=1.9, zorder=6)
                ax.plot([c], [max(yv, 0.03)], marker="v", ms=11, color=acol,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=7)
                status = "RECOVERED" if recovered else "MISSED (nearest call > %d bp)" % args.tol
                box, txtcol = (GBOX, GREEN) if recovered else (RBOX, RED)
                ax.text(0.985, 0.90,
                        "scorer call = %d\n|%d − %d| = %d bp\n→ %s\n"
                        "(call = center of above-threshold\nfootprint [%d–%d])"
                        % (c, c, mid, dist, status, a, b),
                        transform=ax.transAxes, ha="right", va="top", fontsize=9.2,
                        color=txtcol, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.4", fc=box[0], ec=box[1], lw=1))
            else:
                wi = int(np.argmax(tr))
                ax.text(0.985, 0.90,
                        "local ABF1 max = %.3f at %d\n(%d bp from midpoint)\n"
                        "below call threshold %.2f\n→ no footprint → MISSED"
                        % (float(tr[wi]), int(pos[wi]), abs(int(pos[wi]) - mid), thr),
                        transform=ax.transAxes, ha="right", va="top", fontsize=9.0,
                        color=RED, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.4", fc=RBOX[0], ec=RBOX[1], lw=1))

        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=6)
        ax.set_ylim(0, 1.06); ax.set_xlim(disp_lo, disp_hi)
        ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#b8b7b2")
        ax.tick_params(length=0)

    for ax in axes[n:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:n]):
        if i % ncol == 0:
            ax.set_ylabel("ABF1 posterior", fontsize=10.5)
        if i >= n - ncol:
            ax.set_xlabel("%s position (bp)" % chrm, fontsize=10.5)

    leg = [
        Patch(fc=BAND, alpha=0.55, label="MacIsaac motif span (%d–%d)" % (m_start, m_end)),
        Line2D([0], [0], color=INK, ls="--", lw=1.5,
               label="motif midpoint %d  (reference anchor)" % mid),
        Line2D([0], [0], color="#555", lw=1.9, marker="v", ms=8, markeredgecolor="white",
               label="scorer call = footprint center (from score_robocop.score)"),
        Line2D([0], [0], color=RED, ls=":", lw=1.2,
               label="scorer call threshold (0.30 × chrI-max)"),
    ]
    fig.legend(handles=leg, frameon=False, fontsize=9.2, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=2)

    ttl = args.title or ("ABF1 locus %s:%d–%d  (midpoint %d)  —  %d models"
                         % (chrm, m_start, m_end, mid, n))
    fig.suptitle(ttl, fontsize=14.5, fontweight="bold", y=0.995)
    fig.text(0.5, 0.955,
             "Panels are drawn from score_robocop.score() output (return_abf1_tracks=True). "
             "Arrows = the scorer's ABF1 calls; change the scorer's peak-caller and these move.",
             ha="center", fontsize=9.2, color=MUTED)

    plt.tight_layout(rect=[0, 0.05, 1, 0.945])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
