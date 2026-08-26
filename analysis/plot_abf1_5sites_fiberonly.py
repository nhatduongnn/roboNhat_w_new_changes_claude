"""All 5 chrI MacIsaac ABF1 sites for ONE decode (default: Fiber-only / ABF1-only).

Same single-source-of-truth discipline as plot_abf1_locus.py: this does NOT
re-implement peak calling. It calls score_robocop.score(..., return_abf1_tracks=True)
once with all 5 site windows as regions, and draws exactly what the scorer returns --
the ABF1 posterior track, its call threshold, and EVERY call (above-threshold
footprint) in the window, with each call's center marked.

Two DIFFERENT measurements are reported per site, because they separate two failure modes:
  on-motif   = posterior inside the 14 bp MacIsaac span itself
  within60   = best posterior anywhere within +/-60 bp
on-motif high + call center off  => broad/merged footprint, a localisation problem.
on-motif ~0 + within60 high      => the model put the footprint somewhere ELSE entirely
(true of sites #2 and #5: the posterior ON the motif is 0.000, and the nearby peak that
makes within60 look good is a separate footprint 33-46 bp away).

Three annotation rows above each panel show where each source places the 14 bp ABF1 chain:
MacIsaac motif (grey), where the Murphy PWM actually matches (purple = the sequence layer's
choice), and where the Fiber layer wants the footprint (amber). The purple row sits +1 bp
from grey at all 5 sites -- MacIsaac and Murphy agree on location. The amber row does not:
it is -24/+32/-8/-2/-39 bp away, often on the opposite strand. Since emission is a PRODUCT
over layers at ONE shared alignment, that disagreement is what costs the combined decode.

    python plot_abf1_5sites_fiberonly.py
    python plot_abf1_5sites_fiberonly.py --outdir robocop_chrI_seq_maskon --key seq_abf1
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
import pickle
import numpy as np
import h5py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import binom
import robocop
import score_robocop as S

THRESH_CACHE = "abf1_thresholds.json"
LEGACY_CACHE = "site4_thresholds.json"

# repo plot palette (matches plot_abf1_locus.py) -- one data series, so no
# categorical palette is in play; green/red are STATUS only and always carry a word.
BLUE   = "#2a78d6"
INK    = "#0b0b0b"; MUTED = "#6b6a66"; GRID = "#e7e6e2"; BAND = "#cfcec9"
GREEN  = "#1a7a44"; GBOX = ("#eafaf1", "#1baf7a")
RED    = "#a2322b"; RBOX = ("#fdecec", "#eb6834")
PURPLE = "#6a3fb5"   # where the Murphy PWM actually matches (the sequence layer's choice)
AMBER  = "#c8860d"   # where the Fiber layer wants to put the footprint


# --------------------------------------------------------------------------
# Where does each LAYER independently want to place the 14 bp ABF1 chain?
# Reconstructed from the decode's own inputs (nucleotides + Fiber counts in
# tmpDir/info*.h5, PWM in pwm.p, footprint p-vector in inputs/), scored exactly
# the way robocop.py does: sequence = product of PWM/background over the motif;
# fiber = product of binom(k,n,p_state)/binom(k,n,p_bg), watson on reference A,
# crick on reference T.
# --------------------------------------------------------------------------
def _segment_index(outDir):
    idx = {}
    for f in glob.glob(os.path.join(outDir, "tmpDir", "info_*.h5")):
        with h5py.File(f, "r") as h:
            for k in h.keys():
                if k.startswith("segment_"):
                    idx[int(k.split("_")[1])] = f
    return idx


def placement_scan(outDir, chrm, lo, hi, half=45):
    """Return (murphy_best, fiber_best); each = dict(start,strand,lr) or None."""
    try:
        pwm = pickle.load(open(os.path.join(outDir, "pwm.p"), "rb"))
        A = pwm["Abf1_murphy"][:4, :]
        bgv = np.asarray(pwm["background"]).ravel()[:4]
        Ar = robocop.reverse_complement(pwm["Abf1_murphy"])[:4, :]
        lp = pickle.load(open("inputs/all_TFs_1000pealVal_params_pseudo.pkl", "rb"))
        bgp = pickle.load(open("inputs/bg_params.pkl", "rb"))
        PW = np.asarray(lp["p"]["Abf1_murphy"]["watson_signal"]["A"])
        PC = np.asarray(lp["p"]["Abf1_murphy"]["crick_signal"]["A"])
        bw = float(np.ravel(bgp["p"]["watson_signal"]["A"])[0])
        bc = float(np.ravel(bgp["p"]["crick_signal"]["A"])[0])
        co = pd.read_csv(os.path.join(outDir, "coords.tsv"), sep="\t")
        mid = (lo + hi) // 2
        cand = co[(co.chr == chrm) & (co.start <= mid - half - 20) & (co.end >= mid + half + 20)]
        if cand.empty:
            return None, None
        seg = int(cand.index[0])
        segf = _segment_index(outDir).get(seg)
        if segf is None:
            return None, None
        with h5py.File(segf, "r") as h:
            K = "segment_%d/" % seg
            nt = robocop.get_sparse_todense(h, K + "nucleotides").astype(int)
            kw = robocop.get_sparse_todense(h, K + "Fiber_count_meth_watson")
            nw = robocop.get_sparse_todense(h, K + "Fiber_count_A_watson")
            kc = robocop.get_sparse_todense(h, K + "Fiber_count_meth_crick")
            nc = robocop.get_sparse_todense(h, K + "Fiber_count_A_crick")
            base = int(h["segment_%d" % seg].attrs["start"]) - 1
    except Exception as e:                        # never let annotation break the plot
        print("  [placement_scan skipped: %s]" % e)
        return None, None

    L = A.shape[1]
    bm = bf = None
    for g in range(mid - half, mid + half):
        s = g - base
        if s < 0 or s + L >= len(nt):
            continue
        for rev in (False, True):
            M = Ar if rev else A
            sl, bad = 1.0, False
            for j in range(L):
                b = nt[s + j]
                if b > 3:
                    bad = True; break
                sl *= M[b, j] / bgv[b]
            if bad:
                continue
            # Reverse orientation needs BOTH the mirror and the Watson<->Crick cross,
            # mirroring robocop.py:648-649. p was fitted in the motif's own frame
            # (combine_motif_counts_binom crosses channels when pooling minus-strand
            # training sites), so 'watson_signal' = the motif's own strand. Mirroring
            # within-channel would score the reverse placement against the wrong strand.
            Pw = PC[::-1] if rev else PW
            Pc = PW[::-1] if rev else PC
            fl = 1.0
            for j in range(L):
                i = s + j
                if nt[i] == 0:
                    den = binom.pmf(kw[i], nw[i], bw)
                    if den > 0: fl *= binom.pmf(kw[i], nw[i], Pw[j]) / den
                if nt[i] == 3:
                    den = binom.pmf(kc[i], nc[i], bc)
                    if den > 0: fl *= binom.pmf(kc[i], nc[i], Pc[j]) / den
            rec = dict(start=g, end=g + L, strand="rev" if rev else "fwd", lr=sl, flr=fl)
            if bm is None or sl > bm["lr"]:  bm = dict(rec, lr=sl)
            if bf is None or fl > bf["flr"]: bf = dict(rec, flr=fl)
    return bm, bf

# the 5 MacIsaac ABF1 sites on chrI (match_PWM bed), in genomic order
SITES = [
    (1, "chrI", 45318,  45332,  "site #1"),
    (2, "chrI", 45498,  45512,  "site #2"),
    (3, "chrI", 61163,  61177,  "site #3   upstream of ERV46"),
    (4, "chrI", 62657,  62671,  "site #4   downstream of ERV46"),
    (5, "chrI", 108788, 108802, "site #5"),
]


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
    """Whole-chrI ABF1 max for this run -> drives the 0.30x call threshold."""
    if key in cache and cache[key].get("global_max"):
        return float(cache[key]["global_max"])
    print("  [cache miss] scoring %s whole-chrI once for the global ABF1 max..." % key,
          flush=True)
    res = S.score(outDir, return_abf1_tracks=True, label=key)
    gm = 0.0
    for reg in res["_per_region"]:
        d = reg.get("abf1_detail")
        if d:
            gm = max(gm, float(d["global_max"]))
    cache[key] = {"global_max": gm, "call_threshold": S.abf1_call_threshold(gm)}
    save_cache(cache)
    return gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="robocop_chrI_maskon")
    ap.add_argument("--key", default="fiber_abf1")
    ap.add_argument("--title", default="Fiber-seq only  /  ABF1-only mask")
    ap.add_argument("--window", type=int, default=170, help="display half-width (bp)")
    ap.add_argument("--compute-window", type=int, default=400,
                    help="half-width the scorer sees; must exceed any footprint half-width "
                         "so above-threshold runs are not clipped at the window edge")
    ap.add_argument("--tol", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or ("abf1_5sites_%s.png" % args.key)

    cache = load_cache()
    gmax = global_max(args.outdir, args.key, cache)

    # ONE scorer call, all 5 windows, with the run's real whole-chrI threshold
    regions = [(c, m0 - args.compute_window, m1 + args.compute_window)
               for _, c, m0, m1, _ in SITES]
    res = S.score(args.outdir, regions=regions, abf1_global_max=gmax,
                  return_abf1_tracks=True, label=args.key)
    details = [r.get("abf1_detail") for r in res["_per_region"]]

    n = len(SITES)
    fig, axes = plt.subplots(n, 1, figsize=(13.6, 3.0 * n), sharey=True)
    summary = []

    for ax, (num, chrm, m0, m1, name), det in zip(axes, SITES, details):
        mid = (m0 + m1) // 2
        ax.set_facecolor("white")
        ax.axvspan(m0, m1, color=BAND, alpha=0.55, zorder=0)
        ax.axvline(mid, color=INK, ls="--", lw=1.4, zorder=3)

        # --- three annotation rows above the data: what each source says ---
        murphy, fiber = placement_scan(args.outdir, chrm, m0, m1)
        Y_MAC, Y_PWM, Y_FIB, H = 1.30, 1.19, 1.08, 0.075
        ax.add_patch(plt.Rectangle((m0, Y_MAC), m1 - m0, H, color=MUTED,
                                   alpha=0.85, zorder=6, clip_on=False))
        ax.text(m1 + 3, Y_MAC + H / 2, "MacIsaac motif", va="center", fontsize=8,
                color=MUTED, fontweight="bold", zorder=7, clip_on=False)
        if murphy:
            ax.add_patch(plt.Rectangle((murphy["start"], Y_PWM), murphy["end"] - murphy["start"],
                                       H, color=PURPLE, alpha=0.85, zorder=6, clip_on=False))
            ax.text(murphy["end"] + 3, Y_PWM + H / 2,
                    "Murphy PWM match (%s, LR %.3g)  —  %+d bp vs MacIsaac"
                    % (murphy["strand"], murphy["lr"], murphy["start"] - m0),
                    va="center", fontsize=8, color=PURPLE, fontweight="bold",
                    zorder=7, clip_on=False)
        if fiber:
            ax.add_patch(plt.Rectangle((fiber["start"], Y_FIB), fiber["end"] - fiber["start"],
                                       H, color=AMBER, alpha=0.85, zorder=6, clip_on=False))
            gap = fiber["start"] - (murphy["start"] if murphy else m0)
            ax.text(fiber["end"] + 3, Y_FIB + H / 2,
                    "Fiber layer wants (%s, LR %.3g)  —  %+d bp vs PWM%s"
                    % (fiber["strand"], fiber["flr"], gap,
                       ", OPPOSITE strand" if murphy and fiber["strand"] != murphy["strand"] else ""),
                    va="center", fontsize=8, color=AMBER, fontweight="bold",
                    zorder=7, clip_on=False)

        if det is None:
            ax.text(0.5, 0.5, "no ABF1 column", transform=ax.transAxes,
                    ha="center", color=RED)
            continue

        pos = np.asarray(det["pos"]); tr = np.asarray(det["track"]); thr = det["threshold"]
        calls = det["calls"]

        ax.fill_between(pos, tr, color=BLUE, alpha=0.85, lw=0, zorder=2)
        ax.plot(pos, tr, color=BLUE, lw=1.2, zorder=3)
        ax.axhline(thr, color=RED, ls=":", lw=1.1, zorder=4)

        lo, hi = mid - args.window, mid + args.window
        vis = [c for c in calls if c["end"] >= lo and c["start"] <= hi]
        near = min(calls, key=lambda c: abs(c["center"] - mid)) if calls else None
        dist = abs(near["center"] - mid) if near else None
        ok = near is not None and dist <= args.tol

        # every call visible in the window; the matched one is emphasised
        for c in vis:
            matched = near is not None and c["center"] == near["center"]
            col = (GREEN if ok else RED) if matched else MUTED
            ax.axvspan(c["start"], c["end"], color=col,
                       alpha=0.20 if matched else 0.10, zorder=1)
            yv = float(tr[int(c["center"] - pos[0])]) if pos[0] <= c["center"] <= pos[-1] else 1.0
            ax.plot([c["center"]], [min(max(yv, 0.05), 1.0)], marker="v",
                    ms=12 if matched else 8, color=col,
                    markeredgecolor="white", markeredgewidth=1.2,
                    zorder=6 if matched else 5)
            if matched:
                ax.axvline(c["center"], color=col, lw=1.8, zorder=5)

        # two DIFFERENT measurements -- they tell apart the two failure modes:
        #   on_motif  : posterior inside the 14 bp MacIsaac span itself
        #   near_motif: best posterior anywhere within +/-60 bp
        # on_motif high + center off  => broad/merged footprint, localisation issue
        # on_motif ~0 + near_motif high => the model puts the footprint somewhere ELSE
        on_motif = float(tr[(pos >= m0) & (pos <= m1)].max())
        near_motif = float(tr[(pos >= mid - 60) & (pos <= mid + 60)].max())
        if near is not None:
            if ok:
                verdict = "RECOVERED (≤%d bp)" % args.tol
            elif on_motif >= thr:
                verdict = "MISSED (>%d bp) — signal on motif, center off" % args.tol
            else:
                verdict = "MISSED (>%d bp) — footprint sits OFF the motif" % args.tol
            txt = ("posterior on motif span  %.3f\nbest within ±60 bp     %.3f\n"
                   "call center %d  →  %d bp from midpoint\n%s"
                   % (on_motif, near_motif, near["center"], dist, verdict))
        else:
            txt = ("posterior on motif span  %.3f\nbest within ±60 bp     %.3f\n"
                   "no above-threshold footprint\nMISSED" % (on_motif, near_motif))
        box, tc = (GBOX, GREEN) if ok else (RBOX, RED)
        # place the box on whichever half of the panel carries less signal
        m_vis = (pos >= lo) & (pos <= hi)
        left_load = float(tr[m_vis & (pos < mid)].sum())
        right_load = float(tr[m_vis & (pos >= mid)].sum())
        xa, ha = (0.995, "right") if right_load <= left_load else (0.005, "left")
        ax.text(xa, 0.74, txt, transform=ax.transAxes, ha=ha, va="top",
                fontsize=9.0, color=tc, fontweight="bold", family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.38", fc=box[0], ec=box[1], lw=1))

        ax.set_title("%s      %s:%d–%d   midpoint %d" % (name, chrm, m0, m1, mid),
                     fontsize=11.5, fontweight="bold", loc="left", pad=5)
        ax.set_xlim(lo, hi); ax.set_ylim(0, 1.40)
        ax.set_ylabel("ABF1\nposterior", fontsize=9.5)
        ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):  ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#b8b7b2")
        ax.tick_params(length=0, labelsize=9)
        summary.append((num, on_motif, near_motif,
                        None if near is None else near["center"], dist, ok, len(vis)))

    axes[-1].set_xlabel("chrI position (bp)", fontsize=10.5)

    leg = [
        Patch(fc=BAND, alpha=0.55, label="MacIsaac motif span (14 bp)"),
        Patch(fc=PURPLE, alpha=0.85, label="where the Murphy PWM actually matches (sequence layer's choice)"),
        Patch(fc=AMBER, alpha=0.85, label="where the Fiber layer wants the footprint"),
        Line2D([0], [0], color=INK, ls="--", lw=1.4, label="motif midpoint = reference anchor"),
        Line2D([0], [0], color=RED, ls=":", lw=1.2, label="scorer call threshold (0.30 × chrI max)"),
        Line2D([0], [0], color=GREEN, lw=0, marker="v", ms=9, markeredgecolor="white",
               label="matched call center — RECOVERED"),
        Line2D([0], [0], color=RED, lw=0, marker="v", ms=9, markeredgecolor="white",
               label="matched call center — MISSED"),
        Line2D([0], [0], color=MUTED, lw=0, marker="v", ms=7, markeredgecolor="white",
               label="other calls in window"),
    ]
    fig.legend(handles=leg, frameon=False, fontsize=9.2, loc="lower center",
               bbox_to_anchor=(0.5, -0.012), ncol=3)

    fig.suptitle("All 5 chrI MacIsaac ABF1 sites — %s  (%s)" % (args.title, args.outdir),
                 fontsize=14, fontweight="bold", y=0.996)
    fig.text(0.5, 0.9735,
             "Drawn from score_robocop.score(return_abf1_tracks=True). Arrows are the scorer's "
             "own calls (center of each above-threshold footprint).",
             ha="center", fontsize=9.2, color=MUTED)

    plt.tight_layout(rect=[0, 0.045, 1, 0.9655])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print("\n%-6s %10s %11s %12s %8s %8s %7s"
          % ("site", "on-motif", "within60", "call center", "dist", "status", "#calls"))
    for num, onm, nearm, cc, d, ok, ncalls in summary:
        print("%-6s %10.4f %11.4f %12s %8s %8s %7d"
              % ("#%d" % num, onm, nearm, cc if cc is not None else "-",
                 d if d is not None else "-", "HIT" if ok else "miss", ncalls))


if __name__ == "__main__":
    main()
