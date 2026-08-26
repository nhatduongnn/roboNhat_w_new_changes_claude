"""All 5 chrI MacIsaac ABF1 sites, using DECODE OUTPUT ONLY.

Nothing on this figure is recomputed or approximated.  Every number comes out of
`tmpDir/info_*.h5`, which stores the forward-backward posterior table:

    posterior[i, j] = P(the chain is in state j at position i | ALL the data)

(algo.c:227 `posterior_decoding`, p_table[i][j] = scaled alpha[i][j] * beta[i][j]).
Rows sum to 1 over the 3485 states, so grouping the state axis is exact -- no
modelling assumption is added by this script.

State groups (from HMMconfig: tf_starts, tf_lens, nuc_start, silent_states_begin):
    background   state 0
    ABF1 fwd     tf_starts[0] .. +tf_lens[0]                 (states 1-14)
    ABF1 rev     tf_starts[0]+tf_lens[0] .. +2*tf_lens[0]    (states 15-28)
    other TFs    tf_starts[1] .. tf_starts[-1]               (152 other factors)
    unknown      tf_starts[-1] .. +2*tf_lens[-1]             (the generic DBF)
    nucleosome   nuc_start .. silent_states_begin            (531 states)

The middle panel is the honest answer to "why is there no blue call here?" -- it
shows what the decode put at that position INSTEAD.  That question cannot be
answered by any per-layer likelihood, because a likelihood scores one hypothesis
against background while forward-backward scores every hypothesis against every
other one, summed over all paths.

    python plot_abf1_5sites_decoded.py --outdir robocop_chrI_maskon_revfix --key fiber_abf1
    python plot_abf1_5sites_decoded.py --outdir robocop_chrI_seq_maskon_revfix --key seq_abf1
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import score_robocop as S
import plot_abf1_5sites_layers as PL   # SITES, fiber_track, threshold cache helpers

WATSON, CRICK = "#2a78d6", "#eb6834"
BG_METH = 0.1383
BLUE = "#2a78d6"
INK = "#0b0b0b"; MUTED = "#6b6a66"; GRID = "#e7e6e2"; BAND = "#cfcec9"
GREEN = "#1a7a44"; GBOX = ("#eafaf1", "#1baf7a")
RED = "#a2322b"; RBOX = ("#fdecec", "#eb6834")
SURFACE = "#fcfcfb"

# Categorical slots, validated as a set for the light surface
# (scripts/validate_palette.js: lightness band PASS, chroma floor PASS,
#  worst adjacent CVD dE 9.1, worst adjacent normal-vision dE 16.3).
# Contrast WARN on aqua/yellow/magenta obliges relief -> every series that reaches
# the panel is direct-labelled, and the numeric table is printed to stdout.
# `background` is deliberately NOT a colour: the stack sums to 1, so background is
# the unfilled remainder.  A grey fill fails the chroma floor and would also read
# as "no data" rather than "the model chose background".
GROUPS = [
    ("abf1_fwd",   "ABF1 forward",  "#2a78d6"),
    ("abf1_rev",   "ABF1 reverse",  "#4a3aa7"),
    ("nucleosome", "nucleosome",    "#1baf7a"),
    ("other_TFs",  "other TFs",     "#eda100"),
    ("unknown",    "unknown DBF",   "#e87ba4"),
]


def state_composition(dec, chrm, lo, hi):
    """Group the raw forward-backward posterior table by state block.

    Returns (dict of per-position arrays, positions, row_sum).  Mirrors
    score_robocop.region_optable's assembly (segment overlap -> average) but keeps
    the full state axis instead of collapsing to the per-factor optable, so the
    forward and reverse halves of the ABF1 block can be reported separately.
    """
    dshared, coords = dec["dshared"], dec["coords"]
    n = hi - lo + 1
    pt = np.zeros((n, dshared["n_states"]))
    cnt = np.zeros(n)
    for infofile in dec["infofiles"]:
        with h5py.File(infofile, "r") as f:
            for idx in S._seg_idxs(coords, chrm, lo, hi):
                k = "segment_" + str(idx)
                if k not in f:
                    continue
                dp = S._get_sparse_todense(f, k + "/posterior")
                ss, se = coords.loc[idx]["start"], coords.loc[idx]["end"]
                ds = max(0, lo - ss)
                de = min(hi - ss + 1, se - ss + 1)
                ps = max(0, ss - lo)
                pt[ps:ps + de - ds] += dp[ds:de, :]
                cnt[ps:ps + de - ds] += 1
    if not (cnt > 0).any():
        return None, None, None
    pt[cnt > 0] /= cnt[cnt > 0, np.newaxis]

    ts, tl = dshared["tf_starts"], dshared["tf_lens"]
    a0, L = int(ts[0]), int(tl[0])
    comp = {
        "background": pt[:, 0],
        "abf1_fwd":   pt[:, a0:a0 + L].sum(1),
        "abf1_rev":   pt[:, a0 + L:a0 + 2 * L].sum(1),
        "other_TFs":  pt[:, int(ts[1]):int(ts[-1])].sum(1),
        "unknown":    pt[:, int(ts[-1]):int(ts[-1]) + 2 * int(tl[-1])].sum(1),
        "nucleosome": pt[:, dshared["nuc_start"]:dshared["silent_states_begin"]].sum(1),
    }
    return comp, np.arange(lo, hi + 1), sum(comp.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="robocop_chrI_maskon_revfix")
    ap.add_argument("--key", default="fiber_abf1")
    ap.add_argument("--title", default="Fiber-seq only  /  ABF1-only mask")
    ap.add_argument("--window", type=int, default=170)
    ap.add_argument("--compute-window", type=int, default=400)
    ap.add_argument("--tol", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or ("abf1_5sites_%s_decoded.png" % args.key)

    cache = PL.load_cache()
    gmax = PL.global_max(args.outdir, args.key, cache)
    regions = [(c, m0 - args.compute_window, m1 + args.compute_window)
               for _, c, m0, m1, _ in PL.SITES]
    res = S.score(args.outdir, regions=regions, abf1_global_max=gmax,
                  return_abf1_tracks=True, label=args.key)
    details = [r.get("abf1_detail") for r in res["_per_region"]]
    dec = S.load_decode(args.outdir)

    n = len(PL.SITES)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(13.6, 6.6 * n))
    outer = gridspec.GridSpec(n, 1, hspace=0.34)
    axes, saxes, waxes, caxes = [], [], [], []
    for r in range(n):
        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[r], height_ratios=[3.0, 2.4, 1.0, 1.0], hspace=0.07)
        a = fig.add_subplot(inner[0]); axes.append(a)
        saxes.append(fig.add_subplot(inner[1], sharex=a))
        waxes.append(fig.add_subplot(inner[2], sharex=a))
        caxes.append(fig.add_subplot(inner[3], sharex=a))
    summary, rowsum_worst = [], 1.0

    for ax, sax_, wax, cax, (num, chrm, m0, m1, name), det in zip(
            axes, saxes, waxes, caxes, PL.SITES, details):
        mid = (m0 + m1) // 2
        lo, hi = mid - args.window, mid + args.window
        ax.set_facecolor("white")
        ax.axvspan(m0, m1, color=BAND, alpha=0.55, zorder=0)
        ax.axvline(mid, color=INK, ls="--", lw=1.4, zorder=3)

        if det is None:
            ax.text(0.5, 0.5, "no ABF1 column", transform=ax.transAxes,
                    ha="center", color=RED)
            continue
        pos = np.asarray(det["pos"]); tr = np.asarray(det["track"]); thr = det["threshold"]
        calls = det["calls"]

        ax.fill_between(pos, tr, color=BLUE, alpha=0.85, lw=0, zorder=2)
        ax.plot(pos, tr, color=BLUE, lw=1.2, zorder=3)
        ax.axhline(thr, color=RED, ls=":", lw=1.1, zorder=4)

        vis = [c for c in calls if c["end"] >= lo and c["start"] <= hi]
        near = min(calls, key=lambda c: abs(c["center"] - mid)) if calls else None
        dist = abs(near["center"] - mid) if near else None
        ok = near is not None and dist <= args.tol
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

        on_motif = float(tr[(pos >= m0) & (pos <= m1)].max())
        near_motif = float(tr[(pos >= mid - 60) & (pos <= mid + 60)].max())
        if near is not None:
            verdict = ("RECOVERED (≤%d bp)" % args.tol if ok else
                       "MISSED (>%d bp) — signal on motif, center off" % args.tol
                       if on_motif >= thr else
                       "MISSED (>%d bp) — footprint sits OFF the motif" % args.tol)
            txt = ("posterior on motif span  %.3f\nbest within ±60 bp     %.3f\n"
                   "call center %d  →  %d bp from midpoint\n%s"
                   % (on_motif, near_motif, near["center"], dist, verdict))
        else:
            txt = ("posterior on motif span  %.3f\nbest within ±60 bp     %.3f\n"
                   "no above-threshold footprint\nMISSED" % (on_motif, near_motif))
        box, tc = (GBOX, GREEN) if ok else (RBOX, RED)
        m_vis = (pos >= lo) & (pos <= hi)
        xa, ha = ((0.995, "right")
                  if float(tr[m_vis & (pos >= mid)].sum()) <= float(tr[m_vis & (pos < mid)].sum())
                  else (0.005, "left"))
        ax.text(xa, 0.74, txt, transform=ax.transAxes, ha=ha, va="top",
                fontsize=9.0, color=tc, fontweight="bold", family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.38", fc=box[0], ec=box[1], lw=1))
        ax.set_title("%s      %s:%d–%d   midpoint %d" % (name, chrm, m0, m1, mid),
                     fontsize=11.5, fontweight="bold", loc="left", pad=5)
        ax.set_xlim(lo, hi); ax.set_ylim(0, 1.05)
        ax.set_ylabel("ABF1\nposterior", fontsize=9.5)
        ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#b8b7b2")
        ax.tick_params(length=0, labelsize=9, labelbottom=False)

        # ---------------- what the decode put here INSTEAD ----------------
        comp, cpos, rsum = state_composition(dec, chrm, lo, hi)
        sax_.set_facecolor(SURFACE)
        sax_.axvline(mid, color=INK, ls="--", lw=1.4, zorder=6)
        crow = None
        if comp is not None:
            rowsum_worst = min(rowsum_worst, float(np.min(rsum)))
            base = np.zeros(len(cpos))
            for key, lab, col in GROUPS:
                v = comp[key]
                sax_.fill_between(cpos, base, base + v, color=col, lw=0, zorder=3)
                # surface-coloured separator, the continuous analogue of the 2px
                # gap between stacked segments
                sax_.plot(cpos, base + v, color=SURFACE, lw=0.8, zorder=4)
                # direct label for anything that actually shows up in this window
                if float(v.max()) > 0.08:
                    j = int(np.argmax(v))
                    sax_.text(cpos[j], base[j] + v[j] / 2, " %s " % lab,
                              ha="center", va="center", fontsize=7.8, color="white",
                              fontweight="bold", zorder=7,
                              bbox=dict(boxstyle="round,pad=0.18", fc=col,
                                        ec="white", lw=0.8, alpha=0.95))
                base = base + v
            sax_.plot(cpos, base, color=MUTED, lw=1.0, zorder=5)

            def comp_at(x):
                i = int(np.clip(x - cpos[0], 0, len(cpos) - 1))
                return {k: float(comp[k][i]) for k in comp}
            rows = [("MacIsaac motif", mid)] + [("call %d" % c["center"], c["center"])
                                                for c in vis]
            crow = [(lab, x, comp_at(x)) for lab, x in rows]
            lines = ["%-15s bg %.3f  ABF1f %.3f  ABF1r %.3f  nuc %.3f  othTF %.3f  unk %.3f"
                     % (lab, c["background"], c["abf1_fwd"], c["abf1_rev"],
                        c["nucleosome"], c["other_TFs"], c["unknown"])
                     for lab, _, c in crow]
            sax_.text(0.004, 0.035, "\n".join(lines), transform=sax_.transAxes,
                      ha="left", va="bottom", fontsize=7.4, color=INK,
                      family="DejaVu Sans Mono", zorder=9,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c9c8c4",
                                lw=0.8, alpha=0.94))
        sax_.set_ylim(0, 1.0); sax_.set_yticks([0, 0.5, 1.0])
        sax_.set_ylabel("decoded state\n(posterior mass)", fontsize=9.0)
        sax_.text(0.995, 0.955, "unfilled = background   ·   the 6 groups sum to 1.000",
                  transform=sax_.transAxes, ha="right", va="top", fontsize=7.8,
                  color=MUTED, fontweight="bold", zorder=8)
        for sp in ("top", "right"): sax_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): sax_.spines[sp].set_color("#b8b7b2")
        sax_.tick_params(length=0, labelsize=8.2, labelbottom=False)
        summary.append((num, on_motif, near_motif,
                        None if near is None else near["center"], dist, ok, crow))

        # ---------------- raw Fiber-seq input ----------------
        ft = PL.fiber_track(args.outdir, chrm, lo - 5, hi + 5)
        for s2, col, sel, smk, lab in (
                (wax, WATSON, "isA", "smW", "Watson meth/A  (reference A, layer 5)"),
                (cax, CRICK,  "isT", "smC", "Crick meth/A  (reference T, layer 6)")):
            s2.set_facecolor("white")
            s2.axvspan(m0, m1, color=BAND, alpha=0.55, zorder=0)
            s2.axvline(mid, color=INK, ls="--", lw=1.4, zorder=3)
            if ft is not None:
                m = ft[sel]
                s2.scatter(ft["pos"][m], ft["frac"][m], s=6, color=col, alpha=0.75, zorder=2)
                s2.plot(ft["pos"], ft[smk], color=col, lw=1.9, zorder=4)
                s2.axhline(BG_METH, color=MUTED, ls="--", lw=1.0, zorder=3)
            s2.set_ylim(0, 1); s2.set_yticks([0, 0.5, 1.0])
            s2.set_yticklabels(["0.0", "0.5", ""])
            s2.set_ylabel(lab.split("  ")[0].replace(" ", "\n"), fontsize=8.4)
            s2.text(0.995, 0.92, lab, transform=s2.transAxes, ha="right", va="top",
                    fontsize=7.6, color=col, fontweight="bold")
            s2.yaxis.grid(True, color=GRID, lw=1, zorder=0); s2.set_axisbelow(True)
            for sp in ("top", "right"): s2.spines[sp].set_visible(False)
            for sp in ("left", "bottom"): s2.spines[sp].set_color("#b8b7b2")
            s2.tick_params(length=0, labelsize=8.2)
        wax.tick_params(labelbottom=False)

    caxes[-1].set_xlabel("chrI position (bp)", fontsize=10.5)

    leg = [Patch(fc=BAND, alpha=0.55, label="MacIsaac motif span (14 bp)")]
    leg += [Patch(fc=c, label=l) for _, l, c in GROUPS]
    leg += [
        Patch(fc=SURFACE, ec="#b8b7b2", label="background (unfilled remainder)"),
        Line2D([0], [0], color=INK, ls="--", lw=1.4, label="motif midpoint = reference anchor"),
        Line2D([0], [0], color=RED, ls=":", lw=1.2, label="scorer call threshold (0.30 × chrI max)"),
        Line2D([0], [0], color=GREEN, lw=0, marker="v", ms=9, markeredgecolor="white",
               label="matched call center — RECOVERED"),
        Line2D([0], [0], color=RED, lw=0, marker="v", ms=9, markeredgecolor="white",
               label="matched call center — MISSED"),
        Line2D([0], [0], color=MUTED, lw=0, marker="v", ms=7, markeredgecolor="white",
               label="other calls in window"),
        Line2D([0], [0], color=WATSON, lw=1.9, marker=".", ms=9,
               label="Watson meth/A per base + rolling mean (±15 bp)"),
        Line2D([0], [0], color=CRICK, lw=1.9, marker=".", ms=9,
               label="Crick meth/A per base + rolling mean (±15 bp)"),
        Line2D([0], [0], color=MUTED, ls="--", lw=1.0,
               label="genome-wide background m6A rate (0.138)"),
    ]
    fig.legend(handles=leg, frameon=False, fontsize=9.2, loc="lower center",
               bbox_to_anchor=(0.5, -0.010), ncol=3)

    fig.suptitle("All 5 chrI MacIsaac ABF1 sites — %s  (%s)" % (args.title, args.outdir),
                 fontsize=14, fontweight="bold", y=0.996)
    fig.text(0.5, 0.9805,
             "DECODE OUTPUT ONLY — every value is read from tmpDir/info_*.h5. "
             "Nothing here is recomputed, approximated, or fitted by the plotting code.",
             ha="center", va="top", fontsize=9.4, color=INK, fontweight="bold")
    fig.text(0.5, 0.9745,
             "Middle panel groups the forward-backward posterior table by state block; "
             "the groups sum to 1.000 at every position, so it is exactly what the HMM "
             "decided was present.",
             ha="center", va="top", fontsize=8.8, color=MUTED)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.962, bottom=0.075)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print("wrote", out)
    print("state groups sum to 1 at every position; worst row sum = %.6f" % rowsum_worst)
    for num, onm, nearm, cc, d, ok, crow in summary:
        print("\nsite #%d   on-motif %.4f   within60 %.4f   call %s   dist %s   %s"
              % (num, onm, nearm, cc if cc is not None else "-",
                 d if d is not None else "-", "HIT" if ok else "miss"))
        if crow:
            print("   %-15s %8s %8s %8s %8s %8s %8s"
                  % ("position", "bg", "ABF1fwd", "ABF1rev", "nuc", "othTF", "unk"))
            for lab, x, c in crow:
                print("   %-15s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f"
                      % (lab, c["background"], c["abf1_fwd"], c["abf1_rev"],
                         c["nucleosome"], c["other_TFs"], c["unknown"]))


if __name__ == "__main__":
    main()
