"""All 5 chrI MacIsaac ABF1 sites, with a per-LAYER likelihood-ratio profile.

Successor to plot_abf1_5sites_withfiber.py. Two things change:

1. The placement scan covers the ENTIRE plotted window, not +/-45 bp around the
   MacIsaac motif. The old half=45 window meant a call more than ~45 bp from the
   motif (site #5's call is 72 bp away) got no purple/amber annotation at all,
   which read as "no layer wants anything there" when in fact nothing had been
   computed there.

2. A new middle panel plots log10 LR at EVERY offset in the window, per layer:
     purple = sequence only  (Murphy PWM / background, layer 0)
     amber  = Fiber only     (binom(k,n,p_ABF1) / binom(k,n,p_bg), layers 5/6)
     ink    = their product  -- what a single isolated 14 bp placement contributes
   with the posterior-0.5 bar drawn in.  odds = prior x LR, and ABF1's prior is
   2 * tf_prob = 3.59e-07 (HMMconfig.pkl), so LR must exceed ~2.78e6 for that
   placement to be worth more than even money.  Where the ink curve crosses that
   line is where a blue posterior bar should appear -- and it does.

CAVEAT, stated on the figure too: the profile is a HAND-RECOMPUTED approximation,
not the decode's own arithmetic.  It scores one isolated 14 bp placement against
background.  The real HMM does forward-backward over the whole segment, so every
placement competes with nucleosomes, `unknown`, and every other offset, and the
transition prior enters once per state chain rather than as a flat 2*tf_prob.
Expect the profile to track the posterior's SHAPE and be off on absolute height.

The Fiber counts (and the reference sequence) are read from the FIBER-ONLY run by
default (--fiberdir), so the amber curve is the same input the fiber-only decode
saw.  The counts are identical across runs -- they are input data, not output --
but reading them from the fiber-only run keeps the provenance unambiguous.

    python plot_abf1_5sites_layers.py --outdir robocop_chrI_maskon_revfix --key fiber_abf1
    python plot_abf1_5sites_layers.py --outdir robocop_chrI_seq_maskon_revfix --key seq_abf1 \
        --title "Fiber-seq + sequence  /  ABF1-only mask  -- AFTER reverse-complement fix"
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

WATSON, CRICK = "#2a78d6", "#eb6834"
BG_METH = 0.1383          # genome-wide m6A rate, inputs/bg_params.pkl
THRESH_CACHE = "abf1_thresholds.json"
LEGACY_CACHE = "site4_thresholds.json"

BLUE   = "#2a78d6"
INK    = "#0b0b0b"; MUTED = "#6b6a66"; GRID = "#e7e6e2"; BAND = "#cfcec9"
GREEN  = "#1a7a44"; GBOX = ("#eafaf1", "#1baf7a")
RED    = "#a2322b"; RBOX = ("#fdecec", "#eb6834")
PURPLE = "#6a3fb5"   # sequence layer (Murphy PWM)
AMBER  = "#c8860d"   # Fiber layers 5/6

LOG_FLOOR = -30.0     # clip for display; nothing below this is distinguishable


# --------------------------------------------------------------------------
# per-layer likelihood ratios at every offset in the window
# --------------------------------------------------------------------------
def _segment_index(outDir):
    idx = {}
    for f in glob.glob(os.path.join(outDir, "tmpDir", "info_*.h5")):
        with h5py.File(f, "r") as h:
            for k in h.keys():
                if k.startswith("segment_"):
                    idx[int(k.split("_")[1])] = f
    return idx


def _read_segment(outDir, chrm, lo, hi):
    """nucleotides + Fiber counts covering [lo,hi]; returns dict or None."""
    co = pd.read_csv(os.path.join(outDir, "coords.tsv"), sep="\t")
    cand = co[(co.chr == chrm) & (co.start <= lo) & (co.end >= hi)]
    if cand.empty:
        return None
    seg = int(cand.index[0])
    segf = _segment_index(outDir).get(seg)
    if segf is None:
        return None
    with h5py.File(segf, "r") as h:
        K = "segment_%d/" % seg
        d = dict(
            nt=robocop.get_sparse_todense(h, K + "nucleotides").astype(int),
            kw=robocop.get_sparse_todense(h, K + "Fiber_count_meth_watson"),
            nw=robocop.get_sparse_todense(h, K + "Fiber_count_A_watson"),
            kc=robocop.get_sparse_todense(h, K + "Fiber_count_meth_crick"),
            nc=robocop.get_sparse_todense(h, K + "Fiber_count_A_crick"),
            base=int(h["segment_%d" % seg].attrs["start"]) - 1)
    return d


def _pmf_lr(k, n, pvec, bg):
    """(npos, L) array: binom(k,n,pvec[j]) / binom(k,n,bg), 1 where undefined.
    n == 0 gives pmf 1 under any p, so the ratio is 1 and log-LR 0 -- an
    uncovered position contributes nothing, which is the correct behaviour."""
    den = binom.pmf(k, n, bg)
    out = np.ones((len(k), len(pvec)))
    ok = den > 0
    for j, p in enumerate(pvec):
        num = binom.pmf(k[ok], n[ok], p)
        out[ok, j] = np.where(num > 0, num / den[ok], 1e-300)
    return out


def _diag_sum(M, npos, L):
    """s -> sum_j M[s+j, j], for every valid start s. M is (npos, L) log-space."""
    ns = npos - L + 1
    if ns <= 0:
        return np.full(0, np.nan)
    tot = np.zeros(ns)
    for j in range(L):
        tot += M[j:j + ns, j]
    return tot


def layer_profiles(seqDir, fibDir, chrm, lo, hi, seq_on=True):
    """log10 LR of each layer for a 14 bp ABF1 placement CENTRED at each position
    in [lo,hi].  Returns dict of arrays keyed by centre position, or None.

    Sequence and Fiber counts both come from fibDir (identical input data in every
    run; taking them from one place keeps the provenance clean).  The PWM and the
    fitted footprint p-vector come from the decode being plotted."""
    try:
        pwm = pickle.load(open(os.path.join(seqDir, "pwm.p"), "rb"))
        A = pwm["Abf1_murphy"][:4, :]
        Ar = robocop.reverse_complement(pwm["Abf1_murphy"])[:4, :]
        bgv = np.asarray(pwm["background"]).ravel()[:4]
        lp = pickle.load(open("inputs/all_TFs_1000pealVal_params_pseudo.pkl", "rb"))
        bgp = pickle.load(open("inputs/bg_params.pkl", "rb"))
        PW = np.asarray(lp["p"]["Abf1_murphy"]["watson_signal"]["A"])
        PC = np.asarray(lp["p"]["Abf1_murphy"]["crick_signal"]["A"])
        bw = float(np.ravel(bgp["p"]["watson_signal"]["A"])[0])
        bc = float(np.ravel(bgp["p"]["crick_signal"]["A"])[0])
        L = A.shape[1]
        pad = L + 2
        d = _read_segment(fibDir, chrm, lo - pad, hi + pad)
        if d is None:
            return None
    except Exception as e:
        print("  [layer_profiles skipped: %s]" % e)
        return None

    # slice the segment to [lo-pad, hi+pad] in local coords
    i0, i1 = lo - pad - d["base"], hi + pad - d["base"] + 1
    if i0 < 0 or i1 > len(d["nt"]):
        return None
    nt = d["nt"][i0:i1]
    kw, nw = d["kw"][i0:i1], d["nw"][i0:i1]
    kc, nc = d["kc"][i0:i1], d["nc"][i0:i1]
    gstart = lo - pad                      # genomic coord of local index 0
    npos = len(nt)
    isA, isT = (nt == 0), (nt == 3)
    valid = nt <= 3

    with np.errstate(divide="ignore", invalid="ignore"):
        # ---- sequence: log10 PWM/background at each (position, motif column) ----
        ntc = np.clip(nt, 0, 3)
        Sf = np.log10(np.where(valid[:, None], A[ntc, :] / bgv[ntc][:, None], 1e-300))
        Sr = np.log10(np.where(valid[:, None], Ar[ntc, :] / bgv[ntc][:, None], 1e-300))

        # ---- fiber: forward uses (PW on Watson, PC on Crick); reverse needs BOTH
        # the mirror and the Watson<->Crick cross, exactly as robocop.py:648-649.
        # p was fitted in the motif's OWN frame, so 'watson_signal' means "the
        # motif's own strand", not "the reference plus strand".
        Wf = np.log10(_pmf_lr(kw, nw, PW, bw))
        Cf = np.log10(_pmf_lr(kc, nc, PC, bc))
        Wr = np.log10(_pmf_lr(kw, nw, PC[::-1], bw))
        Cr = np.log10(_pmf_lr(kc, nc, PW[::-1], bc))

    zero = np.zeros((npos, L))
    Ff = np.where(isA[:, None], Wf, zero) + np.where(isT[:, None], Cf, zero)
    Fr = np.where(isA[:, None], Wr, zero) + np.where(isT[:, None], Cr, zero)

    seq_f, seq_r = _diag_sum(Sf, npos, L), _diag_sum(Sr, npos, L)
    fib_f, fib_r = _diag_sum(Ff, npos, L), _diag_sum(Fr, npos, L)
    starts = gstart + np.arange(len(seq_f))
    centres = starts + L // 2

    # The PRODUCT is over the layers this decode actually has switched on.  In the
    # fiber-only runs layer 0 is neutralised to 1.0 (robocopExtras.py), so sequence
    # must NOT enter the product there -- including it would score a hypothesis the
    # decode never evaluated.  The winning strand follows from that same product,
    # and all three curves are then reported on that one strand so they describe a
    # single coherent placement rather than three different ones.
    prod_f = fib_f + (seq_f if seq_on else 0.0)
    prod_r = fib_r + (seq_r if seq_on else 0.0)
    rev_wins = prod_r > prod_f
    keep = (centres >= lo) & (centres <= hi)

    def pick(f, r):
        return np.where(rev_wins, r, f)[keep]

    return dict(
        centre=centres[keep], start=starts[keep], L=L, seq_on=seq_on,
        seq=pick(seq_f, seq_r), fib=pick(fib_f, fib_r), prod=pick(prod_f, prod_r),
        strand=np.where(rev_wins, "rev", "fwd")[keep],
        # the annotation bars answer a different question -- "where would this layer
        # ALONE put the footprint?" -- so they keep their own per-layer argmax strand
        seq_best=np.maximum(seq_f, seq_r)[keep],
        seq_best_strand=np.where(seq_r > seq_f, "rev", "fwd")[keep],
        fib_best=np.maximum(fib_f, fib_r)[keep],
        fib_best_strand=np.where(fib_r > fib_f, "rev", "fwd")[keep])


def argmax_placement(prof, key, strand_key):
    """Best placement in the window for one layer -> dict(start,end,strand,log10lr)."""
    if prof is None or len(prof[key]) == 0:
        return None
    v = prof[key]
    if not np.isfinite(v).any():
        return None
    i = int(np.nanargmax(np.where(np.isfinite(v), v, -np.inf)))
    return dict(start=int(prof["start"][i]), end=int(prof["start"][i] + prof["L"]),
                strand=str(prof[strand_key][i]), log10lr=float(v[i]),
                centre=int(prof["centre"][i]))


def posterior_bar(trainDir, tf_index=0):
    """LR a single ABF1 placement needs for posterior 0.5.  odds = prior x LR and
    the prior is 2 * tf_prob (two orientations), so LR = (1 - prior) / prior."""
    try:
        cfg = pickle.load(open(os.path.join(trainDir, "HMMconfig.pkl"), "rb"))
        prior = 2.0 * float(np.ravel(cfg["tf_prob"])[tf_index])
        return (1.0 - prior) / prior, prior
    except Exception as e:
        print("  [posterior_bar skipped: %s]" % e)
        return None, None


def fiber_track(outDir, chrm, lo, hi):
    """Raw Fiber-seq m6A per base over [lo,hi].  Watson read at reference A,
    Crick at reference T -- exactly the positions layers 5/6 use."""
    try:
        d = _read_segment(outDir, chrm, lo, hi)
        if d is None:
            return None
    except Exception as e:
        print("  [fiber_track skipped: %s]" % e)
        return None
    nt, base = d["nt"], d["base"]
    pos = np.arange(lo, hi + 1)
    i = pos - base
    ok = (i >= 0) & (i < len(nt))
    pos, i = pos[ok], i[ok]
    isA, isT = nt[i] == 0, nt[i] == 3
    k = np.where(isA, d["kw"][i], np.where(isT, d["kc"][i], 0)).astype(float)
    n = np.where(isA, d["nw"][i], np.where(isT, d["nc"][i], 0)).astype(float)
    frac = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    W = 15
    def roll(mask):
        out = np.full(len(pos), np.nan)
        kk_, nn_ = np.where(mask, k, 0.0), np.where(mask, n, 0.0)
        for j in range(len(pos)):
            a, b = max(0, j - W), min(len(pos), j + W + 1)
            if nn_[a:b].sum() > 0:
                out[j] = kk_[a:b].sum() / nn_[a:b].sum()
        return out
    return dict(pos=pos, frac=frac, n=n, isA=isA, isT=isT,
                smW=roll(isA), smC=roll(isT))


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
    ap.add_argument("--outdir", default="robocop_chrI_maskon_revfix")
    ap.add_argument("--fiberdir", default="robocop_chrI_maskon_revfix",
                    help="run whose tmpDir supplies the Fiber counts + reference sequence "
                         "for the amber/purple profile (default: the fiber-only ABF1-only run)")
    ap.add_argument("--traindir", default="robocop_train_fiberonly",
                    help="supplies tf_prob for the posterior-0.5 bar")
    ap.add_argument("--seqlayer", choices=("auto", "on", "off"), default="auto",
                    help="is layer 0 live in this decode? fiber-only runs neutralise it to "
                         "1.0, so sequence must not enter the product there. "
                         "auto = on iff 'seq' appears in --outdir")
    ap.add_argument("--key", default="fiber_abf1")
    ap.add_argument("--title", default="Fiber-seq only  /  ABF1-only mask")
    ap.add_argument("--window", type=int, default=170, help="display half-width (bp)")
    ap.add_argument("--compute-window", type=int, default=400)
    ap.add_argument("--tol", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or ("abf1_5sites_%s_layers.png" % args.key)

    seq_on = ("seq" in os.path.basename(args.outdir.rstrip("/"))
              if args.seqlayer == "auto" else args.seqlayer == "on")
    print("sequence layer (layer 0): %s" % ("ON" if seq_on else "OFF"))

    cache = load_cache()
    gmax = global_max(args.outdir, args.key, cache)

    regions = [(c, m0 - args.compute_window, m1 + args.compute_window)
               for _, c, m0, m1, _ in SITES]
    res = S.score(args.outdir, regions=regions, abf1_global_max=gmax,
                  return_abf1_tracks=True, label=args.key)
    details = [r.get("abf1_detail") for r in res["_per_region"]]

    bar, prior = posterior_bar(args.traindir)
    logbar = np.log10(bar) if bar else None

    n = len(SITES)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(13.6, 7.2 * n))
    outer = gridspec.GridSpec(n, 1, hspace=0.38)
    axes, laxes, waxes, caxes = [], [], [], []
    for r in range(n):
        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[r], height_ratios=[3.0, 3.0, 1.0, 1.0], hspace=0.07)
        a = fig.add_subplot(inner[0]); axes.append(a)
        laxes.append(fig.add_subplot(inner[1], sharex=a))
        waxes.append(fig.add_subplot(inner[2], sharex=a))
        caxes.append(fig.add_subplot(inner[3], sharex=a))
    summary = []

    for ax, lax, wax, cax, (num, chrm, m0, m1, name), det in zip(
            axes, laxes, waxes, caxes, SITES, details):
        mid = (m0 + m1) // 2
        lo, hi = mid - args.window, mid + args.window
        ax.set_facecolor("white")
        ax.axvspan(m0, m1, color=BAND, alpha=0.55, zorder=0)
        ax.axvline(mid, color=INK, ls="--", lw=1.4, zorder=3)

        # ---- scan the WHOLE plotted window, per layer ----
        prof = layer_profiles(args.outdir, args.fiberdir, chrm, lo, hi, seq_on=seq_on)
        murphy = argmax_placement(prof, "seq_best", "seq_best_strand")
        fiber = argmax_placement(prof, "fib_best", "fib_best_strand")

        Y_MAC, Y_PWM, Y_FIB, H = 1.30, 1.19, 1.08, 0.075
        ax.add_patch(plt.Rectangle((m0, Y_MAC), m1 - m0, H, color=MUTED,
                                   alpha=0.85, zorder=6, clip_on=False))
        ax.text(m1 + 3, Y_MAC + H / 2, "MacIsaac motif", va="center", fontsize=8,
                color=MUTED, fontweight="bold", zorder=7, clip_on=False)
        if murphy:
            ax.add_patch(plt.Rectangle((murphy["start"], Y_PWM), murphy["end"] - murphy["start"],
                                       H, color=PURPLE, alpha=0.85, zorder=6, clip_on=False))
            ax.text(murphy["end"] + 3, Y_PWM + H / 2,
                    "best Murphy PWM in window (%s, LR %.1e)  —  %+d bp vs MacIsaac"
                    % (murphy["strand"], 10.0 ** murphy["log10lr"], murphy["start"] - m0),
                    va="center", fontsize=8, color=PURPLE, fontweight="bold",
                    zorder=7, clip_on=False)
        if fiber:
            ax.add_patch(plt.Rectangle((fiber["start"], Y_FIB), fiber["end"] - fiber["start"],
                                       H, color=AMBER, alpha=0.85, zorder=6, clip_on=False))
            gap = fiber["start"] - (murphy["start"] if murphy else m0)
            ax.text(fiber["end"] + 3, Y_FIB + H / 2,
                    "best Fiber in window (%s, LR %.1e)  —  %+d bp vs PWM%s"
                    % (fiber["strand"], 10.0 ** fiber["log10lr"], gap,
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
        ax.tick_params(length=0, labelsize=9, labelbottom=False)
        summary.append((num, on_motif, near_motif,
                        None if near is None else near["center"], dist, ok, len(vis),
                        murphy, fiber))

        # ------------------- per-layer log10 LR profile -------------------
        lax.set_facecolor("white")
        lax.axvspan(m0, m1, color=BAND, alpha=0.55, zorder=0)
        lax.axvline(mid, color=INK, ls="--", lw=1.4, zorder=3)
        for c in vis:
            lax.axvline(c["center"], color=MUTED, lw=0.8, alpha=0.45, zorder=1)
        if prof is not None:
            cen = prof["centre"]
            sq = np.clip(prof["seq"], LOG_FLOOR, None)
            fb_ = np.clip(prof["fib"], LOG_FLOOR, None)
            pd_ = np.clip(prof["prod"], LOG_FLOOR, None)
            lax.axhline(0, color=MUTED, lw=1.0, zorder=2)
            lax.plot(cen, sq, color=PURPLE, lw=1.0, zorder=4,
                     alpha=0.8 if prof["seq_on"] else 0.45,
                     ls="-" if prof["seq_on"] else (0, (4, 2)))
            lax.plot(cen, fb_, color=AMBER, lw=1.0, alpha=0.8, zorder=4)
            lax.plot(cen, pd_, color=INK, lw=2.0, zorder=5)
            if logbar is not None:
                lax.axhline(logbar, color=RED, ls=":", lw=1.4, zorder=6)
                lax.text(0.996, logbar, "posterior 0.5 bar (LR %.2g) " % bar,
                         transform=lax.get_yaxis_transform(), va="bottom", ha="right",
                         fontsize=7.8, color=RED, fontweight="bold", zorder=8,
                         bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
                above = pd_ >= logbar
                if above.any():
                    lax.fill_between(cen, logbar, pd_, where=above,
                                     color=INK, alpha=0.13, zorder=3)

            # readout at the positions that matter: the motif, and every call centre
            def at(x):
                i = int(np.argmin(np.abs(cen - x)))
                return sq[i], fb_[i], pd_[i], prof["strand"][i]
            rows, marks = [], [(mid, "MacIsaac motif", MUTED)]
            for c in vis:
                matched = near is not None and c["center"] == near["center"]
                marks.append((c["center"], "call %d" % c["center"],
                              (GREEN if ok else RED) if matched else MUTED))
            for x, lbl, col in marks:
                if not (cen[0] <= x <= cen[-1]):
                    continue
                s_, f_, p_, st = at(x)
                lax.plot([x], [p_], marker="o", ms=6, color=col,
                         markeredgecolor="white", markeredgewidth=1.1, zorder=9)
                # LINEAR scientific notation, not "1e<log10>" -- writing the
                # exponent of a log-value as if it were a mantissa reads as a
                # different number entirely.
                rows.append("%-16s %3s  seq %8.1e  fiber %8.1e  product %8.1e"
                            % (lbl, st, 10.0 ** s_, 10.0 ** f_, 10.0 ** p_))
            if rows:
                lax.text(0.004, 0.03, "\n".join(rows), transform=lax.transAxes,
                         ha="left", va="bottom", fontsize=7.4, color=INK,
                         family="DejaVu Sans Mono", zorder=9,
                         bbox=dict(boxstyle="round,pad=0.32", fc="white",
                                   ec="#c9c8c4", lw=0.8, alpha=0.93))

            vals = np.concatenate([sq, fb_, pd_])
            vals = vals[np.isfinite(vals)]
            ymin = min(-2.0, float(np.percentile(vals, 3)) - 2)
            ymax = max(float(vals.max()), logbar if logbar else 0) + 2.5
            lax.set_ylim(max(ymin, LOG_FLOOR - 1), ymax)

            # Band showing the ONLY stretch of this axis where the posterior is
            # neither ~0 nor ~1.  posterior = odds/(1+odds) with odds = prior * LR,
            # so posterior p sits at log10 LR = log10( p/(1-p)/prior ).  0.01 -> 0.99
            # spans just 4 decades out of the ~50 the panel covers.  Outside the band
            # the posterior is saturated, which is why the blue track reads as binary
            # and why "just under the bar" is nothing like "just under".
            if prior:
                b_lo = np.log10(0.01 / 0.99 / prior)
                b_hi = np.log10(0.99 / 0.01 / prior)
                lax.axhspan(b_lo, b_hi, color=RED, alpha=0.07, zorder=1)
                lax.text(0.004, b_hi, " approx. posterior 0.01 → 0.99 lives in here;"
                                      " above the band ≈ 1, below ≈ 0",
                         transform=lax.get_yaxis_transform(), va="bottom", ha="left",
                         fontsize=7.4, color=RED, zorder=8)
        lax.set_ylabel("log$_{10}$ LR\nvs background", fontsize=9.0)
        lax.text(0.995, 0.965,
                 "sequence (PWM)   ·   Fiber (layers 5/6)   ·   product"
                 if (prof is None or prof["seq_on"]) else
                 "sequence (PWM) — layer 0 is OFF in this run, shown for reference only"
                 "   ·   Fiber (layers 5/6) = the product",
                 transform=lax.transAxes, ha="right", va="top", fontsize=7.8,
                 color=MUTED, fontweight="bold")
        lax.yaxis.grid(True, color=GRID, lw=1, zorder=0); lax.set_axisbelow(True)
        for sp in ("top", "right"): lax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): lax.spines[sp].set_color("#b8b7b2")
        lax.tick_params(length=0, labelsize=8.2, labelbottom=False)

        # ------- raw Fiber-seq meth/A, Watson and Crick as SEPARATE tracks -------
        ft = fiber_track(args.fiberdir, chrm, lo - 5, hi + 5)
        for sax, col, sel, smk, lab in (
                (wax, WATSON, "isA", "smW", "Watson meth/A  (reference A, layer 5)"),
                (cax, CRICK,  "isT", "smC", "Crick meth/A  (reference T, layer 6)")):
            sax.set_facecolor("white")
            sax.axvspan(m0, m1, color=BAND, alpha=0.55, zorder=0)
            sax.axvline(mid, color=INK, ls="--", lw=1.4, zorder=3)
            if ft is not None:
                m = ft[sel]
                sax.scatter(ft["pos"][m], ft["frac"][m], s=6, color=col, alpha=0.75, zorder=2)
                sax.plot(ft["pos"], ft[smk], color=col, lw=1.9, zorder=4)
                sax.axhline(BG_METH, color=MUTED, ls="--", lw=1.0, zorder=3)
            sax.set_ylim(0, 1); sax.set_yticks([0, 0.5, 1.0])
            sax.set_yticklabels(["0.0", "0.5", ""])
            sax.set_ylabel(lab.split("  ")[0].replace(" ", "\n"), fontsize=8.4)
            sax.text(0.995, 0.92, lab, transform=sax.transAxes, ha="right", va="top",
                     fontsize=7.6, color=col, fontweight="bold")
            sax.yaxis.grid(True, color=GRID, lw=1, zorder=0); sax.set_axisbelow(True)
            for sp in ("top", "right"): sax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"): sax.spines[sp].set_color("#b8b7b2")
            sax.tick_params(length=0, labelsize=8.2)
        wax.tick_params(labelbottom=False)

    caxes[-1].set_xlabel("chrI position (bp)", fontsize=10.5)

    leg = [
        Patch(fc=BAND, alpha=0.55, label="MacIsaac motif span (14 bp)"),
        Patch(fc=PURPLE, alpha=0.85, label="best Murphy PWM placement in the window"),
        Patch(fc=AMBER, alpha=0.85, label="best Fiber-layer placement in the window"),
        Line2D([0], [0], color=PURPLE, lw=1.5, label="log$_{10}$ LR, sequence layer alone"),
        Line2D([0], [0], color=AMBER, lw=1.5, label="log$_{10}$ LR, Fiber layers 5/6 alone"),
        Line2D([0], [0], color=INK, lw=2.0, label="log$_{10}$ LR, product (what the HMM multiplies)"),
        Line2D([0], [0], color=RED, ls=":", lw=1.3,
               label="LR needed for posterior 0.5 given ABF1's prior"),
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
               label="genome-wide background m6A rate (0.138) — dips below it = protection"),
    ]
    fig.legend(handles=leg, frameon=False, fontsize=9.2, loc="lower center",
               bbox_to_anchor=(0.5, -0.010), ncol=3)

    fig.suptitle("All 5 chrI MacIsaac ABF1 sites — %s  (%s)" % (args.title, args.outdir),
                 fontsize=14, fontweight="bold", y=0.996)
    fig.text(0.5, 0.9805,
             "Top: ABF1 posterior from score_robocop (the decode's real output).  "
             "Middle: per-layer log$_{10}$ likelihood ratio at EVERY offset in the window.",
             ha="center", va="top", fontsize=9.2, color=MUTED)
    fig.text(0.5, 0.9735,
             "The LR profile is a HAND-RECOMPUTED single-placement approximation "
             "(Fiber counts from %s; sequence layer %s in this decode) — it tracks the "
             "posterior's shape, not its exact height: the real HMM lets every offset, "
             "nucleosomes and `unknown` compete."
             % (args.fiberdir, "ON" if seq_on else "OFF"),
             ha="center", va="top", fontsize=8.6, color=MUTED, style="italic")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.962, bottom=0.075)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print("wrote", out)
    if bar:
        print("posterior-0.5 bar: prior %.4g  ->  LR %.4g  (log10 %.3f)" % (prior, bar, logbar))
    print("\n%-6s %10s %11s %12s %8s %8s %7s  %-26s %-26s"
          % ("site", "on-motif", "within60", "call center", "dist", "status", "#calls",
             "best PWM in window", "best Fiber in window"))
    for num, onm, nearm, cc, d, ok, ncalls, mu, fb in summary:
        f1 = ("%d %s %.1e" % (mu["start"], mu["strand"], 10.0 ** mu["log10lr"])) if mu else "-"
        f2 = ("%d %s %.1e" % (fb["start"], fb["strand"], 10.0 ** fb["log10lr"])) if fb else "-"
        print("%-6s %10.4f %11.4f %12s %8s %8s %7d  %-26s %-26s"
              % ("#%d" % num, onm, nearm, cc if cc is not None else "-",
                 d if d is not None else "-", "HIT" if ok else "miss", ncalls, f1, f2))


if __name__ == "__main__":
    main()
