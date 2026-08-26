"""Which bases does each layer's ABF1 placement actually use, and how many are shared?

For sites #3 and #4 this draws the two competing 14 bp placements base-by-base:
  row 1 = where the FIBER layer wants the ABF1 chain
  row 2 = where the MOTIF (MacIsaac/Murphy PWM) forces it

Only reference-A bases feed the Watson channel and only reference-T bases feed the
Crick channel (robocop.py:775, nucleotide_ref = 0 for watson / 3 for crick); G and C
positions contribute nothing -- the code zeroes them for ALL states, so they cancel.
Each contributing base is labelled with the likelihood-ratio factor it contributes,
and the product of those factors is the whole "fiber LR" for that placement.

The point: an 8 bp shift at site #3 keeps only 3 of 9 contributing bases, and drops
the single most informative one (chrI 61162, 0 methylated of 76 fibers = 347x).
At site #4 the shift is 2 bp and keeps 8 of 9.

    python plot_abf1_base_overlap.py
"""
import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
import numpy as np
import h5py
import robocop
from scipy.stats import binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

H5 = "robocop_chrI_maskon_revfix/tmpDir/info_3_6.h5"
SEG, BASE = 15, 60000          # segment 15 = chrI 60001-65000; idx i <-> chrI 0-based 60000+i
OUT = "abf1_base_overlap_revfix.png"

WATSON = "#2a78d6"   # reference A -> Watson channel
CRICK  = "#eb6834"   # reference T -> Crick channel
IGN    = "#c9c8c3"   # G/C -> contributes nothing
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e7e6e2"
SHARE  = "#f2e9a8"

# (site, label, fiber placement, motif placement)
CASES = [
    (3, "site #3   upstream of ERV46   —  STILL FAILS after the reverse-complement fix",
        dict(start=61158, rev=True), dict(start=61164, rev=False)),
    (4, "site #4   downstream of ERV46  —  STILL WORKS",
        dict(start=62652, rev=True), dict(start=62658, rev=False)),
]


def load():
    h = h5py.File(H5, "r"); K = "segment_%d/" % SEG
    d = dict(
        nt=robocop.get_sparse_todense(h, K + "nucleotides").astype(int),
        kw=robocop.get_sparse_todense(h, K + "Fiber_count_meth_watson"),
        nw=robocop.get_sparse_todense(h, K + "Fiber_count_A_watson"),
        kc=robocop.get_sparse_todense(h, K + "Fiber_count_meth_crick"),
        nc=robocop.get_sparse_todense(h, K + "Fiber_count_A_crick"))
    h.close()
    lp = pickle.load(open("inputs/all_TFs_1000pealVal_params_pseudo.pkl", "rb"))
    bgp = pickle.load(open("inputs/bg_params.pkl", "rb"))
    d["PW"] = np.asarray(lp["p"]["Abf1_murphy"]["watson_signal"]["A"])
    d["PC"] = np.asarray(lp["p"]["Abf1_murphy"]["crick_signal"]["A"])
    d["bw"] = float(np.ravel(bgp["p"]["watson_signal"]["A"])[0])
    d["bc"] = float(np.ravel(bgp["p"]["crick_signal"]["A"])[0])
    return d


def bases(d, start, rev, L=14):
    """Per-column record for one placement."""
    # Reverse needs BOTH the mirror and the Watson<->Crick cross (robocop.py:648-649).
    # p was fitted in the motif's own frame, so 'watson_signal' = the motif's own strand.
    Pw = (d["PC"][::-1] if rev else d["PW"])
    Pc = (d["PW"][::-1] if rev else d["PC"])
    out = []
    for j in range(L):
        g = start + j; i = g - BASE; b = "ACGTN"[d["nt"][i]]
        if d["nt"][i] == 0:
            n, k, p, pb, ch = d["nw"][i], d["kw"][i], Pw[j], d["bw"], "W"
        elif d["nt"][i] == 3:
            n, k, p, pb, ch = d["nc"][i], d["kc"][i], Pc[j], d["bc"], "C"
        else:
            out.append(dict(pos=g, base=b, ch=None, f=1.0)); continue
        den = binom.pmf(k, n, pb)
        f = float(binom.pmf(k, n, p) / den) if den > 0 else 1.0
        out.append(dict(pos=g, base=b, ch=ch, n=int(n), k=int(k), p=float(p), f=f))
    return out


def main():
    d = load()
    fig, axes = plt.subplots(len(CASES), 1, figsize=(18.0, 6.1 * len(CASES)))
    summary = []
    for ax, (num, title, fib, mot) in zip(np.atleast_1d(axes), CASES):
        rows = [("FIBER layer's placement", bases(d, fib["start"], fib["rev"]), fib),
                ("MOTIF placement (MacIsaac / Murphy)", bases(d, mot["start"], mot["rev"]), mot)]
        lo = min(fib["start"], mot["start"]) - 1
        hi = max(fib["start"], mot["start"]) + 14 + 1
        ov0, ov1 = max(fib["start"], mot["start"]), min(fib["start"], mot["start"]) + 13
        if ov1 >= ov0:
            ax.axvspan(ov0 - 0.5, ov1 + 0.5, color=SHARE, alpha=0.55, zorder=0)
            ax.text((ov0 + ov1) / 2, 2.62, "overlap: %d bp" % (ov1 - ov0 + 1),
                    ha="center", va="bottom", fontsize=10, color="#8a7b1f", fontweight="bold")

        contrib = []
        for r, (lab, bb, pl) in enumerate(rows):
            y = 1.55 - r * 1.15
            used = [x for x in bb if x["ch"]]
            contrib.append({x["pos"] for x in used})
            for x in bb:
                col = IGN if x["ch"] is None else (WATSON if x["ch"] == "W" else CRICK)
                ax.add_patch(Rectangle((x["pos"] - 0.46, y), 0.92, 0.52, facecolor=col,
                                       edgecolor="white", lw=1.4, zorder=3))
                ax.text(x["pos"], y + 0.26, x["base"], ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if x["ch"] else "#8a8984", zorder=4)
                if x["ch"]:
                    ax.text(x["pos"], y - 0.10, "%d/%d" % (x["k"], x["n"]), ha="center",
                            va="center", fontsize=6.6, color=MUTED, zorder=4)
                    big = x["f"] >= 50
                    ax.text(x["pos"], y - 0.26, ("%.0f×" % x["f"]) if x["f"] >= 10 else ("%.1f×" % x["f"]),
                            ha="center", va="center", fontsize=7.6 if big else 6.9,
                            color=RED_IF(x["f"]), fontweight="bold" if big else "normal", zorder=4)
            prod = float(np.prod([x["f"] for x in bb]))
            ax.text(lo - 0.2, y + 0.26,
                    "%s\n%s %d–%d  (%s)" % (lab, "chrI", pl["start"], pl["start"] + 13,
                                            "reverse" if pl["rev"] else "forward"),
                    ha="right", va="center", fontsize=10, fontweight="bold", color=INK)
            ax.text(hi + 0.4, y + 0.26,
                    "%d of 14 bases usable\n(%d A→Watson, %d T→Crick)\nfiber LR = %.4g"
                    % (len(used), sum(1 for x in used if x["ch"] == "W"),
                       sum(1 for x in used if x["ch"] == "C"), prod),
                    ha="left", va="center", fontsize=9.6, color=INK)

        shared = contrib[0] & contrib[1]
        only_f = contrib[0] - contrib[1]
        best = max((x for x in bases(d, fib["start"], fib["rev"]) if x["ch"]), key=lambda x: x["f"])
        note = ("shared usable bases: %d      used ONLY by the fiber placement: %d"
                % (len(shared), len(only_f)))
        if best["pos"] in only_f:
            note += "      ← including chrI %d (%d/%d fibers, %.0f×), the single most informative base" \
                    % (best["pos"], best["k"], best["n"], best["f"])
        ax.text((lo + hi) / 2, -0.22, note, ha="center", va="center", fontsize=10.2,
                color=INK, bbox=dict(boxstyle="round,pad=0.45", fc="#fbfbf9", ec="#c9c8c3"))
        v = VERDICT[num]
        ax.text(0.5, 0.10, v["txt"], transform=ax.transAxes, ha="center", va="center",
                fontsize=9.6, color=v["fg"], fontweight="bold", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.5", fc=v["bg"], ec=v["fg"], lw=1.4))
        summary.append((num, len(contrib[0]), len(contrib[1]), len(shared)))

        ax.set_xlim(lo - 9, hi + 9); ax.set_ylim(-0.85, 2.95)
        ax.set_yticks([])
        ticks = [p for p in range(lo, hi + 1) if p % 5 == 0]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks], fontsize=9)
        ax.set_xlabel("chrI position (bp)", fontsize=10.5)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#b8b7b2"); ax.tick_params(length=0)
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=26)

    leg = [Patch(fc=WATSON, label="reference A  →  read by the Watson channel"),
           Patch(fc=CRICK,  label="reference T  →  read by the Crick channel"),
           Patch(fc=IGN,    label="G or C  →  contributes nothing (cancels for all states)"),
           Patch(fc=SHARE, alpha=0.55, label="positions covered by BOTH placements")]
    fig.legend(handles=leg, frameon=False, fontsize=10, loc="lower center",
               bbox_to_anchor=(0.5, -0.015), ncol=4)
    fig.suptitle("Why site #3 is still not recalled AFTER the reverse-complement fix",
                 fontsize=15, fontweight="bold", y=1.022)
    fig.text(0.5, 0.978,
             "The fix DID move the fiber layer's favourite placement (#3: 61156→61158, #4: 62656 fwd→62652 rev) — but both true motifs are "
             "FORWARD, and the forward block was never affected by the bug.\nWhat decides the call is the fiber LR AT THE MOTIF (row 2), and that "
             "number is bit-for-bit unchanged: 1578 at #3 vs 1.86e+06 at #4, a 1200× gap that predates the fix.",
             ha="center", fontsize=10, color=MUTED)
    plt.tight_layout(rect=[0, 0.035, 1, 0.945])
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)
    print("\n%-6s %14s %14s %10s" % ("site", "fiber usable", "motif usable", "shared"))
    for n, a, b, s in summary:
        print("#%-5d %14d %14d %10d" % (n, a, b, s))


VERDICT = {
 3: dict(fg="#a2322b", bg="#fdeeec", txt=(
    "AT THE MOTIF:  seq 89.2  ×  fiber 1578  =  1.4e+05      bar = 2.1e+06   →  falls 15× short\n"
    "The fiber layer's own favourite (61158 rev) has seq LR 5.6e-09 — it cannot win either\n"
    "posterior:  fiber-only 0.937  →  sequence ON 0.032      (0.032 before the fix — IDENTICAL)")),
 4: dict(fg="#0a7d46", bg="#eaf7f0", txt=(
    "AT THE MOTIF:  seq 9788  ×  fiber 1.86e+06  =  1.8e+10      bar = 2.1e+06   →  clears it 8600×\n"
    "posterior:  fiber-only 0.837  →  sequence ON 0.9997      (0.9998 before the fix — unchanged)")),
}


def RED_IF(f):
    return "#a2322b" if f < 1 else ("#0a7d46" if f >= 50 else MUTED)


if __name__ == "__main__":
    main()
