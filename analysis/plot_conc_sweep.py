"""Figures for the ABF1 concentration sweep -> conc_sweep.png

Reads conc_sweep_metrics.json (score_conc_sweep.py --merge) and the per-site probe from
check_conc_site3.py's probe().

Panel D is the one that settles the question. Raising lambda lifts ABF1's posterior
everywhere, so recall and n_pred rise together; the honest comparison against the JASPAR
PWM swap is therefore at MATCHED n_pred. If Murphy@lambda lands on the same recall-vs-n_pred
curve as JASPAR@lambda=1, the PWM swap bought level, not information.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from check_conc_site3 import SITES, probe

VCOLOR = {"seq_maskon": "tab:blue", "seq_maskoff": "tab:orange"}
VLABEL = {"seq_maskon": "Fiber+seq, ABF1-only", "seq_maskoff": "Fiber+seq, all TFs"}


def load():
    rows = json.load(open("conc_sweep_metrics.json"))
    murphy = [r for r in rows if r["pwm"] == "murphy"]
    jaspar = [r for r in rows if r["pwm"] == "jaspar"]
    return murphy, jaspar


def main():
    murphy, jaspar = load()
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # --- A: recall + n_pred vs lambda ---
    a = ax[0, 0]
    a2 = a.twinx()
    for v in VCOLOR:
        rs = sorted([r for r in murphy if r["variant"] == v], key=lambda r: r["lam"])
        if not rs:
            continue
        lam = [r["lam"] for r in rs]
        a.plot(lam, [r["abf1_tp"] for r in rs], "o-", color=VCOLOR[v], label=VLABEL[v])
        a2.plot(lam, [r["abf1_n_pred"] for r in rs], "s--", color=VCOLOR[v], alpha=0.45)
    for r in jaspar:
        a.axhline(r["abf1_tp"], ls=":", color=VCOLOR[r["variant"]], alpha=0.7)
    a.set_xscale("log"); a.set_xlabel(r"$\lambda$ (ABF1 concentration multiplier)")
    a.set_ylabel("MacIsaac sites recovered (of 5)"); a.set_ylim(-0.2, 5.2)
    a2.set_ylabel("predicted ABF1 peaks (dashed)")
    a.set_title("A  recall (solid) and precision cost (dashed)\ndotted = JASPAR PWM at $\\lambda$=1")
    a.legend(loc="upper left", fontsize=8)

    # --- B: AUROC vs lambda -- the discrimination test ---
    a = ax[0, 1]
    for v in VCOLOR:
        rs = sorted([r for r in murphy if r["variant"] == v], key=lambda r: r["lam"])
        if not rs:
            continue
        a.plot([r["lam"] for r in rs], [r["abf1_auroc"] for r in rs], "o-",
               color=VCOLOR[v], label=VLABEL[v])
    for r in jaspar:
        a.axhline(r["abf1_auroc"], ls=":", color=VCOLOR[r["variant"]], alpha=0.7)
    a.set_xscale("log"); a.set_xlabel(r"$\lambda$"); a.set_ylabel("ABF1 AUROC")
    a.set_title("B  threshold-free discrimination\nflat $\\Rightarrow$ $\\lambda$ buys level, not information")
    a.legend(fontsize=8)

    # --- C: per-site posterior vs lambda ---
    a = ax[1, 0]
    rs = sorted([r for r in murphy if r["variant"] == "seq_maskon"], key=lambda r: r["lam"])
    lams, per_site = [], {n: [] for n in SITES}
    for r in rs:
        if not os.path.isdir(r["outDir"]):
            continue
        res = probe(r["outDir"])
        lams.append(r["lam"])
        for n in SITES:
            per_site[n].append(max(res[n][0], 1e-5))
    for n in SITES:
        if lams:
            a.plot(lams, per_site[n], "o-", label="site #%d (%d)" % (n, SITES[n]))
    a.axhline(0.30, color="k", ls="--", lw=1, label="call threshold 0.30")
    a.axhline(1e-4, color="grey", ls=":", lw=1, label="storage floor 1e-4")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"$\lambda$"); a.set_ylabel("max ABF1 posterior within $\\pm$25 bp")
    a.set_title("C  per-site response (Fiber+seq, ABF1-only)")
    a.legend(fontsize=7)

    # --- D: recall vs n_pred -- matched-precision comparison ---
    a = ax[1, 1]
    for v in VCOLOR:
        rs = sorted([r for r in murphy if r["variant"] == v], key=lambda r: r["abf1_n_pred"])
        if not rs:
            continue
        a.plot([r["abf1_n_pred"] for r in rs], [r["abf1_tp"] for r in rs], "o-",
               color=VCOLOR[v], label="Murphy + $\\lambda$ — " + VLABEL[v])
        for r in rs:
            a.annotate(r"$\lambda$=%g" % r["lam"], (r["abf1_n_pred"], r["abf1_tp"]),
                       fontsize=6, xytext=(3, 4), textcoords="offset points")
    for r in jaspar:
        a.plot(r["abf1_n_pred"], r["abf1_tp"], "*", ms=18, color=VCOLOR[r["variant"]],
               mec="k", label="JASPAR PWM, $\\lambda$=1 — " + VLABEL[r["variant"]])
    a.set_xscale("log"); a.set_xlabel("predicted ABF1 peaks on chrI (precision cost)")
    a.set_ylabel("sites recovered (of 5)"); a.set_ylim(-0.2, 5.2)
    a.set_title("D  matched-precision test\nJASPAR star ON the Murphy curve $\\Rightarrow$ swap bought level only")
    a.legend(fontsize=7, loc="lower right")

    fig.suptitle("ABF1 concentration sweep on chrI — native Murphy PWM throughout, "
                 "only the transition prior changes", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("conc_sweep.png", dpi=150)
    print("wrote conc_sweep.png")


if __name__ == "__main__":
    main()
