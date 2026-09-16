"""Tuner v2 summary across campaigns: conc_tuning/tw_summary.{tsv,png}.

Reads conc_tuning/<run>/validation.tsv for each run (tw_validate.py output) and, if present,
conc_tuning/tw_legacy_validation.tsv (u001/m001 on the same 58 groups and chromosomes).

tsv: one row per (campaign, round, set, ref) with fitted=all: calls, sites, matched, P, R, F1,
     within2x (T>=5, of N), n_capped_under, n_falling_over, nucleosome copies and delta% vs r0.
png: 4 panels by round, one line per campaign: F1 vs MacIsaac, F1 vs Rossi _CX (tuning chroms;
     chrIV holdout drawn as hollow markers), within-2x, nucleosome delta%. u001/m001 rounds 0 and 7
     are grey reference markers.

Usage:  python tw_summary.py fw01 sw01 bw01
"""
import csv
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "conc_tuning")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, fixed order
LEGACY_MARK = {"u001": "s", "m001": "D"}
COLS = ["campaign", "round", "set", "ref", "calls", "sites", "matched", "P", "R", "F1", "within2x",
        "n_T5", "n_capped_under", "n_falling_over", "nuc_copies", "nuc_rel_r0"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(path, campaign=None):
    if not os.path.exists(path):
        return []
    rows = [r for r in csv.DictReader(open(path), delimiter="\t") if r["fitted"] == "all"]
    for r in rows:
        r["campaign"] = campaign or r["run"]
    return rows


def main():
    runs = sys.argv[1:] or ["fw01", "sw01", "bw01"]
    rows = []
    for run in runs:
        got = load(os.path.join(ROOT, run, "validation.tsv"), run)
        if not got:
            print("note: %s has no validation.tsv yet" % run)
        rows += got
    legacy = load(os.path.join(ROOT, "tw_legacy_validation.tsv"))
    allrows = sorted(rows + legacy, key=lambda r: (r["campaign"], int(r["round"]), r["set"], r["ref"]))
    out = os.path.join(ROOT, "tw_summary.tsv")
    with open(out, "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for r in allrows:
            fh.write("\t".join(str(r.get(c, "")) for c in COLS) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    panels = [("F1 vs MacIsaac", "macisaac", "F1"), ("F1 vs Rossi ChExMix _CX", "rossi_cx", "F1"),
              ("groups within 2x of MacIsaac count (T >= 5)", "macisaac", "within2x"),
              ("nucleosome copies, % vs round 0", "macisaac", "nuc_rel_r0")]
    for ax, (title, ref, key) in zip(axes.flat, panels):
        for i, run in enumerate(runs):
            for which, style in (("tune", dict(marker="o", ls="-")),
                                 ("holdout", dict(marker="o", ls="none", mfc="none", ms=9))):
                if key in ("within2x",) and which == "holdout":
                    continue
                pts = sorted((int(r["round"]), f(r[key])) for r in rows
                             if r["campaign"] == run and r["ref"] == ref and r["set"] == which)
                if not pts:
                    continue
                ys = [100 * y if key == "nuc_rel_r0" else y for _, y in pts]
                ax.plot([x for x, _ in pts], ys, color=COLORS[i], lw=2, ms=style.pop("ms", 6),
                        label="%s %s" % (run, "chrXIV+chrII" if which == "tune" else "chrIV holdout"),
                        **style)
        for lg, mk in LEGACY_MARK.items():
            pts = sorted((int(r["round"]), f(r[key])) for r in legacy
                         if r["campaign"] == lg and r["ref"] == ref and r["set"] == "tune")
            if pts and key != "nuc_rel_r0":
                ax.plot([x for x, _ in pts], [y for _, y in pts], color="#8a8a86", marker=mk, ls="none",
                        lw=1, ms=6, label="%s (genome-tuned, same groups/chroms)" % lg)
        if key == "nuc_rel_r0":
            for y in (-10, 10):
                ax.axhline(y, color="#b0aea5", lw=1, ls="--")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("round")
        ax.grid(True, color="#e6e4dc", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    png = os.path.join(ROOT, "tw_summary.png")
    fig.savefig(png, dpi=130)
    plt.close("all")
    print("wrote %s (%d rows) and %s" % (out, len(allrows), png))


if __name__ == "__main__":
    main()
