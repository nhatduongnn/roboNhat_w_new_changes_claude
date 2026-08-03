"""Grouped bar chart: nucleosome recall (Chereji + Brogaard) and ABF1 recall
across the 5 full-chrI runs. Reads chrI_5run_metrics.json, writes chrI_5run_recall.png."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("chrI_5run_metrics.json") as fh:
    M = json.load(fh)

labels = [e["label"] for e in M]
def col(key):  # recall as float, 0 if missing
    return [float(e[key]) if e.get(key) is not None else 0.0 for e in M]

chereji  = col("chereji_recall")
brogaard = col("brogaard_recall")
abf1     = col("abf1_recall")

# n_ref (constant across runs; take first non-null)
def nref(key):
    for e in M:
        if e.get(key) is not None:
            return e[key]
    return "?"
n_ch, n_br, n_ab = nref("chereji_nref"), nref("brogaard_nref"), nref("abf1_nref")

# categorical palette slots 1-3 (validated default theme): blue / orange / aqua
C = {"chereji": "#2a78d6", "brogaard": "#eb6834", "abf1": "#1baf7a"}
series = [
    (f"Nucleosome recall vs Chereji +1/-1  (n={n_ch})",   chereji,  C["chereji"]),
    (f"Nucleosome recall vs Brogaard top-2000  (n={n_br})", brogaard, C["brogaard"]),
    (f"ABF1 (TF) recall vs MacIsaac  (n={n_ab})",          abf1,     C["abf1"]),
]

x = np.arange(len(labels))
w = 0.26
fig, ax = plt.subplots(figsize=(13, 6.6))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

for i, (name, vals, color) in enumerate(series):
    off = (i - 1) * w
    bars = ax.bar(x + off, vals, w * 0.92, label=name, color=color,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, color="#0b0b0b", zorder=4)

ax.set_ylim(0, 1.08)
ax.set_ylabel("recall  (fraction of reference sites recovered, tol=20 bp)", fontsize=10.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_title("Full chrI — nucleosome & TF recall  (2x2: layers x mask)",
             fontsize=13.5, fontweight="bold", pad=14)
ax.text(0.5, 1.015,
        "Nucleosome recall is unaffected by the TF mask (as expected); the mask only gates TF states. "
        "ABF1 n=5 on chrI, so its recall is coarse.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.7, color="#52514e")

ax.yaxis.grid(True, color="#e7e6e2", linewidth=1, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#b8b7b2")
ax.tick_params(length=0)
ax.legend(frameon=False, fontsize=9.3, loc="upper center",
          bbox_to_anchor=(0.5, -0.09), ncol=3)

plt.tight_layout()
plt.savefig("chrI_5run_recall.png", dpi=150, bbox_inches="tight")
print("written: chrI_5run_recall.png")
