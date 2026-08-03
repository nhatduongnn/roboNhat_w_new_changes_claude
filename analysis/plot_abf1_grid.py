"""5 MacIsaac ABF1 sites x 4 models recovery matrix (midpoint-to-midpoint).
Predicted anchor = footprint CENTER (Option A). A site is RECOVERED if the
predicted footprint center is within 20 bp of the MacIsaac motif midpoint.
Reads abf1_sites_per_run_centered.txt so the numbers match the scorer exactly."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

REPORT = "abf1_sites_per_run_centered.txt"
TOL = 20

# ---- parse the report ------------------------------------------------------
sites = []          # (num, center)
models = []         # display label
recall = {}         # label -> "3/5"
cells = {}          # (label, site_num) -> dict(hit=bool, dist=int)

cur = None
with open(REPORT) as fh:
    for line in fh:
        m = re.match(r"\s*#(\d+)\s+chrI:\d+-\d+\s+center=(\d+)", line)
        if m:
            sites.append((int(m.group(1)), int(m.group(2))))
            continue
        m = re.match(r"=== (.+?)\s+\(robocop", line)
        if m:
            cur = m.group(1).strip()
            models.append(cur)
            continue
        m = re.match(r"\s*site #(\d+).*RECOVERED.*dist=(\d+)bp", line)
        if m:
            cells[(cur, int(m.group(1)))] = dict(hit=True, dist=int(m.group(2)))
            continue
        m = re.match(r"\s*site #(\d+).*missed.*?(\d+)bp away", line)
        if m:
            cells[(cur, int(m.group(1)))] = dict(hit=False, dist=int(m.group(2)))
            continue
        m = re.match(r"\s*site #(\d+).*missed.*no ABF1 peaks", line)
        if m:
            cells[(cur, int(m.group(1)))] = dict(hit=False, dist=None)
            continue
        m = re.match(r"\s*recall = (\d+/\d+)", line)
        if m and cur:
            recall[cur] = m.group(1)

# ---- colors: status green ramp (closer = darker) + neutral miss ------------
def hit_fill(d):
    if d <= 2:   return "#0a7d46", "white"     # bullseye
    if d <= 9:   return "#3aa06a", "white"
    return "#9bd3b4", "#0a3d24"                 # within tol but loose
MISS_FILL, MISS_INK = "#eceef1", "#9aa0a6"
INK, MUT = "#1b1f24", "#6b7280"

nS, nM = len(sites), len(models)
fig_w = 2.15 * nM + 2.4
fig_h = 1.02 * nS + 2.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, nM); ax.set_ylim(0, nS + 1.15)
ax.invert_yaxis(); ax.set_axis_off()

# column headers (model + recall)
for j, lab in enumerate(models):
    two = lab.replace(" / ", "\n")
    ax.text(j + 0.5, -0.02, two, ha="center", va="bottom", fontsize=11,
            fontweight="bold", color=INK)
    if lab in recall:
        ax.text(j + 0.5, 0.60, "recall " + recall[lab], ha="center", va="bottom",
                fontsize=10, color=MUT)

# row labels
for i, (num, cen) in enumerate(sites):
    y = i + 1.15 + 0.5
    ax.text(-0.06, y, "Site #%d" % num, ha="right", va="center",
            fontsize=11, fontweight="bold", color=INK)
    ax.text(-0.06, y + 0.30, "chrI:%d" % cen, ha="right", va="center",
            fontsize=8.6, color=MUT)

# cells
for i, (num, cen) in enumerate(sites):
    for j, lab in enumerate(models):
        c = cells.get((lab, num), dict(hit=False, dist=None))
        y0 = i + 1.15
        if c["hit"]:
            fill, ink = hit_fill(c["dist"])
            head, sub = u"✓ %d bp" % c["dist"], "overlap"
        else:
            fill, ink = MISS_FILL, MISS_INK
            if c["dist"] is None:
                head, sub = u"✗", "no call"
            else:
                head, sub = u"✗", "%d bp off" % c["dist"]
        ax.add_patch(FancyBboxPatch((j + 0.04, y0 + 0.06), 0.92, 0.88,
                     boxstyle="round,pad=0,rounding_size=0.06",
                     facecolor=fill, edgecolor="white", linewidth=2, zorder=1))
        ax.text(j + 0.5, y0 + 0.42, head, ha="center", va="center",
                fontsize=15, fontweight="bold", color=ink, zorder=2)
        ax.text(j + 0.5, y0 + 0.70, sub, ha="center", va="center",
                fontsize=9, color=ink, zorder=2)

# legend
lg_y = nS + 1.15 + 0.30
sw = 0.30
items = [("#0a7d46", u"✓ ≤2 bp"), ("#3aa06a", u"✓ 3–9 bp"),
         ("#9bd3b4", u"✓ 10–20 bp"), (MISS_FILL, u"✗ missed (>20 bp)")]
x = 0.02
for fill, txt in items:
    ax.add_patch(Rectangle((x, lg_y), sw, 0.34, facecolor=fill,
                 edgecolor="white", linewidth=1.5, clip_on=False))
    ax.text(x + sw + 0.06, lg_y + 0.17, txt, va="center", fontsize=9, color=INK)
    x += sw + 1.05 * (len(txt) * 0.11 + 0.5)

fig.suptitle("ABF1 recovery: 5 MacIsaac sites × 4 models  (midpoint-to-midpoint, ≤20 bp = hit)\n"
             "predicted anchor = footprint center (Option A)",
             fontsize=13, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0.02, 0.0, 1, 0.95])
plt.savefig("abf1_recovery_grid.png", dpi=150, bbox_inches="tight")
print("wrote abf1_recovery_grid.png")
# echo parsed table for sanity
for num, cen in sites:
    print("site #%d (%d):" % (num, cen),
          {lab.split(" / ")[-1]: cells.get((lab, num)) for lab in models})
