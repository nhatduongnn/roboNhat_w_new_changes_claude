"""Crop the single-fiber figures into slide-sized panels for the Nov 3 deck (slides 5, 6, 7, 7b).

Panel edges are the matplotlib frame rows found by scanning each figure for dark full-width lines
(frames of erv46 full composite: 1484-2039 raw clustered, 2757-3312 P(nuc), 3394-3802 P(ABF1);
compare_abf1_3way: 786-1357 before, 1440-2010 local, 2094-2664 global). Titles are cut off: the slide labels
each row in HTML. Full figures are copied (downscaled) for the click-to-enlarge view.
Outputs go to img/ next to this file and are published as `files` beside the deck page.

    python make_images.py
"""
import os
from PIL import Image

H = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(H, "..", "..", "analysis", "singlefiber", "figures")
OUT = os.path.join(H, "img")
os.makedirs(OUT, exist_ok=True)

ERV = os.path.join(FIG, "erv46_2026-11-02", "filtnordna", "permol_composite_full_erv46_5to3_nordna.png")
SWEEP = os.path.join(FIG, "abf1_prior_sweep_2026-11-03", "abf1_prior_sweep_summary.png")
CMP = os.path.join(FIG, "erv46_localprior_2026-11-03", "compare_abf1_3way_erv46.png")


def save(im, name, width=None):
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im = im.convert("RGB")
    p = os.path.join(OUT, name)
    im.save(p, optimize=True)
    print("wrote", name, im.size, os.path.getsize(p) // 1024, "KB")


erv = Image.open(ERV)
assert erv.size == (2800, 4079), erv.size
for name, (y0, y1) in {"erv46_genes.png": (110, 262), "erv46_raw.png": (1482, 2042),
                       "erv46_nuc.png": (2755, 3315), "erv46_abf1.png": (3392, 3905)}.items():
    save(erv.crop((0, y0, 2800, y1)), name, 2200)
save(erv, "erv46_full.png", 2000)

sw = Image.open(SWEEP)
assert sw.size == (1950, 570), sw.size
save(sw.crop((0, 48, 1950, 570)), "abf1_sweep_summary.png")   # drop the suptitle; the slide has its own

cmp_ = Image.open(CMP)
assert cmp_.size == (2800, 3800), cmp_.size
for name, (y0, y1) in {"cmp_before.png": (785, 1360), "cmp_local.png": (1438, 2013),
                       "cmp_global.png": (2092, 2765)}.items():
    save(cmp_.crop((0, y0, 2800, y1)), name, 2200)
save(cmp_.crop((0, 2760, 2560, 3480)), "cmp_scatter.png", 2400)   # right of x=2560 is empty
save(cmp_, "cmp_full.png", 2000)
