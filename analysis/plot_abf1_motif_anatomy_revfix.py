"""Why the sequence layer keeps ABF1 site #4 and rejects site #3 — the motif anatomy.

Top     : sequence logo of the model's OWN Abf1_murphy PWM (from pwm.p). Letter
          height = information content, so tall columns are the ones the PWM insists on.
Middle  : the two genomic 14-mers aligned under the logo, letter-by-letter, each tinted
          by how much that base helps or hurts.
Bottom  : the per-column likelihood ratio p(base|PWM)/p(base|background) on a log axis.
          The model multiplies these 14 numbers together; the product is what decides
          whether ABF1 can be placed there at all.

    python plot_abf1_motif_anatomy.py
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties

PWM_P = "robocop_chrI_seq_maskon/pwm.p"
FASTA = "inputs/SacCer3.fa"
MOTIF = "Abf1_murphy"
OUT = "abf1_motif_anatomy_revfix.png"

# nucleotide colors: identity is carried by the LETTER itself, so color is a
# redundant cue here, not the sole encoding.
NT = {"A": "#2e8b57", "C": "#2a6fb5", "G": "#d2860b", "T": "#c0392b"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e7e6e2"
GOOD, BAD, BAND = "#1a7a44", "#a2322b", "#cfcec9"
IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
FP = FontProperties(family="DejaVu Sans", weight="bold")


def letter_patch(ch, x, y, w, h, color, ax, alpha=1.0):
    """Draw a single glyph stretched into the box (x, y, w, h)."""
    tp = TextPath((0, 0), ch, size=1, prop=FP)
    bb = tp.get_extents()
    if bb.width == 0 or bb.height == 0:
        return
    tr = (Affine2D()
          .translate(-bb.x0, -bb.y0)
          .scale(w / bb.width, h / bb.height)
          .translate(x, y))
    ax.add_patch(PathPatch(tr.transform_path(tp), fc=color, ec="none",
                           alpha=alpha, zorder=4))


def main():
    pwm = pickle.load(open(PWM_P, "rb"))
    A = pwm[MOTIF][:4, :]
    bg = np.asarray(pwm["background"]).ravel()[:4]
    L = A.shape[1]

    seq, name, buf = {}, None, []
    for line in open(FASTA):
        if line.startswith(">"):
            if name:
                seq[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    seq[name] = "".join(buf)
    chrI = seq["chrI"].upper()

    # bed start + 1 is where the 14-mer actually aligns to the forward PWM
    sites = [("site #3  —  upstream of ERV46   chrI:61163-61177", chrI[61164:61164 + L],
              "STILL REJECTED after the reverse-complement fix   (0.937 fiber-only -> 0.032 with seq;  was 0.922 -> 0.032)"),
             ("site #4  —  downstream of ERV46  chrI:62657-62671", chrI[62658:62658 + L],
              "STILL KEPT   (0.837 fiber-only -> 0.9997 with seq;  was 0.708 -> 0.9998)")]

    ic = np.array([2 + np.sum((A[:, j] / A[:, j].sum())
                              * np.log2(A[:, j] / A[:, j].sum() + 1e-12)) for j in range(L)])
    SPACER = list(range(5, 10))          # cols 6-10, the canonical NNNNN spacer

    fig = plt.figure(figsize=(13.6, 9.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1.0, 1.0, 2.6], hspace=0.42)

    # ---------------- 1. the PWM as a logo ---------------------------------
    ax = fig.add_subplot(gs[0])
    for j in range(L):
        col = A[:, j] / A[:, j].sum()
        y = 0.0
        for i in np.argsort(col):                     # small letters first
            h = col[i] * ic[j]
            if h > 0.01:
                letter_patch("ACGT"[i], j + 0.06, y, 0.88, h, NT["ACGT"[i]], ax)
            y += h
    ax.axvspan(SPACER[0], SPACER[-1] + 1, color=BAND, alpha=0.5, zorder=0)
    ax.text((SPACER[0] + SPACER[-1] + 1) / 2, 2.06, "SPACER  (canonical NNNNN)",
            ha="center", va="bottom", fontsize=10.5, color=MUTED, fontweight="bold")
    for a, b, lab in [(0, 5, "half-site 1"), (10, 14, "half-site 2")]:
        ax.text((a + b) / 2, 2.06, lab + "  (the part ABF1 grips)", ha="center",
                va="bottom", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlim(0, L); ax.set_ylim(0, 2.35)
    ax.set_ylabel("information\n(bits)", fontsize=10)
    ax.set_xticks(np.arange(L) + 0.5); ax.set_xticklabels(range(1, L + 1), fontsize=9)
    ax.set_yticks([0, 1, 2]); ax.tick_params(length=0, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("The model's ABF1 motif (Abf1_murphy PWM) — tall letters = positions the PWM insists on",
                 fontsize=12.5, fontweight="bold", loc="left", pad=20)

    # ---------------- 2-3. the two genomic sequences -----------------------
    lrs = {}
    for row, (title, s, verdict) in enumerate(sites):
        ax = fig.add_subplot(gs[1 + row])
        lr = np.array([A[IDX[c], j] / bg[IDX[c]] for j, c in enumerate(s)])
        lrs[title] = lr
        ax.axvspan(SPACER[0], SPACER[-1] + 1, color=BAND, alpha=0.5, zorder=0)
        for j, c in enumerate(s):
            helps = lr[j] >= 1.0
            letter_patch(c, j + 0.10, 0.12, 0.80, 0.74,
                         GOOD if helps else BAD, ax, alpha=0.95 if helps else 0.85)
        ax.set_xlim(0, L); ax.set_ylim(0, 1.0)
        ax.set_yticks([]); ax.set_xticks([])
        for s_ in ("top", "right", "left", "bottom"): ax.spines[s_].set_visible(False)
        ax.set_title("%s          %s" % (title, verdict),
                     fontsize=11, fontweight="bold", loc="left", pad=4,
                     color=BAD if row == 0 else GOOD)

    # ---------------- 4. per-column likelihood ratio -----------------------
    ax = fig.add_subplot(gs[3])
    ax.axvspan(SPACER[0], SPACER[-1] + 1, color=BAND, alpha=0.5, zorder=0)
    ax.axhline(0.0, color=INK, lw=1.2, zorder=5)
    w = 0.38
    for k, (title, lr) in enumerate(lrs.items()):
        col = BAD if k == 0 else GOOD
        lab = ("site #3 (upstream, rejected)" if k == 0 else "site #4 (downstream, kept)")
        ax.bar(np.arange(L) + 0.5 + (k - 0.5) * w, np.log10(lr), width=w * 0.92,
               color=col, alpha=0.9, zorder=3, label=lab)
    ax.set_ylim(-1.15, 0.95); ax.set_xlim(0, L)
    ax.set_xticks(np.arange(L) + 0.5)
    ax.set_xticklabels(["%d" % (j + 1) for j in range(L)], fontsize=9)
    ax.set_xlabel("motif position", fontsize=10.5, labelpad=2)
    ax.set_ylabel("evidence for ABF1\nfrom this base, log₁₀(LR)\n▲ helps      ▼ hurts",
                  fontsize=9.5)
    ax.set_yticks([-1, -0.5, 0, 0.5])
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.tick_params(length=0, labelsize=9)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))

    tot3 = float(np.prod(lrs[sites[0][0]])); tot4 = float(np.prod(lrs[sites[1][0]]))
    sp3 = float(np.prod(lrs[sites[0][0]][SPACER])); sp4 = float(np.prod(lrs[sites[1][0]][SPACER]))
    half = [j for j in range(L) if j not in SPACER]
    h3 = float(np.prod(lrs[sites[0][0]][half])); h4 = float(np.prod(lrs[sites[1][0]][half]))
    fig.text(0.5, 0.012,
             "add the 14 bars up  →   whole motif:  #3 = %.0f  vs  #4 = %.0f  (#3 loses %.0f×)        "
             "half-sites only:  #3 = %.0f  vs  #4 = %.0f  (#3 WINS)        "
             "spacer only:  #3 = %.4f  vs  #4 = %.3f  (#3 loses %.0f×)"
             % (tot3, tot4, tot4 / tot3, h3, h4, sp3, sp4, sp4 / sp3),
             ha="center", va="bottom", fontsize=10, color=INK,
             bbox=dict(boxstyle="round,pad=0.5", fc="#fbfbf9", ec="#c9c8c3"))

    fig.suptitle("Why site #3 is STILL rejected after the reverse-complement fix",
                 fontsize=15, fontweight="bold", y=1.028)
    fig.text(0.5, 0.965,
             "The fix corrected the REVERSE state block; both of these motifs are FORWARD, so every number below is bit-for-bit identical before and after.\n"
             "The cause is unchanged — both sites have near-perfect half-sites, and the whole difference is the 5 spacer bases that ABF1 does not read.",
             ha="center", va="top", fontsize=10.2, color=MUTED, linespacing=1.5)
    plt.tight_layout(rect=[0, 0.045, 1, 0.90])
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)
    for (title, s, _), lr in zip(sites, lrs.values()):
        print("\n%s\n  %s" % (title, "  ".join(s)))
        print("  " + "  ".join("%s" % ("+" if v >= 1 else "-") for v in lr))
        print("  total LR %.1f   half-sites %.1f   spacer %.5f"
              % (np.prod(lr), np.prod(lr[half]), np.prod(lr[SPACER])))


if __name__ == "__main__":
    main()
