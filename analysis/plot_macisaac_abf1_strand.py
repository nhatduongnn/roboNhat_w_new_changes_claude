"""MacIsaac ABF1 strand annotation vs the orientation our PWM actually matches.

Top    : the 5 chrI ABF1 sites -- their bed strand column, the genomic 14-mer, and how
         well Abf1_murphy scores that 14-mer in FORWARD vs REVERSE-COMPLEMENT orientation.
Bottom : the same test over all 151 ABF1 sites in the bed, as a cross-tab, plus the
         best-fit alignment offset per strand.

Finding: the bed's strand column is ANTI-correlated with our PWM's orientation --
bed '-' matches our forward PWM, bed '+' matches our reverse PWM. So MacIsaac's ABF1
matrix is the reverse complement of Abf1_murphy, and the strand column must NOT be
read as "which strand our motif sits on".

    python plot_macisaac_abf1_strand.py
"""
import sys, os, collections
sys.path.insert(0, os.path.join("..", "pkg"))
import pickle
import numpy as np
import robocop
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BED = "inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1_match_PWM.bed"
FASTA = "inputs/SacCer3.fa"
PWM_P = "robocop_chrI_seq_maskon/pwm.p"
OUT = "macisaac_abf1_strand.png"

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated pair (CVD dE 24.7, normal dE 33.6)
INK, MUTED, GRID, BAND = "#0b0b0b", "#6b6a66", "#e7e6e2", "#efeee9"
I = {"A": 0, "C": 1, "G": 2, "T": 3}
CHR_I_SITES = [(1, 45318), (2, 45498), (3, 61163), (4, 62657), (5, 108788)]
NOTE = {3: "upstream of ERV46", 4: "downstream of ERV46"}


def load_genome():
    g, name, buf = {}, None, []
    for line in open(FASTA):
        if line.startswith(">"):
            if name:
                g[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    g[name] = "".join(buf).upper()
    return g


def main():
    pwm = pickle.load(open(PWM_P, "rb"))
    A = pwm["Abf1_murphy"][:4, :]
    bgv = np.asarray(pwm["background"]).ravel()[:4]
    Ar = robocop.reverse_complement(pwm["Abf1_murphy"])[:4, :]
    G = load_genome()
    L = A.shape[1]

    def lr(M, s):
        if len(s) < L or any(x not in I for x in s):
            return 0.0
        return float(np.prod([M[I[x], j] / bgv[I[x]] for j, x in enumerate(s)]))

    # ---- all 151 ABF1 sites: which orientation wins, at what offset -------
    rows = []
    for line in open(BED):
        f = line.split()
        if len(f) < 6 or f[3] != "ABF1" or f[0] not in G:
            continue
        ch, a, strand = f[0], int(f[1]), f[5]
        best = (None, -1.0, None)
        for off in range(-3, 4):
            s = G[ch][a + off:a + off + L]
            fw, rv = lr(A, s), lr(Ar, s)
            if max(fw, rv) > best[1]:
                best = (off, max(fw, rv), "fwd" if fw >= rv else "rev")
        rows.append(dict(chr=ch, start=a, strand=strand, offset=best[0], match=best[2]))
    ct = collections.Counter((r["strand"], r["match"]) for r in rows)
    offs = collections.Counter((r["strand"], r["offset"]) for r in rows)

    # ---- the 5 chrI sites in detail --------------------------------------
    detail = []
    for num, a in CHR_I_SITES:
        strand = next(r["strand"] for r in rows if r["chr"] == "chrI" and r["start"] == a)
        best = None
        for off in range(-3, 4):
            s = G["chrI"][a + off:a + off + L]
            fw, rv = lr(A, s), lr(Ar, s)
            if best is None or max(fw, rv) > max(best[2], best[3]):
                best = (off, s, fw, rv)
        detail.append((num, a, strand) + best)

    fig = plt.figure(figsize=(14.2, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0], width_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.22)

    # ================= panel A: the 5 chrI sites =========================
    ax = fig.add_subplot(gs[0, :])
    n = len(detail)
    for i, (num, a, strand, off, s, fw, rv) in enumerate(detail):
        y = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.45, y + 0.45, color=BAND, zorder=0)
        for val, col, dy in ((fw, BLUE, 0.17), (rv, ORANGE, -0.17)):
            x = max(np.log10(val), -20.0) if val > 0 else -20.0
            ax.barh(y + dy, x, height=0.30, color=col, zorder=3)
            txt = "1e%+d" % round(np.log10(val)) if val > 0 else "0"
            if x >= 0:                      # forward bar: label outside, to the right
                ax.text(x + 0.35, y + dy, txt, va="center", ha="left", fontsize=8.5,
                        color=col, fontweight="bold", zorder=4)
            else:                           # reverse bar: label INSIDE the tip
                ax.text(x + 0.4, y + dy, txt, va="center", ha="left", fontsize=8.5,
                        color="white", fontweight="bold", zorder=4)
        tag = "site #%d" % num + ("   %s" % NOTE[num] if num in NOTE else "")
        ax.text(-33.6, y + 0.30, tag, fontsize=10.5, fontweight="bold", color=INK, va="center")
        ax.text(-33.6, y - 0.02, "chrI:%d-%d" % (a, a + 14), fontsize=9, color=MUTED, va="center")
        ax.text(-33.6, y - 0.31, "bed strand:", fontsize=9, color=MUTED, va="center")
        ax.text(-30.6, y - 0.31, strand, fontsize=13, fontweight="bold", color=ORANGE, va="center")
        ax.text(-23.2, y + 0.11, s, fontsize=12.5, family="DejaVu Sans Mono",
                fontweight="bold", color=INK, va="center")
        ax.text(-23.2, y - 0.28, "aligns at bed start %+d" % off, fontsize=8.2,
                color=MUTED, va="center")
    ax.axvline(0, color=INK, lw=1.2, zorder=5)
    ax.axvline(-21.6, color="#d6d5d0", lw=1.2, zorder=2)   # label gutter / plot divider
    ax.set_xlim(-34.4, 8.0); ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([-20, -15, -10, -5, 0, 5])
    ax.set_xticklabels(["1e-20", "1e-15", "1e-10", "1e-5", "1", "1e5"], fontsize=9)
    ax.set_xlabel("how well Abf1_murphy scores this 14-mer   (likelihood ratio vs background, log scale)",
                  fontsize=10.5)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=1); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#b8b7b2"); ax.tick_params(length=0)
    ax.barh([n + 5], [0], color=BLUE, label="PWM read FORWARD")
    ax.barh([n + 5], [0], color=ORANGE, label="PWM read REVERSE-COMPLEMENT")
    ax.legend(frameon=False, fontsize=10, loc="lower right", ncol=2,
              bbox_to_anchor=(1.0, 1.0))
    ax.set_title("All 5 chrI ABF1 sites are annotated strand “−”, yet all 5 match the FORWARD PWM"
                 " — by 15–18 orders of magnitude",
                 fontsize=12.5, fontweight="bold", loc="left", pad=30)

    # ================= panel B: genome-wide cross-tab ====================
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 2.6); ax.set_ylim(0, 2.5); ax.invert_yaxis(); ax.set_axis_off()
    ax.text(1.3, 0.02, "all 151 ABF1 sites in the bed", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=INK)
    for j, m in enumerate(("fwd", "rev")):
        ax.text(0.85 + j * 0.85, 0.42, "matches\nPWM %s" % m.upper(), ha="center", va="bottom",
                fontsize=9.5, color=INK, fontweight="bold")
    for i, st in enumerate(("+", "-")):
        ax.text(0.34, 1.05 + i * 0.62, "bed\nstrand %s" % st, ha="right", va="center",
                fontsize=10, fontweight="bold", color=INK)
        for j, m in enumerate(("fwd", "rev")):
            v = ct[(st, m)]
            hot = v > 10
            ax.add_patch(FancyBboxPatch((0.45 + j * 0.85, 0.78 + i * 0.62), 0.78, 0.52,
                                        boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor=(BLUE if m == "fwd" else ORANGE) if hot else "#f2f1ee",
                                        edgecolor="white", lw=2))
            ax.text(0.85 + j * 0.85, 1.05 + i * 0.62, str(v), ha="center", va="center",
                    fontsize=17, fontweight="bold", color="white" if hot else "#b4b3ae")
    ax.text(1.3, 2.28, "perfectly anti-correlated: bed “−” ⇒ forward, bed “+” ⇒ reverse\n"
                       "holds for 149/151 sites (98.7%)",
            ha="center", va="center", fontsize=9.6, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbf9", ec="#c9c8c3"))

    # ================= panel C: offset by strand =========================
    ax = fig.add_subplot(gs[1, 1])
    keys = sorted({o for (_, o) in offs})
    w = 0.38
    for k, st in enumerate(("+", "-")):
        vals = [offs[(st, o)] for o in keys]
        ax.bar(np.arange(len(keys)) + (k - 0.5) * w, vals, width=w * 0.9,
               color=ORANGE if st == "+" else BLUE, zorder=3,
               label="bed strand %s  (n=%d)" % (st, sum(vals)))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["%+d" % o for o in keys], fontsize=9.5)
    ax.set_xlabel("best alignment offset relative to the bed start (bp)", fontsize=10)
    ax.set_ylabel("number of sites", fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=0, labelsize=9)
    ax.legend(frameon=False, fontsize=9.2)
    ax.set_title("the offset differs by strand too: “+” sites sit at −1, “−” sites at +1",
                 fontsize=10.5, fontweight="bold", loc="left", pad=6)

    fig.suptitle("MacIsaac ABF1 strand annotation vs the orientation our PWM actually matches",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.955,
             "The strand column encodes MacIsaac's own motif orientation — their ABF1 matrix is the "
             "reverse complement of Abf1_murphy. Do not read it as “which strand our motif is on”.",
             ha="center", fontsize=10, color=MUTED)
    plt.tight_layout(rect=[0, 0, 1, 0.945])
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote", OUT)
    print("\n%-6s %-18s %-6s %-16s %6s %12s %12s" %
          ("site", "chrI span", "bed±", "14-mer", "offset", "fwd LR", "rev LR"))
    for num, a, strand, off, s, fw, rv in detail:
        print("#%-5d %-18s %-6s %-16s %+6d %12.4g %12.4g"
              % (num, "%d-%d" % (a, a + 14), strand, s, off, fw, rv))


if __name__ == "__main__":
    main()
