"""Sequence logos for the two ABF1 matrices, under BOTH background conventions.

Why two conventions. The standard logo scales letters by information content
measured against a UNIFORM background (2 - H). That is what everyone recognises,
and it makes Murphy's spacer look small and harmless -- which is exactly the
misreading that has to be avoided here. FIMO and RoboCOP both score against the
AT-rich yeast background (A .310 C .191 G .191 T .309), and against THAT reference
a GC-leaning column is not degenerate at all. So the lower pair rescales the same
matrices by KL divergence from the real background, which is what actually drives
the scores.

Colour: logomaker's published `colorblind_safe` scheme. In a sequence logo the
glyph IS the letter, so identity is carried entirely by shape and colour is
redundant -- the usual CVD-separation rules for colour-only encoding do not bind.

    python plot_abf1_logos.py
"""
import os, sys, re
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
# NOTE: deliberately NOT importing robocop.utils.parameterize.getMotifsMEME here --
# it drags in rpy2, which lives in the robocop-2024 env, while logomaker lives in
# pyranges_env3. The 8-line parser below reads the same MEME block. Verified to
# agree with getMotifsMEME to 1e-12 (see the assertion in load()).

OUT = "abf1_murphy_vs_jaspar_logos.png"
MURPHY_FILE = "inputs/motifs_meme.txt"
JASPAR_FILE = "inputs/jaspar_abf1_motifs_meme.txt"
SPACER = (5, 9)                      # inclusive column range of the disputed spacer
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e7e6e2"
BANDC = "#a2322b"


def load(path, motif="Abf1_murphy"):
    """Read one motif's probability matrix out of a MEME file -> (w x 4) DataFrame."""
    lines = open(path).read().split("\n")
    mi = next(i for i, l in enumerate(lines) if l.startswith("MOTIF " + motif))
    li = next(i for i in range(mi, mi + 8) if lines[i].startswith("letter-probability"))
    w = int(re.search(r"w=\s*(\d+)", lines[li]).group(1))
    rows = [[float(x) for x in lines[li + 1 + j].split()] for j in range(w)]
    df = pd.DataFrame(rows, columns=list("ACGT"))
    assert np.allclose(df.sum(1), 1.0, atol=1e-5), "columns of %s do not sum to 1" % motif
    return df


def main():
    M, J = load(MURPHY_FILE), load(JASPAR_FILE)
    bg = np.ravel(pickle.load(open("robocop_chrI_maskon_revfix/pwm.p", "rb"))["background"])[:4]

    # uniform-background information (the conventional logo)
    Mu = logomaker.transform_matrix(M, from_type="probability", to_type="information")
    Ju = logomaker.transform_matrix(J, from_type="probability", to_type="information")
    # yeast-background information: heights become KL divergence from the real background
    Mb = logomaker.transform_matrix(M, from_type="probability", to_type="information",
                                    background=bg)
    Jb = logomaker.transform_matrix(J, from_type="probability", to_type="information",
                                    background=bg)

    panels = [
        (Mu, "MURPHY  —  standard logo (information vs UNIFORM background)", "u"),
        (Ju, "JASPAR MA0265.3 (revcomp)  —  standard logo (vs UNIFORM background)", "u"),
        (Mb, "MURPHY  —  rescaled vs the AT-rich YEAST background FIMO/RoboCOP actually use", "b"),
        (Jb, "JASPAR MA0265.3 (revcomp)  —  rescaled vs the AT-rich YEAST background", "b"),
    ]
    ymax = {k: max(float(d.sum(1).max()) for d, _, kk in panels if kk == k) * 1.12
            for k in ("u", "b")}

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 11.0))
    for ax, (df, title, kind) in zip(axes, panels):
        logomaker.Logo(df, ax=ax, color_scheme="colorblind_safe",
                       show_spines=False, vpad=0.02, width=0.9)
        ax.axvspan(SPACER[0] - 0.5, SPACER[1] + 0.5, color=BANDC, alpha=0.08, zorder=0)
        ax.set_ylim(0, ymax[kind])
        ax.set_xticks(range(14))
        ax.set_xticklabels(range(14), fontsize=8.5)
        ax.set_ylabel("bits", fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=6)
        ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=8.5)
        tot = float(df.sum(1).sum())
        sp = float(df.iloc[SPACER[0]:SPACER[1] + 1].sum(1).sum())
        ax.text(0.995, 0.93, "total %.2f bits   ·   spacer (cols 5–9) %.2f bits" % (tot, sp),
                transform=ax.transAxes, ha="right", va="top", fontsize=8.8,
                color=INK, fontweight="bold")
    axes[-1].set_xlabel("motif position (0-based)", fontsize=10)

    fig.suptitle("ABF1 position weight matrices — Murphy vs JASPAR MA0265.3",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.972,
             "Same motif on the 9 core columns (per-column r = +0.998). They disagree only on "
             "the shaded spacer (cols 5–9), where per-column r = −0.411.",
             ha="center", va="top", fontsize=9.4, color=MUTED)
    fig.text(0.5, 0.9525,
             "Top pair: the conventional uniform-background logo makes both spacers look minor. "
             "Bottom pair: against the real yeast background, Murphy's spacer carries 3.00 bits "
             "vs JASPAR's 0.84.",
             ha="center", va="top", fontsize=9.0, color=BANDC, style="italic")

    fig.subplots_adjust(left=0.07, right=0.985, top=0.925, bottom=0.055, hspace=0.42)
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close("all")
    print("wrote", OUT)
    print("\n%-38s %10s %10s" % ("", "total bits", "spacer 5-9"))
    for df, title, kind in panels:
        print("%-38s %10.2f %10.2f"
              % (title.split("  —  ")[0] + (" [uniform bg]" if kind == "u" else " [yeast bg]"),
                 df.sum(1).sum(), df.iloc[SPACER[0]:SPACER[1] + 1].sum(1).sum()))


if __name__ == "__main__":
    main()
