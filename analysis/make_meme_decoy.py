"""Append a sequence-neutral ABF1 decoy motif to a MEME file (decoy campaigns, 2026-09-21).

Plan: ~/.claude/plans/okay-so-i-want-humming-hearth.md, Build step 2.

The decoy `Zz_decoy_abf1` has L columns (14 for the plain model, 23 for the 7/2 model), every
column EQUAL to the background row of the source trainDir's pwm.p, written at full precision
(%.17g, which round-trips a float64 exactly). Consequences:
  - sequence emission of the decoy block = background at every position -> sequence LR exactly 1;
  - calculateKD(decoy) = 10**sum(log10 bg[i] - log10 bg[i]) = 10**0 = 1.0 exactly.
The name sorts after every real motif ('Zms1_badis') and before 'unknown', so dshared['tfs']
(sorted, robocop.py:83-84) keeps every existing motif's index and state block.

THE SOURCE FILE IS NEVER MODIFIED (CLAUDE.md rule 2). The output is refused if it exists.
Gate: stripping the appended block from the output recovers the source byte-for-byte, and
re-parsing the output with getMotifsMEME gives the decoy exactly equal to the background row.

    python make_meme_decoy.py --src inputs/motifs_meme.txt      --width 14 --out inputs/motifs_meme_decoy14.txt
    python make_meme_decoy.py --src inputs/motifs_meme_wide.txt --width 23 --out inputs/motifs_meme_wide_decoy23.txt
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DECOY = "Zz_decoy_abf1"
BG_TRAINDIR = "robocop_train_fiberonly"     # its pwm.p background = computeBackground(SacCer3.fa)


def get_motifs_meme(path):
    """Verbatim logic of pkg/robocop/utils/parameterize.py getMotifsMEME (no robocop import, so
    no embedded R)."""
    motifLines = [x for x in open(path).readlines() if x != "\n"]
    found, d = 0, {}
    for i in motifLines:
        l = i.strip()
        if l[:4] == "URL ":
            continue
        if l[:5] == "MOTIF":
            found = l.split()[1]
            d[found] = [[], [], [], [], []]
            continue
        if found != 0 and l[:6] != "letter":
            l = l.split()
            for j in range(4):
                d[found][j] = d[found][j] + [float(l[j])]
            d[found][4] = d[found][4] + [0]
    return {k: np.array(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, required=True)
    a = ap.parse_args()
    src = os.path.join(HERE, a.src) if not os.path.isabs(a.src) else a.src
    out = os.path.join(HERE, a.out) if not os.path.isabs(a.out) else a.out
    if os.path.exists(out):
        sys.exit("%s already exists; refusing to overwrite (rule 2)" % out)
    bg = np.ravel(pickle.load(open(os.path.join(HERE, BG_TRAINDIR, "pwm.p"), "rb"))["background"])
    bg4 = [float(x) for x in bg[:4]]
    print("background row (%s/pwm.p): %s" % (BG_TRAINDIR, [repr(x) for x in bg4]))
    text = open(src).read()
    if not text.endswith("\n\n"):
        sys.exit("%s does not end with a blank line; the append format assumes it" % src)
    if ("MOTIF %s" % DECOY) in text:
        sys.exit("%s already contains %s" % (src, DECOY))
    row = "".join("  %.17g\t" % v for v in bg4)
    block = ("MOTIF %s %s\n\nletter-probability matrix: alength= 4 w= %d nsites= 20 E= 0\n"
             % (DECOY, DECOY, a.width)) + "".join(row + "\n" for _ in range(a.width)) + "\n"
    new = text + block
    # ---- gates, before writing ----
    assert new[:len(text)] == text and new[len(text):] == block, "append is not a pure suffix"
    assert new[:-len(block)] == text, "stripping the decoy block does NOT recover the source"
    tmp = out + ".tmp"
    open(tmp, "w").write(new)
    m_src, m_new = get_motifs_meme(src), get_motifs_meme(tmp)
    assert list(m_new)[:-1] == list(m_src) and list(m_new)[-1] == DECOY, "motif order changed"
    assert all(np.array_equal(m_src[k], m_new[k]) for k in m_src), "a source motif changed on re-parse"
    dec = m_new[DECOY]
    assert dec.shape == (5, a.width), dec.shape
    want = np.tile(np.asarray(bg, dtype=float).reshape(5, 1), (1, a.width))
    assert np.array_equal(dec, want), "decoy re-parsed != background bit-for-bit"
    assert sorted(list(m_new) + ["unknown"])[-2:] == [DECOY, "unknown"], "decoy does not sort before unknown"
    os.rename(tmp, out)
    print("wrote %s: %s + %s (%d columns = background, %%.17g)" % (a.out, os.path.basename(src), DECOY, a.width))
    print("gate ok: stripping the appended block recovers %s byte-for-byte" % os.path.basename(src))
    print("gate ok: re-parsed %d source motifs unchanged; decoy == background bit-for-bit; sorts "
          "after %s and before unknown" % (len(m_src), sorted(m_src)[-1]))


if __name__ == "__main__":
    main()
