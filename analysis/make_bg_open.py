"""Background at the `combined_low_count` level -- "unbound, open DNA".

Why
---
The shipped background (inputs/bg_params.pkl, 0.13827 W / 0.13841 C) is NOT genome-average
DNA. `computeLinkers` (abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:910-927) takes two 15 bp
windows per Chereji +1/-1 dyad -- [dyad-88, dyad-73] and [dyad+73, dyad+88] -- i.e. the band
just outside the 147 bp nucleosome core, fitted on the barcode01-only pileup (:1078).

`combined_low_count` (0.24894 W / 0.26468 C) is the scalar 142 of 154 TF states use. It was
fitted over the SAME motif spans as the 12 fitted TFs, and its +/-50 bp profile is FLAT
(flank 0.2613, motif 0.2500, flank 0.2520 -- inputs/all_TFs_1000pealVal_params_pseudo_pm50bp.pkl),
so it has no footprint at all: it measures open promoter DNA at mostly-unbound motif matches.

Setting background to that level makes the background state mean what it is supposed to mean
-- unbound, accessible DNA -- instead of nucleosome-flanking linker.

Writes inputs/bg_params_open.pkl. The two A values are copied BIT-EXACTLY from
combined_low_count and asserted; every other entry is asserted identical to the source
bg pkl. Does NOT touch either source (CLAUDE.md rule 2).

    python make_bg_open.py
"""
import copy
import pickle

import numpy as np

SRC_BG = "inputs/bg_params.pkl"
SRC_TF = "inputs/all_TFs_1000pealVal_params_pseudo.pkl"
OUT = "inputs/bg_params_open.pkl"

bg = pickle.load(open(SRC_BG, "rb"), encoding="latin1")
tf = pickle.load(open(SRC_TF, "rb"), encoding="latin1")
out = copy.deepcopy(bg)

clc = tf["p"]["combined_low_count"]
for ch in ("watson_signal", "crick_signal"):
    old = np.asarray(bg["p"][ch]["A"])
    new = copy.deepcopy(clc[ch]["A"])
    assert np.ravel(new).size == 1, "combined_low_count %s is not a scalar: %r" % (ch, new)
    assert np.ravel(old).size == 1, "background %s is not a scalar: %r" % (ch, old)
    out["p"][ch]["A"] = new
    print("%-14s background %.8f -> %.8f  (x%.3f)"
          % (ch, float(np.ravel(old)[0]), float(np.ravel(new)[0]),
             float(np.ravel(new)[0]) / float(np.ravel(old)[0])))

# the two A values must be bit-identical to combined_low_count
for ch in ("watson_signal", "crick_signal"):
    a = np.ravel(np.asarray(out["p"][ch]["A"], dtype=float))
    b = np.ravel(np.asarray(clc[ch]["A"], dtype=float))
    assert a.shape == b.shape and np.array_equal(a, b), \
        "%s: %r != combined_low_count %r" % (ch, a, b)

# nothing else may differ from the source background pkl
changed = []
assert set(out) == set(bg) == {"p"}, (sorted(out), sorted(bg))
assert set(out["p"]) == set(bg["p"]), (sorted(out["p"]), sorted(bg["p"]))
for ch in bg["p"]:
    assert set(out["p"][ch]) == set(bg["p"][ch]), ch
    for base in bg["p"][ch]:
        a = np.ravel(np.asarray(bg["p"][ch][base], dtype=float))
        c = np.ravel(np.asarray(out["p"][ch][base], dtype=float))
        if a.shape != c.shape or not np.array_equal(a, c):
            changed.append("%s/%s" % (ch, base))
assert changed == ["watson_signal/A", "crick_signal/A"], changed
print("\nonly these entries differ from %s: %s" % (SRC_BG, changed))

# C/G/T stay whatever the source background had (zeros) -- the decoder reads only ['A']
# (robocop.py:631) and only emits at reference-A positions (robocop.py:792).
for ch in ("watson_signal", "crick_signal"):
    for base in ("C", "G", "T"):
        assert float(np.ravel(np.asarray(out["p"][ch][base], dtype=float))[0]) == 0.0, (ch, base)

with open(OUT, "wb") as f:
    pickle.dump(out, f)
print("wrote %s" % OUT)

# read it back and re-verify, so a bad dump cannot pass silently
rb = pickle.load(open(OUT, "rb"))
for ch in ("watson_signal", "crick_signal"):
    a = np.ravel(np.asarray(rb["p"][ch]["A"], dtype=float))
    b = np.ravel(np.asarray(clc[ch]["A"], dtype=float))
    assert np.array_equal(a, b), (ch, a, b)
print("re-read OK: %.8f W / %.8f C"
      % (float(np.ravel(rb["p"]["watson_signal"]["A"])[0]),
         float(np.ravel(rb["p"]["crick_signal"]["A"])[0])))
