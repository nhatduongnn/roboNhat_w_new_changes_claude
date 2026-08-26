"""Cap combined_low_count at the background rate.

combined_low_count is a SINGLE scalar (tf_len == 1) that robocop.py broadcasts across the
whole motif block of every TF without an individual Fiber-seq fit -- 141 of the 153 PWMs.
Shipped it sits at 0.2489/0.2647, i.e. 1.8x ABOVE background, so those 141 states are pure
level detectors: they win at any highly-methylated position (linkers, promoters) on
methylation level alone, with no footprint shape to match. Capping it at bg makes them
fiber-NEUTRAL, so the sequence layer decides them instead.

Cap value = the merged-pileup background refit (inputs/bg_params_merged.pkl), not the
shipped single-barcode one.

Writes inputs/all_TFs_1000pealVal_params_pseudo_caplow.pkl. Verifies every other entry is
bit-identical to the source. Does NOT touch the source pkl.
"""
import copy
import pickle

import numpy as np

SRC = "inputs/all_TFs_1000pealVal_params_pseudo.pkl"
BG = "inputs/bg_params_merged.pkl"
OUT = "inputs/all_TFs_1000pealVal_params_pseudo_caplow.pkl"

d = pickle.load(open(SRC, "rb"), encoding="latin1")
bg = pickle.load(open(BG, "rb"))
out = copy.deepcopy(d)

for ch in ("watson_signal", "crick_signal"):
    b = float(np.ravel(bg["p"][ch]["A"])[0])
    old = np.ravel(out["p"]["combined_low_count"][ch]["A"]).astype(float)
    new = np.minimum(old, b)
    out["p"]["combined_low_count"][ch]["A"] = new
    print("%-14s low_count %.6f -> %.6f   (bg %.6f)" % (ch, old[0], new[0], b))

# every other entry must be untouched
changed = []
for k in d["p"]:
    for ch in ("watson_signal", "crick_signal"):
        for base in ("A", "C", "G", "T"):
            a = np.ravel(np.asarray(d["p"][k][ch][base], dtype=float))
            c = np.ravel(np.asarray(out["p"][k][ch][base], dtype=float))
            if a.shape != c.shape or not np.array_equal(a, c):
                changed.append("%s/%s/%s" % (k, ch, base))
assert changed == ["combined_low_count/watson_signal/A",
                   "combined_low_count/crick_signal/A"], changed
assert d["mu"] == out["mu"] and d["phi"] == out["phi"]
print("\nonly these entries differ from the source: %s" % changed)

with open(OUT, "wb") as f:
    pickle.dump(out, f)
print("wrote %s" % OUT)
