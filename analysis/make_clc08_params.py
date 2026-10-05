"""`combined_low_count` at a plausible bound-footprint depth (default 0.08).

Why
---
`combined_low_count` is the single scalar that 142 of 154 TF states use: the 141 motifs with no
individual Fiber-seq fit, plus `unknown` (which is a `tf_prob` key absent from the pkl, so it
takes the `else` branch at robocop.py:671-676 and inherits this value automatically -- it needs
no separate handling).

Shipped it sits at 0.24894 W / 0.26468 C, i.e. 1.80x ABOVE background, because it was fitted by
pooling 62 low-count TFs' motif columns at mostly-UNBOUND Rossi matches; its +/-50 bp profile is
flat (0.2613 / 0.2500 / 0.2520), so it has no footprint at all and acts as a level detector for
open promoter DNA. A state meaning "a TF is bound here" should emit LESS m6A than background.

This sets it to a constant standing for "a bound factor's footprint". 0.08 is a round number
deliberately NOT equal to any fitted TF's mean -- in particular not ABF1's 0.08431, which is what
inputs/all_TFs_1000pealVal_params_pseudo_lowabf1.pkl uses (make_low_abf1.py:52-57). Using ABF1's
own mean would make the generic fallback numerically indistinguishable from the factor the run is
scored against.

Read this alongside inputs/bg_params_open.pkl (background moved to 0.24894/0.26468, the level
combined_low_count used to occupy). Neither file alone is the experiment.

Writes a NEW pkl; the shipped one is never touched (CLAUDE.md rule 2). Asserts that ONLY the two
combined_low_count 'A' entries differ, then re-reads the dump and re-verifies.

    python make_clc08_params.py                 # -> inputs/..._clc08.pkl at 0.08 / 0.08
    python make_clc08_params.py --value 0.06 --out inputs/..._clc06.pkl
"""
import argparse
import copy
import pickle

import numpy as np

SRC = "inputs/all_TFs_1000pealVal_params_pseudo.pkl"
OUT = "inputs/all_TFs_1000pealVal_params_pseudo_clc08.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--value", type=float, default=0.08,
                    help="combined_low_count rate, both strands (default 0.08)")
    ap.add_argument("--crick", type=float, default=None,
                    help="different Crick value; default = --value")
    a = ap.parse_args()
    assert a.out != a.src, "refusing to overwrite the source pkl"
    w_val = a.value
    c_val = a.crick if a.crick is not None else a.value

    d = pickle.load(open(a.src, "rb"), encoding="latin1")
    out = copy.deepcopy(d)

    for ch, val in (("watson_signal", w_val), ("crick_signal", c_val)):
        cur = np.asarray(out["p"]["combined_low_count"][ch]["A"])
        assert np.ravel(cur).size == 1, "combined_low_count %s is not a scalar: %r" % (ch, cur)
        # full_like keeps the container's shape (1,) and dtype, so the broadcast at
        # robocop.py:675-676 behaves exactly as before
        out["p"]["combined_low_count"][ch]["A"] = np.full_like(cur, val, dtype=float)
        print("%-14s combined_low_count %.8f -> %.8f" % (ch, float(np.ravel(cur)[0]), val))

    # nothing else may differ
    assert set(out) == set(d), (sorted(out), sorted(d))
    assert set(out["p"]) == set(d["p"]), "TF key set changed"
    changed = []
    for k in d["p"]:
        for ch in ("watson_signal", "crick_signal"):
            for base in ("A", "C", "G", "T"):
                x = np.ravel(np.asarray(d["p"][k][ch][base], dtype=float))
                y = np.ravel(np.asarray(out["p"][k][ch][base], dtype=float))
                if x.shape != y.shape or not np.array_equal(x, y):
                    changed.append("%s/%s/%s" % (k, ch, base))
    assert changed == ["combined_low_count/watson_signal/A",
                       "combined_low_count/crick_signal/A"], changed
    assert d["mu"] == out["mu"] and d["phi"] == out["phi"]
    print("\nonly these entries differ from %s: %s" % (a.src, changed))

    with open(a.out, "wb") as fh:
        pickle.dump(out, fh)
    print("wrote %s" % a.out)

    # read back, so a bad dump cannot pass silently
    rb = pickle.load(open(a.out, "rb"))
    for ch, val in (("watson_signal", w_val), ("crick_signal", c_val)):
        got = float(np.ravel(np.asarray(rb["p"]["combined_low_count"][ch]["A"], dtype=float))[0])
        assert got == val, (ch, got, val)
    n_fallback = 154 - len(d["p"])          # 154 TF states, 12 fitted + clc key
    print("re-read OK: %.8f W / %.8f C   (used by %d of 154 TF states, incl. `unknown`)"
          % (w_val, c_val, n_fallback + 1))


if __name__ == "__main__":
    main()
