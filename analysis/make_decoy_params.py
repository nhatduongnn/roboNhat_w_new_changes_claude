"""Add a flat ABF1-rate Fiber-seq footprint for the decoy motif `Zz_decoy_abf1` (2026-09-21).

Plan: ~/.claude/plans/okay-so-i-want-humming-hearth.md, Build step 3 (pattern: make_caplow_params.py).

Decoy rate, per strand = the UNWEIGHTED mean of the columns of that width's own Abf1_murphy
vector in the source pkl (user decision: "ABF1's own average, flattened"):
    14 bp: from all_TFs_1000pealVal_params_pseudo.pkl       (Abf1_murphy L 14)
    23 bp: from all_TFs_1000pealVal_params_pseudo_wide.pkl  (Abf1_murphy L 23, the 7/2 model)
p['Zz_decoy_abf1'][{watson_signal, crick_signal}][A,C,G,T] = np.full(L, mean). robocop.py reads
only 'A' (the fiber loop at robocop.py:654-670); C/G/T are filled the same so no reader ever
finds a missing or mismatched key.

Every other entry of the source pkl is asserted bit-identical; the source is never written
(CLAUDE.md rule 2) and the output is refused if it exists.

    python make_decoy_params.py --src inputs/all_TFs_1000pealVal_params_pseudo.pkl \
        --out inputs/all_TFs_1000pealVal_params_pseudo_decoy14.pkl --width 14
    python make_decoy_params.py --src inputs/all_TFs_1000pealVal_params_pseudo_wide.pkl \
        --out inputs/all_TFs_1000pealVal_params_pseudo_wide_decoy23.pkl --width 23
"""
import argparse
import copy
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DECOY = "Zz_decoy_abf1"
TARGET = "Abf1_murphy"
CHANNELS = ("watson_signal", "crick_signal")
BASES = ("A", "C", "G", "T")


def same(a, b):
    """Recursive bit-identity of the pkl's nested dict/array structure."""
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    return type(a) is type(b) and a == b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, required=True)
    a = ap.parse_args()
    src = os.path.join(HERE, a.src)
    out = os.path.join(HERE, a.out)
    if os.path.exists(out):
        sys.exit("%s already exists; refusing to overwrite (rule 2)" % a.out)
    d = pickle.load(open(src, "rb"), encoding="latin1")
    if DECOY in d["p"]:
        sys.exit("%s already has %s" % (a.src, DECOY))
    new = copy.deepcopy(d)
    entry = {}
    for ch in CHANNELS:
        v = np.asarray(d["p"][TARGET][ch]["A"], dtype=np.float64)
        if v.shape != (a.width,):
            sys.exit("%s %s A has shape %s, expected (%d,)" % (TARGET, ch, v.shape, a.width))
        m = float(v.mean())
        entry[ch] = {b: np.full(a.width, m, dtype=np.float64) for b in BASES}
        print("%-14s %s L %d: unweighted column mean %.17g (min %.4f max %.4f) -> flat decoy"
              % (ch, TARGET, a.width, m, v.min(), v.max()))
    new["p"][DECOY] = entry

    # everything but the new key is bit-identical to the source
    assert list(new) == list(d), (list(new), list(d))
    for k in d:
        if k != "p":
            assert same(d[k], new[k]), k
    assert list(new["p"]) == list(d["p"]) + [DECOY], list(new["p"])
    for k in d["p"]:
        assert same(d["p"][k], new["p"][k]), "entry %s changed" % k
    for ch in CHANNELS:
        for b in BASES:
            x = new["p"][DECOY][ch][b]
            assert x.shape == (a.width,) and np.all(x == x[0]) and x[0] == np.asarray(
                d["p"][TARGET][ch]["A"], dtype=np.float64).mean()
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(new, f)
    back = pickle.load(open(tmp, "rb"))
    assert same(back["p"][DECOY], new["p"][DECOY]) and all(same(d["p"][k], back["p"][k]) for k in d["p"])
    os.rename(tmp, out)
    print("only new entry vs %s: p['%s'] (%d keys -> %d); every other entry bit-identical"
          % (a.src, DECOY, len(d["p"]), len(new["p"])))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
