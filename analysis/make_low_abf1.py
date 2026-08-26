"""VARIANT 2 (low_abf1): lower combined_low_count to ABF1's mean footprint p.

Keeps the linker-derived background (0.1383) exactly as shipped. The problem being
tested here is the other side of the inversion: combined_low_count is 0.2489/0.2647,
fit at Rossi peak CENTRES which sit in wide-open NDRs, so it describes accessible
chromatin rather than a protein footprint. 142 of 154 TFs fall back to it, so those
states outscore background wherever DNA is open.

Setting it to ABF1's mean (0.0843 watson / ~0.080 crick) makes the fallback describe an
actual footprint -- below background, i.e. protective -- which is what a TF state should
be. ABF1 is a reasonable donor: it is one of only 3 of the 12 fitted TFs whose entire
vector lies below background.

Only combined_low_count['A'] changes. The 12 fitted TFs, the C/G/T entries and every
other key are copied through untouched, and that is asserted before writing.

Writes inputs/all_TFs_1000pealVal_params_pseudo_lowabf1.pkl.
Does not modify inputs/all_TFs_1000pealVal_params_pseudo.pkl.

    python make_low_abf1.py            # any env with numpy -- no rpy2 needed
"""
import argparse
import copy
import pickle

import numpy as np

SRC = "inputs/all_TFs_1000pealVal_params_pseudo.pkl"
OUT = "inputs/all_TFs_1000pealVal_params_pseudo_lowabf1.pkl"
DONOR = "Abf1_murphy"
STRANDS = ("watson_signal", "crick_signal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default=DONOR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    old = pickle.load(open(SRC, "rb"))
    new = copy.deepcopy(old)

    bg = pickle.load(open("inputs/bg_params.pkl", "rb"))
    bgv = {s: float(np.ravel(bg["p"][s]["A"])[0]) for s in STRANDS}

    means = {}
    for s in STRANDS:
        v = np.asarray(old["p"][args.donor][s]["A"], dtype=float)
        means[s] = float(v.mean())
        # keep the original container's shape/type so downstream indexing is unchanged;
        # combined_low_count is length 1 and gets broadcast across all 2*tf_len states
        cur = np.asarray(old["p"]["combined_low_count"][s]["A"], dtype=float)
        new["p"]["combined_low_count"][s]["A"] = np.full_like(cur, means[s])

    print("%-28s %10s %10s" % ("", "watson", "crick"))
    print("%-28s %10.4f %10.4f" % ("background", bgv["watson_signal"], bgv["crick_signal"]))
    print("%-28s %10.4f %10.4f" % ("combined_low_count OLD",
          float(np.ravel(old["p"]["combined_low_count"]["watson_signal"]["A"])[0]),
          float(np.ravel(old["p"]["combined_low_count"]["crick_signal"]["A"])[0])))
    print("%-28s %10.4f %10.4f" % ("combined_low_count NEW (= %s mean)" % args.donor,
          means["watson_signal"], means["crick_signal"]))

    # --- assertions: nothing but combined_low_count['A'] may have moved ---
    changed = []
    for tf in old["p"]:
        for s in STRANDS:
            for b in old["p"][tf][s]:
                a = np.asarray(old["p"][tf][s][b], dtype=float)
                c = np.asarray(new["p"][tf][s][b], dtype=float)
                if a.shape != c.shape or not np.allclose(a, c, equal_nan=True):
                    changed.append("%s/%s/%s" % (tf, s, b))
    expected = {"combined_low_count/%s/A" % s for s in STRANDS}
    print("\nkeys that changed: %s" % sorted(changed))
    assert set(changed) == expected, "unexpected keys changed: %s" % (set(changed) ^ expected)
    assert set(new["p"]) == set(old["p"]), "TF key set changed"
    print("assertion PASSED: only combined_low_count['A'] differs; all 12 TF vectors intact")

    print("\nrelation to background (a TF footprint should be BELOW background):")
    for s in STRANDS:
        print("   %-14s new low_count %.4f vs bg %.4f  ->  %s"
              % (s, means[s], bgv[s], "below (protective)" if means[s] < bgv[s] else "ABOVE"))

    with open(args.out, "wb") as f:
        pickle.dump(new, f)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
