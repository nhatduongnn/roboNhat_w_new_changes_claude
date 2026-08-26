"""JASPAR vs Murphy ABF1 matrix: decoded posterior at the 5 chrI MacIsaac sites.

All numbers are DECODE OUTPUT, read from each run's tmpDir/info_*.h5 posterior table
via plot_abf1_5sites_decoded.state_composition (ABF1 forward = states 1..14, reverse =
15..28). Nothing is recomputed.

Also prints the ABF1 prior from each trainDir's HMMconfig.pkl, because
parameterize.calculateKD depends on the matrix -- swapping to JASPAR moves tf_prob as
well as pwm_emission, and the two effects must not be conflated.

    python compare_jaspar_vs_murphy.py
"""
import os, sys, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
import numpy as np
import score_robocop as S
import plot_abf1_5sites_decoded as PD

SITES = [(1, 45318, 45332), (2, 45498, 45512), (3, 61163, 61177),
         (4, 62657, 62671), (5, 108788, 108802)]

RUNS = [
    ("Murphy  seq-only",   "robocop_chrI_seqonly_maskon_revfix", "robocop_train_fiberonly"),
    ("JASPAR  seq-only",   "robocop_chrI_seqonly_maskon_JASPAR", "robocop_train_jaspar"),
    ("Murphy  fiber+seq",  "robocop_chrI_seq_maskon_revfix",     "robocop_train_fiberonly"),
    ("JASPAR  fiber+seq",  "robocop_chrI_seq_maskon_JASPAR",     "robocop_train_jaspar"),
    ("Murphy  fiber-only", "robocop_chrI_maskon_revfix",         "robocop_train_fiberonly"),
]


def abf1_prior(trainDir):
    try:
        cfg = pickle.load(open(os.path.join(trainDir, "HMMconfig.pkl"), "rb"))
        tfs = list(cfg["tfs"])
        i = tfs.index("Abf1_murphy")
        return 2.0 * float(np.ravel(cfg["tf_prob"])[i])
    except Exception as e:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, default=60, help="near-motif half-window")
    args = ap.parse_args()

    print("ABF1 prior (2 x tf_prob) from each trainDir's HMMconfig.pkl")
    seen = {}
    for _, _, td in RUNS:
        if td in seen:
            continue
        seen[td] = abf1_prior(td)
        print("   %-26s %s" % (td, "%.4g" % seen[td] if seen[td] else "MISSING"))
    tds = [t for t in seen if seen[t]]
    if len(tds) == 2:
        a, b = seen[tds[0]], seen[tds[1]]
        print("   ratio %s / %s = %.3fx  <-- prior moves too; do not attribute this to the matrix alone"
              % (tds[1], tds[0], b / a))

    avail = [(lab, od, td) for lab, od, td in RUNS if os.path.isdir(os.path.join(od, "tmpDir"))]
    missing = [od for lab, od, td in RUNS if not os.path.isdir(os.path.join(od, "tmpDir"))]
    if missing:
        print("\nNOT YET AVAILABLE: %s" % ", ".join(missing))

    print("\nABF1 posterior (forward + reverse) ON the 14 bp MacIsaac span")
    print("%-20s" % "run" + "".join("%12s" % ("site #%d" % n) for n, _, _ in SITES) + "%10s" % "n>=0.5")
    onmot = {}
    for lab, od, td in avail:
        dec = S.load_decode(od)
        row, hits = [], 0
        for n, m0, m1 in SITES:
            comp, pos, _ = PD.state_composition(dec, "chrI", m0 - args.pad, m1 + args.pad)
            m = (pos >= m0) & (pos <= m1)
            v = float((comp["abf1_fwd"] + comp["abf1_rev"])[m].max())
            row.append(v); hits += v >= 0.5
        onmot[lab] = row
        print("%-20s" % lab + "".join("%12.4f" % v for v in row) + "%10d/5" % hits)

    print("\nsame, but best ABF1 posterior anywhere within +/-%d bp" % args.pad)
    print("%-20s" % "run" + "".join("%12s" % ("site #%d" % n) for n, _, _ in SITES))
    for lab, od, td in avail:
        dec = S.load_decode(od)
        row = []
        for n, m0, m1 in SITES:
            comp, pos, _ = PD.state_composition(dec, "chrI", m0 - args.pad, m1 + args.pad)
            row.append(float((comp["abf1_fwd"] + comp["abf1_rev"]).max()))
        print("%-20s" % lab + "".join("%12.4f" % v for v in row))

    if "Murphy  seq-only" in onmot and "JASPAR  seq-only" in onmot:
        print("\nSITE #3 (chrI:61163-61177) -- the primary question")
        for k in ("Murphy  seq-only", "JASPAR  seq-only",
                  "Murphy  fiber+seq", "JASPAR  fiber+seq"):
            if k in onmot:
                v = onmot[k][2]
                print("   %-20s %.4f   %s" % (k, v, "RECOVERED" if v >= 0.5 else "not recovered"))


if __name__ == "__main__":
    main()
