"""What raising bg actually does to the fiber-layer likelihood ratio.

robocop.py's update_data_emission_matrix_using_binomial_fiber_seq multiplies, at every
reference position i whose base is A (watson) / T (crick), and for EVERY state j:

    emission[i, j] *= Binomial.pmf(k_i, n_i, ps[j])

ps[0] (background) is bg_params['p'][strand]['A']; ps over a TF block is that TF's fitted
per-column p, or a flat combined_low_count for the ~141 PWMs with no individual fit.

So the fiber layer's whole contribution to "TF vs background" at one position is

    LR = (p_tf/p_bg)^k * ((1-p_tf)/(1-p_bg))^(n-k)

which is MONOTONE in k/n. A flat-p state carries no shape information at all: it wins
purely on level. Crossover (LR = 1) is at

    t* = ln((1-p_bg)/(1-p_tf)) / ln( p_tf(1-p_bg) / (p_bg(1-p_tf)) )

p_tf > p_bg  ->  the TF wins ABOVE t*  (the bug: TFs get placed in open DNA)
p_tf < p_bg  ->  the TF wins BELOW t*  (a footprint detector, as intended)

This script prints t* for every bg candidate, then does the same arithmetic on real
counts: a real MacIsaac ABF1 site on chrI and a real high-methylation promoter position.

    python bg_likelihood_arithmetic.py
"""
import pickle

import numpy as np
import pandas as pd
from scipy.stats import binom

from make_bg_tss import PILEUP
from make_params_pm50 import Pileup

MACISAAC = "inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1_match_PWM.bed"
TSS = "inputs/Park_2014_TSS.csv"

BG_CANDIDATES = [
    ("shipped (linkers, dyad+/-73..88)", 0.13827, 0.13841),
    ("TSS -200..-100 mean", 0.1619, 0.1619),
    ("TSS -200..-100 top25% within", 0.3535, 0.3546),
    ("TSS -200..-100 top10% within", 0.4402, 0.4427),
    ("TSS -200..-100 top5%  within", 0.4853, 0.4886),
    ("TSS -200..-100 max    within", 0.5356, 0.5374),
]


def crossover(p_bg, p_tf):
    if abs(p_bg - p_tf) < 1e-12:
        return np.nan
    num = np.log((1 - p_bg) / (1 - p_tf))
    den = np.log((p_tf * (1 - p_bg)) / (p_bg * (1 - p_tf)))
    return num / den


def main():
    lp = pickle.load(open("inputs/all_TFs_1000pealVal_params_pseudo.pkl", "rb"))
    low_w = float(np.ravel(lp["p"]["combined_low_count"]["watson_signal"]["A"])[0])
    low_c = float(np.ravel(lp["p"]["combined_low_count"]["crick_signal"]["A"])[0])
    abf1_w = np.asarray(lp["p"]["Abf1_murphy"]["watson_signal"]["A"], float)
    abf1_c = np.asarray(lp["p"]["Abf1_murphy"]["crick_signal"]["A"], float)
    nuc = np.asarray(pickle.load(open("inputs/nucleosome_params.pkl", "rb"))
                     ["p"]["watson_signal"]["A"], float)

    print("combined_low_count  watson %.4f  crick %.4f   (flat, tf_len = 1)" % (low_w, low_c))
    print("Abf1_murphy         watson mean %.4f (%.4f..%.4f)" % (abf1_w.mean(), abf1_w.min(), abf1_w.max()))
    print("nucleosome          watson mean %.4f (%.4f..%.4f)\n" % (nuc.mean(), nuc.min(), nuc.max()))

    print("=== crossover methylation fraction k/n at which each state ties background ===")
    print("%-34s %8s %-38s %-38s" % ("bg candidate", "bg_w", "combined_low_count", "Abf1 (mean p)"))
    for name, bw, bc in BG_CANDIDATES:
        t_low = crossover(bw, low_w)
        t_abf = crossover(bw, abf1_w.mean())
        s_low = "wins ABOVE k/n=%.3f" % t_low if low_w > bw else "wins BELOW k/n=%.3f" % t_low
        s_abf = "wins ABOVE k/n=%.3f" % t_abf if abf1_w.mean() > bw else "wins BELOW k/n=%.3f" % t_abf
        print("%-34s %8.4f %-38s %-38s" % (name, bw, s_low, s_abf))

    # ------------------------------------------------------------------ real counts
    pile = Pileup(PILEUP)
    bed = pd.read_csv(MACISAAC, sep="\t", header=None,
                      names=["chr", "start", "end", "tf", "score", "strand"])
    sites = bed[(bed.chr == "chrI") & (bed.tf == "ABF1")]

    def window_counts(chrm, lo, hi, sym="+"):
        sl = pile.slice(chrm, lo, hi)
        if sl is None:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        m = (sl["strand"] == sym) & (sl["base"] == "A") & (sl["trials"] > 0)
        return sl["succ"][m], sl["trials"][m]

    print("\n=== real chrI MacIsaac ABF1 sites, watson channel, 14 bp motif span ===")
    print("%-16s %6s %8s %8s %8s" % ("site", "nA", "sum k", "sum n", "k/n"))
    picked = None
    for _, r in sites.iterrows():
        k, n = window_counts("chrI", int(r.start), int(r.end))
        if len(k) == 0:
            continue
        print("%-16s %6d %8d %8d %8.4f"
              % ("chrI:%d" % r.start, len(k), k.sum(), n.sum(), k.sum() / n.sum()))
        if picked is None or k.sum() / n.sum() < picked[2].sum() / picked[3].sum():
            picked = (int(r.start), int(r.end), k, n)

    # a real, highly methylated promoter position: the most-methylated well-covered
    # position in the -200..-100 window of the first chrI Watson gene
    tss = pd.read_csv(TSS, sep="\t")
    tss = tss[(tss.chr == "chrI") & (tss.internal == 0) & (tss.flag == 0)
              & tss.ORF.str.contains(r"^Y[A-P][LR]\d{3}W")]
    prom = None
    for _, r in tss.iterrows():
        c = int(r.coordinate)
        sl = pile.slice("chrI", c - 200, c - 100)
        if sl is None:
            continue
        m = (sl["strand"] == "+") & (sl["base"] == "A") & (sl["trials"] >= 20)
        if m.sum() == 0:
            continue
        rr = sl["succ"][m] / sl["trials"][m]
        i = int(np.argmax(rr))
        cand = (int(sl["pos"][m][i]), int(sl["succ"][m][i]), int(sl["trials"][m][i]), r.ORF)
        if prom is None or cand[1] / cand[2] > prom[1] / prom[2]:
            prom = cand

    print("\n=== a real promoter position (chrI, TSS-200..-100, watson) ===")
    print("chrI:%d  %s   k=%d  n=%d  k/n=%.4f" % (prom[0], prom[3], prom[1], prom[2], prom[1] / prom[2]))

    print("\nper-position log10 LR at that promoter position:")
    print("%-34s %14s %14s" % ("bg candidate", "lowcount vs bg", "Abf1col vs bg"))
    for name, bw, bc in BG_CANDIDATES:
        lr_low = (np.log10(binom.pmf(prom[1], prom[2], low_w))
                  - np.log10(binom.pmf(prom[1], prom[2], bw)))
        lr_abf = (np.log10(binom.pmf(prom[1], prom[2], abf1_w.mean()))
                  - np.log10(binom.pmf(prom[1], prom[2], bw)))
        print("%-34s %14.3f %14.3f" % (name, lr_low, lr_abf))

    # whole 14 bp block, real ABF1 site vs the flat low-count state
    st, en, k, n = picked
    print("\n=== the most-protected chrI ABF1 site, chrI:%d-%d ===" % (st, en))
    print("k = %s" % list(map(int, k)))
    print("n = %s" % list(map(int, n)))
    print("\ntotal log10 likelihood of the 14 bp block under each state:")
    print("%-34s %10s %10s %10s %12s %12s"
          % ("bg candidate", "L(bg)", "L(Abf1)", "L(low)", "Abf1-bg", "low-bg"))
    sl = pile.slice("chrI", st, en)
    m = (sl["strand"] == "+") & (sl["base"] == "A") & (sl["trials"] > 0)
    cols = (sl["pos"][m] - st).astype(int)
    kk, nn = sl["succ"][m], sl["trials"][m]
    for name, bw, bc in BG_CANDIDATES:
        Lbg = np.log10(binom.pmf(kk, nn, bw)).sum()
        Labf = np.log10(binom.pmf(kk, nn, abf1_w[cols])).sum()
        Llow = np.log10(binom.pmf(kk, nn, low_w)).sum()
        print("%-34s %10.3f %10.3f %10.3f %12.3f %12.3f"
              % (name, Lbg, Labf, Llow, Labf - Lbg, Llow - Lbg))

    # how much of chrI currently prefers the flat low-count state over background
    print("\n=== fraction of chrI watson A positions (n>=10) where the FLAT")
    print("    combined_low_count state beats background on the fiber layer ===")
    d = pile.by_chrom["chrI"]
    m = (d["strand"] == "+") & (d["base"] == "A") & (d["trials"] >= 10)
    kk, nn = d["succ"][m], d["trials"][m]
    print("%-34s %10s" % ("bg candidate", "frac"))
    for name, bw, bc in BG_CANDIDATES:
        won = (binom.logpmf(kk, nn, low_w) > binom.logpmf(kk, nn, bw))
        print("%-34s %10.4f" % (name, won.mean()))


if __name__ == "__main__":
    main()
