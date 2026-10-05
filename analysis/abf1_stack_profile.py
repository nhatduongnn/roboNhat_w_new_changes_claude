#!/usr/bin/env python
"""Per-call sequence / fiber / competition / reference measurements for the ABF1 call sets.

For every call in abf1_stack_sets.py's partition this computes, in one pass per decode:

REGISTER (from the decode itself, not guessed)
  The ABF1 block is 28 states: forward 1..14 (state 1+j at motif column j) and reverse
  15..28 (state 15+j at column j of the reverse-complement PWM) -- robocop.py:157 /
  singlefiber/collapse.collapse_matrix. So the posterior mass of a FORWARD motif starting at
  genomic p is  Sfwd[p] = sum_j post[p+j, 1+j], and of a REVERSE motif  Srev[p] =
  sum_j post[p+j, 15+j].  argmax over (strand, p) in a +/-40 bp window gives the register and
  strand the decode actually used. (The collapsed column score_robocop reports is
  Sfwd+Srev smeared over the 14 positions, which cannot give either.)

SEQUENCE
  FIMO log-odds of Abf1_murphy from <trainDir>/pwm.p, pseudo=0.1 spread by RoboCOP's own
  genome background, split into CORE = cols 0-4 + 10-13 and SPACER = cols 5-9, exactly as
  explain_abf1_score_split.py:28-30,69-73 defines them.

FIBER
  Binomial log-likelihood ratio, in nats, of ABF1's fitted per-column p against the decode's
  background p, summed over the 14 motif columns and both layers, replicating
  robocop.update_data_emission_matrix_using_binomial_fiber_seq (robocop.py:775-806): the
  WATSON layer contributes only where the reference base is A (code 0) and the CRICK layer
  only where it is T (code 3); the binomial coefficient cancels in the ratio.
  Strand mapping is robocop.py:645-680: a forward block takes watson_signal for the Watson
  layer and crick_signal for the Crick layer; a reverse block takes them mirrored AND crossed.

COMPETITION
  The other run's collapsed 160-column posterior, max over the call's 14 bp, so "what took it".

REFERENCE
  MacIsaac (inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed, as shipped -- CLAUDE.md rule 5)
  and Rossi (inputs/rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed) ABF1
  centers, (start+end)//2 as score_robocop.load_abf1:301 / score_factors.load_rossi:84.

Also writes a +/-50 bp profile stack (sequence score, m6A ratio, both runs' ABF1 posterior).
"""
import argparse
import glob
import json
import os
import pickle
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# RoboCOP's own genome background (parameterize.computeBackground), copied from
# explain_abf1_score_split.py:28
BG = np.array([0.30980641, 0.19088229, 0.19059636, 0.30871494])
CORE = list(range(0, 5)) + list(range(10, 14))
SPACER = list(range(5, 10))
L = 14
HALF = 50
WIN = 40            # +/- bp searched for the decode's own register
MACISAAC = os.path.join(HERE, "inputs", "MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed")
ROSSI = os.path.join(HERE, "inputs",
                     "rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed")


def log_odds(pwm, pseudo=0.1):
    """explain_abf1_score_split.log_odds, verbatim. pwm is (L, 4) probabilities."""
    p = (pwm + pseudo * BG[None, :]) / (1.0 + pseudo)
    p = p / p.sum(1, keepdims=True)
    return np.log2(p / BG[None, :])


def sparse_todense(f, k):
    g = f[k]
    v = np.array(sparse.csr_matrix(
        (g['data'][:], g['indices'][:], g['indptr'][:]), g.attrs['shape']).todense())
    return v[0] if v.shape[0] == 1 else v


def ref_centers(chrom):
    m = pd.read_csv(MACISAAC, sep="\t", header=None,
                    names=["chr", "start", "end", "name", "score", "strand"])
    m = m[(m["chr"] == chrom) & (m["name"].str.upper() == "ABF1")]
    r = pd.read_csv(ROSSI, sep="\t")
    r = r[(r["chr"] == chrom) & (r["TF"] == "Abf1_murphy")]
    return (np.sort(((m["start"] + m["end"]) // 2).values),
            np.sort(((r["start"] + r["end"]) // 2).values))


def nearest(arr, x):
    if len(arr) == 0:
        return 10 ** 9
    i = int(np.searchsorted(arr, x))
    best = 10 ** 9
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(arr):
            best = min(best, abs(int(arr[j]) - x))
    return best


def register_pass(outDir, chrom, centers, dshared):
    """-> {center: (strand, start, mass)} from the raw per-state posterior."""
    coords = pd.read_csv(os.path.join(outDir, "coords.tsv"), sep="\t")
    sel = coords[coords["chr"] == chrom]
    centers = np.sort(np.asarray(centers))
    acc = {c: dict(f=np.zeros(2 * WIN + 1), r=np.zeros(2 * WIN + 1), n=0) for c in centers}
    for infofile in sorted(glob.glob(os.path.join(outDir, "tmpDir", "info*.h5"))):
        with h5py.File(infofile, "r") as f:
            for k in f.keys():
                idx = int(k.split("_")[1])
                if idx not in sel.index:
                    continue
                s, e = int(coords.loc[idx, "start"]), int(coords.loc[idx, "end"])
                lo, hi = np.searchsorted(centers, [s + WIN + L, e - WIN - L])
                if hi <= lo:
                    continue
                dp = sparse_todense(f, k + "/posterior")
                for c in centers[lo:hi]:
                    i0 = c - WIN - s          # array index of (c - WIN)
                    a = acc[c]
                    for j in range(L):
                        a["f"] += dp[i0 + j:i0 + j + 2 * WIN + 1, 1 + j]
                        a["r"] += dp[i0 + j:i0 + j + 2 * WIN + 1, 15 + j]
                    a["n"] += 1
                del dp
    out = {}
    for c, a in acc.items():
        if a["n"] == 0:
            out[c] = (None, None, 0.0)
            continue
        fv, rv = a["f"] / a["n"], a["r"] / a["n"]
        if fv.max() >= rv.max():
            out[c] = ("+", int(c - WIN + int(np.argmax(fv))), float(fv.max()))
        else:
            out[c] = ("-", int(c - WIN + int(np.argmax(rv))), float(rv.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--dir-a", required=True, nargs=3, help="bo09 decode dirs: chrII chrXIV chrIV")
    ap.add_argument("--dir-b", required=True, nargs=3)
    ap.add_argument("--a", default="bo09")
    ap.add_argument("--b", default="bu01")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prof", required=True)
    args = ap.parse_args()

    CHROMS = ["chrII", "chrXIV", "chrIV"]
    DIRS = {args.a: dict(zip(CHROMS, args.dir_a)), args.b: dict(zip(CHROMS, args.dir_b))}

    pwm = pickle.load(open(os.path.join(HERE, "robocop_train_tw_bo09_48", "pwm.p"), "rb"))
    P = np.asarray(pwm["Abf1_murphy"])[:4].T            # (14, 4) A,C,G,T
    assert P.shape == (L, 4), P.shape
    LO = log_odds(P)                                     # (14, 4) log2 odds
    LOrc = log_odds(P[::-1, ::-1])                        # reverse complement

    par = pickle.load(open(os.path.join(HERE, "inputs",
                                       "all_TFs_1000pealVal_params_pseudo.pkl"), "rb"))["p"]
    pW = np.asarray(par["Abf1_murphy"]["watson_signal"]["A"], dtype=float)
    pC = np.asarray(par["Abf1_murphy"]["crick_signal"]["A"], dtype=float)
    bg = pickle.load(open(os.path.join(HERE, "inputs", "bg_params_open.pkl"), "rb"))["p"]
    bW = float(np.asarray(bg["watson_signal"]["A"])[0])
    bC = float(np.asarray(bg["crick_signal"]["A"])[0])
    clc = 0.08                                            # bu01's clc08 / `unknown` level
    print("ABF1 p: W mean %.6f C mean %.6f | bg W %.8f C %.8f | clc08 %.2f"
          % (pW.mean(), pC.mean(), bW, bC, clc))

    sets = pd.read_csv(args.sets, sep="\t")
    rows, profs = [], []
    for chrom in CHROMS:
        z = {t: np.load(os.path.join(args.stack, "%s_%s.npz" % (t, chrom)), allow_pickle=True)
             for t in (args.a, args.b)}
        cols = list(z[args.a]["cols"])
        full = {t: np.load(os.path.join(args.stack, "%s_%s_full.npy" % (t, chrom)),
                           mmap_mode="r") for t in (args.a, args.b)}
        nt = z[args.a]["nt"]
        assert np.array_equal(nt, z[args.b]["nt"]), "nucleotides differ between runs"
        mw, mc, aw, ac = (z[args.a]["meth_w"].astype(float), z[args.a]["meth_c"].astype(float),
                          z[args.a]["A_w"].astype(float), z[args.a]["A_c"].astype(float))
        for w in ("meth_w", "meth_c", "A_w", "A_c"):
            assert np.array_equal(z[args.a][w], z[args.b][w]), "fiber counts differ: %s" % w
        lo_pos = int(z[args.a]["pos"][0])
        okA, okB = z[args.a]["ok"], z[args.b]["ok"]
        trA, trB = z[args.a]["abf1"], z[args.b]["abf1"]
        mac, ros = ref_centers(chrom)
        sub = sets[sets["chrom"] == chrom]
        print("%s: %d calls in the partition | MacIsaac ABF1 %d | Rossi ABF1 %d"
              % (chrom, len(sub), len(mac), len(ros)))

        # the register comes from whichever run called the site (for `both`, run A)
        owner = {}
        for _, r in sub.iterrows():
            owner.setdefault(args.a if r["set"] != "only_%s" % args.b else args.b,
                             []).append(int(r["center"]))
        reg = {}
        for t, cs in owner.items():
            reg[t] = register_pass(DIRS[t][chrom], chrom, cs, None)
            print("  register pass %s: %d centers, %d resolved"
                  % (t, len(cs), sum(1 for v in reg[t].values() if v[0])))

        def idx(c):
            return c - lo_pos

        for _, r in sub.iterrows():
            c = int(r["center"])
            t = args.a if r["set"] != "only_%s" % args.b else args.b
            strand, start, mass = reg[t][c]
            if strand is None:
                continue
            i = idx(start)
            if i < 0 or i + L > len(nt):
                continue
            seq = nt[i:i + L].astype(int)
            if (seq > 3).any():
                continue
            mat = LO if strand == "+" else LOrc
            per = np.array([mat[j, seq[j]] for j in range(L)])
            # fiber: per-column p in the reference frame
            if strand == "+":
                pw_col, pc_col = pW, pC
            else:
                pw_col, pc_col = pC[::-1], pW[::-1]

            def llr(pb_w, pb_c):
                s = 0.0
                for j in range(L):
                    k_, n_ = mw[i + j], aw[i + j]
                    if n_ > 0 and seq[j] == 0:
                        s += k_ * np.log(pw_col[j] / pb_w) + (n_ - k_) * np.log(
                            (1 - pw_col[j]) / (1 - pb_w))
                    k_, n_ = mc[i + j], ac[i + j]
                    if n_ > 0 and seq[j] == 3:
                        s += k_ * np.log(pc_col[j] / pb_c) + (n_ - k_) * np.log(
                            (1 - pc_col[j]) / (1 - pb_c))
                return float(s)

            nA = int(((seq == 0) & (aw[i:i + L] > 0)).sum())
            nT = int(((seq == 3) & (ac[i:i + L] > 0)).sum())
            trials = float(aw[i:i + L][(seq == 0)].sum() + ac[i:i + L][(seq == 3)].sum())
            meth = float(mw[i:i + L][(seq == 0)].sum() + mc[i:i + L][(seq == 3)].sum())
            # competition: the OTHER run's max posterior per column over the 14 bp
            other = args.b if t == args.a else args.a
            blk = np.asarray(full[other][i:i + L, :])
            top = np.argsort(-blk.max(axis=0))[:4]
            rows.append(dict(
                set=r["set"], chrom=chrom, center=c, start=start, strand=strand,
                owner=t, mass=mass,
                post_a=float(trA[idx(c)]), post_b=float(trB[idx(c)]),
                seq="".join("ACGT"[b] for b in seq),
                score_total=float(per.sum()), score_core=float(per[CORE].sum()),
                score_spacer=float(per[SPACER].sum()),
                llr_bg=llr(bW, bC), llr_clc=llr(clc, clc),
                n_A=nA, n_T=nT, trials=trials, meth=meth,
                m6a=(meth / trials if trials > 0 else float("nan")),
                other_top1=cols[top[0]], other_top1_p=float(blk[:, top[0]].max()),
                other_top2=cols[top[1]], other_top2_p=float(blk[:, top[1]].max()),
                other_unknown=float(blk[:, cols.index("unknown")].max()),
                other_nuc=float(blk[:, cols.index("nucleosome")].max()),
                other_bg=float(blk[:, cols.index("background")].max()),
                other_abf1=float(blk[:, cols.index("Abf1_murphy")].max()),
                d_macisaac=nearest(mac, c), d_rossi=nearest(ros, c)))

            # profile +/- HALF around the motif start, in the motif's strand frame
            if i - HALF >= 0 and i + L + HALF < len(nt):
                sl = slice(i - HALF, i + L + HALF)
                sc = np.full(L + 2 * HALF, np.nan)
                for o in range(-HALF, HALF + 1):
                    jj = i + o
                    s2 = nt[jj:jj + L].astype(int)
                    if len(s2) == L and (s2 <= 3).all():
                        sc[o + HALF] = sum(mat[q, s2[q]] for q in range(L))
                tot = aw[sl] + ac[sl]
                rat = np.where(tot > 0, (mw[sl] + mc[sl]) / np.maximum(tot, 1), np.nan)
                pa, pb_ = trA[sl].astype(float), trB[sl].astype(float)
                if strand == "-":
                    sc, rat, pa, pb_ = sc[::-1], rat[::-1], pa[::-1], pb_[::-1]
                profs.append(dict(set=r["set"], chrom=chrom, center=c, score=sc,
                                  m6a=rat, post_a=pa, post_b=pb_))
        del full

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False, float_format="%.5f")
    print("wrote %s (%d rows)" % (args.out, len(df)))
    np.savez_compressed(
        args.prof,
        set=np.array([p["set"] for p in profs]),
        chrom=np.array([p["chrom"] for p in profs]),
        center=np.array([p["center"] for p in profs]),
        score=np.vstack([p["score"] for p in profs]),
        m6a=np.vstack([p["m6a"] for p in profs]),
        post_a=np.vstack([p["post_a"] for p in profs]),
        post_b=np.vstack([p["post_b"] for p in profs]), half=HALF, L=L)
    print("wrote %s (%d profiles)" % (args.prof, len(profs)))


if __name__ == "__main__":
    main()
