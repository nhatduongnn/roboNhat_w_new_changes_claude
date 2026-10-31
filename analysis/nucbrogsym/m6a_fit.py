#!/usr/bin/env python
"""Nucleosome m6A profile on Brogaard dyads, strand-symmetrised (2026-10-28).

Same estimator as abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:compute_Fiber_seq_nucleosome
(dist="binomial", successes_col=11 = N_mod, trials_col=9 = N_valid_cov; fit_binomial_parameters
:622-651 = sum(successes)/sum(trials) per position, strand, base), on the SAME barcode01 pileup.
Only the sites and the centre change. offset o = centre at 1-based base (BED start + o); the old
fit used o = +1 (window 0-based [S-73, S+74), :951-953).

Steps
  1. reproduce inputs/nucleosome_params.pkl from Chereji at o=+1 (validation of this code);
  2. Watson-vs-mirrored-Crick symmetry at o = -3..+3 on Brogaard (and Chereji);
  3. fit Brogaard at the chosen offset (--offset, default 0, chosen by centring_seq.py), then
     p_sym_W[k] = (p_W[k] + p_C[146-k]) / 2 and p_sym_C[k] = p_sym_W[146-k];
  4. write --out (a NEW file; never inputs/nucleosome_params.pkl).
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nucmodel as NM  # noqa: E402

PILEUP = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/Mar_20_25/"
          "no_trim_dorado_v0.9.1/03202025_barcode01_sup_model_sorted_pileup_all_chr")
OUTDIR = "/usr/project/xtmp/nd141/permol_proto_logfull/nucbrog"
BASES = ["A", "C", "G", "T"]


def load_pileup(path, sizes):
    """{(chrom, '+'/'-', base): (succ int64[len], trials int64[len])}, 0-based index = Start."""
    proc = subprocess.Popen(["tr", " ", "\t"], stdin=open(path), stdout=subprocess.PIPE)
    df = pd.read_csv(proc.stdout, sep="\t", header=None, usecols=[0, 1, 3, 5, 9, 11],
                     dtype={0: str, 1: np.int64, 3: str, 5: str, 9: np.int64, 11: np.int64})
    proc.wait()
    arr = {}
    for (chrom, base, strand), sub in df.groupby([0, 3, 5]):
        if chrom not in sizes:
            continue
        b = base.upper()
        s = np.zeros(sizes[chrom], np.int64)
        t = np.zeros(sizes[chrom], np.int64)
        np.add.at(s, sub[1].to_numpy(), sub[11].to_numpy())
        np.add.at(t, sub[1].to_numpy(), sub[9].to_numpy())
        arr[(chrom, strand, b)] = (s, t)
    bases_seen = sorted(set(k[2] for k in arr))
    return arr, bases_seen, len(df)


def profile(arr, rows, offset):
    """sums over sites: succ[strand][base] (147,), trials[...] (147,)."""
    out = {st: {b: [np.zeros(147, np.int64), np.zeros(147, np.int64)] for b in BASES}
           for st in ("watson_signal", "crick_signal")}
    n_used = 0
    for chrom, s, e in rows:
        lo = s + offset - 1 - 73                 # 0-based coordinate of position 0
        used = False
        for strand, st in (("+", "watson_signal"), ("-", "crick_signal")):
            for b in BASES:
                key = (chrom, strand, b)
                if key not in arr:
                    continue
                S, T = arr[key]
                if lo < 0 or lo + 147 > len(S):
                    continue
                out[st][b][0] += S[lo:lo + 147]
                out[st][b][1] += T[lo:lo + 147]
                used = True
        n_used += used
    return out, n_used


def rates(prof):
    p = {st: {} for st in prof}
    for st in prof:
        for b in BASES:
            s, t = prof[st][b]
            with np.errstate(divide="ignore", invalid="ignore"):
                p[st][b] = np.where(t > 0, s / np.maximum(t, 1), 0.0)
    return p


def mirror_test(prof, base="A"):
    """Two-proportion z for Watson[k] vs Crick[146-k]."""
    sw, tw = prof["watson_signal"][base]
    sc, tc = prof["crick_signal"][base]
    sc, tc = sc[::-1], tc[::-1]
    pw, pc = sw / tw, sc / tc
    pp = (sw + sc) / (tw + tc)
    z = (pw - pc) / np.sqrt(pp * (1 - pp) * (1 / tw + 1 / tc))
    return dict(chi2=float((z ** 2).sum()), n_abs_z_gt3=int((np.abs(z) > 3).sum()),
                max_abs_z=float(np.abs(z).max()),
                mean_abs_diff=float(np.abs(pw - pc).mean()), z=z.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(NM.ANALYSIS, "inputs", "nucleosome_params_brogaard_sym.pkl"))
    # ALL7 (2026-10-30): pileup and json path selectable; defaults = the barcode01 run above.
    ap.add_argument("--pileup", default=PILEUP)
    ap.add_argument("--json", default=os.path.join(OUTDIR, "m6a_fit.json"))
    a = ap.parse_args()
    assert os.path.basename(a.out) != "nucleosome_params.pkl", "never overwrite the shipped pkl"

    g = NM.load_genome()
    sizes = {k: len(v) for k, v in g.items()}
    arr, bases_seen, nrows = load_pileup(a.pileup, sizes)
    print("pileup rows %d, bases %s" % (nrows, bases_seen))
    res = dict(pileup=a.pileup, n_pileup_rows=nrows, bases_seen=bases_seen)

    # 1. validation against the shipped pkl
    shipped = pickle.load(open(os.path.join(NM.ANALYSIS, "inputs", "nucleosome_params.pkl"), "rb"))
    ch_rows = NM.read_bed(NM.CHEREJI)
    pr, _ = profile(arr, ch_rows, +1)
    pch = rates(pr)
    val = {}
    for st in ("watson_signal", "crick_signal"):
        for b in BASES:
            val["%s_%s" % (st, b)] = float(np.abs(np.asarray(shipped["p"][st][b]) - pch[st][b]).max())
    res["reproduce_shipped_chereji_o+1_max_abs_diff"] = val
    print("reproduce shipped (Chereji, o=+1):", json.dumps(val))

    # 2. mirror symmetry by offset
    br_rows = NM.read_bed(NM.BROGAARD)
    res["mirror_by_offset"] = {}
    for name, rows in (("brogaard", br_rows), ("chereji", ch_rows)):
        res["mirror_by_offset"][name] = {}
        for o in range(-3, 4):
            prof, n = profile(arr, rows, o)
            m = mirror_test(prof)
            m["n_sites"] = n
            res["mirror_by_offset"][name][o] = m
            print("%-8s o=%+d chi2=%.1f |z|>3: %d  max|z| %.2f" % (name, o, m["chi2"], m["n_abs_z_gt3"], m["max_abs_z"]))

    # 3. Brogaard fit at the chosen offset, symmetrised
    prof, n = profile(arr, br_rows, a.offset)
    p = rates(prof)
    sym = {"watson_signal": {}, "crick_signal": {}}
    pooled_diff = {}
    for b in BASES:
        w, c = p["watson_signal"][b], p["crick_signal"][b]
        sw = 0.5 * (w + c[::-1])
        sym["watson_signal"][b] = sw
        sym["crick_signal"][b] = sw[::-1].copy()
        s1, t1 = prof["watson_signal"][b]
        s2, t2 = prof["crick_signal"][b]
        tt = t1 + t2[::-1]
        pooled = np.where(tt > 0, (s1 + s2[::-1]) / np.maximum(tt, 1), 0.0)
        pooled_diff[b] = float(np.abs(pooled - sw).max())
    res["brogaard_fit"] = dict(offset=a.offset, n_sites=n,
                               trials_A_watson_min=int(prof["watson_signal"]["A"][1].min()),
                               trials_A_crick_min=int(prof["crick_signal"]["A"][1].min()),
                               mirror=mirror_test(prof),
                               avg_vs_pooled_counts_max_abs=pooled_diff,
                               pW_A=p["watson_signal"]["A"].tolist(),
                               pC_A=p["crick_signal"]["A"].tolist(),
                               psym_W_A=sym["watson_signal"]["A"].tolist(),
                               shipped_W_A=np.asarray(shipped["p"]["watson_signal"]["A"]).tolist(),
                               shipped_C_A=np.asarray(shipped["p"]["crick_signal"]["A"]).tolist())
    w = sym["watson_signal"]["A"]
    assert np.all(np.isfinite(w)) and w.min() > 0 and w.max() < 1
    assert np.array_equal(sym["crick_signal"]["A"], w[::-1])
    out = {"p": sym}
    if os.path.exists(a.out):
        sys.exit("refusing to overwrite %s" % a.out)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    res["written"] = a.out
    json.dump(res, open(a.json, "w"), indent=1)
    print("wrote", a.out, "min %.5f max %.5f" % (w.min(), w.max()))


if __name__ == "__main__":
    main()
