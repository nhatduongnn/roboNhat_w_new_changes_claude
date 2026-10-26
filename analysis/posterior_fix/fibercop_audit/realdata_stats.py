#!/usr/bin/env python
"""Where real Fiber-seq coverage sits relative to the binom.pmf underflow / C1 danger zone.

Usage: python realdata_stats.py [--pileup PATH] [--out PATH]
Writes build/realdata.json next to this script by default.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
PILEUP = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/combine_sequencing_runs/"
          "merged_Mar20_barcode01_Jun25_barcode21-24_May07_barcode03-04_sup_model_sorted_pileup_all_chr")
DEPTH = ("/usr/project/xtmp/nd141/programs/robocop_c_code_test/robocop_c_audit/walkthrough/"
         "variants/depth.json")
FASTA = os.path.join(HERE, "..", "..", "inputs", "SacCer3.fa")
RDNA = ("chrXII", 451435, 468729)  # 1-based inclusive
P_LO, P_HI = 0.0242, 0.2647
N_THR = [1, 10, 50, 100, 200, 300, 500, 1000]
KN_THR = 0.8
BINS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
QS = [50, 90, 99, 99.9, 99.99]

METHOD = (
    "Pileup parsed as pkg/robocop/utils/getReads.py:188-218: whitespace-split columns; col0 chrom, "
    "col1 0-based start, col3 mod code (keep upper()=='A'), col5 strand ('+'=Watson, '-'=Crick), "
    "n=int(col9) valid coverage, k=int(col11) modified count. Rows with the same (chrom,start,strand) "
    "are summed, as getReads does with +=. Rows are then kept only if the reference base (analysis/inputs/"
    "SacCer3.fa via Bio.SeqIO) is A for Watson or T for Crick: robocop.py's fiber emission scores only those "
    "positions and zeroes the rest (nucleotide_ref 0/3), so mismatched rows (~30%, mostly n=1) never reach "
    "binom.pmf; their count is reported under rows_dropped_ref_base_mismatch. 'Scored base' = one (chrom,pos,strand) with n>0 (n=0 rows "
    "are counted separately; they carry no evidence). 1-based pos = col1+1. rDNA = chrXII 1-based "
    "451,435-468,729 inclusive; 'genome_ex_rdna' = all chromosomes in the file (incl. chrM) minus rDNA. "
    "Runs: per position keep the strand with larger n (Watson on tie); a run is a maximal stretch of "
    "consecutive positions each with a scored base. A 4-window is any 4 consecutive positions inside a "
    "run (overlapping windows all counted); its min_n is the min of the 4 n values; 'kn>=0.8' requires "
    "all four k/n>=0.8. 'runs_with' counts maximal runs containing >=1 qualifying window. Underflow: "
    "scipy.stats.binom.pmf(k,n,p)==0.0 in float64 for p=0.0242 or p=0.2647 (either one => at least one "
    "state floored to 1e-30). Quantiles use numpy linear interpolation; n histogram bins are [lo,hi) "
    "with a final [10000,inf)."
)


def load(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1, 3, 5, 9, 11],
                     names=["chrom", "start", "mod", "strand", "n", "k"],
                     dtype={"chrom": str, "start": np.int64, "mod": str, "strand": str,
                            "n": np.int64, "k": np.int64}, engine="c")
    nrows = len(df)
    df = df[df["mod"].str.upper() == "A"]
    df = df[df["strand"].isin(["+", "-"])]
    df = df.groupby(["chrom", "start", "strand"], sort=False, as_index=False)[["n", "k"]].sum()
    df["pos"] = df["start"] + 1
    # reference base: the emission only scores Watson at ref A and Crick at ref T
    # (robocop.py update_data_emission_matrix_using_binomial_fiber_seq, nucleotide_ref 0/3)
    from Bio import SeqIO
    ref = {r.id: np.frombuffer(str(r.seq).upper().encode(), dtype="S1") for r in SeqIO.parse(FASTA, "fasta")}
    base = np.empty(len(df), dtype="S1")
    for c, idx in df.groupby("chrom", sort=False).indices.items():
        base[idx] = ref[c][df["start"].to_numpy()[idx]]
    match = ((df["strand"] == "+").to_numpy() & (base == b"A")) | ((df["strand"] == "-").to_numpy() & (base == b"T"))
    mm = df[~match]
    mismatch = {"rows": int(len(mm)), "rows_n_ge_20": int((mm["n"] >= 20).sum()),
                "n_quantiles": {"p50": float(np.percentile(mm["n"], 50)), "p99": float(np.percentile(mm["n"], 99)),
                                "max": int(mm["n"].max())} if len(mm) else None}
    return df[match].reset_index(drop=True), nrows, mismatch


def qdict(x):
    if len(x) == 0:
        return None
    out = {f"p{q}": float(np.percentile(x, q)) for q in QS}
    out["max"] = float(np.max(x))
    out["mean"] = float(np.mean(x))
    return out


def hist(n):
    edges = BINS + [np.inf]
    c, _ = np.histogram(n, bins=edges)
    return {(f"[{edges[i]},{edges[i+1]})" if np.isfinite(edges[i + 1]) else f"[{edges[i]},inf)"): int(c[i])
            for i in range(len(c))}


def base_stats(d):
    zero = int((d["n"] == 0).sum())
    s = d[d["n"] > 0]
    n = s["n"].to_numpy(); kn = (s["k"] / s["n"]).to_numpy()
    return {
        "n_scored_bases": int(len(s)),
        "n_rows_with_n0": zero,
        "watson": int((s["strand"] == "+").sum()), "crick": int((s["strand"] == "-").sum()),
        "coverage_n": qdict(n), "k_over_n": qdict(kn), "n_hist": hist(n),
        "n_ge": {str(t): int((n >= t).sum()) for t in [100, 200, 230, 500, 1000, 4000]},
        "n_ge_230_and_kn_ge_0.8": int(((n >= 230) & (kn >= 0.8)).sum()),
        "underflow_bases": int(s["uf"].sum()),
        "underflow_bases_p0.0242": int(s["uf_lo"].sum()),
        "underflow_bases_p0.2647": int(s["uf_hi"].sum()),
    }


def windows(d):
    """d: one chromosome, scored bases. Returns per-position arrays and 4-window info."""
    d = d[d["n"] > 0].sort_values(["pos", "n", "strand"], ascending=[True, False, True])
    both = int(d.duplicated("pos").sum())
    d = d.drop_duplicates("pos", keep="first")
    pos = d["pos"].to_numpy(); n = d["n"].to_numpy(); k = d["k"].to_numpy()
    kn = k / n
    m = len(pos)
    if m < 4:
        return dict(both=both, pos=pos, wstart=np.array([], int), wmin=np.array([]), wkn=np.array([], bool),
                    runid=np.array([], int), n=n, kn=kn, runlen=np.array([], int))
    brk = np.r_[True, np.diff(pos) != 1]
    runid = np.cumsum(brk) - 1
    runlen = np.bincount(runid)
    ok = (pos[3:] - pos[:-3]) == 3
    idx = np.nonzero(ok)[0]
    wmin = np.minimum.reduce([n[idx], n[idx + 1], n[idx + 2], n[idx + 3]])
    wkn = (kn[idx] >= KN_THR) & (kn[idx + 1] >= KN_THR) & (kn[idx + 2] >= KN_THR) & (kn[idx + 3] >= KN_THR)
    return dict(both=both, pos=pos, wstart=idx, wmin=wmin, wkn=wkn, runid=runid[idx], n=n, kn=kn,
                runlen=runlen)


def window_stats(parts):
    """parts: list of (chrom, windows-dict) restricted to the region."""
    out = {"positions_with_both_strands": sum(w["both"] for _, w in parts),
           "n_4windows": int(sum(len(w["wstart"]) for _, w in parts))}
    rl = np.concatenate([w["runlen"] for _, w in parts]) if parts else np.array([], int)
    out["run_length_hist"] = {f">={L}": int((rl >= L).sum()) for L in [1, 2, 4, 6, 8, 10, 15, 20]}
    out["max_run_length"] = int(rl.max()) if len(rl) else 0
    for lab, need_kn in [("min_n_ge", False), ("min_n_ge_and_all_kn_ge_0.8", True)]:
        cw, cr = {}, {}
        for t in N_THR:
            nw = nr = 0
            for _, w in parts:
                sel = w["wmin"] >= t
                if need_kn:
                    sel &= w["wkn"]
                nw += int(sel.sum()); nr += int(len(np.unique(w["runid"][sel])))
            cw[str(t)] = nw; cr[str(t)] = nr
        out[f"windows_{lab}"] = cw
        out[f"runs_with_{lab}"] = cr
    best = None
    for c, w in parts:
        for need_kn, key in [(False, "max_window_min_n"), (True, "max_window_min_n_all_kn_ge_0.8")]:
            sel = np.nonzero(w["wkn"])[0] if need_kn else np.arange(len(w["wmin"]))
            if len(sel) == 0:
                continue
            j = sel[np.argmax(w["wmin"][sel])]
            i = w["wstart"][j]
            rec = {"min_n": int(w["wmin"][j]), "chrom": c, "pos_1based": [int(p) for p in w["pos"][i:i + 4]],
                   "n": [int(x) for x in w["n"][i:i + 4]], "k_over_n": [round(float(x), 4) for x in w["kn"][i:i + 4]]}
            if key not in out or rec["min_n"] > out[key]["min_n"]:
                out[key] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pileup", default=PILEUP)
    ap.add_argument("--out", default=os.path.join(HERE, "build", "realdata.json"))
    a = ap.parse_args()

    df, nrows, mismatch = load(a.pileup)
    # underflow on unique (k,n)
    kn_u = df[["k", "n"]].drop_duplicates()
    with np.errstate(all="ignore"):
        lo = binom.pmf(kn_u["k"].to_numpy(), kn_u["n"].to_numpy(), P_LO) == 0.0
        hi = binom.pmf(kn_u["k"].to_numpy(), kn_u["n"].to_numpy(), P_HI) == 0.0
    kn_u = kn_u.assign(uf_lo=lo, uf_hi=hi)
    df = df.merge(kn_u, on=["k", "n"], how="left")
    df["uf"] = df["uf_lo"] | df["uf_hi"]
    df.loc[df["n"] == 0, ["uf", "uf_lo", "uf_hi"]] = False
    uf_pairs = kn_u[(kn_u["uf_lo"] | kn_u["uf_hi"]) & (kn_u["n"] > 0)]
    min_uf_n = {
        "p0.0242": int(kn_u.loc[kn_u["uf_lo"] & (kn_u["n"] > 0), "n"].min()) if kn_u["uf_lo"].any() else None,
        "p0.2647": int(kn_u.loc[kn_u["uf_hi"] & (kn_u["n"] > 0), "n"].min()) if kn_u["uf_hi"].any() else None,
    }

    in_rdna = (df["chrom"] == RDNA[0]) & (df["pos"] >= RDNA[1]) & (df["pos"] <= RDNA[2])
    regions = {"genome_ex_rdna": df[~in_rdna], "rdna_chrXII_451435_468729": df[in_rdna],
               "chrXIV": df[df["chrom"] == "chrXIV"]}

    # windows per chromosome; rDNA split by window start inside rDNA
    win = {c: windows(g) for c, g in df.groupby("chrom", sort=False)}

    def restrict(c, w, keep):  # keep: bool over window starts by pos
        p = w["pos"][w["wstart"]] if len(w["wstart"]) else np.array([], int)
        mk = keep(p)
        ww = dict(w); ww["wstart"] = w["wstart"][mk]; ww["wmin"] = w["wmin"][mk]
        ww["wkn"] = w["wkn"][mk]; ww["runid"] = w["runid"][mk]
        # run lengths / both-strand counts restricted by position
        ppos = w["pos"]; pk = keep(ppos)
        if len(ppos):
            brk = np.r_[True, np.diff(ppos) != 1] | np.r_[True, ~pk[:-1]]
            rid = np.cumsum(brk) - 1
            ww["runlen"] = np.bincount(rid[pk]) if pk.any() else np.array([], int)
            ww["runlen"] = ww["runlen"][ww["runlen"] > 0]
        ww["both"] = 0
        return (c, ww)

    def rd(p):
        return (p >= RDNA[1]) & (p <= RDNA[2])

    parts = {
        "genome_ex_rdna": [restrict(c, w, (lambda p: ~rd(p)) if c == RDNA[0] else (lambda p: np.ones(len(p), bool)))
                           for c, w in win.items()],
        "rdna_chrXII_451435_468729": [restrict(RDNA[0], win[RDNA[0]], rd)],
        "chrXIV": [("chrXIV", win["chrXIV"])],
    }

    res = {"method": METHOD, "pileup": a.pileup, "pileup_rows_total": int(nrows),
           "n_scored_rows_after_ref_filter": int(len(df)),
           "rows_dropped_ref_base_mismatch": mismatch,
           "underflow_min_n_seen": min_uf_n, "n_unique_kn_pairs_underflowing": int(len(uf_pairs)),
           "regions": {}}
    for r, d in regions.items():
        st = base_stats(d)
        ws = window_stats(parts[r])
        both = int(d[d["n"] > 0].duplicated(["chrom", "pos"]).sum())
        ws["positions_with_both_strands"] = both
        st["runs"] = ws
        res["regions"][r] = st

    per = {}
    for c, g in df.groupby("chrom", sort=False):
        s = g[g["n"] > 0]
        w = win[c]
        j = int(np.argmax(s["n"].to_numpy()))
        per[c] = {"n_scored_bases": int(len(s)), "max_n": int(s["n"].iloc[j]),
                  "max_n_pos_1based": int(s["pos"].iloc[j]), "max_n_strand": s["strand"].iloc[j],
                  "p99.99_n": float(np.percentile(s["n"], 99.99)),
                  "max_4window_min_n": int(w["wmin"].max()) if len(w["wmin"]) else 0,
                  "max_4window_min_n_all_kn_ge_0.8": int(w["wmin"][w["wkn"]].max()) if w["wkn"].any() else 0,
                  "underflow_bases": int(s["uf"].sum())}
    res["per_chrom"] = per

    dep = json.load(open(DEPTH))
    res["mnase_reference"] = {"source": DEPTH, "field": "reads", "reads": dep["reads"],
                              "note": "original audit: real MNase data peaks at 22 reads at two neighbouring "
                                      "positions (chrI long-fragment adjmax=22)"}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
