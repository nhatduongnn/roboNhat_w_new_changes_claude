#!/usr/bin/env python
"""Shared (run-independent) viewer tracks per chromosome: m6A, A-depth, sequence, genes, reference sites.

    python viewer_site/extract_shared.py [chrom ...] [--out DIR] [--ref-dir robocop_genome_ct_u001_00]

Writes <out>/shared/<chrom>.json.gz:
    chrom, length, ref_decode, blocks [{start, len, mw, mc, aw, ac, seq}]   (windows.tsv windows)
    genes  [{name, start, end, strand}]          whole chromosome (make_posterior_viewer.load_genes)
    refs   {key: {label, kind, color, sites}}    whole chromosome
             kind "interval": MacIsaac ABF1/REB1 bed as shipped ([start, end, strand], 1-based)
             kind "center":   per tuner-v2 group, MacIsaac c1 site centres and Rossi ChExMix _CX
                              summits (rossi_validate.load_references, as shipped)
and, for the Fiber-seq sameness check, /usr/project/xtmp/nd141/scratch_viewer/fiber_ref/<chrom>.npz:
the reference decode's counts over each window +/- 5 kb (first segment wins at each position).

Counts come from ONE reference decode (default robocop_genome_ct_u001_00, genome-wide, complete);
fiber_check.py verifies every run's per-segment hashes against it.
"""
import argparse
import glob
import gzip
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.dirname(HERE)
sys.path.insert(0, A)
OUT_DEFAULT = "/usr/project/xtmp/nd141/viewer_site"
SCRATCH = "/usr/project/xtmp/nd141/scratch_viewer"
PAD = 5000
KEYS = ("count_meth_watson", "count_meth_crick", "count_A_watson", "count_A_crick")
SHORT = ("mw", "mc", "aw", "ac")


def ref_counts(ref_dir, chrom, lo, hi):
    """-> (counts int32 4 x n over [lo, hi], covered bool n) from the reference decode."""
    import score_robocop as S
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(A, ref_dir, "config.ini"))
    tech2 = cfg.get("main", "tech2")
    coords = pd.read_csv(os.path.join(A, ref_dir, "coords.tsv"), sep="\t")
    want = coords[(coords["chr"] == chrom) & (coords["start"] <= hi) & (coords["end"] >= lo)]
    n = hi - lo + 1
    out = np.zeros((4, n), dtype=np.int64)
    cov = np.zeros(n, dtype=bool)
    for p in sorted(glob.glob(os.path.join(A, ref_dir, "tmpDir", "info_*_*.h5"))):
        with h5py.File(p, "r") as f:
            for idx, r in want.iterrows():
                k = "segment_%d" % idx
                if k not in f:
                    continue
                g = f[k]
                nrow = int(g["posterior"].attrs["shape"][0])
                s = int(r["start"])
                arrs = [np.asarray(S._get_sparse_todense(g, "%s_%s" % (tech2, kk)))[:nrow] for kk in KEYS]
                a, b = max(s, lo), min(s + nrow - 1, hi)
                sl = slice(a - lo, b - lo + 1)
                new = ~cov[sl]                        # first segment wins
                for j in range(4):
                    out[j, sl][new] = arrs[j][a - s:b - s + 1][new]
                cov[sl] = True
    return out, cov


def load_seq(chrom):
    from Bio import SeqIO                     # SeqIO, not faidx: the shipped .fai was broken
    for rec in SeqIO.parse(os.path.join(A, "inputs", "SacCer3.fa"), "fasta"):
        if rec.id == chrom:
            return str(rec.seq).upper()
    raise KeyError(chrom)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chroms", nargs="*")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--ref-dir", default="robocop_genome_ct_u001_00")
    a = ap.parse_args()
    import make_posterior_viewer as MPV
    import rossi_validate as RV
    wins = pd.read_csv(os.path.join(HERE, "windows.tsv"), sep="\t")
    chroms = a.chroms or list(dict.fromkeys(wins["chrom"]))
    sizes = dict(l.split()[:2] for l in open(os.path.join(A, "inputs", "sacCer3.chrom.sizes")))
    st = json.load(open(os.path.join(A, "conc_tuning", "bw01", "state.json")))
    refs_all, _ = RV.load_references(st["groups"])
    os.makedirs(os.path.join(a.out, "shared"), exist_ok=True)
    os.makedirs(os.path.join(SCRATCH, "fiber_ref"), exist_ok=True)
    for chrom in chroms:
        L = int(sizes[chrom])
        seq = load_seq(chrom)
        assert len(seq) == L, (chrom, len(seq), L)
        blocks, fr = [], {}
        for _, w in wins[wins["chrom"] == chrom].iterrows():
            ws, we = int(w["start"]), int(w["end"])
            lo, hi = max(1, ws - PAD), min(L, we + PAD)
            cnt, cov = ref_counts(a.ref_dir, chrom, lo, hi)
            fr["%d-%d" % (lo, hi)] = (lo, cnt, cov)
            sl = slice(ws - lo, we - lo + 1)
            if not cov[sl].all():
                print("WARNING %s: reference decode leaves %d bp of %d-%d uncovered"
                      % (chrom, int((~cov[sl]).sum()), ws, we))
            b = dict(start=ws, len=we - ws + 1, seq=seq[ws - 1:we])
            for j, k in enumerate(SHORT):
                b[k] = cnt[j, sl].astype(int).tolist()
            blocks.append(b)
        np.savez_compressed(os.path.join(SCRATCH, "fiber_ref", chrom + ".npz"),
                            spans=np.array([[v[0], v[0] + v[1].shape[1] - 1] for v in fr.values()]),
                            **{"cnt_%d" % i: v[1] for i, v in enumerate(fr.values())},
                            **{"cov_%d" % i: v[2] for i, v in enumerate(fr.values())},
                            ref_dir=np.array(a.ref_dir))
        refs = {}
        for key, bedname, color, label, _on in MPV.REF_TRACKS:
            sites = MPV.load_ref_sites(chrom, 1, L)[key]
            refs[key] = dict(label=label + " (bed)", kind="interval", color=color, sites=sites, group=bedname)
        for g, r in sorted(refs_all.items()):
            mac = sorted(r["macisaac"].get(chrom, [])) if r["macisaac"] else []
            if mac:
                refs["mac_" + g] = dict(label="MacIsaac c1 %s" % g, kind="center", color=MPV.to_hex(MPV.color_for_name(g)),
                                        sites=[int(x) for x in mac], group=g)
            cx = sorted(r["rossi_cx"].get(chrom, [])) if r["rossi_cx"] else []
            if cx:
                refs["cx_" + g] = dict(label="Rossi _CX %s" % g, kind="center", color=MPV.to_hex(MPV.color_for_name(g)),
                                       sites=[int(x) for x in cx], group=g)
        out = dict(chrom=chrom, length=L, ref_decode=a.ref_dir, blocks=blocks,
                   genes=MPV.load_genes(chrom, 1, L), refs=refs)
        with gzip.open(os.path.join(a.out, "shared", chrom + ".json.gz"), "wt") as fh:
            json.dump(out, fh, separators=(",", ":"))
        print("%s: %d blocks, %d genes, %d ref tracks" % (chrom, len(blocks), len(out["genes"]), len(refs)))
    # factor colours (plotRoboCOP._color_for_name recipe, via make_posterior_viewer) + motif -> group
    import pickle
    tfs = list(pickle.load(open(os.path.join(A, "robocop_train_fiberonly", "pwm.p"), "rb")).keys())
    names = set(tfs) | {"Zz_decoy_abf1", "unknown"}
    colors = {n: MPV.to_hex(MPV.color_for_name(n)) for n in sorted(names)}
    colors.update({k: v for k, v in MPV.SPECIAL.items()})
    colors["other TFs"] = "#6b7785"
    groups = {m: g for g, ms in st["groups"].items() for m in ms}
    groups["Zz_decoy_abf1"] = "ABF1"
    with open(os.path.join(a.out, "colors.json"), "w") as fh:
        json.dump(dict(colors=colors, groups=groups,
                       note="colour = make_posterior_viewer.color_for_name (plotRoboCOP._color_for_name); "
                            "groups = conc_tuning/bw01/state.json groups"), fh)


if __name__ == "__main__":
    main()
