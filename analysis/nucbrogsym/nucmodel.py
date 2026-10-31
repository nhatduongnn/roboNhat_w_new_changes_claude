#!/usr/bin/env python
"""Shared helpers for the strand-symmetric Brogaard nucleosome model (2026-10-28).

Coordinates. A BED row `chrIV 437453 437454` has start value S = 437453. "offset o" means the
nucleosome centre is the 1-based base S + o, i.e. 0-based index S + o - 1, and the 147-mer is
0-based [S+o-1-73, S+o-1+74). Conventions in use before this work:
  * sequence model, calc_dinucleotide.py:131,147-149: dyad = int((s+e)/2) = S,
    seq = chr[S-1-73 : S+73]  ->  o = 0
  * m6A profile, abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:951-953 with the 0-based modkit
    pileup: Start = S-73 (0-based) is position 0, so the centre is 0-based S -> o = +1

The model layout is upstream's (calc_dinucleotide.py): 9 single positions (0..8), a start
distribution at position 9, 127 position-specific dinucleotide transitions 9->10 ... 135->136,
then 10 single positions (137..146). 531 states. Fitting reuses upstream's own count code
(constructNucEmission / constructNucTransition / printDiNuc), loaded from the file.
"""
import importlib.util
import os

import numpy as np
from Bio import SeqIO

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(ANALYSIS)
FASTA = os.path.join(ANALYSIS, "inputs", "SacCer3.fa")
BROGAARD = ("/usr/project/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/Mar_19_24/"
            "Brogaard_nuc_positions.sacCer3.top2000.1bp_window.bed")
CHEREJI = os.path.join(ANALYSIS, "inputs", "Chereji_2018_+1_-1_nucs.bed")
ERV46 = ("chrI", 60001, 65000)
IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = str.maketrans("ACGTN", "TGCAN")

_spec = importlib.util.spec_from_file_location(
    "calc_dinucleotide_upstream", os.path.join(REPO, "pkg", "robocop", "nucleosome", "calc_dinucleotide.py"))
CD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CD)


def revcomp(s):
    return s.translate(COMP)[::-1]


def load_genome(fasta=FASTA):
    return {r.id: str(r.seq).upper() for r in SeqIO.parse(fasta, "fasta")}


def read_bed(path):
    rows = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 3:
                rows.append((t[0], int(t[1]), int(t[2])))
    return rows


def extract(rows, genome, offset):
    """147-mers centred on 1-based base S + offset. Drops short / N-containing windows."""
    seqs, dropped = [], 0
    for chrom, s, e in rows:
        c0 = s + offset - 1
        lo, hi = c0 - 73, c0 + 74
        g = genome.get(chrom)
        if g is None or lo < 0 or hi > len(g):
            dropped += 1
            continue
        q = g[lo:hi]
        if len(q) != 147 or "N" in q:
            dropped += 1
            continue
        seqs.append(q)
    return seqs, dropped


# ---------------------------------------------------------------------------
# symmetry statistics on a set of centred 147-mers
# ---------------------------------------------------------------------------
def base_counts(seqs):
    n = np.zeros((147, 4))
    for q in seqs:
        n[np.arange(147), [IDX[c] for c in q]] += 1
    return n


def dinuc_counts(seqs):
    n = np.zeros((146, 4, 4))
    for q in seqs:
        a = np.array([IDX[c] for c in q])
        np.add.at(n, (np.arange(146), a[:-1], a[1:]), 1)
    return n


def symmetry_chi2(seqs):
    """Pearson chi2 for 'base b at x  ==  complement(b) at -x' (single bases) and
    'dinucleotide ab starting at x == rc(ab) starting at -x-1' (dinucleotides).
    Each mirror pair counted once. Returns (chi2_base, df_base, chi2_dinuc, df_dinuc)."""
    nb = base_counts(seqs)
    c1, df1 = 0.0, 0
    for x in range(0, 74):                       # index 73+x vs 73-x
        for b in range(4):
            if x == 0 and b > 1:                 # centre: A<->T, C<->G, count each pair once
                continue
            o, m = nb[73 + x, b], nb[73 - x, 3 - b]
            if o + m > 0:
                c1 += (o - m) ** 2 / (o + m)
                df1 += 1
    nd = dinuc_counts(seqs)
    c2, df2 = 0.0, 0
    seen = set()
    for i in range(146):                         # dinuc at index i (i,i+1) <-> index 145-i
        j = 145 - i
        for a in range(4):
            for b in range(4):
                key = tuple(sorted([(i, a, b), (j, 3 - b, 3 - a)]))
                if key in seen or (i, a, b) == (j, 3 - b, 3 - a):
                    continue
                seen.add(key)
                o, m = nd[i, a, b], nd[j, 3 - b, 3 - a]
                if o + m > 0:
                    c2 += (o - m) ** 2 / (o + m)
                    df2 += 1
    return c1, df1, c2, df2


# ---------------------------------------------------------------------------
# model fit and scoring
# ---------------------------------------------------------------------------
def fit_counts(seqs):
    """Raw counts in upstream's layout: fst (9,4), strt (4,), dinuc (127,4,4), lst (10,4)."""
    fst = np.zeros((9, 4))
    lst = np.zeros((10, 4))
    strt = np.zeros(4)
    din = np.zeros((127, 4, 4))
    for q in seqs:
        a = np.array([IDX[c] for c in q])
        fst[np.arange(9), a[:9]] += 1
        lst[np.arange(10), a[137:]] += 1
        strt[a[9]] += 1
        np.add.at(din, (np.arange(127), a[9:136], a[10:137]), 1)
    return dict(fst=fst, strt=strt, dinuc=din, lst=lst)


def normalise(cnt):
    return dict(fst=cnt["fst"] / cnt["fst"].sum(1, keepdims=True),
                strt=cnt["strt"] / cnt["strt"].sum(),
                dinuc=cnt["dinuc"] / cnt["dinuc"].sum(2, keepdims=True),
                lst=cnt["lst"] / cnt["lst"].sum(1, keepdims=True))


def check_against_upstream(seqs, model):
    """Upstream's own functions on the same sequences must give the same numbers."""
    em = CD.constructNucEmission(seqs)
    strt, din = CD.constructNucTransition(seqs)
    assert np.allclose(em[:9, :4], model["fst"], rtol=0, atol=1e-15)
    assert np.allclose(em[-10:, :4], model["lst"], rtol=0, atol=1e-15)
    assert np.allclose(strt, model["strt"], rtol=0, atol=1e-15)
    assert np.allclose(din, model["dinuc"], rtol=0, atol=1e-15)
    return True


def nuc_emission_from(model):
    """Same array constructNucEmission builds (calc_dinucleotide.py:31-48)."""
    fst = np.hstack((model["fst"], np.zeros((9, 1))))
    lst = np.hstack((model["lst"], np.zeros((10, 1))))
    return np.vstack((fst, [[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0, 0, 1.0, 0, 0],
                            [0, 0, 0, 1.0, 0]] * 128, lst))


def model_from_trainfiles(dinuc_txt, emission_npy):
    """Read a trainDir's nuc_dinucleotide_model.txt + nuc_emission.npy back into a model."""
    em = np.load(emission_npy)
    strt = np.zeros(4)
    with open(dinuc_txt) as f:
        lines = [l.split() for l in f if not l.startswith("#")]
    # lines 0..7: 1->2 ... 8->9 (prob 1); lines 8..11: 9 -> 10..13 = strt
    for i in range(4):
        assert int(lines[8 + i][0]) == 9
        strt[i] = float(lines[8 + i][2])
    din = CD.read_dinuc_file(dinuc_txt)
    return dict(fst=em[:9, :4].copy(), strt=strt, dinuc=din, lst=em[-10:, :4].copy())


def logp(seq, model):
    """log P(147-mer | nucleosome model), natural log."""
    a = np.array([IDX[c] for c in seq])
    with np.errstate(divide="ignore"):
        v = (np.log(model["fst"][np.arange(9), a[:9]]).sum()
             + np.log(model["strt"][a[9]])
             + np.log(model["dinuc"][np.arange(127), a[9:136], a[10:137]]).sum()
             + np.log(model["lst"][np.arange(10), a[137:]]).sum())
    return v


def logp_from_hmm(seq, cfg):
    """log P(147-mer) read straight off an HMMconfig: transition_matrix + pwm_emission along the
    nucleosome path. The authoritative score of what the decoder uses."""
    T, E, ns = cfg["transition_matrix"], cfg["pwm_emission"], int(cfg["nuc_start"])
    a = [IDX[c] for c in seq]
    v = 0.0
    prev = None
    for p in range(147):
        if p < 9:
            j = ns + p
        elif p < 137:
            j = ns + 9 + 4 * (p - 9) + a[p]
        else:
            j = ns + 9 + 512 + (p - 137)
        if prev is not None:
            v += np.log(T[prev, j])
        v += np.log(E[j, a[p]])
        prev = j
    return v


def window_kmers(genome, chrom=ERV46[0], start1=ERV46[1], end1=ERV46[2]):
    g = genome[chrom][start1 - 1:end1]
    return [g[i:i + 147] for i in range(len(g) - 146) if "N" not in g[i:i + 147]]


def asymmetry(kmers, model=None, cfg=None):
    """mean |log10 P(s) - log10 P(rc s)| over the k-mers."""
    f = (lambda s: logp(s, model)) if model is not None else (lambda s: logp_from_hmm(s, cfg))
    d = np.array([f(s) - f(revcomp(s)) for s in kmers]) / np.log(10)
    return float(np.mean(np.abs(d))), d
