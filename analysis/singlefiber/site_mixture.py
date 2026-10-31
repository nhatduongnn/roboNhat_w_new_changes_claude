#!/usr/bin/env python
"""Per-site fraction-bound model for single-fiber TF calls (SINGLEFIBER_PLAN step 5a, 2026-10-30).

Built from /usr/project/xtmp/nd141/tf_power/mixture.py (the quick ERV46 version).

Idea
----
One read has only ~4-5 informative m6A calls inside a 14 bp ABF1 motif, so no single read can
beat a genome-wide TF prior (tf_power/evidence.json). But the ~150 reads at ONE site share that
site's occupancy. So fit the site's fraction bound f from all its reads, then call each read
with f as its prior.

Model, per site
---------------
Every read that spans [motif - FL, motif + FL] contributes its own-strand calls (Watson read:
reference A; Crick read: reference T), exactly the calls the single-fiber decoder sees
(read_calls.read_molecules). Each read belongs to one of three classes:

  bound       motif positions ~ Bernoulli(p_bound)   (the deconvolved footprint: the
              protection of a BOUND read, not the shipped 0.084 mean, which is diluted
              by unbound reads); flanks ~ Bernoulli(p_open)
  free        motif and flanks ~ Bernoulli(p_open)   (the site's own open rate, fitted)
  nucleosome  the 147 bp nucleosome m6A profile (strand-specific, nucleosome_params*.pkl),
              placed at every offset that covers the whole window, offsets averaged
              (the placement is not known per read)

EM fits (f_bound, f_free, f_nuc), p_open (floored at the background rate, else EM can relabel
a protected region as 'free' with p_open -> 0) and, if asked, p_bound. Bootstrap over reads
gives a CI on f_bound. Each read's posterior bound is its EM responsibility, i.e. Bayes with f
as the prior.

p_bound (2026-10-30, tf_compare)
--------------------------------
Fitting p_bound freely is weakly identified with ~4 motif calls per read: per site it lands on
the 0.001 floor at 30 of 42 MacIsaac sites, and a shared fit across sites does too (bound then
means 'zero m6A in the motif'). Default use: FIX p_bound = 0.02, the per-site fit at the two
best-studied ERV46 sites (0.024, 0.019). f is insensitive to the choice (0.02 vs 0.001: MacIsaac
median f 0.449 vs 0.425, Rossi AUROC 0.963 both; /usr/project/xtmp/nd141/tf_compare/mixture*/).
Caveat: the 3-class model has no 'nucleosome partly in the window' class.

Sequence term
-------------
The PWM score is identical for every read at a site, so it cannot separate reads: it would
multiply every read's bound odds by the same constant, which the fitted f absorbs. It is used
only to choose and gate which sites are tested (`pwm_best`, `scan_pwm`).

API
---
    S = Site(chrom, start1, end1, strand, name)          # motif interval, 1-based inclusive
    data = load_site(S, bam=BAM, fl=20)                  # SiteData: per-read calls + barcode
    fit = fit_site(data, p_bound=None|float, n_boot=200) # dict: f, p_open, p_bound, CI, per-read
    fit['per_read']['post_bound']                        # array, one per read in data.reads
    pb = fit_shared_pbound([data1, data2, ...])          # one p_bound across many sites
    cv = heldout_barcode(data, p_bound)                  # fit on 6 barcodes, score the 7th
    pwm_best(chrom, lo1, hi1)                            # best ABF1 placement in an interval
    scan_pwm(chrom)                                      # log10 LR ABF1 vs background, both strands

Run as a script it does nothing; the step-5 analysis lives in
/usr/project/xtmp/nd141/tf_compare/scripts/run_site_mixture.py.
"""
import os
import pickle
import sys
from dataclasses import dataclass, field

import numpy as np
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import read_calls as RC  # noqa: E402

BAM = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/combine_sequencing_runs/"
       "merged_Mar20_barcode01_Jun25_barcode21-24_May07_barcode03-04.bam")
FASTA = os.path.join(ANALYSIS, "inputs", "SacCer3.fa")
TRAINDIR = os.path.join(ANALYSIS, "robocop_train_fiberonly")
NUC_FILE = os.path.join(ANALYSIS, "inputs", "nucleosome_params.pkl")
BG_FILE = os.path.join(ANALYSIS, "inputs", "bg_params.pkl")

# Read group -> barcode. The merged BAM (combine_sequencing_runs/readme.txt) is a samtools
# merge of 7 per-barcode BAMs; colliding RG IDs got a hash suffix. Mapping verified by looking
# up the first chrI:62000-62700 read of each source BAM in the merged BAM (2026-10-30).
RG_BARCODE = {
    "3ffdd574fd4145ab4a8dc7b33935b88ebfe03a5c_dna_r10.4.1_e8.2_400bps_sup@v5.0.0": "Mar20_bc01",
    "17fcb8e6bc729434e1fdaff94773a0ed3140208b_dna_r10.4.1_e8.2_400bps_sup@v5.0.0": "Jun25_bc21",
    "17fcb8e6bc729434e1fdaff94773a0ed3140208b_dna_r10.4.1_e8.2_400bps_sup@v5.0.0-4908647B": "Jun25_bc22",
    "17fcb8e6bc729434e1fdaff94773a0ed3140208b_dna_r10.4.1_e8.2_400bps_sup@v5.0.0-5E498CC1": "Jun25_bc23",
    "17fcb8e6bc729434e1fdaff94773a0ed3140208b_dna_r10.4.1_e8.2_400bps_sup@v5.0.0-6EFDB6F6": "Jun25_bc24",
    "7499519127370e847160e7df34d4edd145921c58_dna_r10.4.1_e8.2_400bps_sup@v5.0.0": "May07_bc03",
    "7499519127370e847160e7df34d4edd145921c58_dna_r10.4.1_e8.2_400bps_sup@v5.0.0-54557EED": "May07_bc04",
}
BARCODES = sorted(set(RG_BARCODE.values()))
NUC_LEN = 147


# ---------------------------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------------------------
_cache = {}


def _fasta():
    if "fa" not in _cache:
        _cache["fa"] = pysam.FastaFile(FASTA)
    return _cache["fa"]


def ref_codes(chrom, start0, end0):
    """A=0 C=1 G=2 T=3 N=4 for reference [start0, end0)."""
    s = _fasta().fetch(chrom, start0, end0).upper()
    lut = {"A": 0, "C": 1, "G": 2, "T": 3}
    return np.array([lut.get(c, 4) for c in s], dtype=np.int64)


def nuc_profile(nuc_file=NUC_FILE):
    if ("nuc", nuc_file) not in _cache:
        p = pickle.load(open(nuc_file, "rb"))["p"]
        _cache[("nuc", nuc_file)] = {"watson": np.ravel(p["watson_signal"]["A"]).astype(float),
                                     "crick": np.ravel(p["crick_signal"]["A"]).astype(float)}
    return _cache[("nuc", nuc_file)]


def bg_rate(bg_file=BG_FILE):
    p = pickle.load(open(bg_file, "rb"))["p"]
    return float(np.mean([np.ravel(p["watson_signal"]["A"])[0], np.ravel(p["crick_signal"]["A"])[0]]))


def pwm_tables(traindir=TRAINDIR, tf="Abf1_murphy"):
    """log10 PWM-vs-background tables (fwd, rev), each (L, 5), from the trainDir's pwm_emission:
    the exact sequence emission the decoder uses (a decode never re-reads the MEME file)."""
    key = ("pwm", traindir, tf)
    if key not in _cache:
        d = pickle.load(open(os.path.join(traindir, "HMMconfig.pkl"), "rb"))
        i = list(d["tfs"]).index(tf)
        s, L = int(d["tf_starts"][i]), int(d["tf_lens"][i])
        e = d["pwm_emission"]
        with np.errstate(divide="ignore", invalid="ignore"):   # column 4 (N) is 0 everywhere
            bg = np.log10(e[0])
            fwd = np.log10(e[s:s + L]) - bg
            rev = np.log10(e[s + L:s + 2 * L]) - bg
        _cache[key] = (fwd, rev, L)
    return _cache[key]


def scan_pwm(chrom, traindir=TRAINDIR, tf="Abf1_murphy", start0=0, end0=None):
    """(starts1, score_fwd, score_rev): log10 LR at every placement start (1-based)."""
    fwd, rev, L = pwm_tables(traindir, tf)
    if end0 is None:
        end0 = _fasta().get_reference_length(chrom)
    c = ref_codes(chrom, start0, end0)
    n = len(c) - L + 1
    sf = np.zeros(n); sr = np.zeros(n)
    for j in range(L):
        sf += fwd[j, c[j:j + n]]
        sr += rev[j, c[j:j + n]]
    return np.arange(start0 + 1, start0 + 1 + n), sf, sr


def pwm_best(chrom, lo1, hi1, pad=3, traindir=TRAINDIR, tf="Abf1_murphy"):
    """Best placement whose motif overlaps [lo1-pad, hi1+pad]: (start1, end1, strand, log10LR)."""
    fwd, rev, L = pwm_tables(traindir, tf)
    st, sf, sr = scan_pwm(chrom, traindir, tf, start0=max(0, lo1 - 1 - pad - L), end0=hi1 + pad + L)
    ok = (st + L - 1 >= lo1 - pad) & (st <= hi1 + pad)
    st, sf, sr = st[ok], sf[ok], sr[ok]
    i, j = int(np.argmax(sf)), int(np.argmax(sr))
    if sf[i] >= sr[j]:
        return int(st[i]), int(st[i] + L - 1), "+", float(sf[i])
    return int(st[j]), int(st[j] + L - 1), "-", float(sr[j])


# ---------------------------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------------------------
@dataclass
class Site:
    chrom: str
    start1: int          # motif interval, 1-based inclusive
    end1: int
    strand: str = "+"
    name: str = ""

    @property
    def L(self):
        return self.end1 - self.start1 + 1


@dataclass
class SiteData:
    site: Site
    fl: int
    calls: np.ndarray            # (n_reads, L + 2 fl) int8: -1 no obs, 0 canonical, 1 m6A
    strands: np.ndarray          # 'watson' / 'crick'
    barcodes: np.ndarray
    names: np.ndarray
    motif_mask: np.ndarray       # bool over the window: True inside the motif
    info: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.calls)

    def subset(self, idx):
        idx = np.asarray(idx)
        return SiteData(self.site, self.fl, self.calls[idx], self.strands[idx], self.barcodes[idx],
                        self.names[idx], self.motif_mask, dict(self.info))


def load_site(site, bam=BAM, fl=20, min_mapq=0):
    """Reads spanning the whole window [start1 - fl, end1 + fl], own-strand calls."""
    w0 = site.start1 - 1 - fl                       # window [w0, w1) 0-based
    w1 = site.end1 + fl
    codes = ref_codes(site.chrom, w0, w1)
    mols, es, ee, stats = RC.read_molecules(bam, site.chrom, w0 + 1, w1, codes, flank=0,
                                            min_mapq=min_mapq)
    assert es == w0 and ee == w1, (es, ee, w0, w1)
    rg = {}
    with pysam.AlignmentFile(bam) as b:
        for r in b.fetch(site.chrom, w0, w1):
            if not (r.is_secondary or r.is_supplementary):
                rg[r.query_name] = r.get_tag("RG") if r.has_tag("RG") else None
    keep = [m for m in mols if m.ref_start0 <= w0 and m.ref_end0 >= w1]
    calls = np.array([m.calls for m in keep], dtype=np.int8).reshape(len(keep), w1 - w0)
    mask = np.zeros(w1 - w0, bool)
    mask[fl:fl + site.L] = True
    return SiteData(site=site, fl=fl, calls=calls,
                    strands=np.array([m.strand for m in keep]),
                    barcodes=np.array([RG_BARCODE.get(rg.get(m.name), "unknown") for m in keep]),
                    names=np.array([m.name for m in keep]), motif_mask=mask,
                    info=dict(n_overlapping=len(mols), n_spanning=len(keep), reader=stats))


# ---------------------------------------------------------------------------------------------
# likelihoods
# ---------------------------------------------------------------------------------------------
def _ll(k, n, p):
    return k * np.log(p) + (n - k) * np.log1p(-p)


def _counts(d):
    meth = d.calls == RC.METH
    obs = d.calls != RC.NO_OBS
    m = d.motif_mask
    return (meth[:, m].sum(1).astype(float), obs[:, m].sum(1).astype(float),
            meth[:, ~m].sum(1).astype(float), obs[:, ~m].sum(1).astype(float))


def nuc_loglik(d, nuc_file=NUC_FILE):
    """log P(read | nucleosome), the 147 bp profile averaged over every placement that covers
    the whole window (uniform over offsets). Watson reads use the Watson profile, Crick the Crick
    profile, both indexed in reference order (as evidence.py / the aggregate emission do)."""
    prof = nuc_profile(nuc_file)
    W = d.calls.shape[1]
    offs = np.arange(0, NUC_LEN - W + 1)
    out = np.empty(d.n)
    for s in ("watson", "crick"):
        sel = d.strands == s
        if not sel.any():
            continue
        P = np.stack([prof[s][o:o + W] for o in offs])            # (n_off, W)
        lp1, lp0 = np.log(P), np.log1p(-P)
        c = d.calls[sel]
        M1 = (c == RC.METH).astype(float); M0 = (c == RC.CANON).astype(float)
        L = M1 @ lp1.T + M0 @ lp0.T                                 # (reads, n_off)
        mx = L.max(1, keepdims=True)
        out[sel] = (mx[:, 0] + np.log(np.exp(L - mx).mean(1)))
    return out


def _estep(km, nm, kf, nf, lnuc, f, po, pb):
    L = np.vstack([_ll(km, nm, pb) + _ll(kf, nf, po),
                   _ll(km, nm, po) + _ll(kf, nf, po),
                   lnuc]).T
    Lf = L + np.log(np.maximum(f, 1e-300))
    mx = Lf.max(1, keepdims=True)
    R = np.exp(Lf - mx)
    tot = R.sum(1, keepdims=True)
    return R / tot, float((mx[:, 0] + np.log(tot[:, 0])).sum()), L


def em(cnt, lnuc, p_bound=None, p_open_floor=None, it=500, tol=1e-9, f0=None):
    """cnt = (km, nm, kf, nf). p_bound None -> fitted, else fixed. Returns params, R, loglik, L."""
    km, nm, kf, nf = cnt
    floor = bg_rate() if p_open_floor is None else p_open_floor
    f = np.array([1 / 3] * 3) if f0 is None else np.asarray(f0, float)
    po = 0.25
    pb = 0.05 if p_bound is None else float(p_bound)
    prev = -np.inf
    for _ in range(it):
        R, ll, L = _estep(km, nm, kf, nf, lnuc, f, po, pb)
        f = R.mean(0)
        po = (R[:, 0] * kf + R[:, 1] * (km + kf)).sum() / max((R[:, 0] * nf + R[:, 1] * (nm + nf)).sum(), 1e-12)
        po = float(np.clip(po, floor, 0.9))
        if p_bound is None:
            pb = float(np.clip((R[:, 0] * km).sum() / max((R[:, 0] * nm).sum(), 1e-12), 1e-3, po))
        if abs(ll - prev) < tol:
            break
        prev = ll
    R, ll, L = _estep(km, nm, kf, nf, lnuc, f, po, pb)
    return dict(f_bound=float(f[0]), f_free=float(f[1]), f_nuc=float(f[2]),
                p_open=po, p_bound=pb, loglik=ll), R, L


def fit_site(d, p_bound=None, n_boot=200, seed=0, nuc_file=NUC_FILE):
    """Fit one site. Returns a JSON-able dict plus per-read arrays under 'per_read'."""
    cnt = _counts(d)
    lnuc = nuc_loglik(d, nuc_file)
    par, R, L = em(cnt, lnuc, p_bound)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, d.n, d.n)
        boots.append(em(tuple(x[ix] for x in cnt), lnuc[ix], p_bound, it=200)[0]["f_bound"])
    lr_bf = (L[:, 0] - L[:, 1]) / np.log(10)                       # log10 bound : free
    lr_bn = (L[:, 0] - L[:, 2]) / np.log(10)                       # log10 bound : nucleosome
    post = R[:, 0]
    out = dict(par)
    out.update(n_reads=int(d.n), median_calls_motif=float(np.median(cnt[1])) if d.n else 0.0,
               median_calls_flanks=float(np.median(cnt[3])) if d.n else 0.0,
               f_bound_ci90=[float(np.quantile(boots, .05)), float(np.quantile(boots, .95))] if boots else None,
               post_median=float(np.median(post)), post_q90=float(np.quantile(post, .9)),
               frac_post_gt_0p5=float((post > 0.5).mean()), frac_post_gt_0p9=float((post > 0.9).mean()),
               post_sd=float(post.std()),
               per_read=dict(post_bound=post, post_free=R[:, 1], post_nuc=R[:, 2],
                             log10LR_bound_free=lr_bf, log10LR_bound_nuc=lr_bn,
                             names=d.names, barcodes=d.barcodes, strands=d.strands))
    return out


def fit_shared_pbound(datas, it=300, nuc_file=NUC_FILE):
    """One p_bound shared by many sites (each keeps its own f and p_open): alternate per-site EM
    at fixed p_bound with a pooled M-step for p_bound. Returns p_bound."""
    pre = [(_counts(d), nuc_loglik(d, nuc_file)) for d in datas]
    pb = 0.05
    for _ in range(it):
        num = den = 0.0
        for cnt, ln in pre:
            par, R, _ = em(cnt, ln, pb, it=100)
            num += (R[:, 0] * cnt[0]).sum(); den += (R[:, 0] * cnt[1]).sum()
        new = float(np.clip(num / max(den, 1e-12), 1e-3, 0.5))
        if abs(new - pb) < 1e-6:
            pb = new
            break
        pb = new
    return pb


def heldout_barcode(d, p_bound, nuc_file=NUC_FILE, min_reads=5):
    """For each barcode: fit (f, p_open) on the OTHER six barcodes, then on the held-out reads
    report (i) f fitted on the held-out reads alone, (ii) held-out log-likelihood per read under
    the trained model minus that under the same model with f_bound forced to 0 (refit),
    (iii) the held-out reads' posterior bound using the trained f as prior."""
    cnt = _counts(d)
    lnuc = nuc_loglik(d, nuc_file)
    rows = []
    for b in BARCODES:
        te = d.barcodes == b
        tr = ~te
        if te.sum() < min_reads or tr.sum() < 3 * min_reads:
            continue
        sub = lambda m: tuple(x[m] for x in cnt)
        p_tr, _, _ = em(sub(tr), lnuc[tr], p_bound)
        p_te, _, _ = em(sub(te), lnuc[te], p_bound)
        # null: no bound class, trained on the same six barcodes
        p0, _, _ = em(sub(tr), lnuc[tr], p_bound, f0=[1e-12, 0.5, 0.5])
        f_tr = np.array([p_tr["f_bound"], p_tr["f_free"], p_tr["f_nuc"]])
        R1, ll1, _ = _estep(*sub(te), lnuc[te], f_tr, p_tr["p_open"], p_bound)
        f0 = np.array([0.0, p0["f_free"], p0["f_nuc"]]); f0 = f0 / f0.sum()
        _, ll0, _ = _estep(*sub(te), lnuc[te], f0, p0["p_open"], p_bound)
        rows.append(dict(barcode=b, n_test=int(te.sum()), f_train=p_tr["f_bound"], f_test=p_te["f_bound"],
                         dll_per_read=(ll1 - ll0) / te.sum(),
                         test_frac_post_gt_0p5=float((R1[:, 0] > 0.5).mean())))
    return rows
