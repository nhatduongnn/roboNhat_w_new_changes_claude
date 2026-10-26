#!/usr/bin/env python3
"""
fiberCOP version of the C-audit toy model (robocop_c_audit/walkthrough/walkthrough.py:make_model).

Same 12-state HMM (background, TF1 fwd/rev of length 2, 5-state branching nucleosome, CENTRAL and
TF1-entry silent states), same transitions and PWMs. What changes is the data layers, to match
production fiberCOP (pkgvar/seq_maskoff_tune75, the "both" layer set):

  layer 0     sequence PWM (unchanged)
  layers 1-4  MNase/ATAC NB layers: OFF (= 1). robocopExtras.py:81-84 has those calls commented out.
  layer 5     Fiber-seq Watson: binom.pmf(k, n, p_state) at A positions, 0 elsewhere   (robocop.py:789,801,806)
  layer 6     Fiber-seq Crick:  binom.pmf(k, n, p_state) at T positions, 0 elsewhere
  then        zeros in layers 5 and 6 floored to 1e-30                                 (robocopExtras.py:104-106)

Per-state methylation probability p, read from the same pkls production loads (robocop.py:598-607):
  background  bg_params_open.pkl             Watson 0.2489, Crick 0.2647
  TF1         Abf1_murphy columns 2,3 of all_TFs_1000pealVal_params_pseudo_clc08.pkl
              forward block = own strand p; reverse block = other strand p, mirrored (robocop.py:659-670)
  nucleosome  nucleosome_params.pkl, positions NUC_POS (toy N1, N2, N3a, N3b, N4)

Sequence: AATTAA (the audit used AGCTAG). Fiber-seq only scores A (Watson) and T (Crick), and the C1
failure needs a run of neighbouring scored bases, so every position here is an A or a T. Exactly one
strand is scored at each position: Watson at the A's (0, 1, 4, 5), Crick at the T's (2, 3).

Emissions are built three ways:
  E        linear, floored     what production C receives        (n_vars = 7)
  LE_fl    log of E            truth for the floored model        (brute force / log-space "D0")
  LE_ex    exact logpmf        truth for the real model           (log-space "D1", ROBOCOP_LOG_EMISSION=exact)
"""
import os, sys, pickle, ctypes
import numpy as np
from ctypes import c_int, c_double
from numpy.ctypeslib import ndpointer
from scipy.stats import binom

np.long = np.int64                     # walkthrough.py / robocop.py use np.long (removed in NumPy >= 1.24)
HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, "..", ".."))
FORK = os.path.dirname(ANALYSIS)
WT = "/usr/project/xtmp/nd141/programs/robocop_c_code_test/robocop_c_audit/walkthrough"
_argv = sys.argv; sys.argv = ["x"]; sys.path.insert(0, WT)
import walkthrough as W                # read-only import of the audit's model builder
sys.argv = _argv
LONG = np.int64
D_ = ndpointer(np.double, flags="C_CONTIGUOUS"); L_ = ndpointer(LONG, flags="C_CONTIGUOUS")
LD_ = ndpointer(np.longdouble, flags="C_CONTIGUOUS")

SEQ = "AATTAA"
FLOOR = 1e-30
NUC_POS = [0, 36, 73, 73, 110]          # nucleosome_params index for toy N1, N2, N3a, N3b, N4
TF_COLS = [2, 3]                        # Abf1_murphy columns for TF1 p1, p2
N_VARS = 7

# ---- per-state p, exactly as robocop.py:628-683 assembles it, for the toy's 10 emitting states
def state_p():
    inp = os.path.join(ANALYSIS, "inputs")
    bg = pickle.load(open(os.path.join(inp, "bg_params_open.pkl"), "rb"))["p"]
    nu = pickle.load(open(os.path.join(inp, "nucleosome_params.pkl"), "rb"))["p"]
    tf = pickle.load(open(os.path.join(inp, "all_TFs_1000pealVal_params_pseudo_clc08.pkl"), "rb"))["p"]["Abf1_murphy"]
    out = {}
    for strand, other in (("watson_signal", "crick_signal"), ("crick_signal", "watson_signal")):
        p = np.zeros(10)
        p[0] = float(np.ravel(bg[strand]["A"])[0])
        fwd = np.asarray(tf[strand]["A"])[TF_COLS]
        rev = np.asarray(tf[other]["A"])[TF_COLS][::-1]
        p[1:3] = fwd; p[3:5] = rev
        p[5:10] = np.asarray(nu[strand]["A"])[NUC_POS]
        out[strand] = p
    return out

P = state_p()

def make_model(lib, seq=SEQ, n_w=None, k_w=None, n_c=None, k_c=None):
    """n_w/k_w: Watson coverage and methylated count per position (used only where base is A);
       n_c/k_c: Crick (used only where base is T). Returns the audit Model with fiber fields added."""
    T = len(seq)
    n_w = np.zeros(T, int) if n_w is None else np.asarray(n_w, int)
    k_w = np.zeros(T, int) if k_w is None else np.asarray(k_w, int)
    n_c = np.zeros(T, int) if n_c is None else np.asarray(n_c, int)
    k_c = np.zeros(T, int) if k_c is None else np.asarray(k_c, int)
    m = W.make_model(lib, seq=seq, short=(0,) * T, long_=(0,) * T)
    S, N = m.S, m.N
    E = np.ones((N_VARS, T, N))
    E[0] = m.E[0]                                    # sequence layer, built by the C helper
    LE_ex = np.zeros((N_VARS, T, N))
    with np.errstate(divide="ignore"):
        LE_ex[0, :, :S] = np.log(m.E[0, :, :S])
    m.raw_pmf = np.zeros((2, T, S))                  # binom.pmf before the floor, for display
    m.raw_log10 = np.full((2, T, S), np.nan)         # exact log10 pmf at scored positions
    for li, (layer, ref, n_, k_, strand) in enumerate(((5, 0, n_w, k_w, "watson_signal"), (6, 3, n_c, k_c, "crick_signal"))):
        p = P[strand]
        for t in range(T):
            if m.codes[t] == ref:                        # robocop.py:789/792
                E[layer, t, :S] = binom.pmf(k_[t], n_[t], p)            # robocop.py:801
                LE_ex[layer, t, :S] = binom.logpmf(k_[t], n_[t], p)
                m.raw_log10[li, t] = LE_ex[layer, t, :S] / np.log(10)
            else:
                E[layer, t, :S] = 0.0                                    # robocop.py:806 (exact model: log 1 = 0)
            m.raw_pmf[li, t] = E[layer, t, :S]
        E[layer][E[layer] == 0] = FLOOR                                  # robocopExtras.py:104-106
    with np.errstate(divide="ignore"):
        LE_fl = np.log(E)
    LE_fl[:, :, S:] = 0.0
    m.E = E; m.LE_fl = LE_fl; m.LE_ex = LE_ex
    m.n_w, m.k_w, m.n_c, m.k_c = n_w, k_w, n_c, k_c
    m.seq = seq
    return m

# ---- data scenarios -------------------------------------------------------------------------------
# Ordinary data: 25 reads at every position, 6 methylated (k/n = 0.24, background-like).
# The unscored strand's counts are ignored by the emission (robocop.py:789/792), so both are set alike.
ORD = dict(n_w=[25] * 6, k_w=[6] * 6, n_c=[25] * 6, k_c=[6] * 6)

def high(n, frac, positions=(1, 2, 3, 4), base=ORD):
    """Ordinary data, with coverage n and methylated fraction frac at the given positions (both strands)."""
    d = {k: list(v) for k, v in base.items()}
    k = int(round(frac * n))
    for t in positions:
        d["n_w"][t] = n; d["k_w"][t] = k
        d["n_c"][t] = n; d["k_c"][t] = k
    return d

# The scenarios used by every tab (verified 2026-10-13 with sweeps: see README.md)
SCENARIOS = [
    ("ordinary", "Ordinary data", "25 reads at every base, 6 methylated (k/n 0.24).", ORD),
    ("c1", "C1 case: 220 reads, fully methylated", "220 reads at positions 1-4, all 220 methylated. "
           "Fix B is wrong against the floored model; the audit's C1 fix is right.", high(220, 1.0)),
    ("c1x", "Stronger: 250 reads at all 6 bases", "250 reads at every base, all methylated. "
           "Even the C1 fix fails (DBL_MIN padding); only log space holds.", high(250, 1.0, positions=range(6))),
    ("rdna", "rDNA-like: 4,379 reads, unmethylated", "4,379 reads (the rDNA maximum) at positions 1-4, none methylated. "
             "The open states' binomials underflow to 0 and are floored to 1e-30.", high(4379, 0.0)),
]

# ---- libraries ---------------------------------------------------------------------------------------
LIB_PRENORM = os.path.join(FORK, "pkg/robocop/librobocop_prenorm.so")   # original algo.c (pre fix B)
LIB_FIXB = os.path.join(FORK, "pkg/robocop/librobocop.so")              # production fix B
LIB_LOGFULL = os.path.join(FORK, "pkg/robocop/librobocop_logfull.so")   # fork's log-space decoder
LIB_C1FIX = os.path.join(WT, "variants/librobocop_fixed.so")           # audit: bward rescales before silent loop
LIB_TRACED = os.path.join(WT, "build/librobocop_traced.so")              # audit: upstream algo.c + trace records
LOG_ABI = 20261005

def load_lin(path, v2=False):
    lib = W.load(path) if not v2 else ctypes.CDLL(path)
    if v2:
        lib.find_parents_and_children.argtypes = [L_] * 4 + [c_int, c_int, D_]
        lib.fward.argtypes = [D_, D_, D_, D_, c_int, c_int, c_int, c_int, L_, L_, L_, L_, c_int, c_int, c_int, c_int, D_, D_]
        lib.bward.argtypes = [D_, D_, D_, c_int, c_int, c_int, c_int, L_, L_, L_, L_, c_int, c_int, c_int, c_int, D_, D_]
        lib.calc_sr.argtypes = [D_, D_, c_int, LD_]; lib.calc_sr.restype = c_double
        lib.posterior_decoding_v2.argtypes = [D_, D_, D_, LD_, c_int, c_int, c_int, D_]
    lib._v2 = v2
    return lib

def run_lin(lib, m, E=None, trace=None):
    """Mirror of robocop.py posterior_forward_backward_loop for a linear library (n_vars from E)."""
    E = np.ascontiguousarray((m.E if E is None else E).astype(np.double))
    nv, n, N = E.shape; S = m.S
    Tm = m.T.astype(np.double)
    fsf = np.zeros(n); bsf = np.zeros(n); sr = np.zeros(n, dtype=np.longdouble)
    par = np.zeros((N, N), LONG); chi = np.zeros((N, N), LONG); npar = np.zeros(N, LONG); nchi = np.zeros(N, LONG)
    f = np.zeros((n, N)); b = np.zeros((n, N)); p = np.zeros((n, N))
    ms = m.tf_starts.astype(LONG); ml = m.tf_lens.astype(LONG)
    if trace:
        lib.trace_open(trace.encode())
    lib.find_parents_and_children(par, chi, npar, nchi, N, S, Tm)
    lib.fward(m.init.astype(float), Tm, E, m.end.astype(float), N, S, n, nv, par, npar, ms, ml, 1, 1, m.nuc_start, m.nuc_len, f, fsf)
    lib.bward(Tm, E, m.end.astype(float), N, S, n, nv, chi, nchi, ms, ml, 1, 1, m.nuc_start, m.nuc_len, b, bsf)
    ll = lib.calc_sr(fsf, bsf, n, sr)
    if lib._v2:
        lib.posterior_decoding_v2(f, b, bsf, sr, N, n, S, p)
    else:
        lib.posterior_decoding(f, b, bsf, sr, N, n, p)
    if trace:
        lib.trace_close()
    return dict(ftable=f, btable=b, sf=fsf, sb=bsf, sr=sr, post=p, ll=float(ll))

_LG = None
def run_logfull(m, LE):
    """The fork's log-space decoder (algo_log.c): fward_log / bward_log / posterior_decoding_log."""
    global _LG
    if _LG is None:
        _LG = ctypes.CDLL(LIB_LOGFULL)
        _LG.find_parents_and_children.argtypes = [L_] * 4 + [c_int, c_int, D_]
        _LG.fward_log.argtypes = [D_, D_, D_, D_, c_int, c_int, c_int, L_, L_, D_, D_, c_int]; _LG.fward_log.restype = c_int
        _LG.bward_log.argtypes = [D_, D_, D_, c_int, c_int, c_int, L_, L_, D_, D_, c_int]; _LG.bward_log.restype = c_int
        _LG.posterior_decoding_log.argtypes = [D_, D_, c_int, c_int, c_int, D_, c_int]; _LG.posterior_decoding_log.restype = c_int
    n, N, S = LE.shape[1], m.N, m.S
    le = np.ascontiguousarray(LE.sum(0)); le[:, S:] = 0.0
    Tm = m.T.astype(np.double)
    par = np.zeros((N, N), LONG); chi = np.zeros((N, N), LONG); npar = np.zeros(N, LONG); nchi = np.zeros(N, LONG)
    _LG.find_parents_and_children(par, chi, npar, nchi, N, S, Tm)
    lf = np.full((n, N), -np.inf); lb = np.full((n, N), -np.inf); sf = np.zeros(n); sb = np.zeros(n); p = np.zeros((n, N))
    rc = (_LG.fward_log(m.init.astype(float), Tm, le, m.end.astype(float), N, S, n, par, npar, lf, sf, LOG_ABI),
          _LG.bward_log(Tm, le, m.end.astype(float), N, S, n, chi, nchi, lb, sb, LOG_ABI),
          _LG.posterior_decoding_log(lf, lb, N, n, S, p, LOG_ABI))
    assert rc == (LOG_ABI,) * 3, rc        # each entry point returns ROBOCOP_LOG_ABI on success (algo_log.c:22)
    return dict(post=p, ll=float(sf.sum()))

# ---- brute force (the audit's enumerator, fed any log-emission tensor) --------------------------------
_src = open(os.path.join(WT, "variants/empirical.py")).read().split("SCEN = [")[0]
_g = {"__file__": os.path.join(WT, "variants/empirical.py")}
exec(_src, _g)
brute = _g["brute"]                     # brute(m, LE) -> (post, logP, ...)
