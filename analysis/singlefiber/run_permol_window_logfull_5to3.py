#!/usr/bin/env python
"""Single-fiber RoboCOP, log space, every read decoded 5'->3' on its own strand (2026-10-26).

Twin of run_permol_window_logfull.py (left unchanged; its decode_one, posterior grab, savers and
validity check are IMPORTED from it, so `--direction ref --start motif1st` is that driver bit for
bit). Three additions:

1. --direction {5to3, ref}  (default 5to3)
   ref   : every read decoded left->right in reference coordinates; a Crick read's calls go in
           layer 6 (Crick), matched on reference T. Exactly the old driver.
   5to3  : Watson reads unchanged. A Crick read is decoded in ITS OWN 5'->3' frame:
             sequence = reverse complement of the reference segment,
             calls    = the read's calls reversed, placed in layer 5 (Watson): a Crick A is an A
                        of the reverse complement,
             same model (transitions, PWMs, nucleosome model, start/end rule).
           The posterior is mapped back to reference coordinates: rows reversed, every emitting
           state replaced by its mirror (see state_mirror), so permol.npz / info_0_1.h5 have the
           same layout as the ref run.
   --flip-ps {crick_mirror, watson}  (default crick_mirror): the m6A ps used on layer 5 in the
           flipped frame. crick_mirror = ps_crick[mirror(j)], i.e. the SAME per-state Bernoulli
           the ref decode uses for that read (the read's evidence is unchanged; only the decoding
           direction changes). watson = ps_watson as is (one strand-agnostic read-frame model).
           For TF states the two are identical (ps_crick = mirror of ps_watson by construction,
           permol_params.py:84-87); for background and nucleosome they differ (checked at run
           time, printed, and stored in run_stats.json).

2. Consensus flags (consensus_seq.py; user decision 2026-10-28): the decode ALWAYS uses the
   reference sequence, with --consensus on or off. --consensus on only FLAGS positions:
   flagged_alt (coverage >= 10 and one non-reference base carried by > 50% of covering reads),
   flagged_low_cov (coverage < 10) and flagged_indel (majority deletion / insertion). Written to
   consensus_sites.tsv and run_stats.json["consensus_summary"] (n_flagged_alt, ...). Nothing
   else in the decode changes. --consensus off skips the flagging.

3. --start {motif1st, any}  (default any since 2026-10-30, user-approved D1; motif1st = the old
   rule, bc.c:146-200 with only_allow_start_from_motif_1st_position = 1). any = the stationary distribution of the HMM
   over EMITTING states (silent states collapsed out), i.e. the state at a random position of an
   infinitely long chromosome. This is the counterpart of end_probs = 1 (end anywhere,
   robocop.py:918-923): a read is a window cut out of a longer chain at BOTH ends. Only
   dshared['initial_probs'] is overridden; it is read afresh on every decode
   (robocop.py:947) and is not part of the parents/children cache (robocop.py:1014-1023, keyed
   by id(transition_matrix), stores parents/children only). No C or pkgvar change.

4. Single-fiber DEFAULT MODEL (2026-10-30, plan okay-so-what-are-purring-tulip steps 2-3):
     --traindir   robocop_train_fiberonly_nucbrogsym            (Brogaard symmetric sequence model)
     --pkg-tree   pkgvar/permol_seq_maskoff_logfull_nucbrogsym_all7
     --nuc-params nucleosome_params_brogaard_sym_all7.pkl       (Brogaard-centred, W/C-averaged, 7 barcodes)
     --bg-params  bg_params_all7.pkl                            (linker background, 7 barcodes)
     --start any, --direction 5to3
   TF footprint m6A params are unchanged (inputs/all_TFs_1000pealVal_params_pseudo.pkl).
   The tree reads its own nuc/bg files; assert_verbatim refuses a --nuc-params / --bg-params that
   does not match the --pkg-tree.

Usage (from analysis/):

    ROBOCOP_LOG_EMISSION=exact python singlefiber/run_permol_window_logfull_5to3.py --arm A --out <dir>

5. FILTERED, rDNA-FREE DEFAULT MODEL (2026-11-02): tree pkgvar/permol_seq_maskoff_logfull_nucbrogsym_filtnordna,
   params bg_params_filtnordna.pkl / nucleosome_params_brogaard_sym_filtnordna.pkl /
   all_TFs_1000pealVal_params_pseudo_filtnordna.pkl, keep list filtered_refit_nordna/keep_reads.txt (rDNA reads
   chrXII:451,418-468,929 removed before the MAD fence). The intermediate _filt model (rDNA kept) is
   reproducible with --pkg-tree ..._filt, the _filt pkls and --keep-reads /usr/project/xtmp/nd141/filtered_refit/keep_reads.txt.
   Earlier text, kept for the record: tree ..._filt, params
   bg_params_filt.pkl / nucleosome_params_brogaard_sym_filt.pkl / all_TFs_1000pealVal_params_pseudo_filt.pkl
   (all refit on the filtered 7-barcode pileup), read filters --min-mapq 20 --min-aln-len 500 --min-qs 10
   --keep-reads filtered_refit/keep_reads.txt, --call-thresh 0.8105469. Reproduce the 10-30 all7 default:

    ROBOCOP_LOG_EMISSION=exact python singlefiber/run_permol_window_logfull_5to3.py --arm A --out <dir> --pkg-tree pkgvar/permol_seq_maskoff_logfull_nucbrogsym_all7 --nuc-params nucleosome_params_brogaard_sym_all7.pkl --bg-params bg_params_all7.pkl --tf-params all_TFs_1000pealVal_params_pseudo.pkl --min-mapq 0 --min-aln-len 0 --min-qs -1 --keep-reads none

Reproduce an old (pre-2026-10-30 default) run, e.g. the old driver run_permol_window_logfull.py bit for bit:

    ROBOCOP_LOG_EMISSION=exact python singlefiber/run_permol_window_logfull_5to3.py --arm A --out <dir> --traindir robocop_train_fiberonly --pkg-tree pkgvar/permol_seq_maskoff_logfull --nuc-params nucleosome_params.pkl --bg-params bg_params.pkl --start motif1st --direction ref

(drop --direction ref for an old 5to3 run; Brogaard barcode01 runs: --traindir
robocop_train_fiberonly_nucbrogsym --pkg-tree pkgvar/permol_seq_maskoff_logfull_nucbrogsym
--nuc-params nucleosome_params_brogaard_sym.pkl --bg-params bg_params.pkl).

6. --local-prior <sites_w.tsv>  (2026-11-03, default OFF; plan let-s-go-with-mad-snazzy-lollipop Part 2).
   Site-specific TF entry weight w (local_prior.py): both first states of the TF at the motif-start row
   weighted by w, i.e. a w-times larger entry probability at that position only. Written into the per-read
   log emission through decode_one's emission_hook (after --seq-layer's hook), layer 1 (otherwise 0.0), as
   -log(w) on every OTHER emitting state for w > 1 (each layer must be <= 0, robocop.py:970-973). Row/state
   mapping for ref-frame and flipped (Crick 5'->3') reads in local_prior.placements. A table with
   heldout_run rows (Mar20 / May07 / Jun25) gives each read the w fitted with ITS OWN run held out (read
   group -> barcode -> run, site_mixture.RG_BARCODE). w comes from calibrate_local_prior.py. Off = the
   driver is unchanged (no hook, same molecules.tsv columns); run_stats/RUN_INFO record local_prior = null.

Outputs as run_permol_window_logfull.py, plus consensus_sites.tsv; molecules.tsv gains
`frame` (ref / flipped); run_stats.json gains direction, start, flip_ps, consensus (on/off),
consensus_summary, start_info, ps_mirror_vs_watson_max_abs, n_flipped.
TEST-ONLY flags: --seq-layer off (layer 0 = log 1 on emitting states), --ps-symmetric (Crick ps
:= mirror of Watson ps on every state), --cols all (save every collapsed column).
"""
import argparse
import datetime
import json
import os
import pickle
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
os.chdir(ANALYSIS)                      # the package reads 'inputs/...' relatively
sys.path.insert(0, 'pkgvar/permol_seq_maskoff_logfull_nucbrogsym_filtnordna/')   # literal, so write_run_info.driver_tree resolves it
TREE_DEFAULT = "pkgvar/permol_seq_maskoff_logfull_nucbrogsym_filtnordna"   # 2026-11-02 default (rDNA-free); earlier: ..._filt, 10-30: ..._all7
TRAINDIR_DEFAULT = "robocop_train_fiberonly_nucbrogsym"
NUC_PARAMS_DEFAULT = "nucleosome_params_brogaard_sym_filtnordna.pkl"     # 10-30: nucleosome_params_brogaard_sym_all7.pkl
BG_PARAMS_DEFAULT = "bg_params_filtnordna.pkl"                           # 10-30: bg_params_all7.pkl
TF_PARAMS_DEFAULT = "all_TFs_1000pealVal_params_pseudo_filtnordna.pkl"   # 10-30: all_TFs_1000pealVal_params_pseudo.pkl
# Read filters (2026-11-02 defaults; 10-30 = none: --min-mapq 0 --min-aln-len 0 --min-qs -1 --keep-reads none).
# The keep list is the read set the _filt params were fitted on (/usr/project/xtmp/nd141/filtered_refit/:
# nuclear, MAPQ>=20, aligned>=500, qs>=10, pooled median+-4*1.4826*MAD per-read m6A cap). Threshold
# 0.8105469 = modkit sample-probs on the filtered BAM (threshold.txt), equal to read_calls.THRESH.
MIN_MAPQ_DEFAULT, MIN_ALN_LEN_DEFAULT, MIN_QS_DEFAULT = 20, 500, 10.0
KEEP_READS_DEFAULT = "/usr/project/xtmp/nd141/filtered_refit_nordna/keep_reads.txt"   # rDNA reads excluded; earlier: filtered_refit/
CALL_THRESH_DEFAULT = 0.8105469
# --pkg-tree (default: the literal tree above) must be known BEFORE `import robocop`.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--pkg-tree", default=None)
PKG_TREE_OVERRIDE = _pre.parse_known_args()[0].pkg_tree
if PKG_TREE_OVERRIDE:
    assert os.path.isdir(os.path.join(PKG_TREE_OVERRIDE, "robocop")), PKG_TREE_OVERRIDE
    sys.path.insert(0, PKG_TREE_OVERRIDE.rstrip("/") + "/")
sys.path.insert(0, HERE)

import h5py                                                       # noqa: E402
import robocop                                                    # noqa: E402
from robocop.utils.getNucleotides import getNucleotideSequence     # noqa: E402

import collapse as C                                              # noqa: E402
import consensus_seq as CS                                        # noqa: E402
import local_prior as LP                                          # noqa: E402
import permol_params as PP                                        # noqa: E402
import read_calls as RC                                           # noqa: E402
import run_permol_window_logfull as D                             # noqa: E402  (old driver, unchanged)

BAM, FASTA = D.BAM, D.FASTA
TREE = TREE_DEFAULT
if PKG_TREE_OVERRIDE:
    TREE = PKG_TREE_OVERRIDE.rstrip("/")
assert os.path.samefile(os.path.dirname(os.path.dirname(robocop.__file__)), TREE), \
    "imported robocop from %s, expected tree %s" % (robocop.__file__, TREE)
LOGFULL_SO, LOGFULL_SO_MD5 = D.LOGFULL_SO, D.LOGFULL_SO_MD5
ROW_SUM_TOL = D.ROW_SUM_TOL
KEEP_COLS = D.KEEP_COLS
NUC_LEN_BP = 147             # nucleosome positions; states: 9 single, 128 x 4 (one per base), 10 single
NUC_N_FIRST, NUC_N_MID = 9, 128


# ---------------------------------------------------------------------------
# Mirror map between the reference frame and a read's reverse-complement frame
# ---------------------------------------------------------------------------
def nuc_state(ns, p, b=0):
    """State index of nucleosome position p (0..146), base b for the 4-state positions 9..136
    (nuc_emission rows are A,C,G,T, calc_dinucleotide.py constructNucEmission)."""
    if p < NUC_N_FIRST:
        return ns + p
    if p < NUC_N_FIRST + NUC_N_MID:
        return ns + NUC_N_FIRST + 4 * (p - NUC_N_FIRST) + b
    return ns + NUC_N_FIRST + 4 * NUC_N_MID + (p - NUC_N_FIRST - NUC_N_MID)


def nuc_pos(ns, j):
    k = j - ns
    if k < NUC_N_FIRST:
        return k, None
    if k < NUC_N_FIRST + 4 * NUC_N_MID:
        return NUC_N_FIRST + (k - NUC_N_FIRST) // 4, (k - NUC_N_FIRST) % 4
    return NUC_N_FIRST + NUC_N_MID + (k - NUC_N_FIRST - 4 * NUC_N_MID), None


def state_mirror(dshared):
    """sigma: flipped-frame state -> reference-frame state, as an int array over ALL states.

    background -> background; TF i block [s, s+2L): s+k -> s+2L-1-k (forward motif position k
    read on the other strand is reverse-motif position L-1-k; the reverse block's PWM is
    reverse_complement(pwm), robocop.py:278-279, so the sequence emission is mirrored exactly);
    nucleosome position p -> 146-p, base b -> 3-b (complement). The nucleosome model is NOT
    mirror-shaped: 9 single states, 128 x 4, then 10 single. So:
      * flipped p = 9 (4 states) -> ref 137 (single): the 4 states MERGE (sigma not injective);
      * flipped p = 137 (single) -> ref 9 (4 states): no unique target. sigma gives -1 there and
        map_posterior() puts that mass on the ref state whose base is the reference-frame base
        of that row (split 1/4 each if the base is N).
    Silent states map to themselves (their posterior is 0 in log space).
    Returns (sigma, special) with special = the flipped state index of position 137.
    """
    n, ssb, ns = dshared["n_states"], dshared["silent_states_begin"], dshared["nuc_start"]
    assert dshared["nuc_len"] == NUC_N_FIRST + 4 * NUC_N_MID + 10 and ns + dshared["nuc_len"] == ssb
    sig = np.arange(n)
    for i in range(dshared["n_tfs"]):
        s, L = int(dshared["tf_starts"][i]), int(dshared["tf_lens"][i])
        sig[s:s + 2 * L] = np.arange(s + 2 * L - 1, s - 1, -1)
    special = nuc_state(ns, NUC_N_FIRST + NUC_N_MID)        # p = 137
    for j in range(ns, ns + dshared["nuc_len"]):
        p, b = nuc_pos(ns, j)
        q = NUC_LEN_BP - 1 - p
        q_mid = NUC_N_FIRST <= q < NUC_N_FIRST + NUC_N_MID
        if q_mid and b is None:
            assert j == special
            sig[j] = -1
        elif q_mid:
            sig[j] = nuc_state(ns, q, 3 - b)
        else:
            sig[j] = nuc_state(ns, q)
    return sig, special


def mirror_ps(ps, sigma, special, dshared):
    """ps in the other frame: out[j] = ps[sigma(j)]; the special state takes the (shared) value of
    ref position 9. Asserts the 4 states of every nucleosome position share one value."""
    ns = dshared["nuc_start"]
    mid = ps[ns + NUC_N_FIRST: ns + NUC_N_FIRST + 4 * NUC_N_MID].reshape(NUC_N_MID, 4)
    assert (mid == mid[:, :1]).all(), "nucleosome ps differ between the 4 base states of a position"
    sg = sigma[:len(ps)].copy()
    sg[special] = nuc_state(ns, NUC_N_FIRST, 0)
    return ps[sg].copy()


def map_posterior(p_flip, sigma, special, codes_ref, dshared):
    """Flipped-frame posterior (rows = flipped positions) -> reference frame."""
    n_obs, n = p_flip.shape
    ns = dshared["nuc_start"]
    pr = p_flip[::-1]
    sg = sigma.copy()
    keep = sg >= 0
    R = sparse.csr_matrix((np.ones(int(keep.sum())), (np.where(keep)[0], sg[keep])), shape=(n, n))
    out = np.asarray(pr @ R)          # sums merged columns (flipped p=9 -> ref 137)
    col = pr[:, special]
    codes_ref = np.asarray(codes_ref)
    for b in range(4):
        rows = codes_ref == b
        out[rows, nuc_state(ns, NUC_N_FIRST, b)] += col[rows]
    rows = codes_ref > 3
    if rows.any():
        for b in range(4):
            out[rows, nuc_state(ns, NUC_N_FIRST, b)] += col[rows] / 4.0
    return out


def revcomp_codes(c):
    c = np.asarray(c)
    return np.where(c < 4, 3 - c, c)


# ---------------------------------------------------------------------------
# Stationary start
# ---------------------------------------------------------------------------
def stationary_initial(dshared):
    """Stationary distribution of the HMM over emitting states, silent states collapsed.

    One step between emitting positions is T_EE + T_ES (I - T_SS)^-1 T_SE (silent states are
    visited between two positions, fward_log algo_log.c:142-148). pi solves pi P = pi, sum 1.
    Rows of the shipped matrix sum to 1 only to ~4e-8 (nucleosome file is float32), so P is
    not exactly stochastic; the eigenvector for the top eigenvalue is used, found by a linear
    solve with the normalisation replacing one equation, and polished by power iteration.
    """
    T = np.asarray(dshared["transition_matrix"], dtype=np.float64)
    ssb = dshared["silent_states_begin"]
    n = T.shape[0]
    TEE, TES, TSE, TSS = T[:ssb, :ssb], T[:ssb, ssb:], T[ssb:, :ssb], T[ssb:, ssb:]
    P = TEE + TES @ np.linalg.solve(np.eye(n - ssb) - TSS, TSE)
    rs = P.sum(axis=1)
    A = P.T - np.eye(ssb)
    A[-1, :] = 1.0
    rhs = np.zeros(ssb)
    rhs[-1] = 1.0
    pi = np.linalg.solve(A, rhs)
    for _ in range(50):                     # polish (keeps pi >= 0 to rounding)
        pi = pi @ P
        pi = pi / pi.sum()
    resid = float(np.abs(pi @ P / (pi @ P).sum() - pi).max())
    assert pi.min() > 0, "stationary distribution has a non-positive entry (%g)" % pi.min()
    init = np.zeros(n)
    init[:ssb] = pi
    ns = dshared["nuc_start"]
    info = dict(P_row_sum_min=float(rs.min()), P_row_sum_max=float(rs.max()),
                residual_max=resid, pi_min=float(pi.min()),
                pi_background=float(pi[0]),
                pi_nucleosome=float(pi[ns:ns + dshared["nuc_len"]].sum()),
                pi_tfs=float(pi[1:ns].sum()))
    return init, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--pi", type=float, default=None, help="arm B mixing fraction")
    ap.add_argument("--eff", type=int, default=0, choices=[0, 1],
                    help="1 = per-read efficiency correction on")
    ap.add_argument("--window", default="chrI:60001-65000")
    ap.add_argument("--traindir", default=TRAINDIR_DEFAULT,
                    help="default %s (old runs: robocop_train_fiberonly)" % TRAINDIR_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bam", default=BAM)
    ap.add_argument("--flank", type=int, default=400)
    ap.add_argument("--max-molecules", type=int, default=0, help="0 = all; >0 for a smoke test")
    ap.add_argument("--min-span", type=int, default=1,
                    help="skip molecules whose in-window span is shorter than this")
    ap.add_argument("--direction", default="5to3", choices=["5to3", "ref", "rev"],
                    help="rev = TEST ONLY: every read (Watson too) decoded right->left in reference coordinates, for the both-directions check against ref")
    ap.add_argument("--start", default="any", choices=["motif1st", "any"],
                    help="any (default since 2026-10-30) = stationary start; motif1st = old rule")
    ap.add_argument("--flip-ps", default="crick_mirror", choices=["crick_mirror", "watson"])
    ap.add_argument("--consensus", default="on", choices=["on", "off"],
                    help="on: write consensus_sites.tsv flags (flag only; the decode always uses the reference)")
    ap.add_argument("--cols", default="keep", choices=["keep", "all"],
                    help="all = save every collapsed column in permol.npz (tests)")
    ap.add_argument("--seq-layer", default="on", choices=["on", "off"], help="TEST ONLY")
    ap.add_argument("--ps-symmetric", action="store_true", help="TEST ONLY")
    ap.add_argument("--pkg-tree", default=None,
                    help="package tree to import (default %s); " % TREE_DEFAULT +
                         "parsed before import, listed here for --help and RUN_INFO")
    ap.add_argument("--nuc-params", default=NUC_PARAMS_DEFAULT,
                    help="nucleosome m6A profile in inputs/ (default %s). Must be "
                         "the file the --pkg-tree reads (assert_verbatim checks it)" % NUC_PARAMS_DEFAULT)
    ap.add_argument("--bg-params", default=BG_PARAMS_DEFAULT,
                    help="background m6A params in inputs/ (default %s; old runs bg_params.pkl). Must be "
                         "the file the --pkg-tree reads (assert_verbatim checks it)" % BG_PARAMS_DEFAULT)
    ap.add_argument("--tf-params", default=TF_PARAMS_DEFAULT,
                    help="TF footprint m6A params in inputs/ (default %s). Must be the file the "
                         "--pkg-tree reads (assert_verbatim checks it)" % TF_PARAMS_DEFAULT)
    ap.add_argument("--min-mapq", type=int, default=MIN_MAPQ_DEFAULT)
    ap.add_argument("--min-aln-len", type=int, default=MIN_ALN_LEN_DEFAULT,
                    help="minimum aligned bases (query_alignment_length)")
    ap.add_argument("--min-qs", type=float, default=MIN_QS_DEFAULT, help="minimum dorado qs tag")
    ap.add_argument("--keep-reads", default=KEEP_READS_DEFAULT,
                    help="file of read names to keep (the filtered-refit keep list); none = no list")
    ap.add_argument("--call-thresh", type=float, default=CALL_THRESH_DEFAULT,
                    help="m6A call threshold (default read_calls.THRESH)")
    ap.add_argument("--local-prior", default=None,
                    help="sites_w.tsv from calibrate_local_prior.py: site-specific TF entry weights "
                         "(default off; see docstring item 6)")
    a = ap.parse_args(argv)
    if a.keep_reads in ("none", ""):
        a.keep_reads = None
    if a.min_qs is not None and a.min_qs < 0:
        a.min_qs = None
    t_all = time.time()
    mode = robocop.log_emission_mode()      # raises unless ROBOCOP_LOG_EMISSION is floored|exact
    md5 = D.so_md5(LOGFULL_SO)
    assert md5 == LOGFULL_SO_MD5, "librobocop_logfull.so md5 %s != %s" % (md5, LOGFULL_SO_MD5)
    D.install_posterior_grab()

    chrom, rng = a.window.split(":")
    win_start1, win_end1 = (int(x) for x in rng.split("-"))
    n_win = win_end1 - win_start1 + 1
    out = a.out.rstrip("/")
    os.makedirs(os.path.join(out, "tmpDir"), exist_ok=True)

    dshared = pickle.load(open(os.path.join(a.traindir, "HMMconfig.pkl"), "rb"))
    dshared["robocopC"] = LOGFULL_SO
    dshared["tmpDir"] = os.path.join(out, "tmpDir") + "/"
    dshared["nucleotides"] = 1
    dshared["cache_parents_children"] = True   # see robocop.py PERMOL note
    dshared["info_file"] = None
    n_states = dshared["n_states"]
    ssb = dshared["silent_states_begin"]

    # --- start rule -----------------------------------------------------------------
    start_info = dict(start=a.start)
    if a.start == "any":
        init, sinfo = stationary_initial(dshared)
        start_info.update(sinfo)
        dshared["initial_probs"] = init
        print("start=any: stationary initial probs", json.dumps(sinfo))

    # --- parameters ---------------------------------------------------------------
    pf = dict(nuc_file=a.nuc_params, bg_file=a.bg_params, tf_file=a.tf_params)
    PP.assert_verbatim(dshared, "watson_signal", **pf)
    PP.assert_verbatim(dshared, "crick_signal", **pf)
    ps_w, src_w, info_w = PP.arm_ps(dshared, "watson_signal", a.arm, a.pi, **pf)
    ps_c, src_c, info_c = PP.arm_ps(dshared, "crick_signal", a.arm, a.pi, **pf)
    print("m6A params: nucleosome inputs/%s, background inputs/%s (tree %s, traindir %s)" % (
        a.nuc_params, a.bg_params, TREE, a.traindir))
    print("arm params watson:", json.dumps(info_w))
    print("arm params crick :", json.dumps(info_c))
    sigma, special = state_mirror(dshared)
    if a.ps_symmetric:
        ps_c_sym = mirror_ps(ps_w, sigma, special, dshared)
        ts, ns = int(dshared["tf_starts"][0]), dshared["nuc_start"]
        print("ps-symmetric: crick := mirror(watson); max |change| bg %.3g, TFs %.3g, nuc %.3g" % (
            abs(ps_c_sym[0] - ps_c[0]), np.abs(ps_c_sym[ts:ns] - ps_c[ts:ns]).max(),
            np.abs(ps_c_sym[ns:ssb] - ps_c[ns:ssb]).max()))
        ps_c = ps_c_sym
    ps_c_mir = mirror_ps(ps_c, sigma, special, dshared)       # Crick ps seen from the read frame
    ts, ns = int(dshared["tf_starts"][0]), dshared["nuc_start"]
    ps_cmp = dict(bg=float(abs(ps_c_mir[0] - ps_w[0])),
                  tfs=float(np.abs(ps_c_mir[ts:ns] - ps_w[ts:ns]).max()),
                  nucleosome=float(np.abs(ps_c_mir[ns:ssb] - ps_w[ns:ssb]).max()))
    print("max |mirror(ps_crick) - ps_watson|:", json.dumps(ps_cmp))
    ps_flip = ps_c_mir if a.flip_ps == "crick_mirror" else ps_w
    # --direction rev: a flipped WATSON read lands on layer 6 (its A is a T of the reverse
    # complement); crick_mirror's twin is the Watson ps seen through the mirror.
    ps_flip_w = mirror_ps(ps_w, sigma, special, dshared) if a.flip_ps == "crick_mirror" else ps_c
    print("ps range watson [%.6g, %.6g]  crick [%.6g, %.6g]" % (ps_w.min(), ps_w.max(), ps_c.min(), ps_c.max()))

    seq_hook = None
    if a.seq_layer == "off":
        def seq_hook(d):
            d["emission"][0][:, :ssb] = 0.0

    # --- sequence: always the reference; consensus only flags ----------------------
    ext_start0 = max(0, win_start1 - 1 - a.flank)
    ext_end0 = win_end1 + a.flank
    codes_ext = getNucleotideSequence(FASTA, chrom, ext_start0 + 1, ext_end0)
    cons_summary = dict(flagging_run=False, sequence_changed=False)
    if a.consensus == "on":
        t0 = time.time()
        sites, cons_summary = CS.consensus(a.bam, chrom, win_start1, win_end1,
                                           codes_ext, ext_start0)
        cons_summary["flagging_run"] = True
        CS.write_sites(os.path.join(out, "consensus_sites.tsv"), sites)
        print("consensus flags (%.1fs): %s" % (time.time() - t0, json.dumps(cons_summary)))

    # --- molecules ----------------------------------------------------------------
    t0 = time.time()
    keep_names = None
    if a.keep_reads:
        with open(a.keep_reads) as fh:
            keep_names = set(ln.strip() for ln in fh if ln.strip())
    mols, es, ee, rstats = RC.read_molecules(a.bam, chrom, win_start1, win_end1,
                                             codes_ext, flank=a.flank,
                                             min_mapq=a.min_mapq, min_aln_len=a.min_aln_len,
                                             min_qs=a.min_qs, include_names=keep_names,
                                             thresh=a.call_thresh)
    print("read %d molecules in %.1fs; %s" % (len(mols), time.time() - t0,
                                              json.dumps(rstats, sort_keys=True)))
    lp_table, lp_info, mol_run = None, None, {}
    if a.local_prior:
        lp_table = LP.load_table(a.local_prior)
        LP.check_single_parent(dshared)
        lp_runs = sorted(set(lp_table.heldout_run) - {"pooled"})
        if lp_runs:                     # per-read run, from the read group (site_mixture.RG_BARCODE)
            import pysam
            import site_mixture as SMX
            with pysam.AlignmentFile(a.bam) as b:
                for r in b.fetch(chrom, es, ee):
                    if not (r.is_secondary or r.is_supplementary):
                        mol_run[r.query_name] = SMX.barcode_run(
                            SMX.RG_BARCODE.get(r.get_tag("RG") if r.has_tag("RG") else None, "unknown"))
        lp_info = dict(file=os.path.abspath(a.local_prior), md5=LP.md5(a.local_prior),
                       n_rows=int(len(lp_table)), sites=sorted(set(lp_table.name)),
                       tfs=sorted(set(lp_table.tf)), heldout_runs=lp_runs, layer=LP.LAYER,
                       n_reads_weighted=0, n_placements=0)
        print("local prior:", json.dumps(lp_info))
    if a.max_molecules:
        mols = mols[:a.max_molecules]
    pooled_f = PP.pooled_rate(mols, "flank")
    pooled_s = PP.pooled_rate(mols, "span")
    print("pooled flank m6A rate %.5f  span %.5f" % (pooled_f, pooled_s))

    # --- collapse machinery -------------------------------------------------------
    M = C.collapse_matrix(dshared)
    cols = C.column_names(dshared)
    keep = [c for c in KEEP_COLS if c in cols] if a.cols == "keep" else list(cols)
    keep_idx = np.array([cols.index(c) for c in keep])
    print("kept optable columns:", keep if len(keep) < 20 else "%d (all)" % len(keep))

    win_off = (win_start1 - 1) - es          # window start inside the ext arrays
    sum_p = np.zeros((n_win, n_states))
    cov = np.zeros(n_win, dtype=np.int64)
    per_mol = np.zeros((len(mols), n_win, len(keep)), dtype=np.float32)
    per_mol_cov = np.zeros((len(mols), n_win), dtype=bool)
    rows = []
    checked_collapse = False
    worst = dict(n_nan=0, max_row_sum_dev=0.0, max_silent=0.0, min_p=np.inf, max_p=-np.inf)
    t_dec = time.time()

    for mi, m in enumerate(mols):
        s0 = max(m.ref_start0, win_start1 - 1)
        e0 = min(m.ref_end0, win_end1)
        if e0 - s0 < a.min_span:
            rows.append(dict(i=mi, name=m.name, strand=m.strand, skipped="short",
                             span=e0 - s0))
            continue
        seg_start1, seg_end1 = s0 + 1, e0
        lo, hi = s0 - es, e0 - es
        codes_seg = codes_ext[lo:hi]
        calls = m.calls[lo:hi]
        no = np.full(hi - lo, RC.NO_OBS, dtype=np.int8)
        flip = (a.direction == "5to3" and m.strand == "crick") or a.direction == "rev"

        scale, basis = (1.0, "off")
        pw, pc, pf, pfw = ps_w, ps_c, ps_flip, ps_flip_w
        if a.eff:
            scale, basis = PP.read_scale(m, pooled_f, pooled_s)
            pw, pc, pf, pfw = (PP.scaled_ps(ps_w, scale), PP.scaled_ps(ps_c, scale),
                               PP.scaled_ps(ps_flip, scale), PP.scaled_ps(ps_flip_w, scale))

        hook = seq_hook
        lp_names, lp_run = "", ""
        if lp_table is not None:
            lp_run = mol_run.get(m.name, "unknown") if lp_info["heldout_runs"] else "pooled"
            pl = LP.placements(LP.rows_for(lp_table, lp_run), dshared, seg_start1, seg_end1, chrom, flip)
            if pl:
                hook = LP.make_hook(pl, ssb, inner=seq_hook)
                lp_info["n_reads_weighted"] += 1
                lp_info["n_placements"] += len(pl)
            lp_names = ",".join("%s:w=%.6g" % (x[3], x[2]) for x in pl)

        t1 = time.time()
        if not flip:
            if m.strand == "watson":
                calls_w, calls_c = calls, no
            else:
                calls_w, calls_c = no, calls
            p, op, ll = D.decode_one(dshared, chrom, seg_start1, seg_end1, codes_seg,
                                     pw, pc, calls_w, calls_c, M, keep_idx,
                                     emission_hook=hook)
        else:
            codes_f = revcomp_codes(codes_seg)[::-1].copy()
            calls_f = np.ascontiguousarray(calls[::-1])
            if m.strand == "crick":
                pf_raw, _op, ll = D.decode_one(dshared, chrom, seg_start1, seg_end1, codes_f,
                                               pf, pc, calls_f, no, M, keep_idx,
                                               emission_hook=hook)
            else:                                   # --direction rev only
                pf_raw, _op, ll = D.decode_one(dshared, chrom, seg_start1, seg_end1, codes_f,
                                               pw, pfw, no, calls_f, M, keep_idx,
                                               emission_hook=hook)
            vf = D.posterior_validity(pf_raw, ssb)
            assert vf["n_nan"] == 0 and vf["max_row_sum_dev"] < ROW_SUM_TOL and vf["max_silent"] == 0.0, \
                "molecule %d (flipped frame): %r" % (mi, vf)
            p = map_posterior(pf_raw, sigma, special, np.asarray(codes_seg), dshared)
            op = C.collapse(p, M)[:, keep_idx]
        if not checked_collapse:
            d = C.assert_matches(dshared, p[:min(200, p.shape[0])])
            print("vectorised collapse == robocop.sum_for_dbf_probs (max|d| = %g)" % d)
            checked_collapse = True

        ws, we = s0 - (win_start1 - 1), e0 - (win_start1 - 1)
        sum_p[ws:we] += p
        cov[ws:we] += 1
        per_mol[mi, ws:we, :] = op.astype(np.float32)
        per_mol_cov[mi, ws:we] = True
        bad = int(((p < -1e-9) | (p > 1 + 1e-6)).sum())
        v = D.posterior_validity(p, ssb)
        assert v["n_nan"] == 0, "molecule %d: NaN in posterior" % mi
        assert v["max_row_sum_dev"] < ROW_SUM_TOL, "molecule %d: row sum off by %g" % (mi, v["max_row_sum_dev"])
        assert v["max_silent"] == 0.0, "molecule %d: silent column %g" % (mi, v["max_silent"])
        worst["n_nan"] += v["n_nan"]
        worst["max_row_sum_dev"] = max(worst["max_row_sum_dev"], v["max_row_sum_dev"])
        worst["max_silent"] = max(worst["max_silent"], v["max_silent"])
        worst["min_p"] = min(worst["min_p"], v["min_p"])
        worst["max_p"] = max(worst["max_p"], v["max_p"])
        rows.append(dict(i=mi, name=m.name, strand=m.strand, skipped="",
                         span=e0 - s0, seg_start1=seg_start1, seg_end1=seg_end1,
                         n_informative=m.n_informative, n_meth=m.n_meth,
                         n_ambig=m.n_ambig, flank_n=m.flank_n, flank_meth=m.flank_meth,
                         eff_scale=scale, eff_basis=basis, loglik=ll,
                         n_invalid_posterior=bad, max_row_sum_dev=v["max_row_sum_dev"],
                         secs=time.time() - t1, frame="flipped" if flip else "ref"))
        if lp_table is not None:
            rows[-1].update(local_prior_run=lp_run, local_prior_sites=lp_names)
        if (mi + 1) % 25 == 0:
            print("  %d/%d  %.1fs elapsed  peak RSS %.2f GiB" % (
                mi + 1, len(mols), time.time() - t_dec,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0))
            sys.stdout.flush()
    secs_decode = time.time() - t_dec
    done = sum(1 for r in rows if not r.get("skipped"))
    print("decoded %d molecules in %.1fs (%.2fs each)" % (
        done, secs_decode, secs_decode / max(done, 1)))

    # --- molecule average, written as an ordinary decode --------------------------
    avg = np.zeros_like(sum_p)
    ok = cov > 0
    avg[ok] = sum_p[ok] / cov[ok, None]
    pu = RC.pileup_from_molecules(mols, es, ee)
    n_store = n_win + 1
    info_path = os.path.join(out, "tmpDir", "info_0_1.h5")
    if os.path.exists(info_path):
        os.replace(info_path, info_path + ".prev")     # never rm: keep the previous file aside
    with h5py.File(info_path, "w") as f:
        g = f.create_group("segment_0")
        g.attrs["n_obs"] = n_win
        g.attrs["chr"] = chrom
        g.attrs["start"] = win_start1
        g.attrs["end"] = win_end1
        g.attrs["segment"] = 0
        g.attrs["log_likelihood"] = float(np.nansum([r.get("loglik", np.nan) for r in rows]))
        g.attrs["n_molecules"] = done
        D.save_sparse_dense(f, "segment_0/nucleotides",
                            np.asarray(codes_ext[win_off:win_off + n_win]))
        D.save_posterior_truncated(f, "segment_0/posterior", avg)
        for nm, key in (("meth_watson", "k_watson"), ("meth_crick", "k_crick"),
                        ("A_watson", "n_watson"), ("A_crick", "n_crick")):
            D.save_sparse_dense(f, "segment_0/Fiber_count_%s" % nm,
                                pu[key][win_off:win_off + n_store])
        D.save_sparse_dense(f, "segment_0/molecule_depth", cov)
    print("wrote", info_path)

    pd.DataFrame([dict(chr=chrom, start=win_start1, end=win_end1)]).to_csv(
        os.path.join(out, "coords.tsv"), sep="\t", index=False)
    with open(os.path.join(out, "config.ini"), "w") as f:
        f.write("[main]\n")
        f.write("nucFile = inputs/SacCer3.fa\n")
        f.write("nucleosomeFile = inputs/Chereji_2018_+1_-1_nucs.bed\n")
        f.write("tfFile = inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed\n")
        f.write("cshared = ../pkg/robocop/librobocop_logfull.so\n")
        f.write("bamFile = %s\n" % a.bam)
        f.write("nucleotide = A\ntech = MNase\ntech2 = Fiber\n")
        f.write("fragRangeLong = (127, 187)\nfragRangeShort = (0, 80)\n")
        f.write("trainDir = %s\n" % a.traindir)
    label = "permol_logfull_%s_arm%s%s_eff%s_%s_start%s%s%s" % (
        mode, a.arm, ("" if a.pi is None else "_pi%s" % str(a.pi).replace(".", "p")),
        "on" if a.eff else "off", a.direction, a.start,
        ("_flipps%s" % a.flip_ps) if a.direction in ("5to3", "rev") else "",
        "" if a.consensus == "on" else "_nocons")
    if a.seq_layer == "off" or a.ps_symmetric:
        label += "_TEST%s%s" % ("_seqoff" if a.seq_layer == "off" else "",
                                "_symps" if a.ps_symmetric else "")
    if a.nuc_params != PP.NUC_FILE_DEFAULT:
        label += "_nuc%s" % os.path.splitext(a.nuc_params)[0].replace("nucleosome_params_", "")
    if a.bg_params != PP.BG_FILE_DEFAULT:
        label += "_bg%s" % os.path.splitext(a.bg_params)[0].replace("bg_params_", "")
    if a.tf_params != PP.TF_FILE_DEFAULT:
        label += "_tf%s" % os.path.splitext(a.tf_params)[0].replace("all_TFs_1000pealVal_params_pseudo_", "")
    if a.keep_reads or a.min_mapq or a.min_aln_len or a.min_qs is not None:
        label += "_filt"
    if lp_table is not None:
        label += "_localprior"
    run_opts = dict(direction=a.direction, start=a.start, flip_ps=a.flip_ps,
                    consensus=a.consensus, seq_layer=a.seq_layer, ps_symmetric=a.ps_symmetric,
                    nuc_params=a.nuc_params, bg_params=a.bg_params, tf_params=a.tf_params,
                    min_mapq=a.min_mapq, min_aln_len=a.min_aln_len, min_qs=a.min_qs,
                    keep_reads=a.keep_reads, call_thresh=rstats.get("thresh"),
                    defaults_version="2026-11-02 filtnordna", local_prior=lp_info)
    with open(os.path.join(out, "RUN_INFO.json"), "w") as f:
        json.dump(dict(schema=1, outdir=out, label=label,
                       driver="singlefiber/run_permol_window_logfull_5to3.py",
                       pkg_tree=TREE, so=LOGFULL_SO, so_md5=md5,
                       log_emission=mode,
                       pkg_tree_how="--pkg-tree" if PKG_TREE_OVERRIDE else "literal sys.path.insert",
                       traindir=a.traindir, coords="singlefiber (inline)",
                       emission="per-molecule Bernoulli, log space",
                       arm=a.arm, pi=a.pi, efficiency=bool(a.eff),
                       window=a.window, n_molecules=done,
                       bam=a.bam, written=datetime.datetime.now().isoformat(timespec="seconds"),
                       slurm_job_id=os.environ.get("SLURM_JOB_ID"), **run_opts), f, indent=1)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "molecules.tsv"), sep="\t", index=False)
    np.savez_compressed(os.path.join(out, "permol.npz"),
                        per_mol=per_mol, per_mol_cov=per_mol_cov,
                        cols=np.array(keep), cov=cov,
                        names=np.array([m.name for m in mols]),
                        strands=np.array([m.strand for m in mols]),
                        win=np.array([win_start1, win_end1]),
                        avg_optable=C.collapse(avg, M)[:, keep_idx].astype(np.float32))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    stats = dict(label=label, log_emission=mode, so_md5=md5, posterior_validity=worst,
                 arm=a.arm, pi=a.pi, eff=bool(a.eff), window=a.window,
                 traindir=a.traindir, n_molecules_fetched=len(mols),
                 n_molecules_decoded=done, reader=rstats,
                 pooled_flank_rate=float(pooled_f), pooled_span_rate=float(pooled_s),
                 arm_info_watson=info_w, arm_info_crick=info_c,
                 secs_decode=secs_decode, secs_total=time.time() - t_all,
                 secs_per_molecule=secs_decode / max(done, 1),
                 peak_rss_gib=peak,
                 n_positions_uncovered=int((cov == 0).sum()),
                 median_molecule_depth=float(np.median(cov)),
                 n_invalid_posterior=int(df.get("n_invalid_posterior", pd.Series([0])).sum()),
                 n_flipped=int((df.get("frame", pd.Series([])) == "flipped").sum()),
                 ps_mirror_vs_watson_max_abs=ps_cmp, start_info=start_info,
                 consensus_summary=cons_summary, **run_opts)
    with open(os.path.join(out, "run_stats.json"), "w") as f:
        json.dump(stats, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in stats.items() if k != "reader"}, indent=1, default=float))
    print("peak RSS %.2f GiB; total %.1fs" % (peak, time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
