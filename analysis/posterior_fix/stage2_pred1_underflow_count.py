"""Stage 2, prediction 1 (POSTERIOR_FIX_PLAN.md "Phase A verification"): on the chrXII rDNA window,
count the (k, n, p) that the decode actually evaluates for which binom.pmf == 0.0 but binom.logpmf is
finite. Must be > 0, else the premise of the log-space emission fix is void on this window.

    cd analysis && python posterior_fix/stage2_pred1_underflow_count.py [info.h5] [--tree pkgvar/seq_maskoff_tune75]

Reads the (k, n) pairs and nucleotides from a finished decode's info h5 (default: the chrXII Bprod_both
decode), and rebuilds the per-state success-probability vector `ps` exactly as robocop.py:628-673 does
(same three pkls, same strand cross for the reverse block, same nucleosome layout). Every A cell
(watson: nucleotide 0, crick: nucleotide 3; robocop.py:789-792) of every emitting state is evaluated.
Cells in tune75-masked TF blocks are counted separately: the mask overwrites them after the floor, so
their underflow never reaches the decode. Read-only.
"""
import sys, os, ast, pickle, argparse, json
import numpy as np
import h5py
from scipy import sparse
from scipy.stats import binom

ap = argparse.ArgumentParser()
ap.add_argument('h5', nargs='?', default='robocop_rdna_chrXII_Bprod_both/tmpDir/info_0_1.h5')
ap.add_argument('--traindir', default='robocop_train_tw_bv75_49')
ap.add_argument('--tree', default='pkgvar/seq_maskoff_tune75')
a = ap.parse_args()

cfg = pickle.load(open(os.path.join(a.traindir, 'HMMconfig.pkl'), 'rb'), encoding='latin1')
ssb, nuc_start = int(cfg['silent_states_begin']), int(cfg['nuc_start'])
tfs, tf_starts, tf_lens = list(cfg['tfs']), np.array(cfg['tf_starts']), np.array(cfg['tf_lens'])

# the pkls the tree reads (robocop.py:598/602/606 in the tune75 trees)
src = open(os.path.join(a.tree, 'robocop', 'robocop.py')).read()
import re
pk = re.findall(r"open\('(inputs/[^']+\.pkl)', 'rb'\)", src.split('def update_data_emission_matrix_using_fiber_seq_counts_Bionomial')[1].split('def plot_binding_factor')[0])
assert len(pk) == 3, pk
loaded_params = pickle.load(open(pk[0], 'rb'))
nucleosome_params = pickle.load(open(pk[1], 'rb'))
bg_params = pickle.load(open(pk[2], 'rb'))

keep = None
for node in ast.walk(ast.parse(open(os.path.join(a.tree, 'robocop', 'utils', 'robocopExtras.py')).read())):
    if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'KEEP_TUNE75':
        keep = {e.value for e in node.value.elts}
assert keep is not None and len(keep) == 75
masked = np.zeros(ssb, dtype=bool)
for i, name in enumerate(tfs):
    if name not in keep:
        masked[tf_starts[i]:tf_starts[i] + 2 * tf_lens[i]] = True


def build_ps(strand):   # robocop.py:628-673, verbatim logic
    other = 'crick_signal' if strand == 'watson_signal' else 'watson_signal'
    ps = np.zeros(ssb)
    ps[:] = bg_params['p'][strand]['A']
    for i in range(len(tfs)):
        s, e, name = tf_starts[i], tf_starts[i] + 2 * tf_lens[i], tfs[i]
        if name in loaded_params['p']:
            ps[s:e] = np.concatenate((loaded_params['p'][name][strand]['A'],
                                      loaded_params['p'][name][other]['A'][::-1]))
        else:
            ps[s:s + tf_lens[i]] = loaded_params['p']['combined_low_count'][strand]['A']
            ps[s + tf_lens[i]:e] = loaded_params['p']['combined_low_count'][other]['A']
    nuc = nucleosome_params['p'][strand]['A']
    ps[nuc_start:nuc_start + 9] = nuc[0:9]
    for i in range(128):
        ps[nuc_start + 9 + i * 4:nuc_start + 9 + i * 4 + 4] = nuc[i + 9]
    ps[nuc_start + 9 + 128 * 4:nuc_start + 9 + 128 * 4 + 10] = nuc[137:147]
    return ps


def dense(g, k):
    d = g[k]
    v = np.asarray(sparse.csr_matrix((d['data'][:], d['indices'][:], d['indptr'][:]), shape=tuple(d.attrs['shape'])).todense())
    return v[0] if v.shape[0] == 1 else v


h = h5py.File(a.h5, 'r')
triples_live, triples_masked = set(), set()
cells_live = cells_masked = 0
pos_any_live, pos_all_live = set(), set()
max_n = 0
for seg in sorted(h.keys(), key=lambda s: int(s.split('_')[1])):
    g = h[seg]
    start, n_obs = int(g.attrs['start']), int(g.attrs['n_obs'])
    nt = dense(g, 'nucleotides')
    for strand, ref, sig in [('watson', 0, 'watson_signal'), ('crick', 3, 'crick_signal')]:
        ps = build_ps(sig)
        k_all = dense(g, 'Fiber_count_meth_' + strand); n_all = dense(g, 'Fiber_count_A_' + strand)
        for i in range(n_obs):
            if nt[i] != ref:
                continue
            k, n = int(k_all[i]), int(n_all[i])
            max_n = max(max_n, n)
            pmf = binom.pmf(k, n, ps)
            under = (pmf == 0.0) & np.isfinite(binom.logpmf(k, n, ps))
            if not under.any():
                continue
            gpos = (strand, start + i)
            ul, um = under & ~masked, under & masked
            cells_live += int(ul.sum()); cells_masked += int(um.sum())
            for p in np.unique(ps[ul]): triples_live.add((k, n, float(p)))
            for p in np.unique(ps[um]): triples_masked.add((k, n, float(p)))
            if ul.any(): pos_any_live.add(gpos)
            if ul[~masked].all(): pos_all_live.add(gpos)

res = dict(h5=a.h5, tree=a.tree, pkls=pk, segments=len(h.keys()), max_n=max_n,
           distinct_knp_underflow_live_states=len(triples_live),
           distinct_knp_underflow_masked_states_only_extra=len(triples_masked - triples_live),
           distinct_kp_underflow_live_states=len({(k, p) for k, n, p in triples_live}),
           cells_underflow_live=cells_live, cells_underflow_masked=cells_masked,
           strand_positions_with_any_live_underflow=len(pos_any_live),
           strand_positions_all_live_states_underflow=len(pos_all_live),
           note="positions are (strand, genomic) pairs, counted once even where 5 kb segments overlap; "
                "cells count every (segment, position, state) the decode evaluates")
print(json.dumps(res, indent=1))
print("PREDICTION 1 (> 0):", "HOLDS" if len(triples_live) > 0 else "FAILS -- premise void on this window")
