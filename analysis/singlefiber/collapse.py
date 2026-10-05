#!/usr/bin/env python
"""State posterior -> per-factor columns, vectorised, identical to robocop's own collapse.

`robocop.sum_for_dbf_probs` is a Python loop over positions x 154 motifs (~2 s for a 5 kb
segment), which would cost ~15 min over 439 molecules. This does the same slice sums with
one sparse matrix multiply and is verified against the original by `assert_matches`.

Because the collapse is a sum of state-column slices it is LINEAR in the posterior, so
averaging per-molecule posteriors in state space and collapsing once is identical to
collapsing each molecule and averaging -- which is what lets the prototype store a small
per-molecule table and still write a molecule-average that `score_robocop` can read.
"""
import numpy as np
from scipy import sparse


def column_names(dshared):
    header = ["background"] + list(dshared["tfs"])
    if dshared["nuc_present"]:
        header += ["nuc_padding", "nucleosome", "nuc_center", "nuc_start", "nuc_end"]
    return header


def collapse_matrix(dshared):
    """(n_states x ncols) 0/1 matrix M with optable = posterior @ M.

    Mirrors sum_for_dbf_probs exactly, including padding = dshared['padding'] and the
    five nucleosome columns.
    """
    n_states = dshared["n_states"]
    n_tfs = dshared["n_tfs"]
    nuc_present = dshared["nuc_present"]
    pad = dshared["padding"]
    nuc_start, nuc_len = dshared["nuc_start"], dshared["nuc_len"]
    nuc_padding_length = 0
    nuc_padding_end1 = nuc_start + nuc_padding_length
    actual_nuc_start = nuc_padding_end1
    nuc_padding_start2 = nuc_start + nuc_len - nuc_padding_length
    nuc_padding_end2 = nuc_start + nuc_len
    actual_nuc_end = nuc_padding_start2
    nuc_center_start = nuc_padding_end1 + 9 + 4 + 4 * 63
    nuc_center_end = nuc_center_start + 4
    tf_starts, tf_lens = dshared["tf_starts"], dshared["tf_lens"]

    ncols = n_tfs + 1 + 5 * nuc_present
    M = np.zeros((n_states, ncols))
    M[0, 0] = 1.0
    for j in range(n_tfs):
        s, L = int(tf_starts[j]), int(tf_lens[j])
        M[s + pad:s + L - pad, j + 1] = 1.0
        M[s + L:s + 2 * L - pad, j + 1] = 1.0
    if nuc_present:
        M[nuc_start:nuc_padding_end1, n_tfs + 1] = 1.0
        M[nuc_padding_start2:nuc_padding_end2, n_tfs + 1] = 1.0
        M[nuc_padding_end1:nuc_padding_start2, n_tfs + 2] = 1.0
        M[nuc_center_start:nuc_center_end, n_tfs + 3] = 1.0
        M[actual_nuc_start, n_tfs + 4] = 1.0
        M[actual_nuc_end, n_tfs + 5] = 1.0
    return sparse.csr_matrix(M)


def collapse(ptable, M):
    return np.asarray(ptable @ M)


def assert_matches(dshared, ptable):
    """The vectorised collapse must equal robocop.sum_for_dbf_probs on real data."""
    import robocop
    ref = robocop.sum_for_dbf_probs(dshared, ptable)
    mine = collapse(ptable, collapse_matrix(dshared))
    assert ref.shape == mine.shape, (ref.shape, mine.shape)
    d = float(np.abs(ref - mine).max())
    assert d < 1e-12, "max abs difference %g" % d
    return d
