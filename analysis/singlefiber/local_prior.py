#!/usr/bin/env python
"""Site-specific TF entry weight ("local prior") for the single-fiber decode (2026-11-03, plan
let-s-go-with-mad-snazzy-lollipop, Part 2). No C change, no pkgvar change.

Mechanism
---------
A TF's first states (forward tf_starts[i], reverse tf_starts[i] + L) each have ONE parent: the TF's
own silent state (for ABF1: 3331, 0.5 to each orientation), which is entered from the junction (3330)
with tf_prob[i] (ABF1 1.797e-7, robocop_train_fiberonly_nucbrogsym/HMMconfig.pkl; checked for every TF
by check_single_parent). So multiplying the emission of both first states at ONE row t by w multiplies
the weight of every path that starts that TF at t by w and leaves every other path unchanged -- an
unnormalised w-times larger entry probability at that position only. Consequence used as a test:
the odds of "TF starts at row t" change by exactly w (test_local_prior.py).

The log-space tree requires every emission layer <= 0 (robocop.py:970-973 of the _logfull trees:
`nanmax(log emission) <= 0`). So for w > 1 the weight is written as -log(w) on every OTHER emitting
state at that row (a per-row constant cancels in the posterior; the two are the same model); for w <= 1
it is log(w) on the two first states. It goes in layer 1, which is otherwise all 0.0 in the per-molecule
path (robocopExtras.updatePerMoleculeEMMat: "layers 1-4 untouched"); apply() asserts that.

Rows / states
-------------
Motif placed on reference [r, r+L-1] (1-based), read segment [seg_start1, seg_end1]:
  ref frame      row r - seg_start1,            states s, s+L
  flipped frame  row seg_end1 - (r + L - 1),     states s, s+L  (the flipped frame's own first states;
                 state_mirror maps them to the last states of the ref reverse / forward blocks at
                 r+L-1, i.e. the same placement on the reference)
Applied only when the whole motif [r, r+L-1] lies inside the segment.

Table (sites_w.tsv, tab separated): name, chrom, tf, motif_start1, motif_end1, f, w, heldout_run.
heldout_run = 'pooled' (f from all reads) or a run name (Mar20 / May07 / Jun25: f fitted with that run's
reads held out). When the table has run rows, a read uses the rows of ITS OWN run (so its own run did
not inform its prior); else 'pooled'.
"""
import hashlib

import numpy as np
import pandas as pd

LAYER = 1


def load_table(path):
    t = pd.read_csv(path, sep="\t")
    need = {"name", "chrom", "tf", "motif_start1", "motif_end1", "w"}
    assert need <= set(t.columns), "local prior table %s lacks %s" % (path, need - set(t.columns))
    if "heldout_run" not in t.columns:
        t["heldout_run"] = "pooled"
    assert (t.w > 0).all() and np.isfinite(t.w).all(), "local prior w must be finite and > 0"
    return t


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def check_single_parent(dshared):
    """Every TF first state (fwd and rev) has exactly one parent in the transition matrix."""
    T = np.asarray(dshared["transition_matrix"])
    bad = []
    for i in range(dshared["n_tfs"]):
        s, L = int(dshared["tf_starts"][i]), int(dshared["tf_lens"][i])
        for j in (s, s + L):
            if len(np.nonzero(T[:, j])[0]) != 1:
                bad.append((dshared["tfs"][i], j))
    assert not bad, "first states with != 1 parent: %r" % bad[:5]
    return True


def rows_for(table, run):
    """Rows of the table a read of sequencing run `run` uses."""
    runs = set(table.heldout_run) - {"pooled"}
    if runs:
        assert run in runs, "read run %r has no local-prior rows (table runs %s)" % (run, sorted(runs))
        return table[table.heldout_run == run]
    return table[table.heldout_run == "pooled"]


def placements(rows, dshared, seg_start1, seg_end1, chrom, flipped):
    """[(row, (state_fwd_first, state_rev_first), w, name)] for the rows whose motif lies in the segment."""
    tfs = list(dshared["tfs"])
    out = []
    for r in rows.itertuples():
        if r.chrom != chrom:
            continue
        i = tfs.index(r.tf)
        s, L = int(dshared["tf_starts"][i]), int(dshared["tf_lens"][i])
        assert int(r.motif_end1) - int(r.motif_start1) + 1 == L, (r.name, L)
        if int(r.motif_start1) < seg_start1 or int(r.motif_end1) > seg_end1:
            continue
        row = (seg_end1 - int(r.motif_end1)) if flipped else (int(r.motif_start1) - seg_start1)
        out.append((row, (s, s + L), float(r.w), r.name))
    return out


def apply(d, pl, ssb):
    """Write the weights into d['emission'][LAYER] (log space). pl from placements()."""
    E = d["emission"][LAYER]
    if pl:
        assert np.all(E[:, :ssb] == 0.0), "local prior: layer %d is not all 0.0 before the weight" % LAYER
    for row, states, w, _ in pl:
        lw = np.log(w)
        if lw > 0:
            E[row, :ssb] = -lw
            for j in states:
                E[row, j] = 0.0
        else:
            for j in states:
                E[row, j] = lw


def make_hook(pl, ssb, inner=None):
    """emission_hook for decode_one: optional inner hook (e.g. --seq-layer off) then the weights."""
    def hook(d):
        if inner is not None:
            inner(d)
        apply(d, pl, ssb)
    return hook
