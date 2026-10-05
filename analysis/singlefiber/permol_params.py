#!/usr/bin/env python
"""The three footprint arms, and the per-read efficiency correction.

`ps` is the per-state m6A probability vector the Fiber-seq emission uses, length
`silent_states_begin` (3330). The shipped values come from
`inputs/all_TFs_1000pealVal_params_pseudo.pkl`, `inputs/nucleosome_params.pkl` and
`inputs/bg_params.pkl` and are assembled exactly as the aggregate path assembles them;
`robocop.build_fiber_ps` in `pkgvar/permol_seq_maskoff` holds a verbatim copy of that
block, and `build_ps_labelled` below reproduces it while also recording, per state, where
the value came from -- checked bit-for-bit against `build_fiber_ps` by `assert_verbatim`.

Arms
----
A   shipped vectors as-is. The single-variable control: only the emission FORM changes.
B   deconvolved, same shapes.  p_bound = (p_agg - (1-pi) * p_bg) / pi, floored.
    The shipped per-TF vectors were fitted over bound AND unbound molecules, so ABF1's
    mean 0.0843 against background 0.1383 is only a 1.6x contrast. pi in {0.3,0.5,0.7}.
C   flat protection floor 0.024286 = min(nucleosome_params['p']['watson_signal']['A']),
    a measured "fully protected" rate for this enzyme and basecaller.

DECISION (CLAUDE.md rule 7): arms B and C change ONLY the state blocks of the 12 TFs that
have a fitted per-position footprint in all_TFs_1000pealVal_params_pseudo.pkl. background,
the 531 nucleosome states and `combined_low_count` (the single scalar 0.249 shared by the
141 unfitted motifs, including `unknown`) keep their shipped values. Reason: the mixture
inversion needs "each TF's own shape", which only those 12 have, and leaving the 141
competitors fixed is what isolates footprint depth as the variable under test. Changing
combined_low_count too would move ABF1's competitors in the same call.

Per-read efficiency
-------------------
Each molecule's whole ps vector is multiplied by (that molecule's flanking m6A rate) /
(the pooled flanking rate) and clipped into (0, 1). The flank is the +/-400 bp outside
the decoded window, i.e. the candidate region excluded, exactly as planned. Molecules with
fewer than `MIN_FLANK` informative flank calls fall back to their whole-span rate, and
then to no correction. Each emission entry stays a valid Bernoulli likelihood (p and 1-p),
so scaled forward-backward is unaffected -- the ablation measures the effect, it is not
needed for correctness.
"""
import os
import pickle

import numpy as np

SRC_BG = -1          # the single background state
SRC_NUC = -2         # the 531 nucleosome states
SRC_CLC = -3         # combined_low_count (the 141 motifs with no fitted footprint)
MIN_FLANK = 50       # informative flank calls needed to trust a per-read rate
PS_FLOOR = 1e-3      # arm B floor: no position may reach 0
PS_CEIL = 1.0 - 1e-6
ARM_C_P = None       # filled from nucleosome_params at load time


def load_params(params_dir="inputs"):
    with open(os.path.join(params_dir, "all_TFs_1000pealVal_params_pseudo.pkl"), "rb") as f:
        loaded = pickle.load(f)
    with open(os.path.join(params_dir, "nucleosome_params.pkl"), "rb") as f:
        nuc = pickle.load(f)
    with open(os.path.join(params_dir, "bg_params.pkl"), "rb") as f:
        bg = pickle.load(f)
    return loaded, nuc, bg


def build_ps_labelled(dshared, strand, params_dir="inputs"):
    """(ps, source) for one strand layer.

    strand is 'watson_signal' or 'crick_signal'. `source[j]` is the index into
    dshared['tfs'] for a state that took a FITTED per-position TF footprint, or one of
    SRC_BG / SRC_NUC / SRC_CLC. Mirrors robocop.build_fiber_ps line for line; the two ps
    vectors are asserted equal by assert_verbatim().
    """
    loaded_params, nucleosome_params, bg_params = load_params(params_dir)
    tf_starts, tf_lens = dshared["tf_starts"], dshared["tf_lens"]
    nsb = dshared["silent_states_begin"]
    ps = np.zeros(nsb)
    source = np.full(nsb, SRC_BG, dtype=np.int64)
    ps[:] = bg_params["p"][strand]["A"]
    all_tf_from_pwm = dshared["tfs"]
    other_strand = "crick_signal" if strand == "watson_signal" else "watson_signal"

    for i in range(dshared["n_tfs"]):
        tf_start = tf_starts[i]
        tf_end = tf_start + 2 * tf_lens[i]
        tf_name = all_tf_from_pwm[i]
        if tf_name in loaded_params["p"]:
            p_forward = loaded_params["p"][tf_name][strand]["A"]
            p_reverse = loaded_params["p"][tf_name][other_strand]["A"][::-1]
            ps[tf_start:tf_end] = np.concatenate((p_forward, p_reverse))
            source[tf_start:tf_end] = i
        else:
            ps[tf_start:tf_start + tf_lens[i]] = loaded_params["p"]["combined_low_count"][strand]["A"]
            ps[tf_start + tf_lens[i]:tf_end] = loaded_params["p"]["combined_low_count"][other_strand]["A"]
            source[tf_start:tf_end] = SRC_CLC

    nuc_p_params = nucleosome_params["p"][strand]["A"]
    if dshared["nuc_present"]:
        ns = dshared["nuc_start"]
        ps[ns:(ns + 9)] = nuc_p_params[0:9]
        for i in range(128):
            ps[(ns + 9 + i * 4):((ns + 9 + i * 4 + 4))] = nuc_p_params[i + 9]
        ps[(ns + 9 + 128 * 4):(ns + 9 + 128 * 4 + 10)] = nuc_p_params[137:147]
        source[ns:ns + dshared["nuc_len"]] = SRC_NUC
    assert (ps > 0).sum() > 0
    return ps, source


def assert_verbatim(dshared, strand, params_dir="inputs"):
    """build_ps_labelled must equal robocop.build_fiber_ps bit for bit."""
    import robocop
    ref, _bg, _nuc, _lp = robocop.build_fiber_ps(dshared, strand, params_dir=params_dir)
    mine, _src = build_ps_labelled(dshared, strand, params_dir=params_dir)
    assert ref.shape == mine.shape, (ref.shape, mine.shape)
    bad = int((ref != mine).sum())
    assert bad == 0, "%d of %d states differ from the verbatim build" % (bad, ref.size)
    return True


def arm_ps(dshared, strand, arm, pi=None, params_dir="inputs"):
    """ps for one arm, plus a small dict describing what the arm changed."""
    _lp, nuc, bg = load_params(params_dir)
    ps, source = build_ps_labelled(dshared, strand, params_dir=params_dir)
    fitted = source >= 0
    info = dict(arm=arm, pi=pi, strand=strand,
                n_fitted_states=int(fitted.sum()),
                ps_fitted_mean_before=float(ps[fitted].mean()),
                p_bg=float(np.asarray(bg["p"][strand]["A"]).ravel()[0]))
    if arm == "A":
        pass
    elif arm == "B":
        assert pi is not None and 0 < pi <= 1
        p_bg = float(np.asarray(bg["p"][strand]["A"]).ravel()[0])
        new = (ps[fitted] - (1.0 - pi) * p_bg) / pi
        info["n_floored"] = int((new < PS_FLOOR).sum())
        ps[fitted] = np.clip(new, PS_FLOOR, PS_CEIL)
    elif arm == "C":
        flat = float(np.asarray(nuc["p"][strand]["A"]).min())
        info["flat_p"] = flat
        ps[fitted] = flat
    else:
        raise ValueError(arm)
    info["ps_fitted_mean_after"] = float(ps[fitted].mean())
    return ps, source, info


# ---------------------------------------------------------------------------
# per-read efficiency
# ---------------------------------------------------------------------------
def pooled_rate(mols, which="flank"):
    k = sum((m.flank_meth if which == "flank" else m.span_meth) for m in mols)
    n = sum((m.flank_n if which == "flank" else m.span_n) for m in mols)
    return (k / n) if n else np.nan


def read_scale(mol, pooled_flank, pooled_span, min_flank=MIN_FLANK):
    """(scale, basis) for one molecule."""
    if mol.flank_n >= min_flank and np.isfinite(pooled_flank) and pooled_flank > 0:
        return mol.flank_meth / mol.flank_n / pooled_flank, "flank"
    if mol.span_n >= min_flank and np.isfinite(pooled_span) and pooled_span > 0:
        return mol.span_meth / mol.span_n / pooled_span, "span"
    return 1.0, "none"


def scaled_ps(ps, scale):
    return np.clip(np.asarray(ps) * float(scale), 1e-9, PS_CEIL)
