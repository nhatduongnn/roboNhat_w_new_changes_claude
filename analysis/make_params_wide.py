"""Build decode-ready Fiber-seq parameter pkls with per-TF WIDENED footprints.

Why
---
`factor_p_values_pm50/p_values_Abf1_murphy.png` shows ABF1's protection is not confined to
its 14 bp motif: on-motif P(m6A) is 0.61x/0.58x background, but the depletion runs past the
motif edge. RoboCOP gives ABF1 a 14-state block, so the Fiber layers only ever read the 14
motif columns. This script widens the p-vector so the state can carry the footprint.

The motif is NOT touched. Pad columns emit pwm['background'] in the sequence layer -- the
same 5x1 vector the background state emits -- so they cancel exactly in the likelihood ratio
and the sequence contribution stays the 14 bp motif log-odds. Widening is Fiber-layer-only.

Where the widened numbers come from
-----------------------------------
inputs/all_TFs_1000pealVal_params_pseudo_pm50bp.pkl, built by make_params_pm50.py as a
101 bp (centre +/-50) refit and marked PLOTTING ONLY because RoboCOP would mis-index a
101-long vector. It is safe to slice: its centre columns are bit-identical to the shipped
motif-length vectors (gate 1 below re-proves this on every run). So widening is a slice of
an already-validated fit, not a new fit.

REGISTER. make_params_pm50.window_for centres each site at `c = start + L//2` (plus) and
`c = end - L//2 - 1` (minus), which is exactly what makes combine_motif_counts_binom's
[::-1] mirror land minus-strand columns on top of plus-strand ones. The output vector is
therefore in the MOTIF's own frame with the motif at

    lo = 50 - L//2 ,  motif = [lo, lo+L)

and the widened slice is [lo-left, lo+L+right). Asymmetric pads are safe for the same
reason: the mirror already happened during fitting, so robocop.py's reverse block
(`other_strand[::-1]`) maps the motif-frame right pad onto the reference-left side by
itself, with no extra logic.

Outputs
-------
  inputs/all_TFs_1000pealVal_params_pseudo_wide.pkl      fitted pad columns
  inputs/all_TFs_1000pealVal_params_pseudo_widenull.pkl  pad columns = background rate

The second is the CONTROL. Widening a block hands it an implicit prior boost of
background_prob^-(left+right) (0.9332^-9 = 1.9x for ABF1), because covering a base with a
longer TF block is free while covering it with background is not. widenull has identical
geometry and therefore an identical boost, so the boost cancels between the two and
`wide > widenull` isolates the footprint's own contribution.

Never writes inputs/all_TFs_1000pealVal_params_pseudo.pkl or inputs/bg_params.pkl.

    conda activate robocop-2024
    python make_params_wide.py            # writes both pkls
    python make_params_wide.py --dry-run  # gates only, no writes

TFs WITHOUT AN INDIVIDUAL FIT
-----------------------------
Only 12 TFs clear the >=50-site bar that
`abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:54` uses to fit individually; the other 141
share the pooled `combined_low_count`, which the shipped pkl stores as a single scalar
broadcast across the whole block. That scalar cannot fill a widened block, because every TF
has a different motif length and `ps[tf_start:tf_end] = ...` needs an exact length match.

So when `--meme` is given, a widened TF with no individual fit gets a slice of the
`combined_low_count` +/-50bp profile taken at ITS OWN motif length, written under its own
key. robocop's existing `if tf_name in loaded_params['p']` branch then handles all 154 TFs
with no code change, and each of the 141 carries a real pooled footprint instead of a flat
scalar.

    conda activate robocop-2024
    python make_params_wide.py --pads tf_pads_wide10.tsv \
        --meme inputs/motifs_meme.txt \
        --out inputs/all_TFs_1000pealVal_params_pseudo_wide10.pkl

PADS WIDER THAN +/-50
---------------------
The pm50 refit is only 101 columns, so a +/-150 pad has nowhere to come from. --src/--half
point the slice at a wider refit of the same construction, built by the same script
(`make_params_pm50.py --half 200`), whose centre columns pass the very same GATE 1:

    python make_params_wide.py --pads tf_pads_wide150all.tsv \
        --src inputs/all_TFs_1000pealVal_params_pseudo_pm200bp.pkl --half 200 \
        --meme inputs/motifs_meme.txt \
        --out inputs/all_TFs_1000pealVal_params_pseudo_wide150all.pkl

The defaults are unchanged, so every existing variant rebuilds bit-for-bit.
"""
import argparse
import os
import pickle
import sys

import numpy as np

PADS = "tf_pads.tsv"
SHIPPED = "inputs/all_TFs_1000pealVal_params_pseudo.pkl"
PM50 = "inputs/all_TFs_1000pealVal_params_pseudo_pm50bp.pkl"
PM200 = "inputs/all_TFs_1000pealVal_params_pseudo_pm200bp.pkl"
BG = "inputs/bg_params.pkl"
OUT_WIDE = "inputs/all_TFs_1000pealVal_params_pseudo_wide.pkl"
OUT_NULL = "inputs/all_TFs_1000pealVal_params_pseudo_widenull.pkl"
STRANDS = ("watson_signal", "crick_signal")
BASES = ("A", "C", "G", "T")
HALF = 50       # source refit half-window; motif centre sits at index HALF.
SRC_PATH = PM50  # which refit the widened columns are sliced out of.
# Both are overridden by --half/--src.  Pads of +/-150 do not fit in the 101-column pm50
# refit, so wide150/wide150all slice the 401-column pm200 refit instead (--src PM200
# --half 200).  Defaults keep every pre-existing variant bit-for-bit unchanged.
POOLED = "combined_low_count"


def meme_lengths(path):
    """{motif: width} straight off a MEME file, without importing robocop."""
    lens, name = {}, None
    for line in open(path):
        s = line.strip()
        if s.startswith("MOTIF"):
            name = s.split()[1]
            lens[name] = 0
        elif name and s and not s.startswith("letter") and not s.startswith("URL"):
            lens[name] += 1
    return lens


def read_pads(path=PADS):
    """{tf: (left, right)} from the tsv. Comments and blank lines ignored."""
    pads = {}
    for ln, raw in enumerate(open(path), 1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        f = line.split()
        if len(f) != 3:
            sys.exit("%s:%d: expected '<tf> <left> <right>', got %r" % (path, ln, line))
        tf, left, right = f[0], int(f[1]), int(f[2])
        if left < 0 or right < 0:
            sys.exit("%s:%d: pads must be >= 0" % (path, ln))
        if tf in pads:
            sys.exit("%s:%d: %s listed twice" % (path, ln, tf))
        pads[tf] = (left, right)
    return pads


def motif_window(L):
    """(lo, hi) of the motif inside the 101-long pm50 vector, motif frame."""
    lo = HALF - L // 2
    return lo, lo + L


def _slice(pm50, src_tf, L, left, right, bg, null):
    """Cut a length L+left+right vector out of src_tf's +/-HALF profile, motif frame."""
    lo, hi = motif_window(L)
    s, e = lo - left, hi + right
    if s < 0 or e > 2 * HALF + 1:
        sys.exit("%s: L=%d pads %d/%d exceed the +/-%d refit window (need [%d:%d])"
                 % (src_tf, L, left, right, HALF, s, e))
    new = {}
    for sig in STRANDS:
        new[sig] = {}
        for b in BASES:
            src = np.asarray(pm50["p"][src_tf][sig][b], dtype=float)
            if src.size != 2 * HALF + 1:
                # C/G/T are all-zero stubs of some other length; just zero-fill.
                new[sig][b] = np.zeros(e - s)
                continue
            v = src[s:e].copy()
            if null:
                rate = float(np.asarray(bg["p"][sig][b]).ravel()[0])
                v[:left] = rate
                if right:
                    v[len(v) - right:] = rate
            new[sig][b] = v
    return new, s, e


def widen(shipped, pm50, bg, pads, null=False, verbose=True, motif_lens=None):
    """Return a new params dict whose widened TFs carry length L+left+right vectors.

    `motif_lens` (from the MEME file) enables the pooled path: a widened TF with no
    individual fit gets a `combined_low_count` slice cut at its own motif length.
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in shipped.items()}
    out["p"] = {}
    for tf, entry in shipped["p"].items():
        left, right = pads.get(tf, (0, 0))
        if left == 0 and right == 0:
            out["p"][tf] = entry            # untouched -> today's behaviour exactly
            continue
        if tf not in pm50["p"]:
            sys.exit("%s has pads but no +/-%d refit in %s" % (tf, HALF, SRC_PATH))
        L = len(np.asarray(entry["watson_signal"]["A"]))
        new, s, e = _slice(pm50, tf, L, left, right, bg, null)
        out["p"][tf] = new
        if verbose:
            w = np.asarray(new["watson_signal"]["A"])
            print("  %-16s L=%2d pads %d/%d -> %2d bp  window[%d:%d]  "
                  "padL %.3f | core %.3f | padR %s"
                  % (tf, L, left, right, len(w), s, e,
                     w[:left].mean() if left else float("nan"),
                     w[left:left + L].mean(),
                     "%.3f" % w[left + L:].mean() if right else "n/a"))

    # --- widened TFs with no individual fit: pooled combined_low_count, cut to size ---
    pooled = sorted(t for t in pads if t not in out["p"])
    if pooled:
        if motif_lens is None:
            sys.exit("%d widened TFs have no individual fit (%s...). Pass --meme so their "
                     "motif lengths are known and a %s slice can be cut for each."
                     % (len(pooled), ", ".join(pooled[:3]), POOLED))
        if POOLED not in pm50["p"]:
            sys.exit("%s carries no +/-%d profile in %s" % (POOLED, HALF, SRC_PATH))
        for tf in pooled:
            left, right = pads[tf]
            if tf not in motif_lens:
                sys.exit("%s is padded but absent from the MEME file" % tf)
            new, _, _ = _slice(pm50, POOLED, motif_lens[tf], left, right, bg, null)
            out["p"][tf] = new
        w = np.asarray(out["p"][pooled[0]]["watson_signal"]["A"])
        if verbose:
            print("  %-16s %d TFs from the pooled %s profile, each cut to its own "
                  "motif length + %d/%d (e.g. %s -> %d bp, core mean %.3f)"
                  % ("(pooled)", len(pooled), POOLED, pads[pooled[0]][0], pads[pooled[0]][1],
                     pooled[0], len(w), w[pads[pooled[0]][0]:len(w) - pads[pooled[0]][1]].mean()))
    return out


def gate_centre_identity(shipped, pm50):
    """GATE 1: the source refit's centre columns must reproduce the shipped vectors
    exactly. If this ever fails, the two fits have drifted and no slice of pm50 is a
    legitimate widening of the shipped model."""
    bad = []
    for tf, entry in shipped["p"].items():
        if tf not in pm50["p"] or tf == "combined_low_count":
            continue
        L = len(np.asarray(entry["watson_signal"]["A"]))
        lo, hi = motif_window(L)
        for sig in STRANDS:
            a = np.asarray(entry[sig]["A"], dtype=float)
            b = np.asarray(pm50["p"][tf][sig]["A"], dtype=float)[lo:hi]
            if a.shape != b.shape or not np.array_equal(a, b):
                bad.append("%s/%s" % (tf, sig))
    if bad:
        sys.exit("GATE 1 FAILED: %s centre != shipped for %s"
                 % (os.path.basename(SRC_PATH), ", ".join(bad)))
    print("GATE 1 ok: %s[lo:lo+L] is bit-identical to the shipped vector, "
          "all %d fitted TFs, both strands"
          % (os.path.basename(SRC_PATH),
             sum(1 for t in shipped["p"] if t != "combined_low_count")))


def gate_zero_pad_identity(shipped, pm50, bg):
    """GATE 2: with no pads, widen() must return the shipped params untouched -- the
    proof that pads=0/0 reproduces the current model bit-for-bit."""
    z = widen(shipped, pm50, bg, {}, verbose=False)
    for tf in shipped["p"]:
        for sig in STRANDS:
            for b in BASES:
                a = np.asarray(shipped["p"][tf][sig][b], dtype=float)
                c = np.asarray(z["p"][tf][sig][b], dtype=float)
                if a.shape != c.shape or not np.array_equal(a, c):
                    sys.exit("GATE 2 FAILED: zero-pad changed %s/%s/%s" % (tf, sig, b))
    print("GATE 2 ok: pads 0/0 reproduces %s bit-for-bit" % os.path.basename(SHIPPED))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pads", default=PADS)
    ap.add_argument("--src", default=PM50,
                    help="the fixed-window Fiber refit the widened columns are sliced "
                         "out of (default: the 101-column pm50 refit)")
    ap.add_argument("--half", type=int, default=50,
                    help="half-width of --src, so the motif sits at lo = half - L//2. "
                         "Must match --src exactly; checked below. Use --half 200 with "
                         "the 401-column pm200 refit for pads beyond +/-50.")
    ap.add_argument("--out", default=OUT_WIDE, help="fitted-pad pkl to write")
    ap.add_argument("--out-null", default=None,
                    help="background-pad control pkl; omit to skip building one")
    ap.add_argument("--meme", default=None,
                    help="MEME file supplying motif lengths for padded TFs that have no "
                         "individual fit, so a combined_low_count slice can be cut for each")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output (refused by default: a decode may "
                         "still be reading it)")
    ap.add_argument("--dry-run", action="store_true", help="run the gates, write nothing")
    a = ap.parse_args()

    global HALF, SRC_PATH
    HALF, SRC_PATH = a.half, a.src

    shipped = pickle.load(open(SHIPPED, "rb"))
    pm50 = pickle.load(open(SRC_PATH, "rb"))
    bg = pickle.load(open(BG, "rb"))

    # --half must describe --src, or every slice silently lands in the wrong place.
    for tf, entry in pm50["p"].items():
        n = len(np.asarray(entry["watson_signal"]["A"]))
        if n != 2 * HALF + 1:
            sys.exit("%s: %s is %d columns, not 2*%d+1=%d -- --half does not match --src"
                     % (os.path.basename(SRC_PATH), tf, n, HALF, 2 * HALF + 1))
    pads = read_pads(a.pads)
    lens = meme_lengths(a.meme) if a.meme else None

    print("source refit: %s  (+/-%d, %d columns, %d entries)"
          % (SRC_PATH, HALF, 2 * HALF + 1, len(pm50["p"])))
    print("pads from %s: %d TF(s)%s" % (a.pads, len(pads),
                                        "" if pads else "  (none -- baseline)"))
    if lens:
        print("motif lengths from %s: %d motifs" % (a.meme, len(lens)))
    gate_centre_identity(shipped, pm50)
    gate_zero_pad_identity(shipped, pm50, bg)

    outputs = [(a.out, False)]
    if a.out_null:
        outputs.append((a.out_null, True))

    built = []
    for path, is_null in outputs:
        print("\n%s (%s pad columns):"
              % (os.path.basename(path), "background-rate" if is_null else "fitted"))
        built.append((path, widen(shipped, pm50, bg, pads, null=is_null, motif_lens=lens)))

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    for path, obj in built:
        if os.path.abspath(path) == os.path.abspath(SHIPPED):
            sys.exit("refusing to clobber the shipped pkl")
        if os.path.exists(path) and not a.force:
            sys.exit("%s already exists. A running decode may be reading it -- pick a new "
                     "--out, or pass --force if you are certain nothing is using it." % path)
        with open(path, "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        n = len(obj["p"])
        print("wrote %s  (%d TF entries)" % (path, n))


if __name__ == "__main__":
    main()
