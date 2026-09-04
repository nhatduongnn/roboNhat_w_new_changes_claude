"""Write a widened MEME file: TF motifs extended by per-TF flanking columns.

THE IDEA. RoboCOP gives a TF a state block exactly as long as its PWM
(`get_transition_matrix_info`, robocop.py:156: `tf_lens = pwm[tf].shape[1]`). The Fiber-seq
layers therefore only ever read that many columns. To let the model see a wider Fiber-seq
footprint, make the motif itself wider -- then `tf_lens`, `tf_starts`, `n_states`,
`pwm_emission` and the transition matrix all follow with NO code change. `stack_pwms`
vstacks any width and `reverse_complement` mirrors the extra columns correctly onto the
reverse block, so the whole widening is a data change.

The paired Fiber-seq vector must be widened to match, which
`make_params_wide.py` already does from the same tf_pads.tsv:
    inputs/all_TFs_1000pealVal_params_pseudo_wide.pkl

WHAT GOES IN THE EXTRA COLUMNS. The real flanking base composition, estimated by stacking
the TF's own binding sites. For Abf1_murphy that is the 341 sites in
rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed -- the same sites the Fiber-seq
parameters were fit on, so the two layers describe the same population. Those sites are
already motif-anchored (gated below: the genome slice must equal the bed's own `best_seq`).

THE CORE IS NEVER RE-ESTIMATED. The shipped PWM's own rows are copied through byte-for-byte.
Re-estimating from the sites would give a more degenerate core (13.9 bits vs the shipped
16.6; the consensus base agrees at only 10/14 columns) and would change TF detection for
reasons unrelated to the footprint, confounding every comparison against existing runs.

REGISTER. Pads are in the MOTIF's own 5'->3' frame, matching tf_pads.tsv and the pm50 slice:
a plus-strand site contributes [start-Pl, end+Pr); a minus-strand site contributes
[start-Pr, end+Pl) reverse-complemented. Getting this wrong shifts the whole footprint
silently, so check_widememe_traindir.py re-checks it against the built HMMconfig.

KNOWN SIDE EFFECT ON THE PRIOR. `parameterize.calculateKD` scores only each column's argmax.
Estimated pad columns have argmax above background, so RoboCOP reads the wider motif as
tighter-binding -- and then feeds that Kd in as if it were a concentration, which inverts
into a LOWER prior. For ABF1 at 7/2 this is a 5.12x drop on top of the 0.9332^9 = 0.537x
length cancellation, i.e. tf_prob lands at 0.105x baseline. This is expected and accepted;
it can be undone later without retraining via
    make_conc_trainDir.py --tf Abf1_murphy --lam 5.1185

THE ORIGINAL inputs/motifs_meme.txt IS NEVER MODIFIED.

    conda activate robocop-2024
    python make_meme_wide.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "inputs", "motifs_meme.txt")
OUT = os.path.join(HERE, "inputs", "motifs_meme_wide.txt")
FASTA = os.path.join(HERE, "inputs", "SacCer3.fa")
SITES = os.path.join(HERE, "inputs",
                     "rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed")
PADS = os.path.join(HERE, "tf_pads.tsv")

PSEUDO = 1.0                      # Laplace, on the pad columns only
MIN_SITES = 50                    # RoboCOP's own individual-fit threshold
                                  # (abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:54)
BASES = "ACGT"
COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s):
    return s.translate(COMP)[::-1]


def read_pads(path=PADS):
    """-> {tf: (left, right)}; absent TF means 0/0, i.e. today's behaviour."""
    pads = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            f = line.split()
            if len(f) != 3:
                raise ValueError("malformed line in %s: %r" % (path, line))
            pads[f[0]] = (int(f[1]), int(f[2]))
    return pads


def stack_flanks(sub, left, right, fa, label):
    """Base composition of the left/right flanks, in each site's motif 5'->3' frame.

    -> (left_cols, right_cols) as (left,4) and (right,4) arrays over A,C,G,T.

    Works for one TF's sites or for a pooled set spanning many TFs of different motif
    widths: only the outer `left` and `right` columns are counted, so the differing cores
    in between never have to line up.

    GATES that the coordinates really point at the motif the bed claims before trusting
    the stack -- a mis-anchored site set would silently blur the flanks into background.
    """
    if not len(sub):
        raise ValueError("no sites for %s in %s" % (label, SITES))
    agree = 0
    for _, r in sub.iterrows():
        s = fa.fetch(r.chr, r.start, r.end).upper()
        if r.strand == "-":
            s = rc(s)
        agree += (s == r.best_seq)
    if agree != len(sub):
        raise ValueError("%s: only %d/%d sites match their own best_seq -- sites are not "
                         "motif-anchored, refusing to estimate flanks from them"
                         % (label, agree, len(sub)))

    # CHROMOSOME ENDS. pysam.fetch silently clips a window that runs off either end of a
    # contig, and a clipped slice would put the wrong bases under q[j] / q[w-right+j] --
    # the motif's own tail, mistaken for flank. At +/-10 no yeast site is close enough for
    # this to bite; at +/-150 it becomes a real possibility, so the window is checked
    # against the contig length BEFORE fetching and a site that does not fit is dropped
    # outright rather than silently truncated. Every drop is counted and reported, and the
    # existing >=90% usability gate still has to pass afterwards.
    chrom_len = dict(zip(fa.references, fa.lengths))
    cl = np.zeros((left, 4))
    cr = np.zeros((right, 4))
    used = clipped = ambiguous = 0
    for _, r in sub.iterrows():
        lo, hi = r.start - left, r.end + right
        if lo < 0 or hi > chrom_len[r.chr]:
            clipped += 1
            continue
        q = fa.fetch(r.chr, lo, hi).upper()
        if r.strand == "-":
            q = rc(q)
        w = len(q)
        if w != hi - lo:
            # cannot happen once the bounds check above passes; kept as a hard assertion
            # so a pysam behaviour change can never reintroduce a silent truncation.
            raise ValueError("%s: fetch returned %d bp for a %d bp window at %s:%d-%d"
                             % (label, w, hi - lo, r.chr, lo, hi))
        if any(BASES.find(c) < 0 for c in q):
            ambiguous += 1
            continue
        for j in range(left):
            cl[j, BASES.find(q[j])] += 1
        for j in range(right):
            cr[j, BASES.find(q[w - right + j])] += 1
        used += 1
    if clipped or ambiguous:
        print("      %s: dropped %d site(s) within %d/%d bp of a contig end, "
              "%d with non-ACGT bases in the window"
              % (label, clipped, left, right, ambiguous))
    if used < 0.9 * len(sub):
        raise ValueError("%s: only %d of %d sites usable (%d clipped at a contig end, "
                         "%d ambiguous)" % (label, used, len(sub), clipped, ambiguous))

    norm = lambda c: (c + PSEUDO) / (c.sum(1, keepdims=True) + 4 * PSEUDO)
    return norm(cl), norm(cr), used


def fmt(row):
    """Match the shipped file's column style exactly: '  %.6f\\t' per base."""
    return "".join("  %.6f\t" % v for v in row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pads", default=PADS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--min-sites", type=int, default=MIN_SITES,
                    help="a TF with fewer sites than this uses the pooled low-count "
                         "estimate instead of its own (default matches RoboCOP's own "
                         "individual-fit threshold)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output (refused by default: a running "
                         "train job may still be reading it)")
    a = ap.parse_args()

    pads = read_pads(a.pads)
    todo = {t: p for t, p in pads.items() if p != (0, 0)}
    print("pads from %s: %d TF(s)" % (os.path.basename(a.pads), len(todo)))

    est = {}
    if todo:
        fa = pysam.FastaFile(FASTA)
        bed = pd.read_csv(SITES, sep="\t")
        counts = bed.TF.value_counts()
        own = sorted(t for t in todo if counts.get(t, 0) >= a.min_sites)
        pooled = sorted(t for t in todo if t not in own)
        print("estimating flank composition:")
        print("   %d TF(s) with >=%d sites -> own flanks" % (len(own), a.min_sites))
        for tf in own:
            l, r = todo[tf]
            L, R, n = stack_flanks(bed[bed.TF == tf], l, r, fa, tf)
            est[tf] = (L, R)
            print("      %-16s %4d sites  %d + motif + %d" % (tf, n, l, r))

        if pooled:
            # The sequence-layer twin of combined_low_count: one estimate, pooled over the
            # sites of every TF below the individual-fit threshold, shared by all of them.
            # Same >=50/<50 split RoboCOP applies to the Fiber layer
            # (abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:54,147).
            widths = {todo[t] for t in pooled}
            if len(widths) != 1:
                raise ValueError("pooled TFs must share one pad width, got %s"
                                 % sorted(widths))
            l, r = widths.pop()
            low = sorted(t for t in counts.index
                         if counts[t] < a.min_sites and t not in own)
            L, R, n = stack_flanks(bed[bed.TF.isin(low)], l, r, fa, "pooled low-count")
            for tf in pooled:
                est[tf] = (L, R)
            nosite = [t for t in pooled if counts.get(t, 0) == 0]
            print("   %d TF(s) below threshold -> ONE pooled estimate, shared" % len(pooled))
            print("      pooled from %d sites across %d low-count TFs" % (n, len(low)))
            print("      (%d of the %d have no sites at all and could not be done otherwise)"
                  % (len(nosite), len(pooled)))
            print("      left[0]  %s" % np.array2string(L[0], precision=4))
            print("      right[-1]%s" % np.array2string(R[-1], precision=4))

    src = open(SRC).read().split("\n")
    out, i, n_widened = [], 0, 0
    while i < len(src):
        line = src[i]
        if not line.startswith("MOTIF "):
            out.append(line)
            i += 1
            continue
        tf = line.split()[1]
        if tf not in todo:
            out.append(line)
            i += 1
            continue

        left, right = todo[tf]
        lcols, rcols = est[tf]
        out.append(line)
        i += 1
        # pass through everything up to and including the header, fixing only w=
        while not src[i].lstrip().startswith("letter-probability"):
            out.append(src[i])
            i += 1
        hdr = src[i]
        i += 1
        # count the matrix rows so w= is corrected from the real width, not a guess
        j, nrow = i, 0
        while j < len(src) and src[j].strip():
            nrow += 1
            j += 1
        out.append(hdr.replace("w= %d" % nrow, "w= %d" % (nrow + left + right)))
        out.extend(fmt(c) for c in lcols)
        out.extend(src[i:i + nrow])          # the shipped rows, byte-for-byte
        i += nrow
        out.extend(fmt(c) for c in rcols)
        n_widened += 1
        print("   %-16s %d -> %d bp" % (tf, nrow, nrow + left + right))

    text = "\n".join(out)
    if os.path.exists(a.out) and not a.force:
        sys.exit("%s already exists. A running train job may be reading it -- pick "
                 "a new --out, or pass --force if nothing is using it." % a.out)
    open(a.out, "w").write(text)
    print("wrote %s  (%d motif(s) widened)" % (a.out, n_widened))

    # GATE: with no pads this must reproduce the source byte-for-byte.
    if not todo:
        assert text == open(SRC).read(), "zero-pad output is NOT byte-identical to the source"
        print("gate ok: pads 0/0 reproduces %s byte-for-byte" % os.path.basename(SRC))
    else:
        # equivalent check: strip the inserted lines back out and compare
        import copy  # noqa: F401
        chk = []
        k = 0
        lines = text.split("\n")
        while k < len(lines):
            if lines[k].startswith("MOTIF ") and lines[k].split()[1] in todo:
                tf = lines[k].split()[1]
                left, right = todo[tf]
                chk.append(lines[k]); k += 1
                while not lines[k].lstrip().startswith("letter-probability"):
                    chk.append(lines[k]); k += 1
                nrow = 0
                j = k + 1
                while j < len(lines) and lines[j].strip():
                    nrow += 1; j += 1
                core = nrow - left - right
                chk.append(lines[k].replace("w= %d" % nrow, "w= %d" % core)); k += 1
                k += left
                chk.extend(lines[k:k + core]); k += core
                k += right
            else:
                chk.append(lines[k]); k += 1
        assert "\n".join(chk) == open(SRC).read(), \
            "removing the inserted pad rows does NOT recover the source byte-for-byte"
        print("gate ok: stripping the inserted rows recovers %s byte-for-byte"
              % os.path.basename(SRC))


if __name__ == "__main__":
    main()
