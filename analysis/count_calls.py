"""Per-factor predicted site counts and expected occupancy from a decode.

Why this exists
---------------
`score_factors.py` already computes `n_pred` per factor, but only for factors that have a
reference on the one chromosome requested, and only against a reference set. Concentration
calibration needs the same quantity for **every** state, on **every** chromosome, with no
reference involved -- the target comes from `make_conc_targets.py`, not from here. This is
that counter.

Two quantities, and the distinction matters
-------------------------------------------
`occ`  -- expected occupancy: `sum(posterior) / block_len`, i.e. the expected NUMBER OF
          BOUND COPIES of the factor. Smooth, threshold-free, and the natural conjugate of
          the prior being tuned. **This is what the tuning loop must use.**

`n_pred_adaptive` -- calls above `max(0.10, 0.30 * global_max)`, `score_robocop`'s rule, kept
          so this tool can be cross-checked against `score_factors.py`. It is a poor tuning
          signal: the threshold is 30% of the factor's OWN max, so as a concentration falls
          the threshold falls with it and the count does not respond proportionally. A
          fixed-point iteration driven by it would chase its own threshold.

`n_pred_fixed` -- calls above a constant 0.10. Stable across iterations because nothing about
          it moves when the concentration changes, so this is the honest number to plot
          convergence against.

`global_max` is taken over whatever span this invocation covers. With `--chrom X` that is
chromosome X, which is exactly what `score_factors.py` uses -- that equivalence is what makes
the cross-check meaningful. Genome-wide adaptive counts would need a second pass to learn the
genome-wide max first; `--merge` therefore recomputes nothing and simply sums `occ` and
`n_pred_fixed`, and reports adaptive counts as per-chromosome sums (flagged in the header).

MacIsaac match sidecar
----------------------
For every group with a MacIsaac target, the `n_pred_fixed` calls are also matched to MacIsaac
site centers within 30 bp (`macisaac/<out>`: n_macisaac, n_calls, n_matched), so each tuning
round reports not just how many sites were called but how many are where MacIsaac puts them.

Call-position sidecar (--calls)
-------------------------------
`--calls` also writes every factor's `n_pred_fixed` calls, one row per call (factor, chrom,
center, start, end), to `calls/<out>`. Nothing else changes: the counts table and the MacIsaac
sidecar are byte-identical with or without it. It exists so a decode can be scored against any
other reference (Rossi, genic fraction, ...) without re-reading the posterior; the same
`call_abf1` runs feed the counts, the MacIsaac match and this file.

Usage
-----
    python count_calls.py <outDir> --chrom chrXIV --out counts_chrXIV.tsv [--calls]
    python count_calls.py --merge counts_chr*.tsv --out counts_genome.tsv
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_robocop as S

CHUNK = 200000          # same chunking score_factors.py uses
FIXED_THRESHOLD = 0.10
VALID_TOL = 1e-4        # slack on the [0,1] posterior bound, for float32 rounding
MATCH_TOL = 30          # bp between a decoded call and a MacIsaac site center
TARGETS = os.path.join(HERE, "inputs", "conc_targets_macisaac_c1.tsv")


def macisaac_groups(path=TARGETS):
    """motif -> group, for motifs whose group has a MacIsaac target."""
    out = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            if int(r["has_target"]):
                out[r["motif"]] = r["group"]
    return out


def match_macisaac(calls, chrom):
    """Per MacIsaac group on `chrom`: sites, fixed-threshold calls, and calls within MATCH_TOL.

    `calls` is {factor: [call centers]}. A group's motifs are POOLED (Rap1's four motifs are
    one RAP1 call set), mirroring how the tuning loop sums their occupancy. Matching is
    score_robocop's greedy one-to-one `match_peaks`, so n_matched is both "MacIsaac sites with
    a call nearby" and "calls with a MacIsaac site nearby".
    """
    import make_conc_targets as T
    ref = {}
    for c, tf, center in T.load_macisaac_c1_sites():
        if c == chrom:
            ref.setdefault(tf, []).append(center)
    pooled = {}
    for m, g in macisaac_groups().items():
        pooled.setdefault(g, []).extend(calls.get(m, []))
    rows = []
    for g in sorted(pooled):
        agg = S.match_peaks(pooled[g], ref.get(g, []), MATCH_TOL)
        rows.append(dict(group=g, chrom=chrom, n_macisaac=agg["n_ref"],
                         n_calls=agg["n_pred"], n_matched=agg["tp"]))
    return rows


def block_lens(dec):
    """Factor -> state-block width in bp. Non-TF columns get 1.

    `occ = sum(posterior)/block_len` converts posterior MASS into an expected COUNT, so the
    divisor has to be the width the factor's posterior is spread over. For a widened run
    that is the padded block, not the motif -- which is why this is read from the decode's
    own HMMconfig rather than from the MEME file.
    """
    d = dec["dshared"]
    tfs, lens = list(d["tfs"]), np.asarray(d["tf_lens"])
    out = {t: int(lens[i]) for i, t in enumerate(tfs)}
    out["nucleosome"] = 147
    return out


def count_chrom(outDir, chrom, return_calls=False):
    dec = S.load_decode(outDir)
    lens = block_lens(dec)
    regions = [r for r in S.merge_regions(dec["coords"]) if r[0] == chrom]
    if not regions:
        raise ValueError("%s decoded no %s" % (outDir, chrom))

    pos_parts, track_parts, ok_parts = [], {}, []
    for (c, rs, re_) in regions:
        for cs in range(rs, re_ + 1, CHUNK):
            ce = min(cs + CHUNK - 1, re_)
            optable, covered, _ = S.region_optable(dec, c, cs, ce)
            pos_parts.append(np.arange(cs, ce + 1))
            # VALIDITY, not finiteness. Each optable entry is a posterior probability, so it
            # must lie in [0, 1]. Anything outside that is meaningless regardless of whether
            # it is finite, and one bad column invalidates the whole position vector.
            #
            # Finiteness alone is NOT enough, and getting this wrong cost two rounds:
            #   - float64 isfinite() passes 1e171 (a perfectly good double), and the float32
            #     cast then turns it into inf -> every genome total read `inf`;
            #   - float32 isfinite() passes 1e30 (under the 3.4e38 ceiling), which is still
            #     nonsense as a probability -> every genome total read ~1e32.
            # The chrXII rDNA array produces values spanning 1e0 to 1e171, so any threshold
            # short of the actual [0,1] bound lets part of it through.
            vals32 = optable.values.astype(np.float32)
            ok_parts.append(np.isfinite(vals32).all(axis=1)
                            & (vals32 >= -VALID_TOL).all(axis=1)
                            & (vals32 <= 1.0 + VALID_TOL).all(axis=1))
            for i, f in enumerate(optable.columns):
                track_parts.setdefault(f, []).append(vals32[:, i])
    pos = np.concatenate(pos_parts)
    ok = np.concatenate(ok_parts)

    n_bad = int((~ok).sum())
    if n_bad:
        # Do NOT quietly nan_to_num this away. An inf posterior is a broken decode, and
        # summing it would poison every genome-wide total (which is exactly what it did:
        # one bad chrXII region made all 81 targets read `inf`). Drop the positions, and
        # say so loudly enough that it cannot pass unnoticed.
        span = pos[~ok]
        sys.stderr.write(
            "WARNING %s: %d of %d positions (%.3f%%) carry an INVALID posterior "
            "(outside [0,1] or non-finite) and are EXCLUDED; span %d-%d\n"
            % (chrom, n_bad, len(pos), 100.0 * n_bad / len(pos), span.min(), span.max()))

    rows, calls, runs = [], {}, {}
    for f, parts in track_parts.items():
        tr = np.concatenate(parts)[ok]
        p = pos[ok]
        gmax = float(tr.max()) if tr.size else 0.0
        bl = lens.get(f, 1)
        runs[f] = S.call_abf1(tr, p, FIXED_THRESHOLD)
        calls[f] = [c["center"] for c in runs[f]]
        rows.append(dict(
            factor=f, chrom=chrom, block_len=bl,
            post_sum=float(tr.sum()),
            occ=float(tr.sum()) / bl,
            n_pred_fixed=len(calls[f]),
            n_pred_adaptive=len(S.call_abf1(tr, p, S.abf1_call_threshold(gmax))),
            global_max=gmax,
            state_extinct=int(gmax < 1e-6),
            n_excluded=n_bad))
        del tr
    rows.sort(key=lambda r: -r["occ"])
    if return_calls:
        return rows, match_macisaac(calls, chrom), runs
    return rows, match_macisaac(calls, chrom)


COLS = ["factor", "chrom", "block_len", "post_sum", "occ",
        "n_pred_fixed", "n_pred_adaptive", "global_max", "state_extinct", "n_excluded"]


def write(rows, out):
    with open(out, "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for r in rows:
            fh.write("\t".join(
                ("%.6g" % r[c]) if isinstance(r[c], float) else str(r[c])
                for c in COLS) + "\n")


def read(path):
    rows = []
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            r = dict(zip(hdr, f))
            for k in ("post_sum", "occ", "global_max"):
                r[k] = float(r[k])
            for k in ("block_len", "n_pred_fixed", "n_pred_adaptive", "state_extinct",
                      "n_excluded"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def merge(paths, out):
    acc = {}
    for p in paths:
        for r in read(p):
            a = acc.setdefault(r["factor"], dict(
                factor=r["factor"], chrom="genome", block_len=r["block_len"],
                post_sum=0.0, occ=0.0, n_pred_fixed=0, n_pred_adaptive=0,
                global_max=0.0, state_extinct=1, n_excluded=0))
            a["post_sum"] += r["post_sum"]
            a["occ"] += r["occ"]
            a["n_pred_fixed"] += r["n_pred_fixed"]
            a["n_pred_adaptive"] += r["n_pred_adaptive"]
            a["global_max"] = max(a["global_max"], r["global_max"])
            a["n_excluded"] += r["n_excluded"]
            a["state_extinct"] &= r["state_extinct"]
    rows = sorted(acc.values(), key=lambda r: -r["occ"])
    write(rows, out)
    return rows


MATCH_COLS = ["group", "chrom", "n_macisaac", "n_calls", "n_matched"]


def match_path(out):
    """Sidecar location: a `macisaac/` subdir, so `<dir>/*.tsv` globs still see only counts."""
    return os.path.join(os.path.dirname(out) or ".", "macisaac", os.path.basename(out))


CALL_COLS = ["factor", "chrom", "center", "start", "end"]


def calls_path(out):
    """Call-position sidecar: a `calls/` subdir, alongside `macisaac/`."""
    return os.path.join(os.path.dirname(out) or ".", "calls", os.path.basename(out))


def write_calls(runs, chrom, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\t".join(CALL_COLS) + "\n")
        for f in sorted(runs):
            for c in runs[f]:
                fh.write("%s\t%s\t%d\t%d\t%d\n" % (f, chrom, c["center"], c["start"], c["end"]))


def write_match(rows, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\t".join(MATCH_COLS) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in MATCH_COLS) + "\n")


def merge_match(paths, out):
    acc = {}
    for p in paths:
        with open(p) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                r = dict(zip(hdr, line.rstrip("\n").split("\t")))
                a = acc.setdefault(r["group"], dict(group=r["group"], chrom="genome",
                                                    n_macisaac=0, n_calls=0, n_matched=0))
                for k in ("n_macisaac", "n_calls", "n_matched"):
                    a[k] += int(r[k])
    write_match([acc[g] for g in sorted(acc)], out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outDir", nargs="?", default=None)
    ap.add_argument("--chrom", default=None)
    ap.add_argument("--merge", nargs="+", default=None,
                    help="per-chromosome TSVs to sum into one genome-wide table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls", action="store_true",
                    help="also write every fixed-threshold call position to calls/<out>")
    args = ap.parse_args()

    if args.merge:
        paths = [p for pat in args.merge for p in sorted(glob.glob(pat))]
        if not paths:
            sys.exit("--merge matched no files")
        rows = merge(paths, args.out)
        print("merged %d chromosome tables -> %s (%d factors)"
              % (len(paths), args.out, len(rows)))
        sides = [match_path(p) for p in paths]
        if all(os.path.exists(s) for s in sides):
            merge_match(sides, match_path(args.out))
            print("merged MacIsaac match tables -> %s" % match_path(args.out))
    else:
        if not args.outDir or not args.chrom:
            sys.exit("need <outDir> and --chrom, or --merge")
        if args.calls:
            rows, mrows, runs = count_chrom(args.outDir, args.chrom, return_calls=True)
            write_calls(runs, args.chrom, calls_path(args.out))
            print("wrote call positions -> %s" % calls_path(args.out))
        else:
            rows, mrows = count_chrom(args.outDir, args.chrom)
        write(rows, args.out)
        write_match(mrows, match_path(args.out))
        print("%s %s: %d factors -> %s (+ %s)"
              % (args.outDir, args.chrom, len(rows), args.out, match_path(args.out)))

    top = [r for r in rows if r["factor"] not in
           ("background", "nucleosome", "unknown", "nuc_center", "nuc_start", "nuc_end")][:6]
    for r in top:
        print("   %-22s occ %10.1f  fixed %6d  adaptive %6d"
              % (r["factor"], r["occ"], r["n_pred_fixed"], r["n_pred_adaptive"]))


if __name__ == "__main__":
    main()
