"""Targets for the motifs MacIsaac does not cover, from a calibrated FIMO scan.

The problem
-----------
69 of the 153 RoboCOP motifs have no MacIsaac row and no Rossi row, so the concentration
loop has nothing to tune them against. They are not negligible: they carry ~17% of all TF
occupancy, and the single worst over-caller -- Nhp6a, occupancy 1592 on chrXIV, five times
the next factor -- is one of them.

Why raw FIMO counts cannot be used directly
-------------------------------------------
FIMO counts motif MATCHES, not bound sites. At p<1e-4 over 12 Mb on two strands a motif
collects ~2400 hits by chance alone, far above any real site count. Handing those numbers to
the tuner as targets would tell the model to call MORE sites, which is the opposite of the
correction needed.

The calibration
---------------
MacIsaac p005_c1 is itself a motif scan plus a cross-species conservation filter -- the same
kind of object as a FIMO scan, just stricter. So for the 84 motifs that have both we can ask:
at what FIMO stringency does the hit count reproduce the MacIsaac site count? If that
stringency is consistent across motifs, it transfers to the motifs MacIsaac lacks.

Two mappings are fitted and compared, and the script reports both so the choice is made on
evidence rather than taste:

  THRESHOLD  find the single FIMO p-value cut whose hit counts best match MacIsaac counts
             across the 84. One number, no per-motif freedom, and directly interpretable
             ("MacIsaac ~= FIMO at p<X"). Fails if different motifs need very different cuts.

  REGRESSION log(MacIsaac count) ~ a + b*log(FIMO count at a fixed cut) + c*(motif width),
             fitted on the 84 and applied to the rest. Absorbs a systematic width effect that
             one shared threshold cannot, at the cost of two more free parameters.

Whichever is used, this touches Rossi not at all -- Rossi stays entirely free for validation.

Honest limits
-------------
A predicted target is an extrapolation, not a measurement. The output marks each motif's
`target_source` (macisaac / fimo_fit) so downstream work can weight them differently or
report them apart, and carries the fit's residual spread so the extrapolation's uncertainty
is visible rather than implied.

Usage
-----
    python fimo_targets.py --counts                     # summarise the scan, per motif
    python fimo_targets.py --fit                        # fit + report both mappings
    python fimo_targets.py --emit inputs/conc_targets_macisaac_fimo.tsv
"""
import argparse
import collections
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HITS = os.path.join(HERE, "fimo_genome", "hits_*.tsv")
MEME = os.path.join(HERE, "inputs", "motifs_meme.txt")
MAC_TARGETS = os.path.join(HERE, "inputs", "conc_targets_macisaac_c1.tsv")

# post-filter cuts available from the 1e-3 scan
CUTS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]


def motif_widths():
    """Motif name -> width, from the MEME 'letter-probability matrix' header.

    This file writes `w= 14` with a space, so the width is the token AFTER 'w=', not a
    suffix of it. Both spellings are accepted since MEME emits either depending on version.
    """
    w, name = {}, None
    with open(MEME) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                name = line.split()[1]
            elif line.startswith("letter-probability") and name:
                toks = line.split()
                for i, tok in enumerate(toks):
                    if tok == "w=" and i + 1 < len(toks):
                        w[name] = int(toks[i + 1])
                        break
                    if tok.startswith("w=") and len(tok) > 2:
                        w[name] = int(tok[2:])
                        break
    return w


def load_hits():
    """motif -> sorted array of p-values. chrM dropped (not assayed by MacIsaac or Rossi)."""
    files = sorted(glob.glob(HITS))
    if not files:
        sys.exit("no FIMO output at %s -- run sbatch_fimo_genome.sh first" % HITS)
    per = collections.defaultdict(list)
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line or line.startswith("motif_id") or line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) < 8:
                    continue
                if p[2] == "chrM":
                    continue
                try:
                    per[p[0]].append(float(p[7]))
                except ValueError:
                    continue
    return {m: np.sort(np.asarray(v)) for m, v in per.items()}


def counts_at(pv, cut):
    return int(np.searchsorted(pv, cut, side="right"))


def load_mac():
    rows = {}
    with open(MAC_TARGETS) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            rows[r["motif"]] = r
    return rows


def fit(hits, mac, widths):
    """Fit both mappings on the motifs that have a MacIsaac target."""
    both = [m for m in hits
            if m in mac and mac[m]["has_target"] == "1" and int(mac[m]["n_total"]) > 0]
    # Rap1's four motifs share one target; keeping all four would quadruple its weight.
    seen, keep = set(), []
    for m in sorted(both):
        g = mac[m]["group"]
        if g in seen:
            continue
        seen.add(g)
        keep.append(m)
    both = keep

    print("calibrating on %d motifs with both a FIMO scan and a MacIsaac target\n" % len(both))

    # --- mapping 1: one shared threshold ---
    print("THRESHOLD mapping -- how well does one FIMO cut reproduce MacIsaac counts?")
    print("  %-10s %10s %10s %10s" % ("cut", "median F/M", "RMS log10", "within 2x"))
    best = None
    for c in CUTS:
        r = np.array([counts_at(hits[m], c) / max(1, int(mac[m]["n_total"])) for m in both],
                     dtype=float)
        r = r[r > 0]
        if len(r) < 10:
            continue
        lr = np.log10(r)
        rms = float(np.sqrt(np.mean(lr ** 2)))
        w2 = float(np.mean(np.abs(lr) < math.log10(2)))
        print("  %-10.0e %10.2f %10.3f %9.0f%%" % (c, float(np.median(r)), rms, 100 * w2))
        if best is None or rms < best[1]:
            best = (c, rms, float(np.median(r)))
    print("  best cut %.0e  (RMS %.3f in log10, median ratio %.2f)\n" % best)

    # --- mapping 2: regression at a fixed cut ---
    CUT = 1e-4
    x1 = np.array([math.log(max(1, counts_at(hits[m], CUT))) for m in both])
    x2 = np.array([widths.get(m, 8) for m in both], dtype=float)
    y = np.array([math.log(int(mac[m]["n_total"])) for m in both])
    A = np.vstack([np.ones_like(x1), x1, x2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = y - pred
    print("REGRESSION mapping at cut %.0e" % CUT)
    print("  log(mac) = %.3f + %.3f*log(fimo) + %.4f*width" % tuple(coef))
    print("  residual sd %.3f in log units  (= %.2fx typical error)"
          % (float(resid.std()), math.exp(float(resid.std()))))
    print("  within 2x: %.0f%%\n" % (100 * float(np.mean(np.abs(resid) < math.log(2)))))
    return both, best, (CUT, coef, float(resid.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--emit", default=None, help="write a combined target table here")
    ap.add_argument("--method", default="regression", choices=["regression", "threshold"])
    args = ap.parse_args()

    hits, mac, widths = load_hits(), load_mac(), motif_widths()
    print("FIMO scan: %d motifs, %d hits at p<1e-3 (chrM excluded)\n"
          % (len(hits), sum(len(v) for v in hits.values())))

    if args.counts:
        print("%-20s %6s %8s %8s %8s %8s" % ("motif", "width", "1e-3", "1e-4", "1e-5", "1e-6"))
        for m in sorted(hits, key=lambda k: -counts_at(hits[k], 1e-4))[:25]:
            print("%-20s %6d %8d %8d %8d %8d" % (
                m, widths.get(m, 0), counts_at(hits[m], 1e-3), counts_at(hits[m], 1e-4),
                counts_at(hits[m], 1e-5), counts_at(hits[m], 1e-6)))
        return

    both, best, reg = fit(hits, mac, widths)

    if args.emit:
        CUT, coef, sd = reg
        cut_t, _, _ = best
        out = []
        for m in sorted(set(list(mac) + list(hits))):
            r = mac.get(m)
            if r and r["has_target"] == "1":
                out.append((m, r["group"], int(r["n_total"]), "macisaac", ""))
            elif m in hits:
                f = counts_at(hits[m], CUT)
                if args.method == "regression":
                    v = math.exp(coef[0] + coef[1] * math.log(max(1, f)) + coef[2] * widths.get(m, 8))
                else:
                    v = counts_at(hits[m], cut_t)
                out.append((m, m.split("_")[0].upper(), int(round(max(1.0, v))), "fimo_fit",
                            "%.3f" % sd))
            else:
                out.append((m, m.split("_")[0].upper(), 0, "none", ""))
        with open(args.emit, "w") as fh:
            fh.write("motif\tgroup\tn_total\ttarget_source\tlog_sd\n")
            for row in out:
                fh.write("\t".join(str(x) for x in row) + "\n")
        n_mac = sum(1 for r in out if r[3] == "macisaac")
        n_fim = sum(1 for r in out if r[3] == "fimo_fit")
        print("wrote %s\n  %d macisaac targets, %d fimo-fitted, %d with neither"
              % (args.emit, n_mac, n_fim, len(out) - n_mac - n_fim))


if __name__ == "__main__":
    main()
