#!/usr/bin/env python
"""How much does the choice of Rossi motif change the answer?

JASPAR ships exactly one matrix per gene, but Rossi yields up to 11 (several
ChExMix samples per factor, up to 3 motifs each).  Picking the Rossi matrix
NEAREST TO NATIVE -- which is what the sheet did originally -- is a biased rule
twice over: it shrinks the native-rossi distance by construction, and it lets
native choose the matrix used for the jaspar-rossi leg, which was supposed to be
the leg native has no say in.

This compares four selection rules and reports how often the verdict moves.

  nearest   nearest to native, by KL          (biased -- the original)
  evalue    best MEME E-value                 (native-independent)
  nsites    most binding sites in the fit     (native-independent)
  medoid    smallest mean KL to the other     (native-independent)
            Rossi candidates for that gene

It also reports the full min-max spread of each distance over all candidates,
which is the honest error bar on any single-pick number.
"""
import os
import numpy as np
from motif_distance_sheet import (parse_meme, best_align, rc, add_pseudo,
                                  kld_col, call_odd_one_out, SETS, PAIRS,
                                  source_of, FITTED, DB)

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_meme_meta(path):
    """-> {motif_id: (gene, pwm, evalue, nsites)}"""
    out, mid, gene, ev, ns, rows = {}, None, None, None, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                if mid and rows:
                    out[mid] = (gene, np.array(rows), ev, ns)
                p = line.split()
                mid, gene = p[1], (p[2] if len(p) > 2 else p[1])
                gene = gene.upper()
                ev, ns, rows = None, None, []
                continue
            if mid is None:
                continue
            if line.startswith("letter-probability"):
                for tok, key in (("E=", "e"), ("nsites=", "n")):
                    if tok in line:
                        v = line.split(tok)[1].split()[0]
                        if key == "e":
                            ev = float(v)
                        else:
                            ns = int(float(v))
                continue
            p = line.split()
            if len(p) == 4:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    pass
    if mid and rows:
        out[mid] = (gene, np.array(rows), ev, ns)
    return out


def main():
    native = parse_meme(os.path.join(DB, "shipped.meme"))
    jaspar = parse_meme(os.path.join(DB, "jaspar.meme"))
    rossi = parse_meme_meta(os.path.join(DB, "rossi.meme"))

    jas_by_gene, ros_by_gene = {}, {}
    for mid, (g, pwm) in jaspar.items():
        jas_by_gene.setdefault(g, []).append((mid, pwm))
    for mid, (g, pwm, ev, ns) in rossi.items():
        ros_by_gene.setdefault(g, []).append((mid, pwm, ev, ns))

    rules = ("nearest", "evalue", "nsites", "medoid")
    flips = {r: 0 for r in rules}
    rows_out = []
    n_multi = 0

    for mid, (gene, npwm) in sorted(native.items()):
        cands = ros_by_gene.get(gene, [])
        jl = jas_by_gene.get(gene, [])
        if not cands or not jl:
            continue
        jid, jpwm = jl[0]
        kl_nj = best_align(npwm, jpwm)[0]

        # every candidate, scored against BOTH native and jaspar
        rec = []
        for rid, rpwm, ev, ns in cands:
            kn = best_align(npwm, rpwm)[0]
            kj = best_align(jpwm, rpwm)[0]
            rec.append(dict(id=rid, pwm=rpwm, ev=ev, ns=ns, kn=kn, kj=kj))
        if len(rec) > 1:
            n_multi += 1

        # medoid: smallest mean KL to the other Rossi candidates
        for i, a in enumerate(rec):
            others = [b for j, b in enumerate(rec) if j != i]
            a["med"] = (np.mean([best_align(a["pwm"], b["pwm"])[0] for b in others])
                        if others else 0.0)

        picks = {
            "nearest": min(rec, key=lambda r: r["kn"]),
            "evalue": min(rec, key=lambda r: (r["ev"] if r["ev"] is not None else 1e9)),
            "nsites": max(rec, key=lambda r: (r["ns"] if r["ns"] is not None else -1)),
            "medoid": min(rec, key=lambda r: r["med"]),
        }
        verds = {}
        for rule, p in picks.items():
            d = {frozenset(("native", "jaspar")): kl_nj,
                 frozenset(("native", "rossi")): p["kn"],
                 frozenset(("jaspar", "rossi")): p["kj"]}
            verds[rule] = call_odd_one_out(d)[2]
        base = verds["nearest"]
        for rule in rules:
            if verds[rule] != base:
                flips[rule] += 1

        kns = [r["kn"] for r in rec]
        kjs = [r["kj"] for r in rec]
        rows_out.append(dict(
            motif=mid, gene=gene, source=source_of(mid),
            fitted="yes" if mid in FITTED else "", n=len(rec),
            kn_min=min(kns), kn_max=max(kns),
            kj_min=min(kjs), kj_max=max(kjs),
            pick_nearest=picks["nearest"]["id"], pick_evalue=picks["evalue"]["id"],
            same_pick=picks["nearest"]["id"] == picks["evalue"]["id"],
            v_nearest=verds["nearest"], v_evalue=verds["evalue"],
            v_nsites=verds["nsites"], v_medoid=verds["medoid"],
        ))

    print("genes with all three datasets: %d   of which >1 Rossi matrix: %d"
          % (len(rows_out), n_multi))

    print("\nHow often does the pick change the verdict, vs the biased "
          "'nearest to native' rule?")
    for r in rules:
        print("   %-9s %3d / %d verdicts differ" % (r, flips[r], len(rows_out)))

    same = sum(1 for r in rows_out if r["same_pick"])
    print("\n'nearest to native' and 'best E-value' choose the SAME Rossi matrix "
          "for %d of %d genes (%.0f%%)." % (same, len(rows_out),
                                            100 * same / len(rows_out)))

    multi = [r for r in rows_out if r["n"] > 1]
    sp_n = np.array([r["kn_max"] - r["kn_min"] for r in multi])
    sp_j = np.array([r["kj_max"] - r["kj_min"] for r in multi])
    print("\nSpread across Rossi candidates (n=%d multi-motif genes) -- the honest"
          % len(multi))
    print("error bar on any single pick:")
    print("   native<->rossi  median %.3f  p90 %.3f  max %.3f"
          % (np.median(sp_n), np.percentile(sp_n, 90), sp_n.max()))
    print("   jaspar<->rossi  median %.3f  p90 %.3f  max %.3f"
          % (np.median(sp_j), np.percentile(sp_j, 90), sp_j.max()))

    print("\nThe 6 'native is odd' calls, re-tested under every rule:")
    print("   %-16s %-4s %-3s %-16s %-16s %s"
          % ("motif", "fit", "n", "nearest", "evalue", "nsites / medoid"))
    for r in rows_out:
        if r["v_nearest"] == "native_is_odd" or r["v_evalue"] == "native_is_odd":
            print("   %-16s %-4s %-3d %-16s %-16s %s / %s"
                  % (r["motif"], r["fitted"] or "-", r["n"], r["v_nearest"],
                     r["v_evalue"], r["v_nsites"], r["v_medoid"]))

    print("\nWorst single-pick sensitivity (largest native<->rossi spread):")
    print("   %-16s %-3s %8s %8s %8s  %s"
          % ("motif", "n", "kn_min", "kn_max", "spread", "verdict(nearest/evalue)"))
    for r in sorted(multi, key=lambda x: -(x["kn_max"] - x["kn_min"]))[:10]:
        print("   %-16s %-3d %8.3f %8.3f %8.3f  %s / %s"
              % (r["motif"], r["n"], r["kn_min"], r["kn_max"],
                 r["kn_max"] - r["kn_min"], r["v_nearest"], r["v_evalue"]))


if __name__ == "__main__":
    main()
