"""Genic vs intergenic for EVERY TF in Rossi's merged ChExMix set -- a validation table.

Same single rule as rossi_genic.py, applied to all 381 merged peak files instead of 12:

    genic       the summit lies between some gene's ATG and its stop codon
    intergenic  it does not

WHY THIS EXISTS. It is a target to score a decode against. For any TF the model emits, count
how many of its calls land genic and how many intergenic, and compare that split with the
row for the same TF here. Rossi is then a validation set at the level of a *distribution*
rather than a site list, which is the weaker but much broader claim: a model can be wrong
site-by-site and still be asked whether it puts the right FRACTION of each factor inside
genes. The per-peak file is written too, so the stricter site-level comparison stays open.

TWO PEAK SETS, both reported, because they answer different questions:

  cx      04_ChExMix_Peaks/{TF}_CX.bed -- Rossi's merged calls, one 1 bp summit per peak.
          ChExMix v0.31 re-run over all replicates with the blacklist, the 15-experiment
          BY4741 no-tag control, >=1.5-fold and BH p<0.01. 381 TFs, ~182k peaks.
          This is what "the Rossi call set" means with nothing of ours added.

  motif   inputs/rossi_peak_w_strand_all_TFs.bed -- the same merged calls kept only where a
          YEP FIMO motif sits within 30 bp. 358 TFs, 29k peaks. Smaller and cleaner, but the
          filter is not neutral: it removes genic peaks preferentially, so genic% under
          `motif` runs below genic% under `cx` for most factors. Compare a decode against
          whichever set the decode itself resembles -- if the model only calls a TF where its
          PWM matches, `motif` is the fair target.

THE NULL. 73% of this genome is inside an ORF, so a raw genic% says nothing. Every row
carries genic_vs_null = genic% / 73.0%; 1.00 means "indistinguishable from throwing darts".

    conda activate robocop-2024
    python rossi_genic_all.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from rossi_genic import (ARABIC2ROMAN, COND, GTF, N_NULL, ROSSI, SEED, Genome,
                         read_orfs, read_sizes)

HERE = os.path.dirname(os.path.abspath(__file__))
CXDIR = "/usr/project/xtmp/nd141/projects/data/rossi_strand"
MEME = os.path.join(HERE, "inputs", "motifs_meme.txt")
OUTDIR = os.path.join(HERE, "rossi_genic")


def model_tfs():
    """-> the TF names RoboCOP has a PWM for, lowercased ('Abf1_murphy' -> 'abf1')."""
    out = set()
    for line in open(MEME):
        if line.startswith("MOTIF "):
            out.add(line.split()[1].split("_")[0].lower())
    return out


def read_cx(cxdir):
    """-> every merged ChExMix summit, one row per peak, chromosomes in roman."""
    rows = []
    for path in sorted(glob.glob(os.path.join(cxdir, "*_CX.bed"))):
        tf = os.path.basename(path)[:-len("_CX.bed")]
        d = pd.read_csv(path, sep="\t", header=None,
                        names=["chr", "start", "end", "name", "score", "strand"])
        d["TF"] = tf
        rows.append(d[["chr", "start", "end", "TF"]])
    d = pd.concat(rows, ignore_index=True)
    d["chr"] = d["chr"].map(ARABIC2ROMAN).fillna(d["chr"])
    return d


def tabulate(peaks, gg, label):
    """-> per-TF genic/intergenic counts for one peak set."""
    peaks = peaks.copy()
    peaks["genic"] = gg.genic(peaks["chr"].to_numpy(), peaks.start.to_numpy())
    g = peaks.groupby("TF").genic.agg(["size", "sum"])
    t = pd.DataFrame({"TF": g.index,
                      "n_%s" % label: g["size"].astype(int).to_numpy(),
                      "genic_%s" % label: g["sum"].astype(int).to_numpy()})
    t["intergenic_%s" % label] = t["n_%s" % label] - t["genic_%s" % label]
    t["genic_pct_%s" % label] = 100.0 * t["genic_%s" % label] / t["n_%s" % label]
    return t, peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxdir", default=CXDIR)
    ap.add_argument("--normal-only", action="store_true",
                    help="drop motif-set peaks annotated from a heat-shock sample. Only the "
                         "motif set can be filtered: the merged CX calls are pooled across "
                         "replicates and carry no sample attribution to filter on.")
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    sizes = read_sizes()
    gg = Genome(read_orfs())

    cx = read_cx(a.cxdir)
    bad = sorted(set(cx["chr"]) - set(sizes))
    if bad:
        print("dropping merged peaks on unmapped contigs: %s" % bad)
        cx = cx[cx["chr"].isin(sizes)]
    print("merged ChExMix: %d peaks over %d TFs" % (len(cx), cx.TF.nunique()))

    mot = pd.read_csv(ROSSI, sep="\t")
    mot["chr"] = mot["chr"].map(ARABIC2ROMAN).fillna(mot["chr"])
    mot = mot[mot["chr"].isin(sizes)]
    # the annotated bed lowercases TF names; the CX filenames are capitalised
    mot["TF"] = mot.TF.str.lower()
    print("motif-annotated : %d peaks over %d TFs" % (len(mot), mot.TF.nunique()))
    if a.normal_only:
        cond = pd.read_csv(COND, sep="\t")
        ok = set(zip(cond.loc[cond.condition == "normal", "sample_id"],
                     cond.loc[cond.condition == "normal", "replicate"]))
        keep = [t in ok for t in zip(mot.sample_id, mot.replicate)]
        hs = sorted(set(mot.loc[[not k for k in keep], "TF"]))
        print("   normal-condition annotating samples only: %d -> %d rows (heat shock in %s)"
              % (len(mot), sum(keep), ", ".join(hs) or "none"))
        mot = mot[keep]
    print()

    t_cx, cx_pk = tabulate(cx, gg, "cx")
    t_mo, mo_pk = tabulate(mot[["chr", "start", "TF"]], gg, "motif")

    t_cx["key"] = t_cx.TF.str.lower()
    tab = t_cx.merge(t_mo.rename(columns={"TF": "key"}), on="key", how="outer")
    tab["TF"] = tab.TF.fillna(tab.key.str.capitalize())

    rng = np.random.default_rng(SEED)
    ch = np.array(sorted(sizes))
    w = np.array([sizes[c] for c in ch], float)
    nc = rng.choice(len(ch), N_NULL, p=w / w.sum())
    npos = (rng.random(N_NULL) * w[nc]).astype(int)
    nullg = 100.0 * gg.genic(ch[nc], npos).mean()

    tab["genic_vs_null_cx"] = tab.genic_pct_cx / nullg
    tab["genic_vs_null_motif"] = tab.genic_pct_motif / nullg
    tab["in_robocop"] = tab.key.isin(model_tfs())
    tab["null_genic_pct"] = round(nullg, 3)

    cols = ["TF", "in_robocop",
            "n_cx", "genic_cx", "intergenic_cx", "genic_pct_cx", "genic_vs_null_cx",
            "n_motif", "genic_motif", "intergenic_motif", "genic_pct_motif",
            "genic_vs_null_motif", "null_genic_pct"]
    tab = tab[cols].sort_values("TF").reset_index(drop=True)
    for c in ("genic_pct_cx", "genic_pct_motif"):
        tab[c] = tab[c].round(2)
    for c in ("genic_vs_null_cx", "genic_vs_null_motif"):
        tab[c] = tab[c].round(3)

    out = os.path.join(a.outdir, "rossi_genic_all_TFs.tsv")
    tab.to_csv(out, sep="\t", index=False, na_rep="NA")
    cx_pk.to_csv(os.path.join(a.outdir, "rossi_peaks_genic_all_cx.tsv"),
                 sep="\t", index=False)
    mo_pk.to_csv(os.path.join(a.outdir, "rossi_peaks_genic_all_motif.tsv"),
                 sep="\t", index=False)

    have = tab.dropna(subset=["n_cx"])
    print("random genome: %.1f%% genic (%d draws)\n" % (nullg, N_NULL))
    print("merged set, %d TFs with >=1 peak:" % len(have))
    print("   pooled genic%%        %.1f  (%d of %d peaks)"
          % (100.0 * have.genic_cx.sum() / have.n_cx.sum(),
             int(have.genic_cx.sum()), int(have.n_cx.sum())))
    print("   per-TF genic%% median %.1f   IQR %.1f - %.1f   range %.1f - %.1f"
          % (have.genic_pct_cx.median(),
             have.genic_pct_cx.quantile(.25), have.genic_pct_cx.quantile(.75),
             have.genic_pct_cx.min(), have.genic_pct_cx.max()))
    print("   at or above the null  %d of %d TFs"
          % (int((have.genic_pct_cx >= nullg).sum()), len(have)))
    print("   %d of the 153 RoboCOP motif TFs have a Rossi row"
          % int(tab.in_robocop.sum()))

    show = have[have.n_cx >= 100].sort_values("genic_pct_cx")
    hdr = "%-12s %7s %7s %7s %8s   %7s %7s %8s" % (
        "TF", "n_cx", "genic", "interg", "genic%", "n_mot", "genic%", "vs null")
    for title, sub in (("most intergenic", show.head(15)),
                       ("most genic", show.tail(15))):
        print("\n%s (>=100 merged peaks):" % title)
        print(hdr)
        print("-" * len(hdr))
        for _, r in sub.iterrows():
            nm = "-" if pd.isna(r.n_motif) else "%d" % r.n_motif
            gm = "-" if pd.isna(r.genic_pct_motif) else "%.1f" % r.genic_pct_motif
            print("%-12s %7d %7d %7d %7.1f%%   %7s %7s %7.2fx"
                  % (r.TF, r.n_cx, r.genic_cx, r.intergenic_cx, r.genic_pct_cx,
                     nm, gm, r.genic_vs_null_cx))

    print("\nwrote %s" % out)
    print("      %s/rossi_peaks_genic_all_{cx,motif}.tsv" % os.path.basename(a.outdir))


if __name__ == "__main__":
    main()
