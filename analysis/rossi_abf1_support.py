"""Are Abf1's gene-body peaks weaker or less reproducible than its promoter peaks?

The merged calls ({TF}_CX.bed) carry no score -- every row is 1000 -- so "did this peak
survive the filter" cannot be read off the file. It can be reconstructed. Each replicate's
zip ships {id}_chexmix_allevents.tabular: one row per candidate event with signal reads,
control reads, log2 fold enrichment over control and log2 p-value, PLUS
{id}_chexmix_filtered_peaks.bed telling us which of those events passed q<0.01 and the
blacklist. Matching every merged Abf1 summit back to those three replicates gives, per peak:

  n_pass    how many of the 3 replicates independently CALLED it (filtered bed, <=TOL bp)
  n_seen    how many had a candidate event there at all (allevents, <=TOL bp)
  fold/logp best log2 enrichment and log2 p across the replicates that saw it

The question is not whether gene-body peaks are significant -- everything in _CX.bed is, by
construction. It is whether they are significant in the SAME WAY promoter peaks are, or
whether they are the marginal tail that only the pooled multi-replicate analysis rescued.

    conda activate robocop-2024 && python rossi_abf1_support.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rossi_locus_class as R

ROSSI_DIR = "/usr/xtmp/nd141/projects/data/rossi_strand"
OUTDIR = os.path.join(R.HERE, "rossi_abf1_support")
TARGET = "Abf1"
TOL = 30                      # bp; summits are re-derived by the merged run, not copied


def read_allevents(sid):
    rows = []
    for line in open("%s/%d/%d_YEP/%d_chexmix_allevents.tabular" % (ROSSI_DIR, sid, sid, sid)):
        if not line.startswith("chr"):
            continue
        f = line.rstrip("\n").split("\t")
        c, p = f[0].split(":")
        rows.append((c, int(p) - 1, float(f[1]), float(f[2]), float(f[3]),
                     float(f[4]) if f[4] != "-Infinity" else -np.inf))
    return pd.DataFrame(rows, columns=["chr", "start", "sig", "ctrl", "fold", "logp"])


def read_filtered(sid):
    return pd.read_csv("%s/%d/%d_YEP/%d_chexmix_filtered_peaks.bed" % (ROSSI_DIR, sid, sid, sid),
                       sep="\t", header=None, names=["chr", "start", "end", "name", "v"])


def host_genes(genes, df):
    """Name(s) of the gene whose body each position falls inside."""
    out = []
    for _, r in df.iterrows():
        g = genes[(genes["chr"] == r["chr"]) & (genes.start <= r.start) & (genes.end > r.start)]
        out.append(",".join(g.gene.astype(str)) if len(g) else "?")
    return out


def match(a, b, tol):
    """For each row of a, index into b of the nearest same-chromosome row within tol, else -1."""
    out = np.full(len(a), -1)
    pos_a = a.start.to_numpy()
    for c, idx in a.groupby("chr").groups.items():
        sub = b[b["chr"] == c]
        if not len(sub):
            continue
        o = np.argsort(sub.start.to_numpy())
        t = sub.start.to_numpy()[o]
        bi = sub.index.to_numpy()[o]
        p = pos_a[a.index.get_indexer(idx)]
        j = np.clip(np.searchsorted(t, p), 1, len(t) - 1)
        left, right = j - 1, j
        pick = np.where(np.abs(p - t[left]) <= np.abs(p - t[right]), left, right)
        d = np.abs(p - t[pick])
        out[a.index.get_indexer(idx)] = np.where(d <= tol, bi[pick], -1)
    return out


def geometry(genes, ann, cx):
    """Are the gene-body peaks really INTERNAL? An independent re-check of the classifier.

    The two confounders this was written to expose are now handled upstream, in
    rossi_locus_class: incomplete TSS coverage, by anchoring on the GTF ATG rather than
    Park; and terminator NDRs, by the dedicated 'gene end' class. This block stays as the
    check on that, because it asks the question a different way -- nearest annotated
    boundary of ANY gene, ignoring windows and priority entirely.

    Agreement is the result to look for: the same 15 peaks survive here as survived the
    three-class version, so reclassifying the other 40 was the annotation being fixed and
    not the answer being moved.
    """
    g = genes.copy()
    g["five"] = np.where(g.strand == "+", g.start, g.end)
    g["three"] = np.where(g.strand == "+", g.end, g.start)

    def nearest(pos, c, col):
        h = g[g["chr"] == c]
        return np.min(np.abs(h[col].to_numpy() - pos)) if len(h) else np.nan

    print("GEOMETRY (motif-bearing peaks only)")
    print("  %-12s %5s  %10s %10s %10s" % ("", "n", "med|dTSS|", "med d 5'", "med d 3'"))
    for name in ["promoter", "gene body"]:
        s = cx[(cx.cls_prom_first == name) & cx.has_motif].copy()
        s["d5"] = [nearest(p, c, "five") for p, c in zip(s.start, s["chr"])]
        s["d3"] = [nearest(p, c, "three") for p, c in zip(s.start, s["chr"])]
        print("  %-12s %5d  %10.0f %10.0f %10.0f"
              % (name, len(s), np.nanmedian(np.abs(s.d_tss)),
                 np.nanmedian(s.d5), np.nanmedian(s.d3)))
        if name == "gene body":
            near5, near3 = s.d5 <= 500, s.d3 <= 500
            print("     within 500 bp of some GTF 5' end (outside the promoter window):  %d/%d"
                  % (near5.sum(), len(s)))
            print("     within 500 bp of some GTF 3' end (outside the gene-end window):  %d/%d"
                  % (near3.sum(), len(s)))
            print("     >500 bp from BOTH -- genuinely internal:                        %d/%d"
                  % ((~near5 & ~near3).sum(), len(s)))
            deep = s[~near5 & ~near3]
            print("     the internal ones, by host gene:")
            for _, r in deep.sort_values(["chr", "start"]).iterrows():
                print("       %-7s %9d  npass %d  log2fold %5.2f  d5 %5.0f  d3 %5.0f"
                      % (r["chr"], r.start, r.n_pass, r.best_fold, r.d5, r.d3))
    return


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    sizes = {}
    for line in open(R.SIZES):
        f = line.split()
        if len(f) == 2:
            sizes[f[0] if f[0].startswith("chr") else "chr" + f[0]] = int(f[1])

    genes, rnas = R.read_gtf()
    ann = R.Annot(genes, rnas, sizes)

    cond = pd.read_csv(os.path.join(R.HERE, "inputs", "rossi_sample_conditions.tsv"), sep="\t")
    sids = cond[(cond.target == TARGET) & (cond.condition == "normal")].sample_id.tolist()
    print("%s replicates: %s\n" % (TARGET, sids))

    cx = pd.read_csv("%s/%s_CX.bed" % (ROSSI_DIR, TARGET), sep="\t", header=None,
                     names=["chr", "start", "end", "name", "v", "strand"])
    cx["chr"] = cx["chr"].map(R.ARABIC2ROMAN).fillna(cx["chr"])
    cx = cx[cx["chr"].isin(sizes)].reset_index(drop=True)

    cl = ann.classify(cx["chr"].to_numpy(), cx.start.to_numpy()).drop(columns=["chr", "pos"])
    cx = pd.concat([cx, cl.reset_index(drop=True)], axis=1)

    n_pass = np.zeros(len(cx), int)
    n_seen = np.zeros(len(cx), int)
    fold = np.full((len(cx), len(sids)), np.nan)
    logp = np.full((len(cx), len(sids)), np.nan)
    for k, sid in enumerate(sids):
        ev = read_allevents(sid)
        ev["chr"] = ev["chr"].map(R.ARABIC2ROMAN).fillna(ev["chr"])
        fi = read_filtered(sid)
        fi["chr"] = fi["chr"].map(R.ARABIC2ROMAN).fillna(fi["chr"])
        je = match(cx, ev, TOL)
        jf = match(cx, fi, TOL)
        n_seen += (je >= 0)
        n_pass += (jf >= 0)
        hit = je >= 0
        fold[hit, k] = ev.loc[je[hit], "fold"].to_numpy()
        logp[hit, k] = ev.loc[je[hit], "logp"].to_numpy()
    cx["n_pass"] = n_pass
    cx["n_seen"] = n_seen
    with np.errstate(all="ignore"):
        cx["best_fold"] = np.nanmax(np.where(np.isnan(fold), -np.inf, fold), axis=1)
        cx["best_logp"] = np.nanmin(np.where(np.isnan(logp), np.inf, logp), axis=1)
    cx.loc[~np.isfinite(cx.best_fold), "best_fold"] = np.nan
    cx.loc[~np.isfinite(cx.best_logp), "best_logp"] = np.nan

    # is this peak in the model's motif-annotated subset?
    mot = pd.read_csv(os.path.join(R.HERE, "inputs", "rossi_peak_w_strand_all_TFs.bed"), sep="\t")
    mot = mot[mot.TF == TARGET.lower()].copy()
    mot["chr"] = mot["chr"].map(R.ARABIC2ROMAN).fillna(mot["chr"])
    cx["has_motif"] = match(cx, mot.reset_index(drop=True), TOL) >= 0

    cx.to_csv(os.path.join(OUTDIR, "abf1_peak_support.tsv"), sep="\t", index=False)

    cls = cx.cls_prom_first
    print("Merged %s_CX.bed: %d peaks, %d with a FIMO motif within %d bp\n"
          % (TARGET, len(cx), cx.has_motif.sum(), TOL))

    print("REPLICATE SUPPORT -- how many of the %d replicates called each merged peak "
          "(filtered bed, <=%d bp)" % (len(sids), TOL))
    print("                      n     0 reps   1 rep   2 reps  3 reps   mean")
    for name in ["promoter", "gene body", "other intergenic"]:
        for sub, tag in ((cx[cls == name], name),
                         (cx[(cls == name) & cx.has_motif], name + " +motif")):
            if not len(sub):
                continue
            v = sub.n_pass.value_counts()
            print("  %-22s %4d  %s  %5.2f" % (
                tag, len(sub),
                "  ".join("%5.1f%%" % (100 * v.get(i, 0) / len(sub)) for i in range(4)),
                sub.n_pass.mean()))
    print()

    print("STRENGTH in the replicate that saw it best (allevents; NaN = no candidate event)")
    print("                      n    med log2fold   med log2P    no event in any rep")
    for name in ["promoter", "gene body", "other intergenic"]:
        s = cx[cls == name]
        print("  %-22s %4d   %8.2f     %10.1f   %6.1f%%" % (
            name, len(s), np.nanmedian(s.best_fold), np.nanmedian(s.best_logp),
            100 * (s.n_seen == 0).mean()))
    print()

    body = cx[(cls == "gene body") & cx.has_motif].copy()
    print("THE %d MOTIF-BEARING GENE-BODY PEAKS, strongest first" % len(body))
    print("  %-6s %9s  %5s %5s  %8s %9s  %7s  %s"
          % ("chr", "summit", "npass", "nseen", "log2fold", "log2P", "d_tss", "host gene"))
    body["host"] = host_genes(genes, body)
    for _, r in body.sort_values("best_fold", ascending=False).iterrows():
        print("  %-6s %9d  %5d %5d  %8.2f %9.1f  %7.0f  %s"
              % (r["chr"], r.start, r.n_pass, r.n_seen,
                 r.best_fold if np.isfinite(r.best_fold) else float("nan"),
                 r.best_logp if np.isfinite(r.best_logp) else float("nan"),
                 r.d_tss, r["host"]))
    print()
    geometry(genes, ann, cx)
    return cx


if __name__ == "__main__":
    main()
