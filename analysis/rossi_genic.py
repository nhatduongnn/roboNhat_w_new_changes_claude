"""Are Rossi's peaks for our 12 fitted TFs inside genes, or between them?

THE RULE, and there is only one:

    genic       the position lies between a gene's ATG and its stop codon
    intergenic  it does not

No promoter window, no terminator window, no priority order, no ties to break. A position
either falls inside an ORF or it does not, and that is the whole classification. Everything
this replaces -- promoter/gene-end/gene-body windows, and the ordering needed to arbitrate
between them -- introduced choices that moved the answer by several points without adding
any fact. The bare question does not.

WHY SGD/ENSEMBL COORDINATES ANSWER IT DIRECTLY. inputs/sacCer3.gtf is Ensembl R64-1-1,
which is SGD R64. Its `gene` feature for a protein-coding gene runs ATG -> stop codon and
nothing else, verified across all 6,692 of them:

    gene.start - CDS.start   is always  -3 or 0
    gene.end   - CDS.end     is always   0 or 3
    + strand:  gene.end   == stop_codon.end    for every gene
    - strand:  gene.start == stop_codon.start  for every gene

The 3 bp is only whether the stop codon is counted inside the CDS. No UTRs are annotated,
so the gene span IS the ORF, with no interpretation applied. Median length 1,086 bp.

Strand does not enter into it. "Between the ATG and the stop" is the same interval whether
the gene reads left-to-right or right-to-left, so unlike anything anchored on one end this
needs no direction-of-transcription bookkeeping and cannot be silently mis-registered.

SOURCE. inputs/rossi_peak_w_strand_all_TFs.bed -- Rossi's MERGED ChExMix calls
(04_ChExMix_Peaks/{TF}_CX.bed, one 1 bp summit per peak), each annotated with the nearest
YEP FIMO motif within 30 bp. One row per merged peak; sample_id/replicate record which
sample's FIMO scan supplied the motif, not which replicate called the peak.

CONDITION. Only Spt15 has a non-normal sample among the 12 (8600, rep2, 3 min heat shock);
--normal-only drops the 35 rows carrying its annotation.

THE NULL. ~70% of the yeast genome is coding, so a genic fraction is meaningless on its own.
Every number is therefore also computed over 200,000 uniformly random genomic positions.

    conda activate robocop-2024
    python rossi_genic.py --normal-only
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROSSI = os.path.join(HERE, "inputs", "rossi_peak_w_strand_all_TFs.bed")
GTF = os.path.join(HERE, "inputs", "sacCer3.gtf")
SIZES = os.path.join(HERE, "inputs", "sacCer3.chrom.sizes")
COND = os.path.join(HERE, "inputs", "rossi_sample_conditions.tsv")
OUTDIR = os.path.join(HERE, "rossi_genic")

N_NULL = 200000
SEED = 0

TF12 = ["abf1", "cin5", "fhl1", "fkh1", "mcm1", "nhp6a",
        "rap1", "reb1", "sko1", "spt15", "tbf1", "ume6"]
PRETTY = {"abf1": "Abf1", "cin5": "Cin5", "fhl1": "Fhl1", "fkh1": "Fkh1",
          "mcm1": "Mcm1", "nhp6a": "Nhp6a", "rap1": "Rap1", "reb1": "Reb1",
          "sko1": "Sko1", "spt15": "Spt15/TBP", "tbf1": "Tbf1", "ume6": "Ume6"}

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII",
         "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]
ARABIC2ROMAN = {"chr%d" % (i + 1): "chr%s" % r for i, r in enumerate(ROMAN)}

GENIC, INTER = "#b4531f", "#2b6ca8"
INK, MUTED, GRID = "#12161b", "#666e78", "#e3e7eb"


def read_sizes():
    sz = {}
    for line in open(SIZES):
        f = line.split()
        if len(f) == 2:
            sz[f[0] if f[0].startswith("chr") else "chr" + f[0]] = int(f[1])
    return sz


def read_orfs():
    """-> DataFrame of protein-coding ORFs, and the audit that the span really is ATG->stop."""
    rows = []
    for line in open(GTF):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        a = f[8]
        bt = a.split('gene_biotype "', 1)[1].split('"', 1)[0] if 'gene_biotype "' in a else ""
        gid = a.split('gene_id "', 1)[1].split('"', 1)[0] if 'gene_id "' in a else ""
        rows.append((f[2], "chr" + f[0], int(f[3]) - 1, int(f[4]), f[6], gid, bt))
    d = pd.DataFrame(rows, columns=["feat", "chr", "start", "end", "strand", "gene", "bt"])

    orfs = d[(d.feat == "gene") & (d.bt == "protein_coding")].reset_index(drop=True)
    cds = d[d.feat == "CDS"].groupby("gene").agg(cs=("start", "min"), ce=("end", "max"))
    stop = d[d.feat == "stop_codon"].groupby("gene").agg(ss=("start", "min"), se=("end", "max"))
    j = orfs.set_index("gene").join(cds, how="inner").join(stop, how="inner")
    plus, minus = j[j.strand == "+"], j[j.strand == "-"]
    print("ORF audit over %d protein-coding genes with CDS and stop codon annotated:" % len(j))
    print("   gene.start - CDS.start   %s" % np.unique(j.start - j.cs))
    print("   gene.end   - CDS.end     %s" % np.unique(j.end - j.ce))
    print("   + strand  gene.end   - stop_codon.end     %s" % np.unique(plus.end - plus.se))
    print("   - strand  gene.start - stop_codon.start   %s" % np.unique(minus.start - minus.ss))
    print("   -> the gene span is the ORF: ATG to stop codon, no UTRs.")
    print("   %d protein-coding ORFs, median length %d bp\n"
          % (len(orfs), int(np.median(orfs.end - orfs.start))))
    return orfs[["chr", "start", "end", "strand", "gene"]]


class Genome:
    """One question, asked fast: is this position inside any ORF?

    Intervals may overlap (yeast has nested and antisense ORFs), so membership cannot
    assume disjointness. A running maximum of the end coordinates fixes that: after
    sorting by start, the furthest-reaching end at or before a position is enough to
    decide it in one searchsorted.
    """

    def __init__(self, orfs):
        self.iv = {}
        for c, sub in orfs.groupby("chr"):
            s = sub.sort_values("start")
            st = s.start.to_numpy()
            self.iv[c] = (st, np.maximum.accumulate(s.end.to_numpy()))

    def genic(self, chrom, pos):
        out = np.zeros(len(pos), bool)
        df = pd.DataFrame({"chr": chrom, "pos": pos, "i": np.arange(len(pos))})
        for c, sub in df.groupby("chr", sort=False):
            t = self.iv.get(c)
            if t is None:
                continue
            st, cummax = t
            p = sub.pos.to_numpy()
            j = np.searchsorted(st, p, "right") - 1
            ok = j >= 0
            hit = np.zeros(len(p), bool)
            hit[ok] = p[ok] < cummax[j[ok]]
            out[sub.i.to_numpy()] = hit
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-only", action="store_true",
                    help="drop peaks whose FIMO annotation came from a non-normal sample "
                         "(only Spt15 rep2, 3 min heat shock, is affected)")
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    sizes = read_sizes()
    orfs = read_orfs()
    gg = Genome(orfs)

    coding_bp = 0
    for c, sub in orfs.groupby("chr"):
        s = sub.sort_values("start")
        st, en = s.start.to_numpy(), s.end.to_numpy()
        # union of possibly-overlapping intervals, so shared bases are not double counted
        m_end = np.maximum.accumulate(en)
        newseg = np.r_[True, st[1:] > m_end[:-1]]
        gid = np.cumsum(newseg) - 1
        for k in range(gid[-1] + 1):
            sel = gid == k
            coding_bp += m_end[sel].max() - st[sel].min()
    total_bp = sum(sizes.values())
    print("genome %s bp; ORF union %s bp = %.1f%% coding\n"
          % (format(total_bp, ","), format(coding_bp, ","), 100.0 * coding_bp / total_bp))

    bed = pd.read_csv(ROSSI, sep="\t")
    bed["chr"] = bed["chr"].map(ARABIC2ROMAN).fillna(bed["chr"])
    drop = sorted(set(bed["chr"]) - set(sizes))
    if drop:
        print("dropping peaks on unmapped contigs: %s" % drop)
        bed = bed[bed["chr"].isin(sizes)]
    peaks = bed[bed.TF.isin(TF12)].copy()
    print("Rossi calls for the 12 TFs: %d (of %d in the file)" % (len(peaks), len(bed)))
    if a.normal_only:
        cond = pd.read_csv(COND, sep="\t")
        ok = set(zip(cond.loc[cond.condition == "normal", "sample_id"],
                     cond.loc[cond.condition == "normal", "replicate"]))
        n0 = len(peaks)
        peaks = peaks[[t in ok for t in zip(peaks.sample_id, peaks.replicate)]]
        print("normal-condition annotating samples only: %d -> %d rows" % (n0, len(peaks)))
    print()

    peaks["genic"] = gg.genic(peaks["chr"].to_numpy(), peaks.start.to_numpy())

    rng = np.random.default_rng(SEED)
    ch = np.array(sorted(sizes))
    w = np.array([sizes[c] for c in ch], float)
    nc = rng.choice(len(ch), N_NULL, p=w / w.sum())
    npos = (rng.random(N_NULL) * w[nc]).astype(int)
    null_genic = gg.genic(ch[nc], npos)

    rows = []
    for tf in TF12 + ["__null__"]:
        if tf == "__null__":
            v, name = null_genic, "random genome"
        else:
            v, name = peaks.loc[peaks.TF == tf, "genic"].to_numpy(), PRETTY[tf]
        n = len(v)
        rows.append({"TF": name, "n": n,
                     "genic": int(v.sum()), "intergenic": int(n - v.sum()),
                     "genic_pct": 100.0 * v.mean(),
                     "intergenic_pct": 100.0 * (1 - v.mean())})
    tab = pd.DataFrame(rows)
    nullg = float(tab.loc[tab.TF == "random genome", "genic_pct"].iloc[0])
    tab["genic_vs_null"] = tab.genic_pct / nullg
    tab.to_csv(os.path.join(a.outdir, "rossi_genic.tsv"), sep="\t", index=False)
    peaks.to_csv(os.path.join(a.outdir, "rossi_peaks_genic.tsv"), sep="\t", index=False)

    hdr = "%-14s %7s  %7s %7s   %7s %7s   %9s" % (
        "TF", "n", "genic", "interg", "genic%", "interg%", "vs null")
    print(hdr)
    print("-" * len(hdr))
    for _, r in tab.sort_values("genic_pct").iterrows():
        print("%-14s %7d  %7d %7d   %7.1f %7.1f   %8.2fx"
              % (r.TF, r.n, r.genic, r.intergenic,
                 r.genic_pct, r.intergenic_pct, r.genic_vs_null))
    print("\ngenic      = between some gene's ATG and its stop codon (Ensembl R64-1-1)")
    print("intergenic = everywhere else")
    print("vs null    = genic%% divided by the random-genome genic%% (%.1f%%); "
          "below 1 is depletion" % nullg)

    plot(tab, a.outdir, nullg)
    print("\nwrote %s/{rossi_genic.tsv,rossi_peaks_genic.tsv,genic_bars.png}"
          % os.path.basename(a.outdir))


def plot(tab, outdir, nullg):
    order = tab[tab.TF != "random genome"].sort_values("genic_pct")
    nrow = tab[tab.TF == "random genome"].iloc[0]
    labels = ["random genome", ""] + list(order.TF)
    gvals = np.array([nrow.genic_pct, np.nan] + list(order.genic_pct), float)

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    y = np.arange(len(labels))
    ax.barh(y, np.nan_to_num(gvals), height=.66, color=GENIC, zorder=3,
            edgecolor="white", linewidth=.8)
    ax.barh(y, np.nan_to_num(100 - gvals), left=np.nan_to_num(gvals), height=.66,
            color=INTER, zorder=3, edgecolor="white", linewidth=.8)
    for yi, g in enumerate(gvals):
        if np.isnan(g):
            continue
        if g >= 7:
            ax.text(g / 2, yi, "%.0f" % g, ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold", zorder=4)
        else:
            ax.text(g + 1.4, yi, "%.0f%%" % g, ha="left", va="center",
                    color=GENIC, fontsize=8.6, fontweight="bold", zorder=4)
        ax.text(g + (100 - g) / 2, yi, "%.0f" % (100 - g), ha="center", va="center",
                color="white", fontsize=9, fontweight="bold", zorder=4)

    ax.axvline(nullg, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=5, alpha=.75)
    ax.text(nullg + .8, len(labels) - .35, "random genome: %.0f%% genic" % nullg,
            fontsize=8.8, color=INK, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if lab == "random genome":
            t.set_style("italic")
            t.set_color(MUTED)
    ax.set_xlim(0, 100)
    ax.set_ylim(-.7, len(labels) - .3)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels(["%d%%" % v for v in range(0, 101, 20)], fontsize=9.5)
    ax.set_xlabel("share of reported peaks", fontsize=10.5, color=MUTED, labelpad=9)
    ax.xaxis.grid(True, color=GRID, lw=.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED)

    ax.set_title("Genic or intergenic", fontsize=15.5, fontweight="bold",
                 color=INK, loc="left", pad=34)
    ax.text(0, 1.045, "One rule: is the summit between some gene's ATG and its stop codon?",
            transform=ax.transAxes, fontsize=10.2, color=MUTED, va="bottom")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=GENIC, label="genic (inside an ORF)"),
                       Patch(fc=INTER, label="intergenic")],
              loc="lower right", bbox_to_anchor=(1.0, -.175), ncol=2,
              frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "genic_bars.png"), dpi=190,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
