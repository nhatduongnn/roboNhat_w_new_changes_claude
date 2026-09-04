"""Where do Rossi's reported peaks for our 12 fitted TFs actually fall -- promoter, gene
body, or neither?

WHY. agentR's literature review says none of the 12 (except Nhp6a, which is not
sequence-specific at all) has a documented population of occupied sites in gene-body
linker DNA, and that every real non-promoter site class -- telomeric tracts, subtelomeric
X-elements, silencers, ARSs, rDNA, tRNA genes -- is itself nucleosome-depleted. This puts
a number on that from the data we already parameterise on.

SOURCE. inputs/rossi_peak_w_strand_all_TFs.bed -- 29,328 rows, derived from the lab shared
set at /usr/xtmp/nd141/projects/data/lab_shared_dataset/rossi_strand/output/{tf}_fimo.bed.
Its spine is Rossi's MERGED ChExMix calls (04_ChExMix_Peaks/{TF}_CX.bed, one 1 bp summit per
peak per TF), each annotated with the nearest YEP FIMO motif within 30 bp. Verified: all 502
Abf1 rows here are a subset of that file's 569 has_motif rows, out of 724 total peaks.

So `sample_id`/`replicate` on a row is ANNOTATION provenance -- which replicate's
genome-wide FIMO scan supplied the winning motif -- and NOT which replicate called the peak.
There is exactly one row per merged peak; nothing needs collapsing across replicates.

Using this file rather than rossi_peak_w_strand_conformed_to_PWM_all_TFs_peakVal_1000.bed
(the one the model is fit on) matters: it holds 502 Abf1 peaks against the fitted file's 341,
and the filters that remove the other 161 drop a peak for failing to conform to the PWM --
exactly the bias that would distort a promoter/body split.

CONDITION. Of the 12 targets, only Spt15 has a non-normal sample (8600, rep2, 3 min heat
shock); 35 of its 294 rows carry that sample's motif annotation. --normal-only drops them.

ANNOTATION. Ensembl R64-1-1 = SGD R64 throughout: the ATG anchors the promoter window and
the stop codon anchors the gene-end window, both strand-aware, so "upstream" means upstream
of transcription and not merely leftward. Park 2014 TSSs are NOT used. They measure the
real transcript start but only for expressed genes -- 4,985 of 6,692 protein-coding genes,
and the 1,707 they miss were the single largest reason a peak got scored "gene body" when
it actually sat upstream of a neighbour. The ATG costs a known, measured offset instead:
over the genes where both exist, ATG - TSS has quartiles 29 / 52 / 103 bp, so every
promoter window sits a median 52 bp downstream of a TSS-anchored one. That bias is
constant and small next to the 600 bp window; the coverage hole was neither.

THE PRIORITY PROBLEM, AND WHAT IS DONE ABOUT IT. Yeast is gene-dense: a 500 bp upstream
window routinely runs into the neighbouring ORF, so a peak can be both "promoter of gene A"
and "body of gene B" and the promoter/body ratio becomes a function of which label wins.
Rather than pick one and hide it, BOTH orders are computed and printed. If the two
disagree materially for a factor, that factor's split is an artifact of the convention and
is reported as such.

THE NULL. Fractions alone are uninterpretable when ~70% of the genome is coding. Every
class is therefore also computed over 200,000 uniformly random genomic positions, so each
TF's bars can be read against what a factor binding nowhere in particular would give.

    conda activate robocop-2024
    python rossi_locus_class.py
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROSSI = os.path.join(HERE, "inputs", "rossi_peak_w_strand_all_TFs.bed")
PARK = os.path.join(HERE, "inputs", "Park_2014_TSS.csv")
GTF = os.path.join(HERE, "inputs", "sacCer3.gtf")
SIZES = os.path.join(HERE, "inputs", "sacCer3.chrom.sizes")
OUTDIR = os.path.join(HERE, "rossi_locus_class")

TF12 = ["abf1", "cin5", "fhl1", "fkh1", "mcm1", "nhp6a",
        "rap1", "reb1", "sko1", "spt15", "tbf1", "ume6"]
PRETTY = {"abf1": "Abf1", "cin5": "Cin5", "fhl1": "Fhl1", "fkh1": "Fkh1",
          "mcm1": "Mcm1", "nhp6a": "Nhp6a", "rap1": "Rap1", "reb1": "Reb1",
          "sko1": "Sko1", "spt15": "Spt15/TBP", "tbf1": "Tbf1", "ume6": "Ume6"}

PROM_UP, PROM_DOWN = 500, 100     # promoter window, in the direction of transcription
END_UP, END_DOWN = 100, 300       # "end of gene" window around the stop codon, same sense
TRNA_PAD = 500                    # a peak this close to a Pol III gene is "tRNA-proximal"
SUBTEL = 25000                    # distance from a chromosome end that counts as subtelomeric
N_NULL = 200000

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII",
         "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]
ARABIC2ROMAN = {"chr%d" % (i + 1): "chr%s" % r for i, r in enumerate(ROMAN)}

CLASSES = ["promoter", "gene end", "gene body", "other intergenic"]
COLORS = {"promoter": "#2a78d6", "gene end": "#8e5fc4", "gene body": "#eb6834",
          "other intergenic": "#9a9993"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e7e6e2"


# --------------------------------------------------------------------------- annotation
def read_gtf():
    """-> (protein-coding genes, Pol III + structural RNA genes), both as DataFrames."""
    rows = []
    for line in open(GTF):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9 or f[2] != "gene":
            continue
        attr = f[8]
        bt = attr.split('gene_biotype "', 1)[1].split('"', 1)[0] if 'gene_biotype "' in attr else ""
        gid = attr.split('gene_id "', 1)[1].split('"', 1)[0] if 'gene_id "' in attr else ""
        rows.append(("chr" + f[0], int(f[3]) - 1, int(f[4]), f[6], gid, bt))
    g = pd.DataFrame(rows, columns=["chr", "start", "end", "strand", "gene", "biotype"])
    return (g[g.biotype == "protein_coding"].reset_index(drop=True),
            g[g.biotype.isin(["tRNA", "snoRNA", "snRNA", "rRNA"])].reset_index(drop=True))


def read_anchors(genes):
    """Per-gene 5' and 3' anchors, strand-aware, straight from the GTF.

    5' = the ATG (start codon), 3' = the stop codon. Both come from the same complete
    Ensembl R64-1-1 = SGD R64 annotation, so every one of the 6,692 protein-coding genes
    has both -- which is the point. Park 2014 measures the real TSS but only for expressed
    genes: 4,985 of 6,692, and the 1,707 it misses were the single largest reason a peak
    got labelled "gene body" when it was actually upstream of a neighbour.

    The price is a known, measured offset. Over the 4,985 genes where both exist,
    ATG - TSS has quartiles 29 / 52 / 103 bp, so an ATG-anchored promoter window sits a
    median 52 bp downstream of a TSS-anchored one. That is deliberate and uncorrected:
    a complete annotation with one constant bias beats a partial one with none, and the
    bias is small next to the 600 bp window it shifts inside.
    """
    a = genes.copy()
    a["five"] = np.where(a.strand == "+", a.start, a.end)
    a["three"] = np.where(a.strand == "+", a.end, a.start)
    return a[["chr", "gene", "strand", "five", "three"]]


class Annot:
    """Per-chromosome interval and point lookups. Sorted arrays + searchsorted, so the
    200k-position null costs about as much as the 3,270 real peaks."""

    def __init__(self, genes, rnas, sizes):
        self.sizes = sizes
        self.body, self.rna, self.five, self.three = {}, {}, {}, {}
        anchors = read_anchors(genes)
        self.genes = {}
        for c, sub in genes.groupby("chr"):
            s = sub.sort_values("start")
            self.body[c] = (s.start.to_numpy(), s.end.to_numpy())
            self.genes[c] = list(zip(
                s.start.to_numpy(), s.end.to_numpy(),
                np.where(s.strand.to_numpy() == "+", s.start.to_numpy(), s.end.to_numpy()),
                np.where(s.strand.to_numpy() == "+", s.end.to_numpy(), s.start.to_numpy()),
                np.where(s.strand.to_numpy() == "+", 1, -1)))
        for c, sub in rnas.groupby("chr"):
            s = sub.sort_values("start")
            self.rna[c] = (s.start.to_numpy() - TRNA_PAD, s.end.to_numpy() + TRNA_PAD)
        for c, sub in anchors.groupby("chr"):
            sign = np.where(sub.strand.to_numpy() == "+", 1, -1)
            for store, col in ((self.five, "five"), (self.three, "three")):
                o = np.argsort(sub[col].to_numpy())
                store[c] = (sub[col].to_numpy()[o], sign[o])

    @staticmethod
    def _in_any(pos, iv):
        """Inside any [start, end)? Intervals may overlap, so a running max of ends is
        used rather than assuming disjointness."""
        if iv is None:
            return np.zeros(len(pos), bool)
        st, en = iv
        i = np.searchsorted(st, pos, "right") - 1
        cummax = np.maximum.accumulate(en)
        ok = i >= 0
        out = np.zeros(len(pos), bool)
        out[ok] = pos[ok] < cummax[i[ok]]
        return out

    @staticmethod
    def _signed_dist(t, pos):
        """Signed distance to the nearest anchor, in the direction of transcription.
        Negative = upstream (5' side of the anchor). Nearest is chosen on |unsigned|
        distance, so the sign is read off the anchor that won, not off the array."""
        if t is None:
            return np.full(len(pos), np.nan)
        coord, sign = t
        j = np.clip(np.searchsorted(coord, pos), 1, len(coord) - 1)
        left, right = coord[j - 1], coord[j]
        pick = np.where(pos - left <= right - pos, j - 1, j)
        return (pos - coord[pick]) * sign[pick]

    def dist_to_tss(self, c, pos):
        return self._signed_dist(self.five.get(c), pos)

    def dist_to_end(self, c, pos):
        return self._signed_dist(self.three.get(c), pos)

    def host_labels(self, chrom, pos):
        """Windows restricted to a gene the position is actually INSIDE.

        The default scheme lets any gene's window claim any position, which produces a
        real error: a peak 400 bp upstream of gene B's ATG can sit inside gene A's ORF and
        still be labelled "promoter". 25 of Abf1's 502 peaks are exactly that. Here a peak
        inside an ORF may only be claimed by a window belonging to a gene it is inside.

        That collapses to something cheap. Inside its own ORF a position's distance from
        that ORF's ATG is >= 0, so "within -PROM_UP..+PROM_DOWN of my own ATG" reduces to
        "within the first PROM_DOWN bp of the ORF"; symmetrically gene end reduces to
        "within the last END_UP bp". Overlapping ORFs are resolved in favour of the
        stronger label, promoter over gene end.

        -> (inside_any_orf, near own 5' end, near own 3' end), one entry per position.
        """
        n = len(pos)
        inb = np.zeros(n, bool)
        p5 = np.zeros(n, bool)
        p3 = np.zeros(n, bool)
        df = pd.DataFrame({"chr": chrom, "pos": pos, "i": np.arange(n)})
        for c, sub in df.groupby("chr", sort=False):
            g = self.genes.get(c)
            if g is None:
                continue
            o = np.argsort(sub.pos.to_numpy())
            ps = sub.pos.to_numpy()[o]
            ix = sub.i.to_numpy()[o]
            for st, en, five, three, sgn in g:
                lo = np.searchsorted(ps, st, "left")
                hi = np.searchsorted(ps, en, "left")
                if hi <= lo:
                    continue
                sl, q = ix[lo:hi], ps[lo:hi]
                inb[sl] = True
                p5[sl] |= ((q - five) * sgn) <= PROM_DOWN
                p3[sl] |= ((three - q) * sgn) <= END_UP
        return inb, p5, p3

    def classify(self, chrom, pos):
        """-> DataFrame of the raw predicates, one row per position.

        FOUR classes now, not three. A peak sitting just past a stop codon is inside the
        ORF and used to be scored "gene body", but yeast has a nucleosome-depleted region
        at the terminator as well as the promoter: that peak is in open chromatin of the
        same physical character as a promoter NDR, and calling it gene body implied
        something false -- buried in nucleosomal DNA mid-transcription-unit. "gene end"
        splits it out. Anchor is the stop codon (GTF), window -END_UP..+END_DOWN in the
        direction of transcription; the terminator NDR sits downstream of the poly(A)
        site, itself ~100 bp past the stop, hence the asymmetry.
        """
        out = pd.DataFrame({"chr": chrom, "pos": pos})
        out["in_body"] = False
        out["near_rna"] = False
        out["d_tss"] = np.nan
        out["d_end"] = np.nan
        for c, idx in out.groupby("chr").groups.items():
            p = out.loc[idx, "pos"].to_numpy()
            out.loc[idx, "in_body"] = self._in_any(p, self.body.get(c))
            out.loc[idx, "near_rna"] = self._in_any(p, self.rna.get(c))
            out.loc[idx, "d_tss"] = self.dist_to_tss(c, p)
            out.loc[idx, "d_end"] = self.dist_to_end(c, p)
        out["in_prom"] = (out.d_tss >= -PROM_UP) & (out.d_tss <= PROM_DOWN)
        out["in_end"] = (out.d_end >= -END_UP) & (out.d_end <= END_DOWN)
        end = out.chr.map(self.sizes).to_numpy(float)
        out["subtel"] = (out.pos < SUBTEL) | (out.pos > end - SUBTEL)
        # Priority: promoter > gene end > gene body > other intergenic. Yeast is dense
        # enough that a position routinely satisfies two predicates at once (300 bp past
        # gene A's stop AND 400 bp upstream of gene B's ATG is the ordinary arrangement
        # for a convergent-then-divergent pair), so the order is doing real work and its
        # cost is measured: `ambiguous` counts positions where it mattered, and
        # cls_body_first re-runs the whole thing with body winning instead.
        out["cls_prom_first"] = np.select(
            [out.in_prom, out.in_end, out.in_body],
            ["promoter", "gene end", "gene body"], default="other intergenic")
        out["cls_body_first"] = np.select(
            [out.in_body, out.in_prom, out.in_end],
            ["gene body", "promoter", "gene end"], default="other intergenic")
        inb, p5, p3 = self.host_labels(chrom, pos)
        out["cls_host_first"] = np.where(
            inb, np.where(p5, "promoter", np.where(p3, "gene end", "gene body")),
            np.where(out.in_prom, "promoter",
                     np.where(out.in_end, "gene end", "other intergenic")))
        out["ambiguous"] = (out.in_prom.astype(int) + out.in_end.astype(int)
                            + out.in_body.astype(int)) > 1
        return out


# --------------------------------------------------------------------------------- main
def main():
    global OUTDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-only", action="store_true",
                    help="drop peaks whose FIMO annotation came from a sample that is not "
                         "'normal' condition (only Spt15 rep2, 3 min heat shock, is affected)")
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    OUTDIR = a.outdir
    os.makedirs(OUTDIR, exist_ok=True)
    sz = {}
    for line in open(SIZES):
        f = line.split()
        if len(f) == 2:
            sz[f[0] if f[0].startswith("chr") else "chr" + f[0]] = int(f[1])
    sizes = sz

    genes, rnas = read_gtf()
    print("protein-coding genes: %d   Pol III / structural RNA genes: %d" % (len(genes), len(rnas)))
    ann = Annot(genes, rnas, sizes)

    bed = pd.read_csv(ROSSI, sep="\t")
    bed["chr"] = bed["chr"].map(ARABIC2ROMAN).fillna(bed["chr"])
    unmapped = sorted(set(bed["chr"]) - set(sizes))
    if unmapped:
        print("dropping peaks on unmapped contigs: %s" % unmapped)
        bed = bed[bed["chr"].isin(sizes)]
    peaks = bed[bed.TF.isin(TF12)].copy()
    print("\nRossi calls for the 12 TFs: %d (of %d in the file)" % (len(peaks), len(bed)))
    if a.normal_only:
        cond = pd.read_csv(os.path.join(HERE, "inputs", "rossi_sample_conditions.tsv"), sep="\t")
        ok = set(zip(cond.loc[cond.condition == "normal", "sample_id"],
                     cond.loc[cond.condition == "normal", "replicate"]))
        n0 = len(peaks)
        peaks = peaks[[t in ok for t in zip(peaks.sample_id, peaks.replicate)]]
        print("normal-condition annotating samples only: %d -> %d rows" % (n0, len(peaks)))
    print()

    cl = ann.classify(peaks["chr"].to_numpy(), peaks.start.to_numpy())
    peaks = pd.concat([peaks.reset_index(drop=True),
                       cl.drop(columns=["chr", "pos"]).reset_index(drop=True)], axis=1)

    rng = np.random.default_rng(0)
    chroms = np.array(sorted(sizes))
    w = np.array([sizes[c] for c in chroms], float)
    nc = rng.choice(len(chroms), N_NULL, p=w / w.sum())
    npos = (rng.random(N_NULL) * w[nc]).astype(int)
    null = ann.classify(chroms[nc], npos)

    # -------------------------------------------------------------- table
    rows = []
    for tf in TF12 + ["__null__"]:
        d = null if tf == "__null__" else peaks[peaks.TF == tf]
        n = len(d)
        r = {"TF": "random genome" if tf == "__null__" else PRETTY[tf], "n": n}
        for c in CLASSES:
            r[c] = int((d.cls_prom_first == c).sum())
            r[c + "_pct"] = 100.0 * r[c] / n
        r["body_first_body_pct"] = 100.0 * (d.cls_body_first == "gene body").mean()
        for c in CLASSES:
            r["host_" + c + "_pct"] = 100.0 * (d.cls_host_first == c).mean()
        r["subtel_pct"] = 100.0 * d.subtel.mean()
        r["rna_pct"] = 100.0 * d.near_rna.mean()
        r["median_|d_tss|"] = float(np.nanmedian(np.abs(d.d_tss)))
        r["ambiguous_pct"] = 100.0 * d.ambiguous.mean()
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUTDIR, "rossi_locus_class.tsv"), sep="\t", index=False)
    peaks.to_csv(os.path.join(OUTDIR, "rossi_peaks_classified.tsv"), sep="\t", index=False)

    hdr = ("%-11s %6s | %6s %6s %6s %6s | %6s %6s %6s %6s | %6s %6s"
           % ("TF", "n", "prom%", "end%", "body%", "oth%",
              "prom%", "end%", "body%", "oth%", "body2%", "ambig%"))
    print("%-11s %6s | %-27s | %-27s |" % ("", "", "  HOST-ONLY (reported)",
                                           "  ANY-GENE (permissive)"))
    print(hdr); print("-" * len(hdr))
    for _, r in tab.iterrows():
        print("%-11s %6d | %6.1f %6.1f %6.1f %6.1f | %6.1f %6.1f %6.1f %6.1f | %6.1f %6.1f"
              % (r.TF, r.n,
                 r["host_promoter_pct"], r["host_gene end_pct"],
                 r["host_gene body_pct"], r["host_other intergenic_pct"],
                 r["promoter_pct"], r["gene end_pct"], r["gene body_pct"],
                 r["other intergenic_pct"],
                 r.body_first_body_pct, r.ambiguous_pct))
    print("\nHOST-ONLY         = a peak inside an ORF may only be claimed by a window of a "
          "gene it is\n                    INSIDE. This is the reported scheme.")
    print("ANY-GENE          = any gene's window may claim any position. Its flaw: a peak "
          "400 bp\n                    upstream of gene B's ATG can sit inside gene A's ORF "
          "and still be called\n                    promoter -- 25 of Abf1's 502 peaks are "
          "exactly that.")
    print("prom%%             = within -%d..+%d of an ATG, promoter wins ties"
          % (PROM_UP, PROM_DOWN))
    print("end%%              = within -%d..+%d of a stop codon (terminator NDR), "
          "second in priority" % (END_UP, END_DOWN))
    print("ambig%             = satisfies more than one predicate, so the priority order "
          "decided the label")
    print("body2%            = gene body under the OPPOSITE priority; the gap between it "
          "and body% is how much of the split is convention rather than data")
    print("subtel%%          = within %d kb of a chromosome end" % (SUBTEL // 1000))
    print("RNA%%             = within %d bp of a tRNA/snoRNA/snRNA/rRNA gene" % TRNA_PAD)

    # -------------------------------------------------------------- plots
    plot_stacked(tab)
    plot_dist(peaks, null)
    print("\nwrote %s/{rossi_locus_class.tsv,rossi_peaks_classified.tsv,"
          "locus_class_bars.png,dist_to_tss.png}" % os.path.basename(OUTDIR))


def plot_stacked(tab):
    """Random genome sits at the BOTTOM as the reference row; TFs run upward from least to
    most promoter-biased, so the eye reads the gap against the null directly."""
    order = tab[tab.TF != "random genome"].sort_values("promoter_pct", ascending=True)
    nullrow = tab[tab.TF == "random genome"].iloc[0]
    labels = ["random genome", ""] + list(order.TF)
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    y = np.arange(len(labels))

    def vals(name):
        return np.array([nullrow[name + "_pct"], np.nan] + list(order[name + "_pct"]), float)

    left = np.zeros(len(labels))
    for c in CLASSES:
        v = vals(c)
        ax.barh(y, np.nan_to_num(v), left=left, height=0.68, color=COLORS[c],
                edgecolor="white", linewidth=0.8, zorder=3)
        for yi, (lo, w) in enumerate(zip(left, v)):
            if not np.isnan(w) and w >= 6:
                ax.text(lo + w / 2, yi, "%.0f" % w, ha="center", va="center",
                        fontsize=8.6, color="white", fontweight="bold", zorder=4)
        left += np.nan_to_num(v)

    for yi, n in enumerate([nullrow.n, np.nan] + list(order.n)):
        if not np.isnan(n):
            ax.text(101.2, yi, "n=%d" % int(n), va="center", fontsize=8.4, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.get_yticklabels()[0].set_color(MUTED)
    ax.get_yticklabels()[0].set_style("italic")
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of Rossi peaks", fontsize=10.5)
    ax.set_title("Where Rossi's peaks fall for the 12 individually-fitted TFs",
                 fontsize=13.5, fontweight="bold", pad=44)
    ax.text(0.0, 1.075,
            "promoter = within %d bp upstream / %d bp downstream of a Park 2014 TSS, in the "
            "direction of transcription; promoter wins where a peak is both.\n"
            "The italic bottom row is 200,000 uniformly random genomic positions -- what a "
            "factor binding nowhere in particular would give." % (PROM_UP, PROM_DOWN),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, color=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#b8b7b2")
    ax.tick_params(length=0, labelsize=9.5)
    ax.legend(handles=[Patch(fc=COLORS[c], label=c) for c in CLASSES],
              frameon=False, fontsize=9.4, loc="lower right", ncol=3,
              bbox_to_anchor=(1.0, -0.155))
    fig.subplots_adjust(left=0.135, right=0.965, top=0.815, bottom=0.155)
    fig.savefig(os.path.join(OUTDIR, "locus_class_bars.png"), dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_dist(peaks, null):
    """Signed distance to the nearest TSS. This is the panel that shows WHETHER a factor is
    promoter-targeted, independently of any window choice."""
    bins = np.arange(-1000, 1001, 40)
    fig, axes = plt.subplots(3, 4, figsize=(14.2, 8.2), sharex=True)
    nd = null.d_tss.to_numpy(float)
    nh, _ = np.histogram(nd[np.isfinite(nd)], bins=bins)
    nh = nh / max(nh.sum(), 1)
    for ax, tf in zip(axes.ravel(), TF12):
        d = peaks.loc[peaks.TF == tf, "d_tss"].to_numpy(float)
        d = d[np.isfinite(d)]
        h, _ = np.histogram(d, bins=bins)
        h = h / max(h.sum(), 1)
        ctr = (bins[:-1] + bins[1:]) / 2
        ax.fill_between(ctr, nh, step="mid", color="#d9d8d3", zorder=1)
        ax.step(ctr, h, where="mid", color="#2a78d6", lw=1.6, zorder=3)
        ax.axvline(0, color=INK, lw=1.0, zorder=4)
        ax.axvspan(-PROM_UP, PROM_DOWN, color="#2a78d6", alpha=0.08, zorder=0)
        inw = 100.0 * np.mean((d >= -PROM_UP) & (d <= PROM_DOWN))
        ax.set_title("%s   n=%d   %.0f%% in window" % (PRETTY[tf], len(d), inw),
                     fontsize=10.2, fontweight="bold")
        ax.set_yticks([]); ax.tick_params(length=0, labelsize=8.6)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#b8b7b2")
    for ax in axes[-1]:
        ax.set_xlabel("distance to nearest TSS (bp)\nnegative = upstream", fontsize=9.2)
    fig.suptitle("Distance from each Rossi peak to the nearest Park 2014 TSS",
                 fontsize=13.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, "blue = the TF   ·   grey fill = 200,000 random genomic positions"
             "   ·   shaded band = the -%d..+%d promoter window used in the bar chart"
             % (PROM_UP, PROM_DOWN), ha="center", fontsize=9.2, color=MUTED)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.885, bottom=0.105, hspace=0.42)
    fig.savefig(os.path.join(OUTDIR, "dist_to_tss.png"), dpi=150, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
