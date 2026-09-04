"""Does the promoter/gene-body split depend on WHICH Rossi peak set you start from?

Three peak sets for the same 12 factors, classified by identical machinery
(rossi_locus_class.Annot), normal-condition samples only:

  zip       per-sample ChExMix calls out of the YEP zips, unioned over a factor's normal
            replicates. Summits from different replicates land a few bp apart, so the union
            is taken with a small merge window rather than on exact equality.
            {id}_chexmix_FILTERED_peaks.bed, not {id}_chexmix_peaks.bed: the filtered file
            is the one the blacklist (rDNA, tRNA genes, telomeres) has been applied to, and
            the merged set is blacklisted too, so this is the apples-to-apples comparison.
            Using the unblacklisted file instead makes every factor look far more random --
            Fhl1 23% promoter rather than 55% -- because rDNA and telomeric repeats generate
            junk calls.
  merged    Rossi's combined calls, {TF}_CX.bed -- the release's own cross-replicate
            merge, one summit per peak. No motif requirement.
  +motif    merged, restricted to peaks carrying a FIMO motif within 30 bp, i.e. the
            file the model is parameterised from
            (inputs/rossi_peak_w_strand_all_TFs.bed).

Also answers, empirically, whether heat-shock reads reached the merged calls: Spt15 is the
only one of the 12 with a non-normal replicate (8600, 3 min heat shock), so its merged peaks
are checked against the normal replicates' calls and against 8600's separately.

    conda activate robocop-2024
    python rossi_zip_vs_merged.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import rossi_locus_class as R

ROSSI_DIR = "/usr/xtmp/nd141/projects/data/rossi_strand"
OUTDIR = os.path.join(R.HERE, os.environ.get("SETCMP_DIR", "rossi_locus_class_setcmp"))
MERGE_BP = 10          # replicate summits of the same event sit within a few bp
SETS = ["zip", "merged", "+motif"]


def union_within(df, bp):
    """Collapse calls within `bp` of one another (per chromosome), keeping the strongest.
    Used ONLY for the zip set, where a real event is called once per replicate."""
    keep = []
    for c, g in df.groupby("chr", sort=False):
        g = g.sort_values("start")
        grp = (g.start.diff().fillna(bp + 1) > bp).cumsum()
        keep.append(g.loc[g.groupby(grp).peakVal.idxmax()])
    return pd.concat(keep, ignore_index=True)


def read_zip(sample_ids):
    out = []
    for sid in sample_ids:
        p = "%s/%d/%d_YEP/%d_chexmix_filtered_peaks.bed" % (ROSSI_DIR, sid, sid, sid)
        d = pd.read_csv(p, sep="\t", header=None, comment="t",
                        names=["chr", "start", "end", "name", "peakVal"])
        d["sample_id"] = sid
        out.append(d)
    return pd.concat(out, ignore_index=True)


def read_cx(target):
    return pd.read_csv("%s/%s_CX.bed" % (ROSSI_DIR, target), sep="\t", header=None,
                       names=["chr", "start", "end", "name", "peakVal", "strand"])


def to_roman(df):
    df = df.copy()
    df["chr"] = df["chr"].map(R.ARABIC2ROMAN).fillna(df["chr"])
    return df[df["chr"].isin(SIZES)]


def frac(ann, df):
    cl = ann.classify(df["chr"].to_numpy(), df.start.to_numpy())
    n = len(cl)
    return dict(n=n,
                prom=100 * (cl.cls_prom_first == "promoter").mean(),
                body=100 * (cl.cls_prom_first == "gene body").mean(),
                oth=100 * (cl.cls_prom_first == "other intergenic").mean(),
                subtel=100 * cl.subtel.mean(),
                dtss=float(np.nanmedian(np.abs(cl.d_tss))))


def nearest_bp(a, b):
    """For each row of a, distance in bp to the nearest row of b on the same chromosome."""
    out = np.full(len(a), np.inf)
    for c, idx in a.groupby("chr").groups.items():
        sub = b[b["chr"] == c]
        if not len(sub):
            continue
        t = np.sort(sub.start.to_numpy())
        p = a.loc[idx, "start"].to_numpy()
        j = np.clip(np.searchsorted(t, p), 1, len(t) - 1)
        out[a.index.get_indexer(idx)] = np.minimum(np.abs(p - t[j - 1]), np.abs(p - t[j]))
    return out


def main():
    global SIZES
    os.makedirs(OUTDIR, exist_ok=True)
    SIZES = {}
    for line in open(R.SIZES):
        f = line.split()
        if len(f) == 2:
            SIZES[f[0] if f[0].startswith("chr") else "chr" + f[0]] = int(f[1])

    genes, rnas = R.read_gtf()
    ann = R.Annot(genes, rnas, SIZES)

    cond = pd.read_csv(os.path.join(R.HERE, "inputs", "rossi_sample_conditions.tsv"), sep="\t")
    motif = pd.read_csv(os.path.join(R.HERE, "inputs", "rossi_peak_w_strand_all_TFs.bed"),
                        sep="\t")
    ok = set(zip(cond.loc[cond.condition == "normal", "sample_id"],
                 cond.loc[cond.condition == "normal", "replicate"]))
    motif = motif[[t in ok for t in zip(motif.sample_id, motif.replicate)]]

    target_of = {t.lower(): t for t in
                 ["Abf1", "Cin5", "Fhl1", "Fkh1", "Mcm1", "Nhp6a",
                  "Rap1", "Reb1", "Sko1", "Spt15", "Tbf1", "Ume6"]}

    rows = []
    for tf in R.TF12:
        target = target_of[tf]
        sids = cond[(cond.target == target) & (cond.condition == "normal")].sample_id.tolist()
        z_raw = to_roman(read_zip(sids))
        z = union_within(z_raw, MERGE_BP)
        m = to_roman(read_cx(target))
        f = to_roman(motif[motif.TF == tf].rename(columns={"chr": "chr"}))
        for name, d, extra in (("zip", z, "%d calls over %d reps" % (len(z_raw), len(sids))),
                               ("merged", m, ""),
                               ("+motif", f, "")):
            r = frac(ann, d)
            r.update(TF=R.PRETTY[tf], set=name, note=extra)
            rows.append(r)

    tab = pd.DataFrame(rows)
    nullc = ann.classify(*random_null(SIZES))
    tab.to_csv(os.path.join(OUTDIR, "set_comparison.tsv"), sep="\t", index=False)

    w = tab.pivot(index="TF", columns="set", values=["n", "prom", "body"])
    print("\n%-11s %-24s %-24s %s" % ("", "n", "promoter %", "gene body %"))
    print("%-11s %7s %7s %7s  %7s %7s %7s  %7s %7s %7s"
          % ("TF", "zip", "merged", "+motif", "zip", "merged", "+motif",
             "zip", "merged", "+motif"))
    print("-" * 84)
    for tf in [R.PRETTY[t] for t in R.TF12]:
        r = w.loc[tf]
        print("%-11s %7d %7d %7d  %7.1f %7.1f %7.1f  %7.1f %7.1f %7.1f"
              % (tf, r[("n", "zip")], r[("n", "merged")], r[("n", "+motif")],
                 r[("prom", "zip")], r[("prom", "merged")], r[("prom", "+motif")],
                 r[("body", "zip")], r[("body", "merged")], r[("body", "+motif")]))
    nrow = frac(ann, pd.DataFrame(dict(zip(["chr", "start"], random_null(SIZES)))))
    print("-" * 84)
    print("%-11s %7s %7s %7d  %7s %7s %7.1f  %7s %7s %7.1f"
          % ("random", "", "", nrow["n"], "", "", nrow["prom"], "", "", nrow["body"]))

    heat_shock_audit(ann, cond, target_of)
    plot(tab, nrow)
    print("\nwrote %s/{set_comparison.tsv,set_comparison.png}" % os.path.basename(OUTDIR))


def random_null(sizes, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    chroms = np.array(sorted(sizes))
    w = np.array([sizes[c] for c in chroms], float)
    nc = rng.choice(len(chroms), n, p=w / w.sum())
    return chroms[nc], (rng.random(n) * w[nc]).astype(int)


def heat_shock_audit(ann, cond, target_of):
    """Did heat-shock reads reach Rossi's merged calls? Spt15 is the only one of the 12 with
    a non-normal replicate, so its merged peaks are matched against the normal replicates'
    own calls and against the heat-shock replicate's separately."""
    print("\n" + "=" * 84)
    print("HEAT-SHOCK AUDIT -- Spt15 (rep2 = sample 8600, 3 min heat shock)")
    sub = cond[cond.target == "Spt15"]
    norm = sub[sub.condition == "normal"].sample_id.tolist()
    hs = sub[sub.condition != "normal"].sample_id.tolist()
    m = to_roman(read_cx("Spt15")).reset_index(drop=True)
    zn = to_roman(read_zip(norm)).reset_index(drop=True)
    zh = to_roman(read_zip(hs)).reset_index(drop=True)
    dn, dh = nearest_bp(m, zn), nearest_bp(m, zh)
    for w in (0, 10, 50):
        only_hs = int(((dn > w) & (dh <= w)).sum())
        print("  within %2d bp: %4d/%4d merged peaks match a NORMAL rep, %4d match the "
              "heat-shock rep, %3d match ONLY heat shock"
              % (w, int((dn <= w).sum()), len(m), int((dh <= w).sum()), only_hs))
    print("  (heat-shock rep has %d calls; normal reps have %d across %d samples)"
          % (len(zh), len(zn), len(norm)))
    print("=" * 84)


def plot(tab, nrow):
    C = {"zip": "#2a6fd6", "merged": "#c9502a", "+motif": "#7a8290"}
    order = (tab[tab.set == "merged"].sort_values("prom", ascending=True).TF.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6), sharey=True)
    y = np.arange(len(order))
    for ax, col, lab, ref in ((axes[0], "prom", "promoter %", nrow["prom"]),
                              (axes[1], "body", "gene body %", nrow["body"])):
        for k, s in enumerate(SETS):
            v = [float(tab[(tab.TF == t) & (tab.set == s)][col].iloc[0]) for t in order]
            ax.barh(y + (1 - k) * 0.26, v, height=0.24, color=C[s], zorder=3,
                    edgecolor="white", linewidth=0.6)
        ax.axvline(ref, color="#0b0b0b", lw=1.2, ls="--", zorder=4)
        ax.text(ref, len(order) - 0.25, " random genome %.0f%%" % ref, fontsize=8.6,
                color="#4b535f", va="bottom")
        ax.set_xlabel(lab, fontsize=10.5)
        ax.xaxis.grid(True, color="#e7e6e2", lw=1, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#b8b7b2")
        ax.tick_params(length=0, labelsize=9.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=10.5)
    axes[0].set_ylim(-0.6, len(order) - 0.1)
    fig.suptitle("Three Rossi peak sets, same classification, normal-condition samples only",
                 fontsize=13.5, fontweight="bold", y=0.975)
    fig.legend(handles=[Patch(fc=C[s], label={"zip": "zip (per-sample ChExMix, unioned)",
                                              "merged": "merged (%s_CX.bed)" % "{TF}",
                                              "+motif": "merged + FIMO motif within 30 bp"}[s])
                        for s in SETS],
               frameon=False, fontsize=9.4, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.905, bottom=0.135, wspace=0.06)
    fig.savefig(os.path.join(OUTDIR, "set_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
