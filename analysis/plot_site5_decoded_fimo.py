"""One locus, +/-200 bp: FIMO motif scan + decoded ABF1 posteriors + raw methylation.

Defaults to MacIsaac ABF1 site #5 (chrI:108788-108802); pass --chrom/--start/--end
--span-label to draw the identical panel set anywhere, e.g. a FIMO hit that is NOT a
MacIsaac site.

PROVENANCE IS THE POINT OF THIS SCRIPT.  Every panel is labelled with where its
numbers came from, and the plotting code invents none of them:

  DECODE OUTPUT  the forward-backward posterior table in tmpDir/info_*.h5.
                 posterior[i, j] = P(state j at position i | all data), rows summing
                 to 1 over 3485 states (algo.c:227).  ABF1 forward = states
                 tf_starts[0] .. +tf_lens[0]; reverse = the next tf_lens[0] states.
                 Summing a contiguous state block is exact, not an approximation.

  RAW INPUT      Fiber_count_{meth,A}_{watson,crick} from the same h5.

  FIMO           an EXTERNAL tool (MEME suite), not decode output.  RoboCOP never
                 computes a per-position motif score -- it fills
                 emission[0][pos][state] = pwm_emission[state][base] and feeds that
                 into forward-backward, so no such score exists to read.  FIMO
                 supplies the missing combining rule.  It scans with the IDENTICAL
                 matrix: inputs/motifs_meme.txt MOTIF Abf1_murphy is np.allclose to
                 both pwm.p['Abf1_murphy'] and HMMconfig['pwm_emission'][1:15].

COORDINATES.  Everything is in the bed 0-based frame the rest of analysis/ uses, so
this figure lines up with the other site plots.  Verified empirically (not assumed):
h5 index = pos_bed - (segment attrs['start'] - 1) reproduces the samtools-faidx
sequence exactly over this window.  samtools and FIMO are 1-based, so their
coordinates are converted with -1 on entry.

    python plot_site5_decoded_fimo.py
    python plot_site5_decoded_fimo.py --start 190559 --end 190573 \
        --span-label "FIMO top chrI hit" --workdir chrI190k_fimo --out chrI190559_decoded_fimo.png
"""
import os, sys, glob, subprocess, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import score_robocop as S
import plot_abf1_5sites_layers as PL          # fiber_track()
import plot_abf1_5sites_decoded as PD         # state_composition()

# Default locus = MacIsaac site #5, straight from inputs/MacIsaac_..._match_PWM.bed:
#   chrI  108788  108802  ABF1  0  -
# Overridable so any locus can be drawn with the identical panel set; the defaults
# keep site5_decoded_fimo.png reproducible with a bare `python plot_site5_decoded_fimo.py`.
D_CHRM, D_M0, D_M1, D_STRAND = "chrI", 108788, 108802, "-"
D_LABEL = "MacIsaac ABF1 site #5"
FIMO_PAD = 25            # room for placements whose midpoint reaches the window edge

FIBER_DIR = "robocop_chrI_maskon_revfix"
SEQ_DIR = "robocop_chrI_seq_maskon_revfix"
SEQONLY_DIR = "robocop_chrI_seqonly_maskon_revfix"   # sequence layer alone, Fiber neutralised
MEME_ENV = "/home/users/nd141/miniconda3/envs/meme/bin"
MEME_FILE = "inputs/motifs_meme.txt"
GENOME = "inputs/SacCer3.fa"

# validated as a 4-set for the light surface (scripts/validate_palette.js:
# lightness PASS, chroma PASS, worst adjacent CVD dE 23.1, normal-vision dE 24.0).
# Contrast WARN on aqua -> every series is direct-labelled and tabulated to stdout.
FWD, REV = "#4a3aa7", "#1baf7a"          # ABF1 orientation
WATSON, CRICK = "#2a78d6", "#eb6834"     # raw channels, unchanged from plotRoboCOP
INK, MUTED, GRID, BAND = "#0b0b0b", "#6b6a66", "#e7e6e2", "#cfcec9"
RED = "#a2322b"
SURFACE = "#fcfcfb"
BG_METH = 0.1383
TAG_DECODE = ("DECODE OUTPUT", "#1a7a44", "#eafaf1")
TAG_RAW = ("RAW INPUT", "#2a78d6", "#eaf2fc")
TAG_EXT = ("FIMO — EXTERNAL TOOL, NOT DECODE OUTPUT", "#a2322b", "#fdecec")


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def faidx(chrm, lo_bed, hi_bed):
    """Sequence over the INCLUSIVE bed-0-based range [lo_bed, hi_bed].
    samtools is 1-based inclusive, so a bed coordinate g is 1-based g+1."""
    r = "%s:%d-%d" % (chrm, lo_bed + 1, hi_bed + 1)
    fa = sh(["samtools", "faidx", GENOME, r])
    return "".join(l.strip() for l in fa.splitlines() if not l.startswith(">")).upper()


def h5_nucleotides(outDir, chrm, lo_bed, hi_bed):
    """Reference bases as the DECODE stored them, for the coordinate assertion."""
    dec = S.load_decode(outDir)
    coords = dec["dshared"], dec["coords"]
    co = dec["coords"]
    cand = co[(co.chr == chrm) & (co.start <= lo_bed) & (co.end >= hi_bed)]
    if cand.empty:
        return None
    seg = int(cand.index[0])
    for f in sorted(glob.glob(os.path.join(outDir, "tmpDir", "info_*.h5"))):
        with h5py.File(f, "r") as h:
            k = "segment_%d" % seg
            if k not in h:
                continue
            import robocop
            nt = robocop.get_sparse_todense(h, k + "/nucleotides").astype(int)
            base = int(h[k].attrs["start"]) - 1
        return "".join("ACGTN"[min(nt[g - base], 4)] for g in range(lo_bed, hi_bed + 1))
    return None


def run_fimo(chrm, lo_bed, hi_bed, workdir):
    """FIMO over [lo_bed, hi_bed], every placement, RoboCOP's background.

    Returns a dict of arrays keyed by strand.  FIMO parses the `chr:start-end`
    FASTA header and reports 1-BASED GENOMIC coordinates, converted to bed here.
    """
    os.makedirs(workdir, exist_ok=True)
    bg = os.path.join(workdir, "robocop_bg.txt")
    stem = "%s_%d_%d" % (chrm, lo_bed, hi_bed)     # region-stamped so runs never collide
    fa = os.path.join(workdir, stem + ".fa")
    tsv = os.path.join(workdir, "fimo_" + stem + ".tsv")

    # background = RoboCOP's own, so FIMO's log-odds share the decode's reference
    import pickle
    pwm = pickle.load(open(os.path.join(FIBER_DIR, "pwm.p"), "rb"))
    b = np.ravel(pwm["background"])[:4]
    with open(bg, "w") as fh:
        fh.write("# order 0\n")
        for letter, v in zip("ACGT", b):
            fh.write("%s %.8f\n" % (letter, v))

    with open(fa, "w") as fh:
        fh.write(sh(["samtools", "faidx", GENOME,
                     "%s:%d-%d" % (chrm, lo_bed + 1, hi_bed + 1)]))

    with open(tsv, "w") as fh:
        subprocess.run([os.path.join(MEME_ENV, "fimo"),
                        "--motif", "Abf1_murphy", "--bfile", bg,
                        "--thresh", "1", "--text", MEME_FILE, fa],
                       check=True, stdout=fh, stderr=subprocess.DEVNULL)

    rows = {"+": [], "-": []}
    with open(tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ci = {name: i for i, name in enumerate(header)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            st = p[ci["strand"]]
            start = int(p[ci["start"]]) - 1        # 1-based genomic -> bed
            stop = int(p[ci["stop"]]) - 1
            rows[st].append((start, stop, (start + stop) // 2,
                             float(p[ci["score"]]), float(p[ci["p-value"]])))
    out = {}
    for st in ("+", "-"):
        a = sorted(rows[st])
        out[st] = dict(start=np.array([r[0] for r in a]),
                       stop=np.array([r[1] for r in a]),
                       mid=np.array([r[2] for r in a]),
                       score=np.array([r[3] for r in a]),
                       pval=np.array([r[4] for r in a]))
    return out, tsv


def tag(ax, spec):
    label, fg, bg = spec
    ax.text(0.004, 0.955, " %s " % label, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.6, color=fg, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.26", fc=bg, ec=fg, lw=0.8))


LOCUS = {}          # filled by main(); keeps style() free of globals-by-accident


def style(ax, ylabel):
    ax.set_facecolor("white")
    ax.axvspan(LOCUS["m0"], LOCUS["m1"], color=BAND, alpha=0.55, zorder=0)
    ax.axvline(LOCUS["mid"], color=INK, ls="--", lw=1.3, zorder=3)
    ax.set_ylabel(ylabel, fontsize=9.0)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b8b7b2")
    ax.tick_params(length=0, labelsize=8.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrom", default=D_CHRM)
    ap.add_argument("--start", type=int, default=D_M0, help="bed 0-based start of the span to annotate")
    ap.add_argument("--end", type=int, default=D_M1, help="bed 0-based end of that span")
    ap.add_argument("--span-label", default=D_LABEL)
    ap.add_argument("--span-strand", default=D_STRAND)
    ap.add_argument("--half", type=int, default=200, help="bp each side of the span midpoint")
    ap.add_argument("--out", default="site5_decoded_fimo.png")
    ap.add_argument("--workdir", default="site5_fimo")
    args = ap.parse_args()
    CHRM, M0, M1, MSTRAND = args.chrom, args.start, args.end, args.span_strand
    MID, HALF = (M0 + M1) // 2, args.half
    LOCUS.update(m0=M0, m1=M1, mid=MID)
    lo, hi = MID - HALF, MID + HALF
    audit = []

    # ---------------- coordinate assertion, before anything is plotted ----------
    flo, fhi = lo - FIMO_PAD, hi + FIMO_PAD
    fasta = faidx(CHRM, flo, fhi)
    stored = h5_nucleotides(FIBER_DIR, CHRM, flo, fhi)
    if stored is None or fasta != stored:
        raise SystemExit("COORDINATE CHECK FAILED: samtools sequence != h5 nucleotides. "
                         "Refusing to plot mis-registered data.")
    print("coordinate check PASSED: samtools chrI:%d-%d == h5 nucleotides (%d bp)"
          % (flo + 1, fhi + 1, len(fasta)))

    # ---------------- FIMO ----------------
    fimo, tsvpath = run_fimo(CHRM, flo, fhi, args.workdir)
    nsig = sum(int((fimo[s]["pval"] < 1e-4).sum()) for s in ("+", "-"))
    print("FIMO: %d placements (+ strand %d, - strand %d), written to %s"
          % (sum(len(fimo[s]["score"]) for s in "+-"),
             len(fimo["+"]["score"]), len(fimo["-"]["score"]), tsvpath))
    print("FIMO: %d placements with p < 1e-4" % nsig)
    best = max((("+", int(np.argmax(fimo["+"]["score"]))),
                ("-", int(np.argmax(fimo["-"]["score"])))),
               key=lambda t: fimo[t[0]]["score"][t[1]])
    bs, bi = best
    print("FIMO: best overall  %d-%d  strand %s  score %+.4f  p %.3g"
          % (fimo[bs]["start"][bi], fimo[bs]["stop"][bi], bs,
             fimo[bs]["score"][bi], fimo[bs]["pval"][bi]))
    audit.append(("FIMO score / p-value", tsvpath,
                  "fimo --motif Abf1_murphy --bfile <robocop bg> --thresh 1 --text"))

    # ---------------- decoded posteriors ----------------
    posts = {}
    for lbl, od in (("Fiber-seq only", FIBER_DIR),
                    ("Fiber-seq + sequence", SEQ_DIR),
                    ("Sequence only", SEQONLY_DIR)):
        dec = S.load_decode(od)
        comp, cpos, rsum = PD.state_composition(dec, CHRM, lo, hi)
        if comp is None:
            raise SystemExit("no posterior for %s" % od)
        posts[lbl] = (comp, cpos, rsum, od)
        print("%-22s worst state-group row sum %.6f (must be ~1.0)" % (lbl, float(rsum.min())))
        audit.append(("ABF1 fwd/rev posterior (%s)" % lbl,
                      os.path.join(od, "tmpDir/info_*.h5"), "posterior table, state blocks"))

        # independent cross-check against score_robocop's own aggregation
        op, _, _ = S.region_optable(dec, CHRM, lo, hi)
        if "Abf1_murphy" in op.columns:
            mine = comp["abf1_fwd"] + comp["abf1_rev"]
            theirs = np.asarray(op["Abf1_murphy"])
            print("   cross-check vs region_optable Abf1_murphy: max abs diff %.2e"
                  % float(np.max(np.abs(mine - theirs))))

    ft = PL.fiber_track(FIBER_DIR, CHRM, lo - 5, hi + 5)
    audit.append(("Watson / Crick meth per bp",
                  os.path.join(FIBER_DIR, "tmpDir/info_*.h5"),
                  "Fiber_count_{meth,A}_{watson,crick}"))
    audit.append(("annotated span", "--start/--end argument",
                  "%s %d %d  %s  strand %s" % (CHRM, M0, M1, args.span_label, MSTRAND)))

    # ---------------- figure ----------------
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(13.0, 17.6))
    gs = gridspec.GridSpec(7, 1, height_ratios=[0.75, 2.5, 2.2, 2.2, 2.2, 1.15, 1.15],
                           hspace=0.16)
    axes = [fig.add_subplot(gs[0])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in range(1, 7)]
    a_ann, a_fimo, a_fib, a_seq, a_sonly, a_w, a_c = axes

    # -- 1. MacIsaac annotation --
    a_ann.set_facecolor("white")
    a_ann.set_xlim(lo, hi); a_ann.set_ylim(0, 1)
    a_ann.axvline(MID, color=INK, ls="--", lw=1.3, zorder=3)
    a_ann.add_patch(plt.Rectangle((M0, 0.46), M1 - M0, 0.44, color=MUTED,
                                  alpha=0.9, zorder=4))
    a_ann.annotate("", xy=(M0, 0.26), xytext=(M1, 0.26),
                   arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    # label offset to the LEFT of the span: the 14 bp arrow is far too short to
    # hold text between its heads at this x-scale
    a_ann.text(M0 - 5, 0.26, "%d bp " % (M1 - M0), ha="right", va="center",
               fontsize=8.2, color=INK, fontweight="bold")
    a_ann.text(M1 + 4, 0.68,
               "%s   %s:%d-%d   %d bp   %s strand"
               % (args.span_label, CHRM, M0, M1, M1 - M0, MSTRAND),
               va="center", fontsize=9.0, color=MUTED, fontweight="bold")
    for s in a_ann.spines.values():
        s.set_visible(False)
    a_ann.set_yticks([]); a_ann.tick_params(length=0, labelbottom=False)
    a_ann.set_ylabel(args.span_label.split()[0], fontsize=9.0)

    # -- 2. FIMO scan --
    style(a_fimo, "FIMO score\n(bits, log-odds)")
    a_fimo.axhline(0, color=MUTED, lw=1.0, zorder=2)
    # the two strands can peak at the same base (a palindromic motif scores on both),
    # so stagger the direct labels vertically instead of letting them overprint
    for k, (st, col, lab) in enumerate((("+", FWD, "forward (+)"), ("-", REV, "reverse (−)"))):
        m = (fimo[st]["mid"] >= lo) & (fimo[st]["mid"] <= hi)
        a_fimo.plot(fimo[st]["mid"][m], fimo[st]["score"][m], color=col, lw=1.6,
                    zorder=4, label=lab)
        j = int(np.argmax(np.where(m, fimo[st]["score"], -np.inf)))
        a_fimo.plot([fimo[st]["mid"][j]], [fimo[st]["score"][j]], marker="o", ms=7,
                    color=col, markeredgecolor="white", markeredgewidth=1.2, zorder=6)
        a_fimo.annotate("%s best %+.2f (p %.3g)  %d-%d"
                        % (lab, fimo[st]["score"][j], fimo[st]["pval"][j],
                           fimo[st]["start"][j], fimo[st]["stop"][j]),
                        xy=(fimo[st]["mid"][j], fimo[st]["score"][j]),
                        xytext=(10, 4 + 13 * (1 - k)), textcoords="offset points",
                        fontsize=7.8, color=col, fontweight="bold", va="bottom", zorder=7)
    if nsig:
        for st, col in (("+", FWD), ("-", REV)):
            sel = fimo[st]["pval"] < 1e-4
            for s0, s1 in zip(fimo[st]["start"][sel], fimo[st]["stop"][sel]):
                a_fimo.axvspan(s0, s1, color=col, alpha=0.20, zorder=1)
    else:
        a_fimo.text(0.004, 0.035,
                    "span track EMPTY — no placement in this %d bp window reaches "
                    "FIMO's p < 1e-4 (best is p = %.3g). Threshold not loosened."
                    % (fhi - flo + 1,
                       min(fimo["+"]["pval"].min(), fimo["-"]["pval"].min())),
                    transform=a_fimo.transAxes, ha="left", va="bottom", fontsize=8.2,
                    color=RED, fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.32", fc="#fdecec", ec=RED, lw=0.9))
    # legend goes bottom-right: the top of this panel carries the two best-score
    # direct labels and the bottom-left carries the empty-span note
    a_fimo.legend(frameon=False, fontsize=8.4, loc="lower right", ncol=2)
    tag(a_fimo, TAG_EXT)
    a_fimo.tick_params(labelbottom=False)

    # -- 3/4. decoded ABF1 posterior --
    for ax, lbl in ((a_fib, "Fiber-seq only"), (a_seq, "Fiber-seq + sequence"),
                    (a_sonly, "Sequence only")):
        comp, cpos, _, od = posts[lbl]
        style(ax, "ABF1 posterior\n%s" % lbl)
        for key, col, lab in (("abf1_fwd", FWD, "ABF1 forward (states 1–14)"),
                              ("abf1_rev", REV, "ABF1 reverse (states 15–28)")):
            ax.plot(cpos, comp[key], color=col, lw=1.7, zorder=4, label=lab)
            ax.fill_between(cpos, comp[key], color=col, alpha=0.18, lw=0, zorder=3)
            if float(comp[key].max()) > 0.02:
                j = int(np.argmax(comp[key]))
                ax.plot([cpos[j]], [comp[key][j]], marker="o", ms=7, color=col,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=6)
                ax.text(cpos[j], comp[key][j], "  %.3f at %d" % (comp[key][j], cpos[j]),
                        fontsize=7.8, color=col, fontweight="bold", va="bottom", zorder=7)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8.4, loc="upper right", ncol=2)
        ax.text(0.004, 0.06, "%s   (segments averaged where they overlap, as in "
                             "score_robocop.region_optable)" % od,
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4, color=MUTED)
        # If ABF1 is flat at ~0 across the annotated span, name what the decode put
        # there instead -- an empty panel otherwise reads as a broken plot.
        insp = (cpos >= M0) & (cpos <= M1)
        if float((comp["abf1_fwd"] + comp["abf1_rev"])[insp].max()) < 0.05:
            top = max(("background", "nucleosome", "other_TFs", "unknown"),
                      key=lambda k: float(comp[k][insp].max()))
            ax.text(MID, 0.52,
                    "ABF1 ≈ 0 over the annotated span;\ndecode puts %s = %.3f here"
                    % (top.replace("_", " "), float(comp[top][insp].max())),
                    ha="center", va="center", fontsize=8.6, color=MUTED,
                    fontweight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c9c8c4", lw=0.9))
        tag(ax, TAG_DECODE)
        ax.tick_params(labelbottom=False)

    # -- 5/6. raw methylation --
    for ax, col, sel, smk, lab in ((a_w, WATSON, "isA", "smW", "Watson meth/A  (reference A)"),
                                   (a_c, CRICK, "isT", "smC", "Crick meth/A  (reference T)")):
        style(ax, lab.split("  ")[0].replace(" ", "\n"))
        if ft is not None:
            m = ft[sel]
            ax.scatter(ft["pos"][m], ft["frac"][m], s=7, color=col, alpha=0.75, zorder=2)
            ax.plot(ft["pos"], ft[smk], color=col, lw=1.9, zorder=4)
            ax.axhline(BG_METH, color=MUTED, ls="--", lw=1.0, zorder=3)
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0])
        ax.text(0.995, 0.92, "%s  ·  dashed = genome-wide 0.138" % lab,
                transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
                color=col, fontweight="bold")
        tag(ax, TAG_RAW)
    a_w.tick_params(labelbottom=False)
    a_c.set_xlabel("chrI position (bp, bed 0-based — same frame as the other site plots)",
                   fontsize=10.0)
    axes[0].set_xlim(lo, hi)

    fig.suptitle("%s — %s:%d–%d ±%d bp" % (args.span_label, CHRM, M0, M1, HALF),
                 fontsize=14.5, fontweight="bold", y=0.997)
    fig.text(0.5, 0.9825,
             "Posteriors and methylation are read from the decode h5 and nothing else. "
             "The FIMO panel is an external tool — RoboCOP computes no per-position motif "
             "score — scanning with the identical Abf1_murphy matrix.",
             ha="center", va="top", fontsize=8.8, color=MUTED)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.972, bottom=0.045)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nwrote", args.out)

    print("\nPROVENANCE OF EVERY PLOTTED VALUE")
    print("%-42s %-46s %s" % ("quantity", "source", "detail"))
    for q, src, det in audit:
        print("%-42s %-46s %s" % (q, src, det))

    print("\nVALUES AT KEY POSITIONS")
    print("%-9s %10s %10s | %10s %10s | %10s %10s | %8s %8s"
          % ("position", "fib fwd", "fib rev", "f+s fwd", "f+s rev",
             "seqonly f", "seqonly r", "FIMO +", "FIMO -"))
    fibc, fpos = posts["Fiber-seq only"][0], posts["Fiber-seq only"][1]
    seqc = posts["Fiber-seq + sequence"][0]
    sonc = posts["Sequence only"][0]
    for x in sorted({MID, M0, M1}):
        i = int(np.clip(x - fpos[0], 0, len(fpos) - 1))
        def fs(st):
            k = np.argmin(np.abs(fimo[st]["mid"] - x))
            return fimo[st]["score"][k]
        print("%-9d %10.6f %10.6f | %10.6f %10.6f | %10.6f %10.6f | %8.2f %8.2f"
              % (x, fibc["abf1_fwd"][i], fibc["abf1_rev"][i],
                 seqc["abf1_fwd"][i], seqc["abf1_rev"][i],
                 sonc["abf1_fwd"][i], sonc["abf1_rev"][i], fs("+"), fs("-")))


if __name__ == "__main__":
    main()
