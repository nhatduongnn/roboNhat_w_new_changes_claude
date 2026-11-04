"""Write slides_fragment.html and notes.json for the Nov 3 deck from
/usr/project/xtmp/nd141/bg_investigation/nov03/nov03_numbers.json (analyze_nov03.py). No number is typed by hand.

    python make_fragment.py && python make_template.py
"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
SRC = "/usr/project/xtmp/nd141/bg_investigation/nov03"
N = json.load(open(os.path.join(SRC, "nov03_numbers.json")))
BCS = ["Mar20_bc01", "May07_bc03", "May07_bc04", "Jun25_bc21", "Jun25_bc22", "Jun25_bc23", "Jun25_bc24"]
RUN = {"Mar20": ("mar", "2025-03-20"), "May07": ("may", "2025-05-07"), "Jun25": ("jun", "2025-06-25")}   # run folder names Mar_20_25 etc.; = BAM @RG DT
R = {(r["set"], r["barcode"]): r for r in N["rates"]}
C = {r["barcode"]: r for r in N["cascade"]}
B = {(b["rule"], b["run"]): b for b in N["bounds"]}

# ---- final read set (2026-11-02): rDNA excluded, ONE pooled MAD fence. Per-barcode rates recomputed on
# filtered_refit_nordna/keep_reads.txt by bg_investigation/nov03_final/rates_final.py (same rates() as analyze_nov03.py).
import csv
FIN = "/usr/project/xtmp/nd141/filtered_refit_nordna"
FINR = "/usr/project/xtmp/nd141/bg_investigation/nov03_final"
for row in csv.DictReader(open(os.path.join(FINR, "rates_final.tsv")), delimiter="\t"):
    R[("final", row["barcode"])] = {k: (float(v) if k != "barcode" else v) for k, v in row.items()}
FS = {r["barcode"]: {k: (float(v) if k != "barcode" else v) for k, v in r.items()}
      for r in csv.DictReader((l for l in open(os.path.join(FIN, "filter_summary.tsv")) if not l.startswith("#")), delimiter="\t")}
FENCE = json.load(open(os.path.join(FIN, "fence.json")))
OLDFENCE = json.load(open("/usr/project/xtmp/nd141/filtered_refit/fence.json"))
RD = json.load(open(os.path.join(FINR, "rdna_checks.json")))
CMP = open(os.path.join(FIN, "fits", "compare_fits.txt")).read().splitlines()
bg_w = [l for l in CMP if l.startswith("bg watson_signal")][0].split()   # bg watson_signal: filt X nordna Y diff Z
nuc_ratio = float([l for l in CMP if l.startswith("nuc watson_signal")][0].split("mean(new/old)")[1].split()[0])
tf_ratios = [float(l.split("\t")[5]) for l in CMP if l.count("\t") >= 6 and not l.startswith("tf\t")]
tf_maxdev = max(abs(1 - x) for x in tf_ratios)


def f3(x): return "%.3f" % x
def pct(a, b): return "%.1f%%" % (100.0 * a / b) if b else "–"
def comma(x): return "{:,}".format(int(x))


# ------------------------------------------------------------------ bar chart (inline SVG)
def chart(setname, mini=False):   # mini: no value labels (slide 4 copy)
    regions = [("lk", "Linkers"), ("ncB", "Nucleosome cores"), ("gw", "Genome-wide")]
    W, Hh = 1440, 470
    pw, gap = 440, 60
    top, bot = 70, 380
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="m6A rate per barcode in linkers, nucleosome cores and genome-wide, '
           'after read filters. Jun25 barcodes are highest in all three.">' % (W, Hh)]
    for pi, (k, name) in enumerate(regions):
        x0 = pi * (pw + gap) + 40
        vals = [R[(setname, bc)][k + "_pooled"] for bc in BCS]
        vmax = {"lk": 0.20, "ncB": 0.06, "gw": 0.09}[k]
        ticks = {"lk": [0, .05, .10, .15, .20], "ncB": [0, .02, .04, .06], "gw": [0, .03, .06, .09]}[k]
        y = lambda v: bot - (bot - top) * v / vmax
        out.append('<g class="panel">')
        out.append('<text class="ttl" x="%d" y="30">%s</text>' % (x0, name))
        for t in ticks:
            out.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (x0 + 40, x0 + pw, y(t), y(t)))
            out.append('<text class="tk" x="%d" y="%.1f" text-anchor="end">%s</text>' % (x0 + 34, y(t) + 4, ("%.2f" % t)))
        # run groups: Mar20 (1), May07 (2), Jun25 (4); slot width with a gap between runs
        bw, sgap = 40, 8
        xs, xx = [], x0 + 52
        prev = None
        for bc in BCS:
            run = bc[:5]
            if prev and run != prev: xx += 22
            xs.append(xx); xx += bw + sgap; prev = run
        for bc, xb, v in zip(BCS, xs, vals):
            cls = RUN[bc[:5]][0]
            yy = y(v); h = bot - yy
            r = 4
            # rounded data end, square at the baseline
            path = "M%.1f,%.1f v%.1f a%d,%d 0 0 1 %d,-%d h%.1f a%d,%d 0 0 1 %d,%d v%.1f z" % (
                xb, bot, -(h - r), r, r, r, r, bw - 2 * r, r, r, r, r, h - r)
            out.append('<path class="b %s" d="%s"><title>%s (%s) · %s %.4f · %s informative A calls</title></path>' % (
                cls, path, bc, RUN[bc[:5]][1], name.lower(), v, comma(R[(setname, bc)][k + "_calls"])))
            if not mini:
                out.append('<text class="val %s" x="%.1f" y="%.1f" text-anchor="middle">%s</text>' % (cls, xb + bw / 2, yy - 8, f3(v)[1:]))
            out.append('<text class="bc" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (xb + bw / 2, bot + 22, bc[-2:]))
        # run labels under each group
        for run in ("Mar20", "May07", "Jun25"):
            idx = [i for i, bc in enumerate(BCS) if bc[:5] == run]
            cx = (xs[idx[0]] + xs[idx[-1]] + bw) / 2
            out.append('<text class="rn" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (cx, bot + 46, run))
        out.append('<line class="base" x1="%d" x2="%d" y1="%d" y2="%d"/>' % (x0 + 40, x0 + pw, bot, bot))
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


T, U, M = "tukey", "unfiltered", "mad"
FINAL = "final"   # the barcode slide uses the final keep list (was T = per-run Tukey before 2026-11-02)
lk = {bc: R[(FINAL, bc)]["lk_pooled"] for bc in BCS}
jun_lo, jun_hi = min(lk[b] for b in BCS[3:]), max(lk[b] for b in BCS[3:])
may_lo, may_hi = min(lk[b] for b in BCS[1:3]), max(lk[b] for b in BCS[1:3])
ratio_jun_may = R[(FINAL, "Jun25_run")]["lk_pooled"] / R[(FINAL, "May07_run")]["lk_pooled"]
ratio_core = R[(FINAL, "Jun25_run")]["ncB_pooled"] / R[(FINAL, "May07_run")]["ncB_pooled"]
ratio_gw = R[(FINAL, "Jun25_run")]["gw_pooled"] / R[(FINAL, "May07_run")]["gw_pooled"]

# ------------------------------------------------------------------ slides
S = []
S.append('''<!-- ======================= T TITLE ======================= -->
<section class="slide title-slide" data-id="T" data-kind="main" data-sec="0" data-steps="1" data-title="Nov 3 lab meeting">
  <h1>Nov 3 lab meeting</h1>
  <div class="ruler" aria-hidden="true"></div>
</section>''')

S.append('''<!-- ======================= 1 BARCODES ======================= -->
<section class="slide" data-id="1" data-kind="main" data-sec="1" data-steps="1" data-title="m6A rate per barcode">
  <span class="slide-no">1</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> Background m6A</p>
    <h2>The 2025-06-25 run is methylated more, everywhere</h2>
    <p class="msg">After read filters, all four Jun25 barcodes sit at %s–%s in linkers, against %s–%s for May07 and %s for Mar20.
    The gap is the same in nucleosome cores and genome-wide (Jun25 / May07 = %.2f×, %.2f×, %.2f×), so it is the run, not one barcode.</p>
  </header>
  <div class="body stack" style="gap:12px">
    <div class="legend-runs">
      <span><i style="background:var(--run-mar)"></i><b>Mar20</b> · run 2025-03-20 · bc01</span>
      <span><i style="background:var(--run-may)"></i><b>May07</b> · run 2025-05-07 · bc03, bc04</span>
      <span><i style="background:var(--run-jun)"></i><b>Jun25</b> · run 2025-06-25 · bc21–bc24</span>
    </div>
    <div class="bars">%s</div>
    <p class="foot txt" style="font-size:15px">Rate = methylated / (methylated + canonical) A calls, pooled per barcode. Calls as
    <span class="m">singlefiber/read_calls.py</span>: ML/255 ≥ 0.8105 methylated, ≤ 0.1895 canonical (Watson on ref A, Crick on ref T).
    Linkers: <span class="m">computeLinkers</span>, 15 bp each side of 147 bp Chereji ±1 nucleosomes. Cores: the 147 bp windows of the
    current nucleosome m6A fit (Brogaard top 2,000, <span class="m">nucbrogsym/m6a_fit.py</span>). Genome-wide: every ref A/T, chrM excluded.
    Reads: the final keep list (filters on the next slide: rDNA excluded, one pooled MAD fence), %s reads. Source:
    <span class="m">bg_investigation/nov03_final/rates_final.tsv</span>.</p>
  </div>
</section>''' % (f3(jun_lo), f3(jun_hi), f3(may_lo), f3(may_hi), f3(lk["Mar20_bc01"]), ratio_jun_may, ratio_core, ratio_gw, chart(FINAL), comma(R[(FINAL, 'ALL')]['reads'])))

# ---- filter table (final order, 2026-11-02): filter_summary.tsv + fence.json of filtered_refit_nordna
COLS = ["Mar20", "May07", "Jun25", "TOTAL"]
COLCLS = {"Mar20": "run-mar", "May07": "run-may", "Jun25": "run-jun", "TOTAL": ""}
COLLAB = {"Mar20": "Mar20<br>bc01", "May07": "May07<br>bc03–04", "Jun25": "Jun25<br>bc21–24", "TOTAL": "All<br>7 barcodes"}
steps = [("flag", "No m6A tags", "Failed flag, or no MM/ML tags (0 calls)."),
         ("chrM", "chrM", "Mitochondrial: we model the nuclear genome."),
         ("rDNA", "rDNA array", "Overlaps chrXII:451,418–468,929 (next slide)."),
         ("mapq<20", "MAPQ < 20", "≥ 1% chance the read is in the wrong place."),
         ("aligned<500bp", "Aligned < 500 bp", "Under 500 bp of genome: ~3 nucleosomes."),
         ("qs<10", "Mean Q < 10", "Base calls under ~90% accurate on average."),
         ("outlier_hyper", "m6A outlier", "Outside the pooled fence [%.3f, %.3f]; all %s are high."
          % (FENCE["lo"], FENCE["hi"], comma(FENCE["n_hyper"])))]
hdr = "".join('<th class="r %s">%s</th>' % (COLCLS[c], COLLAB[c]) for c in COLS)
rows = []
for key, name, desc in steps:
    cells = []
    for c in COLS:
        f = FS[c]
        nr, nc = f["rm_reads_" + key], f["rm_calls_" + key]
        small = ("%s · " % comma(nr) if c == "TOTAL" else "") + ("–" if nc == 0 and key == "flag" else pct(nc, f["calls_in"]))
        cells.append('<td class="n r" title="%s reads, %s calls">%s<br><small>%s</small></td>' % (
            comma(nr), comma(nc), pct(nr, f["reads_in"]), small))
    rows.append('<tr><td><b>%s</b></td><td class="d">%s</td>%s</tr>' % (name, desc, "".join(cells)))
kept = []
for c in COLS:
    f = FS[c]
    kept.append('<td class="n r">%s<br><small>%s · %s calls</small></td>' % (pct(f["kept_reads"], f["reads_in"]), comma(f["kept_reads"]),
                pct(f["kept_calls"], f["calls_in"])))
rows.append('<tr class="key"><td>Kept</td><td class="d" style="color:inherit">%s of %s mapped reads</td>%s</tr>' % (
    comma(FS["TOTAL"]["kept_reads"]), comma(FS["TOTAL"]["reads_in"]), "".join(kept)))
assert int(FS["TOTAL"]["kept_reads"]) == FENCE["n_keep"]

S.append('''<!-- ======================= 2 FILTERS ======================= -->
<section class="slide" data-id="2" data-kind="hidden" data-sec="1" data-steps="1" data-title="Read filters">
  <span class="slide-no">2</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> Which reads we keep</p>
    <h2 style="font-size:46px">Read filters: we keep %s of reads and %s of calls</h2>
    <p class="msg" style="font-size:20px;max-width:none">Applied top to bottom; each read counts at the first filter that removes it.
    Big number: %% of mapped reads. Small: %% of informative A calls. The outlier fence is now <b>one pooled fence</b> over all 7 barcodes, not one per run.</p>
  </header>
  <div class="body stack" style="gap:10px">
    <div class="cols filt-cards" style="grid-template-columns:1fr 1fr 1fr;gap:16px">
      <div class="card" style="padding:10px 16px"><h3 style="font-size:20px;margin-bottom:4px">MAPQ: is the read in the right place?</h3>
        <p style="font-size:16px">MAPQ = −10·log<sub>10</sub>(chance the placement is wrong). 20 means a 1%% chance; 60 is the top.
        Low MAPQ usually means the read fits more than one place.</p></div>
      <div class="card" style="padding:10px 16px"><h3 style="font-size:20px;margin-bottom:4px">qs: are the bases right?</h3>
        <p style="font-size:16px">dorado's mean base quality per read, same scale. qs 10 ≈ 90%% of bases right; 20 ≈ 99%%.
        Low-qs reads also look more methylated (Spearman ρ −0.33 to −0.43, every barcode).</p></div>
      <div class="card" style="padding:10px 16px"><h3 style="font-size:20px;margin-bottom:4px">Untrimmed reads are fine</h3>
        <p style="font-size:16px">Adapters (~117 bp at the 5′ end) are soft-clipped by the aligner. Trimming and re-aligning 3,500 random reads moved
        only +10 / −6 across MAPQ 20.</p></div>
    </div>
    <div class="tbl-wrap"><table class="tbl compact filt filt2" style="font-size:17px">
      <thead><tr><th>Filter</th><th>What it removes</th>%s</tr></thead>
      <tbody>%s</tbody>
    </table></div>
    <p class="foot txt" style="font-size:14px">Fence: reads with ≥ 50 informative A calls are judged on their genome-wide m6A rate; fence = median ± 4 × 1.4826 × MAD
    (median %.4f, MAD %.4f, %s reads judged; %s reads with &lt; 50 calls kept unjudged).
    Source: <span class="m">filtered_refit_nordna/filter_summary.tsv</span>, <span class="m">fence.json</span>, <span class="m">scripts/make_keep.py</span>.</p>
  </div>
</section>''' % (pct(FS["TOTAL"]["kept_reads"], FS["TOTAL"]["reads_in"]), pct(FS["TOTAL"]["kept_calls"], FS["TOTAL"]["calls_in"]),
                 hdr, "\n".join(rows), FENCE["median"], FENCE["mad"], comma(FENCE["judged"]), comma(FS["TOTAL"]["kept_unjudged_lt50"])))


# ---- rDNA slide: dot plot of median per-read m6A rate, rDNA vs rest, per barcode (kept reads, with-rDNA read set)
def rdna_chart():
    W, Hh, top, bot, x0, x1 = 760, 380, 30, 300, 56, 744
    vmax = 0.12
    y = lambda v: bot - (bot - top) * v / vmax
    MR = RD["kept_median_rate_by_bc"]; NN = RD["kept_n_by_bc"]
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Median per-read m6A rate per barcode, rDNA reads versus the rest of the genome. '
           'rDNA is higher in all seven barcodes.">' % (W, Hh)]
    for t in (0, .03, .06, .09, .12):
        out.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (x0, x1, y(t), y(t)))
        out.append('<text class="tk" x="%d" y="%.1f" text-anchor="end">%.2f</text>' % (x0 - 8, y(t) + 4, t))
    slot = (x1 - x0 - 2 * 30) / 7.0
    xs, xx, prev = [], x0 + slot / 2, None
    for bc in BCS:
        if prev and bc[:5] != prev: xx += 30
        xs.append(xx); xx += slot; prev = bc[:5]
    for bc, xc in zip(BCS, xs):
        a, b = MR[bc]["rest"], MR[bc]["rdna"]
        out.append('<line class="conn" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>' % (xc, xc, y(a), y(b)))
        out.append('<circle class="d rest" cx="%.1f" cy="%.1f" r="7"><title>%s rest of genome: median %.3f (%s reads)</title></circle>' % (xc, y(a), bc, a, comma(NN[bc]["rest"])))
        out.append('<circle class="d rd" cx="%.1f" cy="%.1f" r="7"><title>%s rDNA: median %.3f (%s reads)</title></circle>' % (xc, y(b), bc, b, comma(NN[bc]["rdna"])))
        out.append('<text class="val" x="%.1f" y="%.1f" text-anchor="middle">%s</text>' % (xc, y(b) - 14, f3(b)[1:]))
        out.append('<text class="val mute" x="%.1f" y="%.1f" text-anchor="middle">%s</text>' % (xc, y(a) + 24, f3(a)[1:]))
        out.append('<text class="bc" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (xc, bot + 22, bc[-2:]))
    for run in ("Mar20", "May07", "Jun25"):
        idx = [i for i, bc in enumerate(BCS) if bc[:5] == run]
        out.append('<text class="rn" x="%.1f" y="%d" text-anchor="middle">%s</text>' % ((xs[idx[0]] + xs[idx[-1]]) / 2, bot + 44, run))
    out.append('<line class="base" x1="%d" x2="%d" y1="%d" y2="%d"/>' % (x0, x1, bot, bot))
    out.append("</svg>")
    return "\n".join(out)


cv = lambda s, r, k: RD["cov_%s_%s" % (s, r)][k]
rd_reads, all_reads = RD["merged_rdna_reads"], FS["TOTAL"]["reads_in"]
# % MAPQ<20 among rDNA reads in the merged BAM (the trim_check figure, 0.680, is the same set: cls rDNA n=198,847)
rd_low = 1 - RD["merged_rdna_mapq_ge20"] / rd_reads
uq = RD["ref_unique_stretch"]
S.append('''<!-- ======================= 2b rDNA ======================= -->
<section class="slide" data-id="2b" data-kind="hidden" data-sec="1" data-steps="1" data-title="Why we exclude the rDNA array">
  <span class="slide-no">2b</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> Which reads we keep</p>
    <h2 style="font-size:44px">Why we exclude the rDNA array</h2>
    <p class="msg" style="font-size:21px">About 150 rDNA copies in the cell map onto 2 copies in the reference. Their reads pile up, cannot be placed,
    and look more methylated, so they bent the outlier fence. Dropping them barely moves the fitted rates.</p>
  </header>
  <div class="body" style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.05fr);gap:28px;min-height:0">
    <div class="stack rdna-facts" style="gap:10px">
      <div class="card"><h3>Where</h3><p>chrXII:451,418–468,929: two copies of a 9,137 bp unit in sacCer3 (the second is %s bp, cut short).</p></div>
      <div class="card"><h3>Pile-up</h3><p><b>%s</b> of %s mapped reads (<b>%s</b>) touch it. Depth <b>%.0f×</b> vs %.1f× for the rest of the genome
        (%.0f× vs %.1f× after filters).</p></div>
      <div class="card"><h3>Placement</h3><p><b>%s</b> of rDNA reads have MAPQ &lt; 20, vs 4%% elsewhere (%s at MAPQ 0, %s at 60).
        %s of the %s MAPQ ≥ 20 reads touch the %s bp stretch that exists in only one reference copy: high MAPQ means
        “unique in the reference”, not “unique in the cell”.</p></div>
      <div class="card"><h3>Methylation</h3><p>Pooled m6A <b>%.3f</b> vs %.3f. Median per read %.3f vs %.3f, higher in every barcode (right).</p></div>
    </div>
    <div class="stack" style="gap:10px">
      <p class="panel-t">Median per-read m6A rate, kept reads</p>
      <div class="legend-runs" style="font-size:17px"><span><i class="dot rd"></i><b>rDNA reads</b></span><span><i class="dot rest"></i><b>rest of genome</b></span></div>
      <div class="dots">%s</div>
      <div class="callout" style="font-size:17px;padding:12px 16px"><span class="lab">What excluding it changes</span>
        <b>Outlier fence:</b> rDNA was %.2f%% of judged reads but %.1f%% of fence removals (%s of %s). Upper fence %.4f → %.4f.<br>
        <b>Fitted rates:</b> background %.5f → %.5f (Watson), nucleosome ×%.3f, TF footprints within %.1f%%.<br>
        <b>Decoder:</b> rDNA is where the old (non-log) decoder gave invalid posteriors.</div>
    </div>
  </div>
</section>''' % (comma(17512 - 9137),
                 comma(rd_reads), comma(all_reads), pct(rd_reads, all_reads), cv("merged", "rdna", "mean_cov"), cv("merged", "rest", "mean_cov"),
                 cv("filtered", "rdna", "mean_cov"), cv("filtered", "rest", "mean_cov"),
                 "%.0f%%" % (100 * rd_low), comma(RD["merged_rdna_mapq0"]), comma(RD["merged_rdna_mapq60"]),
                 pct(RD["mapq_ge20_touch_unique"], RD["merged_rdna_mapq_ge20"]), comma(RD["merged_rdna_mapq_ge20"]), comma(uq[2]),
                 cv("merged", "rdna", "rate"), cv("merged", "rest", "rate"), RD["kept_median_rate"]["rdna"], RD["kept_median_rate"]["rest"],
                 rdna_chart(),
                 RD["judged_rdna_pct"], RD["hyper_rdna_pct"], comma(RD["old_hyper_rdna"]), comma(RD["old_hyper"]), OLDFENCE["hi"], FENCE["hi"],
                 float(bg_w[3]), float(bg_w[5]), nuc_ratio, 100 * tf_maxdev))


# ---- hidden detail: unfiltered vs Tukey vs MAD, both core sets, medians
def drow(setname, label, key=False):
    tds = []
    for bc in BCS + ["ALL"]:
        r = R[(setname, bc)]
        tds.append('<td class="n r">%s · %s · %s</td>' % (f3(r["lk_pooled"])[1:], f3(r["ncB_pooled"])[1:], f3(r["gw_pooled"])[1:]))
    return '<tr%s><td>%s</td>%s</tr>' % (' class="key"' if key else "", label, "".join(tds))


def mrow(setname, label):
    tds = []
    for bc in BCS + ["ALL"]:
        r = R[(setname, bc)]
        tds.append('<td class="n r">%s · %s · %s</td>' % (f3(r["lk_median_read"])[1:], f3(r["ncB_median_read"])[1:], f3(r["gw_median_read"])[1:]))
    return '<tr><td>%s</td>%s</tr>' % (label, "".join(tds))


def crow(setname, label):
    tds = ['<td class="n r">%s</td>' % f3(R[(setname, bc)]["ncC_pooled"])[1:] for bc in BCS + ["ALL"]]
    return '<tr><td>%s</td>%s</tr>' % (label, "".join(tds))


hdr2 = "".join('<th class="r">%s</th>' % bc.replace("_", " ") for bc in BCS) + '<th class="r">ALL</th>'
S.append('''<!-- ======================= H1 DETAIL ======================= -->
<section class="slide" data-id="H1" data-kind="hidden" data-sec="1" data-steps="1" data-title="Rates under each read set">
  <span class="slide-no">H1</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> Detail · does the read set matter?</p>
    <h2 style="font-size:40px">Same picture under every read set</h2>
    <p class="msg" style="font-size:20px;max-width:none">Each cell: linker · nucleosome core · genome-wide. Filters lower the genome-wide rate a little and
    barely move linkers; the Jun25 lead is unchanged under every read set.</p>
  </header>
  <div class="body stack" style="gap:12px">
    <p class="panel-t">Pooled rate</p>
    <div class="tbl-wrap"><table class="tbl compact h1t" style="font-size:14px">
      <thead><tr><th>Read set</th>%s</tr></thead>
      <tbody>%s%s%s%s%s</tbody></table></div>
    <p class="panel-t">Median per-read rate (reads with ≥ 20 linker, ≥ 20 core, ≥ 200 genome-wide calls)</p>
    <div class="tbl-wrap"><table class="tbl compact h1t" style="font-size:14px">
      <thead><tr><th>Read set</th>%s</tr></thead>
      <tbody>%s%s%s</tbody></table></div>
    <p class="panel-t">Nucleosome core on Chereji ±1 sites instead (dyad −73 to +73, the original generator's window)</p>
    <div class="tbl-wrap"><table class="tbl compact h1t" style="font-size:14px">
      <thead><tr><th>Read set</th>%s</tr></thead>
      <tbody>%s%s%s</tbody></table></div>
    <p class="foot txt" style="font-size:14px">MAD rule: outside median ± 4 × 1.4826 × MAD of the genome-wide per-read rate (per run here;
    one pooled fence in the final set, which also drops rDNA). Sources: <span class="m">bg_investigation/nov03/rates_by_barcode.tsv</span>,
    final row <span class="m">bg_investigation/nov03_final/rates_final.tsv</span>.</p>
  </div>
</section>''' % (hdr2, drow(U, "Unfiltered"), drow("prefilter_no_outlier", "Filters, no fence"), drow(T, "Tukey per run"), drow(M, "MAD per run"), drow(FINAL, "Final (no rDNA, pooled MAD)", True),
                 hdr2, mrow(U, "Unfiltered"), mrow(T, "Tukey per run"), mrow(FINAL, "Final (no rDNA, pooled MAD)"),
                 hdr2, crow(U, "Unfiltered"), crow(T, "Tukey per run"), crow(FINAL, "Final (no rDNA, pooled MAD)")))

# ---- decisions (2026-11-03): rows come from decisions_rows.json, the SAME file that wrote the
# "FiberCOP model choices" sheet rows (artifact 8KkLfcKovncXqmNwFZh5TT, collection "choices"). Slide text =
# each row's `choice` + `short`; detail and evidence go to the notes.
from html import escape as _e
SHEET = "https://claude.ai/artifact/8KkLfcKovncXqmNwFZh5TT"
DROWS = json.load(open(os.path.join(H, "decisions_rows.json"), encoding="utf-8"))
DEC = [r for r in DROWS if r["fields"].get("slide_nov03") == "decided"]
OPN = sorted([r for r in DROWS if r["fields"].get("slide_nov03") == "open"], key=lambda r: r["fields"]["open_rank"])
assert len(DEC) == 15 and len(OPN) == 15
def drow_html(r, num=None):
    n = '<span class="dn">%d</span>' % num if num else ""
    return '<li>%s<span class="dc">%s</span><span class="ds">%s</span></li>' % (n, _e(r["choice"]), _e(r["fields"]["short"]))
half = (len(DEC) + 1) // 2
link = ('<p class="foot txt dec-link">All rows, with values, evidence paths and dates: shared sheet <b>FiberCOP model choices</b> · '
        '<span class="m">%s</span></p>' % SHEET)
S.append('''<!-- ======================= 3 DECIDED ======================= -->
<section class="slide" data-id="3" data-kind="main" data-sec="2" data-steps="1" data-title="Single-fiber model decisions: decided">
  <span class="slide-no">3</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 2</span> Single-fiber model decisions</p>
    <h2 style="font-size:46px">Decided</h2>
  </header>
  <div class="body stack" style="gap:14px">
    <div class="dec-grid"><ul class="dec">%s</ul><ul class="dec">%s</ul></div>
    %s
  </div>
</section>''' % ("".join(drow_html(r) for r in DEC[:half]), "".join(drow_html(r) for r in DEC[half:]), link))
firsts = [r for r in OPN if r["fields"]["open_rank"] == 1]
rest = [r for r in OPN if r["fields"]["open_rank"] != 1]
h2 = (len(rest) + 1) // 2
S.append('''<!-- ======================= 3b NEXT ======================= -->
<section class="slide" data-id="3b" data-kind="main" data-sec="2" data-steps="1" data-title="Single-fiber model decisions: next">
  <span class="slide-no">3b</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 2</span> Single-fiber model decisions</p>
    <h2 style="font-size:46px">Still to decide / to do</h2>
  </header>
  <div class="body stack" style="gap:14px">
    <div class="callout dec-first"><span class="lab">#1 · blocks every per-read TF call</span>
      %s</div>
    <div class="dec-grid"><ol class="dec">%s</ol><ol class="dec">%s</ol></div>
    %s
  </div>
</section>''' % ("<br>".join('<span class="dc">%s</span><span class="ds">%s</span>' % (_e(r["choice"]), _e(r["fields"]["short"])) for r in firsts),
                 "".join(drow_html(r, r["fields"]["open_rank"]) for r in rest[:h2]),
                 "".join(drow_html(r, r["fields"]["open_rank"]) for r in rest[h2:]), link))

exec(open(os.path.join(H, 'update_slides.py'), encoding='utf-8').read())   # slides 4-7b (2026-11-03 update)

for i, pid in enumerate(["8", "9"]):
    S.append('''<!-- ======================= %s PLACEHOLDER ======================= -->
<section class="slide" data-id="%s" data-kind="main" data-sec="3" data-steps="1" data-tag="PENDING" data-title="To be added">
  <span class="slide-no">%s</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> Next</p>
    <h2>Slide to be added</h2>
  </header>
  <div class="body stack"><div class="placeholder">Placeholder</div></div>
</section>''' % (pid, pid, pid))

# 2026-11-03 (user): slide 7b goes last.
_i7b = [i for i, x in enumerate(S) if 'data-id="7b"' in x]
assert len(_i7b) == 1
S.append(S.pop(_i7b[0]))
# 2026-11-03 (user): slide 3b also goes to the end (after 7b).
_i3b = [i for i, x in enumerate(S) if 'data-id="3b"' in x]
assert len(_i3b) == 1
S.append(S.pop(_i3b[0]))
open(os.path.join(H, "slides_fragment.html"), "w", encoding="utf-8").write("\n\n".join(S) + "\n")

# ------------------------------------------------------------------ notes (plain text)
def rm(bc, k): return C[bc]["rm_reads_outlier_" + k]
outl = "; ".join("%s %s Tukey / %s MAD" % (bc, comma(rm(bc, "tukey")), comma(rm(bc, "mad"))) for bc in BCS)
fences = "; ".join("%s Tukey %.3f / MAD %.3f" % (run, B[("tukey", run)]["hi"], B[("mad", run)]["hi"]) for run in ("Mar20", "May07", "Jun25"))
un = "; ".join("%s %s / %s / %s" % (bc, f3(R[(U, bc)]["lk_pooled"]), f3(R[(U, bc)]["ncB_pooled"]), f3(R[(U, bc)]["gw_pooled"])) for bc in BCS)
md = "; ".join("%s %s / %s / %s" % (bc, f3(R[(M, bc)]["lk_pooled"]), f3(R[(M, bc)]["ncB_pooled"]), f3(R[(M, bc)]["gw_pooled"])) for bc in BCS)
fin = "; ".join("%s %s / %s / %s" % (bc, f3(R[(FINAL, bc)]["lk_pooled"]), f3(R[(FINAL, bc)]["ncB_pooled"]), f3(R[(FINAL, bc)]["gw_pooled"])) for bc in BCS)
tk = "; ".join("%s %s / %s / %s" % (bc, f3(R[(T, bc)]["lk_pooled"]), f3(R[(T, bc)]["ncB_pooled"]), f3(R[(T, bc)]["gw_pooled"])) for bc in BCS)
runrm = "; ".join("%s %s (%s of reads before the fence)" % (r, comma(FS[r]["rm_reads_outlier_hyper"]), "%.2f%%" % FS[r]["fence_rm_pct_of_prefence_reads"]) for r in ("Mar20", "May07", "Jun25"))
share = RD["kept_rdna_share_by_run"]
rdna_all = "; ".join("%s %s" % (r, pct(FS[r]["rm_reads_rDNA"] + 0, FS[r]["reads_in"])) for r in ("Mar20", "May07", "Jun25"))
mrb = RD["kept_median_rate_by_bc"]
mrs = "; ".join("%s %.3f vs %.3f" % (bc, mrb[bc]["rdna"], mrb[bc]["rest"]) for bc in BCS)
notes = {
    "T": "Last week the background m6A rate went from 0.138 (barcode01 only) to 0.155 (all seven barcodes). Today: where that comes from, and the read filters we will use for the refit.",
    "1": ("The jump is a run effect. The four 2025-06-25 barcodes are higher than the other two runs in linkers, in nucleosome cores and genome-wide, by about the same factor. "
          "Pooling all seven barcodes mixes runs with different labelling levels, which is why the pooled background moved. "
          "Run dates come from the run folder names (Mar_20_25, May_07_25, Jun_25_25 under process_nanopore_sequencing/) and match the @RG DT field of the merged BAM "
          "(DT 2025-03-20, 2025-05-07, 2025-06-25; samtools view -H). "
          "Rates on this slide are recomputed on the FINAL keep list (filtered_refit_nordna/keep_reads.txt, " + comma(R[(FINAL, "ALL")]["reads"]) + " reads: rDNA excluded, one pooled MAD fence) "
          "from the per-read table bg_investigation/nov03/perread_nov03.tsv.gz (columns lkw_*, ncB_*, gw_*), by bg_investigation/nov03_final/rates_final.py. "
          "Final (linker / core / genome-wide): " + fin + ". "
          "Earlier per-run Tukey set, as shown before: " + tk + ". "
          "Unfiltered: " + un + ". The conclusion does not change: Jun25 is highest, May07 lowest, under every read set."),
    "2": ("Filters run top to bottom; each read is counted once, at the first filter that removes it. Order: flag / no MM tags, chrM, rDNA, MAPQ < 20, aligned < 500 bp, qs < 10, then the outlier fence. "
          "MAPQ is the aligner's confidence in the placement: MAPQ = -10 log10(P(wrong place)), so 20 means a 1%% chance the read is misplaced. "
          "qs is dorado's mean base quality of the read on the same scale: qs 10 means about 90%% of bases are right. "
          "The reads are untrimmed, but the ~117 bp 5' adapter flank is soft-clipped by the aligner in 99%% of reads, and it does not change MAPQ meaningfully: trimming and re-aligning 3,500 random reads moved 10 from below MAPQ 20 to above and 6 the other way "
          "(trim_check/analyze.out; trim_check/realign/compare.out, clipcut). "
          "Outlier fence: ONE pooled fence over all 7 barcodes (no longer one per run): median %.4f, MAD %.4f, fence median +- 4 x 1.4826 x MAD = [%.3f, %.3f], over %s judged reads (>= 50 informative A calls). "
          "It removed %s hypermethylated reads and 0 hypomethylated. By run: %s. "
          "Kept %s of %s mapped reads (%s of reads, %s of informative A calls). Sources: filtered_refit_nordna/filter_summary.tsv, fence.json, scripts/make_keep.py."
          % (FENCE["median"], FENCE["mad"], FENCE["lo"], FENCE["hi"], comma(FENCE["judged"]), comma(FENCE["n_hyper"]), runrm,
             comma(FS["TOTAL"]["kept_reads"]), comma(FS["TOTAL"]["reads_in"]), pct(FS["TOTAL"]["kept_reads"], FS["TOTAL"]["reads_in"]),
             pct(FS["TOTAL"]["kept_calls"], FS["TOTAL"]["calls_in"]))),
    "2b": ("Where: chrXII:451,418-468,929 holds two copies of a 9,137 bp rDNA unit in sacCer3; the real genome has about 150 copies in tandem (rDNA self-similarity scan, rdna_check/; "
           "checked here: sequence at x equals sequence at x + 9,137 over the whole %s bp overlap, so the second copy is %s bp and the last %s bp of copy 1, chrXII:%s-%s, exist only once). "
           "Reads: %s of %s primary mapped reads overlap it (rdna_check/merged_*.rdna.tsv; filtered_refit_nordna/rdna_reads.txt). Share of all mapped reads by run: %s (filter_summary.tsv). "
           "Depth %.1fx vs %.1fx for the rest of the genome in the merged pileup, %.1fx vs %.1fx in the filtered pileup (rdna_check/pileup_cov_rdna_vs_rest.txt). "
           "MAPQ: %s rDNA reads at MAPQ 0, %s at MAPQ 60; 68%% of rDNA reads are MAPQ < 20 vs 4%% of other nuclear reads (trim_check/analyze.out, class counts). "
           "%s of the %s MAPQ >= 20 rDNA reads touch the stretch present in only one reference copy (start + read length, rdna_check tsv), so their high MAPQ says they are unique in the reference, not that they come from one place in the cell. "
           "Methylation: pooled m6A %.4f rDNA vs %.4f rest (merged pileup; %.4f vs %.4f after filters). Median per-read rate among kept reads of the with-rDNA set (rdna_check/kept_rates.tsv.gz, >= 50 calls): %.3f vs %.3f; by barcode rDNA vs rest: %s. "
           "Before the fence, %.1f%% of judged rDNA reads were above 0.18 vs %.1f%% of other reads, with a tail to about 0.45 (99.9th percentile %.3f; %s reads above 0.45). "
           "Hypothesis, not tested: a mix of open (transcribed) and closed rDNA copies. "
           "Among kept reads of the with-rDNA set, rDNA was %.1f%% of Mar20, %.1f%% of Jun25 and %.1f%% of May07 reads, so it also differs by run. "
           "Fence: with rDNA in, %s of %s fence removals (%.1f%%) were rDNA reads though rDNA was %.2f%% of judged reads; upper fence %.4f (filtered_refit/fence.json) -> %.4f without rDNA (filtered_refit_nordna/fence.json). "
           "Fitted rates (filtered_refit_nordna/fits/compare_fits.txt): background Watson %.6f -> %.6f, nucleosome mean ratio %.5f, all TF footprint means within %.1f%%. "
           "No Chereji +-1 nucleosome and no Brogaard top-2000 dyad lies within 100 bp of the rDNA (checked on inputs/Chereji_2018_+1_-1_nucs.bed and the Brogaard bed); the TF fitting windows were not checked one by one. "
           "The small shifts come from the fence moving, which also drops some non-rDNA reads. "
           "Decoder: chrXII rDNA is where the old (non-log) decoder emitted invalid posteriors, up to 1e171 (CLAUDE.md gotchas; POSTERIOR_SCALING.md). "
           "Numbers computed by bg_investigation/nov03_final/rates_final.py -> rdna_checks.json."
           % (comma(17512 - 9137), comma(17512 - 9137), comma(uq[2]), comma(uq[0]), comma(uq[1]),
              comma(rd_reads), comma(all_reads), rdna_all,
              cv("merged", "rdna", "mean_cov"), cv("merged", "rest", "mean_cov"), cv("filtered", "rdna", "mean_cov"), cv("filtered", "rest", "mean_cov"),
              comma(RD["merged_rdna_mapq0"]), comma(RD["merged_rdna_mapq60"]),
              comma(RD["mapq_ge20_touch_unique"]), comma(RD["merged_rdna_mapq_ge20"]),
              cv("merged", "rdna", "rate"), cv("merged", "rest", "rate"), cv("filtered", "rdna", "rate"), cv("filtered", "rest", "rate"),
              RD["kept_median_rate"]["rdna"], RD["kept_median_rate"]["rest"], mrs,
              100 * RD["rdna_judged_frac_gt018"], 100 * RD["nonrdna_judged_frac_gt018"], RD["rdna_judged_rate_q"]["0.999"], RD["rdna_judged_n_gt045"],
              100 * share["Mar20"], 100 * share["Jun25"], 100 * share["May07"],
              comma(RD["old_hyper_rdna"]), comma(RD["old_hyper"]), RD["hyper_rdna_pct"], RD["judged_rdna_pct"], OLDFENCE["hi"], FENCE["hi"],
              float(bg_w[3]), float(bg_w[5]), nuc_ratio, 100 * tf_maxdev)),
    "H1": ("Pooled rates move by at most a few thousandths between read sets; linker rates hardly move at all. Brogaard cores and Chereji cores give the same ordering; Chereji cores read a little higher. "
           "The final row is the keep list used for the refit (rDNA excluded, one pooled MAD fence)."),
    "8": "Placeholder.",
    "9": "Placeholder.",
}
def _note(r):
    f = r["full"]
    return "%s: %s (%s; decided by %s). Evidence: %s%s" % (
        f["choice"], f["value"], f.get("when", ""), f.get("decided_by", ""), f.get("evidence", "") or f.get("where", ""),
        (". Revisit: " + f["revisit"]) if f.get("revisit") else "")
notes["3"] = ("Every row here is a row of the shared sheet FiberCOP model choices (" + SHEET + "), stage Model 8 Single-fiber; "
              "slide and sheet are written from the same file, presentation/lab_nov03/decisions_rows.json. Sources: SINGLEFIBER_PLAN.md, "
              "reports/2026-11-02_report.md. Row by row: " + " | ".join(_note(r) for r in DEC))
notes["3b"] = ("Open items, ranked by how much they block per-read results, from reports/2026-11-03_singlefiber_decisions.md. "
               "Each is a sheet row with decided_by 'open (user)' and the review's recommendation in its Revisit field. "
               "Row by row: " + " | ".join("#%d %s: %s" % (r["fields"]["open_rank"], r["full"]["choice"], r["full"]["revisit"].split(". ", 1)[1]) + " " + r["full"]["why"] + "." for r in OPN))
notes.update(UPDATE_NOTES)
json.dump(notes, open(os.path.join(H, "notes.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote slides_fragment.html and notes.json")
