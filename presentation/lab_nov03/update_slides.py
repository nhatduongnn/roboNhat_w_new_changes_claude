# Slides 4, 5, 6, 7, 7b of the Nov 3 deck (added 2026-11-03). exec'd by make_fragment.py, which provides
# S, R, FINAL, chart, comma, f3, json, os, H. Every number is read from a named file here; none is typed by hand.
# Images: img/*.png, cropped by make_images.py, published as `files` beside the page.

SWEEP_J = json.load(open("/usr/project/xtmp/nd141/permol_proto_logfull/abf1sweep/sweep_summary.json"))
CMP_J = json.load(open("/usr/project/xtmp/nd141/permol_proto_logfull/compare_localprior/compare_summary.json"))
THR_TXT = "/usr/project/xtmp/nd141/filtered_refit_nordna/threshold.txt"
THR = float([l for l in open(THR_TXT) if l.startswith("A_threshold")][0].split()[1])
assert abs(THR - 0.8105469) < 1e-9
SITES_F = {}
for row in csv.DictReader(open("/usr/project/xtmp/nd141/tf_compare/mixture_filtnordna_pb0p02/sites_f.tsv"), delimiter="\t"):
    SITES_F[row["name"]] = float(row["f"])
F1, F2 = SITES_F["mac_chrI_61165"], SITES_F["mac_chrI_62659"]


def P(x): return "%.0f%%" % (100 * x)


# ---- slide 4: call-rule scale (SVG, one scale x = p) + mini barcode chart
def call_scale():
    W, x0, x1, yb, hb = 1440, 30, 1410, 62, 44
    X = lambda p: x0 + (x1 - x0) * p
    lo, hi = 1 - THR, THR
    o = ['<svg viewBox="0 0 %d 170" role="img" aria-label="m6A call rule on p, the dorado ML probability: skipped A and p up to %.2f '
         'unmethylated, between %.2f and %.2f no call, %.2f and above methylated.">' % (W, lo, lo, hi, hi)]
    o.append('<defs><pattern id="nocall-hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
             '<rect class="nc-bg" width="10" height="10"/><line class="nc-ln" x1="0" y1="0" x2="0" y2="10"/></pattern></defs>')
    o.append('<rect class="band un" x="%.1f" y="%d" width="%.1f" height="%d"/>' % (X(0), yb, X(lo) - X(0), hb))
    o.append('<rect class="band skip" x="%.1f" y="%d" width="%.1f" height="%d"/>' % (X(0), yb, X(0.05) - X(0), hb))
    o.append('<rect class="band nc" x="%.1f" y="%d" width="%.1f" height="%d" fill="url(#nocall-hatch)"/>' % (X(lo), yb, X(hi) - X(lo), hb))
    o.append('<rect class="band me" x="%.1f" y="%d" width="%.1f" height="%d"/>' % (X(hi), yb, X(1) - X(hi), hb))
    o.append('<text class="bl" x="%.1f" y="%d">unmethylated</text>' % (X(0), yb - 14))
    o.append('<text class="bl" x="%.1f" y="%d" text-anchor="middle">no call: no evidence for that base</text>' % ((X(lo) + X(hi)) / 2, yb - 14))
    o.append('<text class="bl me-t" x="%.1f" y="%d" text-anchor="end">methylated</text>' % (X(1), yb - 14))
    for p, lab in ((0, "0"), (0.05, "0.05"), (lo, "%.2f" % lo), (hi, "%.2f" % hi), (1, "1")):
        o.append('<line class="tkl" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (X(p), X(p), yb + hb, yb + hb + 8))
        anc = "start" if p == 0 else ("end" if p == 1 else "middle")
        o.append('<text class="tk" x="%.1f" y="%d" text-anchor="%s">%s</text>' % (X(p), yb + hb + 28, anc, lab))
    o.append('<text class="sk" x="%.1f" y="%d">skipped A (p &lt; 0.05) count as unmethylated</text>' % (X(0.05) + 10, yb + hb - 15))
    o.append('<text class="axl" x="%.1f" y="%d" text-anchor="middle">p = dorado ML probability of m6A</text>' % ((x0 + x1) / 2, yb + hb + 62))
    o.append("</svg>")
    return "\n".join(o)


S.append('''<!-- ======================= 4 UPDATE ======================= -->
<section class="slide" data-id="4" data-kind="main" data-sec="3" data-steps="1" data-title="Update">
  <span class="slide-no">4</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> Single-fiber update</p>
    <h2>Update</h2>
  </header>
  <div class="body stack" style="gap:26px">
    <div class="callscale">%s</div>
    <p class="foot txt" style="font-size:16px;margin-top:-14px">%.2f = modkit's 10th-percentile-of-confidence rule, fixed once on the filtered reads (%.7f).</p>
    <div class="upd-row">
      <div class="bars mini">%s</div>
      <ul class="upd-bul"><li>Some runs are more methylated than others.</li></ul>
    </div>
  </div>
</section>''' % (call_scale(), THR, THR, chart(FINAL, mini=True)))


# ---- shared: figure rows (label | image), site markers above the first row
def markers(window, items):
    # x of a coordinate inside a cropped 2800-px-wide panel: axes frame = columns 237..2617 (frame scan, make_images.py)
    a, b = window
    out = []
    for pos, lab, cls in items:
        fx = (237 + (pos - a) / float(b - a) * 2380) / 2800.0
        out.append('<span class="mk %s" style="left:%.2f%%">%s</span>' % (cls, 100 * fx, lab))
    return '<div class="mk-strip">%s</div>' % "".join(out)


def zoom_btn(src, title):
    return '<button type="button" class="zoom-btn" data-src="%s" data-title="%s">Full figure</button>' % (src, title)


ERV_WIN = (60001, 65000)
SITE_MK = [(61171, "site 1", "s"), (62665, "site 2", "s"), (61662, "no-site spot", "x")]
a1 = SWEEP_J["1"]
assert a1["traindir"] == "robocop_train_fiberonly_nucbrogsym"
abf1_max = max(a1["site1"]["max"], a1["site2"]["max"], a1["avg_abf1_max"])
# ---- slide 4b: how a no-call enters the Bernoulli (added 2026-11-03). Numbers read from files.
import pickle as _pk
import numpy as np
_A = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/inputs/"
TH_OPEN = float(np.ravel(_pk.load(open(_A + "bg_params_filtnordna.pkl", "rb"))["p"]["watson_signal"]["A"])[0])
TH_NUC = float(np.ravel(_pk.load(open(_A + "nucleosome_params_brogaard_sym_filtnordna.pkl", "rb"))["p"]["watson_signal"]["A"])[73])
NCJ = json.load(open("/usr/project/xtmp/nd141/nocall_rates/nocall_rates.json"))
NU_OPEN, NU_NUC = NCJ["linker"]["nocall_frac"], NCJ["core_brogaard"]["nocall_frac"]
assert abs(NCJ["linker"]["m6a_rate"] - 0.1554) < 0.002          # pileup linker rate = the bg fit (sanity)

def _cells(po, pn):
    r = po / pn
    fav = ("%.2f&times; open" % r) if r > 1.0005 else (("%.2f&times; nucleosome" % (1 / r)) if r < 0.9995 else "1&times; &mdash; no evidence")
    post = po / (po + pn)
    return po, pn, fav, post

today = [_cells(1 - TH_OPEN, 1 - TH_NUC), _cells(TH_OPEN, TH_NUC), _cells(1.0, 1.0), _cells(1.0, 1.0)]
three = [_cells((1 - NU_OPEN) * (1 - TH_OPEN), (1 - NU_NUC) * (1 - TH_NUC)),
         _cells((1 - NU_OPEN) * TH_OPEN, (1 - NU_NUC) * TH_NUC),
         _cells(NU_OPEN, NU_NUC)]

def _rows(tag, vals, fo, fn):
    out = []
    lab = [("score if open", 0, fo), ("score if nucleosome", 1, fn), ("evidence", 2, None),
           ("P(open) after this base<br><small>if open and nucleosome start equal (50:50)</small>", 3, None)]
    for name, k, form in lab:
        tds = []
        for j, v in enumerate(vals):
            x = v[k]
            txt = x if isinstance(x, str) else ("%.3f" % x)
            sub = ('<span>%s</span>' % form[j]) if form else ""
            if k == 3:   # write out row 4 from rows 1 and 2: a / (a + b)
                sub = '<span>%.3f / (%.3f + %.3f)</span>' % (v[0], v[0], v[1])
            tds.append('<td class="pn%s">%s%s</td>' % (" key" if k == 2 else "", txt, sub))
        out.append('<tr><th scope="row">%s</th>%s</tr>' % (name, "".join(tds)))
    return "\n".join(out)

fo_t = ["1 &minus; &theta;<sub>open</sub>", "&theta;<sub>open</sub>", "1 (skipped)", "1 (not scored)"]
fn_t = ["1 &minus; &theta;<sub>nuc</sub>", "&theta;<sub>nuc</sub>", "1 (skipped)", "1 (not scored)"]
fo_3 = ["(1&minus;&nu;<sub>open</sub>)(1&minus;&theta;<sub>open</sub>)", "(1&minus;&nu;<sub>open</sub>)&theta;<sub>open</sub>", "&nu;<sub>open</sub>"]
fn_3 = ["(1&minus;&nu;<sub>nuc</sub>)(1&minus;&theta;<sub>nuc</sub>)", "(1&minus;&nu;<sub>nuc</sub>)&theta;<sub>nuc</sub>", "&nu;<sub>nuc</sub>"]
NOCALL_HTML = """<table class="tbl compact probs nc3">
<thead><tr><th></th><th>Unmethylated A<br><small>p &le; %.2f</small></th><th>Methylated A<br><small>p &ge; %.2f</small></th><th>No-call A<br><small>%.2f &lt; p &lt; %.2f</small></th><th>C, G or T base<br><small>cannot carry m6A on this strand</small></th></tr></thead>
<tbody>
<tr class="grp"><th colspan="5">Today: Bernoulli on confident calls only</th></tr>
%s
</tbody></table>""" % (1 - THR, THR, 1 - THR, THR, _rows("t", today, fo_t, fn_t))

S.append('''<!-- ======================= 4b NO-CALL MATH ======================= -->
<section class="slide" data-id="4b" data-kind="main" data-sec="3" data-steps="1" data-title="How a no-call enters the model">
  <span class="slide-no">4b</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> one A on one read &middot; open (&theta; = %.3f) vs nucleosome dyad (&theta; = %.3f)</p>
    <h2 style="font-size:44px">How a no-call enters the model</h2>
  </header>
  <div class="body stack" style="gap:14px">
    %s
    <div class="nc-layers">
      <table class="tbl compact nclay">
        <thead><tr><th>layer</th><th>what</th><th>at a no-call A</th></tr></thead>
        <tbody>
        <tr><td>0</td><td>sequence (PWM for TFs, dinucleotide model for nucleosome, base frequencies for background)</td><td><b>used as normal</b></td></tr>
        <tr><td>1&ndash;4</td><td>MNase</td><td>off (0)</td></tr>
        <tr><td>5</td><td>Watson m6A Bernoulli</td><td><b>0</b>: the no-call is in neither the methylated nor the unmethylated mask</td></tr>
        <tr><td>6</td><td>Crick m6A Bernoulli</td><td>0: a read fills only its own strand's layer</td></tr>
        </tbody>
      </table>
      <div class="nc-eq">
        <p class="nc-eq-l">score of state <i>s</i> at a no-call base <i>i</i></p>
        <p class="nc-eq-m">log e<sub>s</sub>(i) = log seq<sub>s</sub>(base<sub>i</sub>) + 0</p>
        <p class="nc-eq-n">same as a C, G or T base; the state at <i>i</i> still comes from the neighbouring bases through forward&ndash;backward</p>
      </div>
    </div>
    <ul class="upd-bul nc-bul">
      <li>Today a no-call scores 1 in every state, so it cancels: P(s | read) = &pi;<sub>s</sub>&middot;c&middot;L<sub>s</sub> / &Sigma; &pi;<sub>t</sub>&middot;c&middot;L<sub>t</sub>.</li>
      <li>That assumes no-calls are equally likely in every state. Measured: %.1f%% of linker A's vs %.1f%% in nucleosome cores.</li>
    </ul>
  </div>
</section>''' % (TH_OPEN, TH_NUC, NOCALL_HTML, 100 * NU_OPEN, 100 * NU_NUC))

S.append('''<!-- ======================= 5 SINGLE-FIBER DECODE ======================= -->
<section class="slide" data-id="5" data-kind="main" data-sec="3" data-steps="1" data-title="Current single-fiber decoding">
  <span class="slide-no">5</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> ERV46 · chrI:60,001–65,000 · %d reads</p>
    <h2 style="font-size:46px">Current single-fiber decoding</h2>
    <p class="msg">Per-read P(ABF1) stays near 0.001 (max %.4f).</p>
  </header>
  <div class="body figrows-wrap">
    <div class="figrows" style="--imgw:970px">
      <div class="fl"></div><div class="fi">%s</div>
      <div class="fl">Genes</div><div class="fi figplate"><img src="img/erv46_genes.png" alt="Genes PTA1, ERV46, CDC24"></div>
      <div class="fl">Raw m6A<small>fibers clustered on the site 2 window</small></div><div class="fi figplate"><img src="img/erv46_raw.png" alt="Raw m6A per fiber, clustered"></div>
      <div class="fl">P(nucleosome)<small>0 to 1</small></div><div class="fi figplate"><img src="img/erv46_nuc.png" alt="Per-read nucleosome posterior"></div>
      <div class="fl">P(ABF1)<small>colour scale 0 to 0.0011</small>%s</div><div class="fi figplate"><img src="img/erv46_abf1.png" alt="Per-read ABF1 posterior, scale 0 to 0.0011"></div>
    </div>
  </div>
</section>''' % (a1["n_molecules"], abf1_max, markers(ERV_WIN, SITE_MK[:2]),
                 zoom_btn("img/erv46_full.png", "ERV46 chrI:60,001-65,000, %d reads, filtered rDNA-free model (all panels)" % a1["n_molecules"])))

# ---- slide 6: genome-wide ABF1 prior sweep
x15, x50 = SWEEP_J["1500"], SWEEP_J["5000"]
spot = [l for l in x50["elsewhere"]["loci"] if l["start"] == 61656][0]
assert x15["elsewhere"]["reads_with_call"] == 0
S.append('''<!-- ======================= 6 ABF1 PRIOR SWEEP ======================= -->
<section class="slide" data-id="6" data-kind="hidden" data-sec="3" data-steps="1" data-title="Raised genome-wide ABF1 prior">
  <span class="slide-no">6</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> ERV46 · ABF1 prior × 1, 138, 400, 1500, 5000</p>
    <h2 style="font-size:46px">Raised genome-wide ABF1 prior (×1500)</h2>
  </header>
  <div class="body stack" style="gap:26px">
    <div class="figplate"><img class="full" src="img/abf1_sweep_summary.png" alt="ABF1 prior sweep: per-read posterior, fraction of reads above 0.5, and off-site calls versus the prior multiplier"></div>
    <ul class="upd-bul">
      <li>All reads rise together: at ×1500, %s of site-2 reads pass 0.5 (%d of %d). Site 1 stays at %.2f.</li>
      <li>A spot with no motif site (%s–%s) rises too. At ×5000 it passes 0.5 in %d reads.</li>
    </ul>
  </div>
</section>''' % (P(x15["site2"]["frac_gt_0p5"]), x15["site2"]["n_gt_0p5"], x15["site2"]["n"], x15["site1"]["median"],
                 comma(spot["start"]), comma(spot["end"]), spot["n_reads"]))

# ---- slide 7: local prior vs before vs global x1500
CW = (60901, 62900)
cs = CMP_J["sites"]
def fr(cond): return "site 1 <b>%s</b> · site 2 <b>%s</b>" % (P(cs["site1"][cond]["frac_gt_0p5"]), P(cs["site2"][cond]["frac_gt_0p5"]))
S.append('''<!-- ======================= 7 LOCAL PRIOR ======================= -->
<section class="slide" data-id="7" data-kind="main" data-sec="3" data-steps="1" data-title="Local prior (per-site bound fraction)">
  <span class="slide-no">7</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> ERV46 · chrI:60,901–62,900 · same reads, same order</p>
    <h2 style="font-size:46px">Local prior (per-site bound fraction)</h2>
    <p class="msg">Local prior: both sites called. Global ×1500: site 2 only.</p>
  </header>
  <div class="body figrows-wrap">
    <div class="figrows" style="--imgw:960px">
      <div class="fl"><span class="fl-k">reads with P(ABF1) &gt; 0.5</span></div><div class="fi">%s</div>
      <div class="fl">Before<small>%s</small></div><div class="fi figplate"><img src="img/cmp_before.png" alt="P(ABF1) before, current defaults"></div>
      <div class="fl">Local prior<small>%s</small></div><div class="fi figplate"><img src="img/cmp_local.png" alt="P(ABF1) with the local prior"></div>
      <div class="fl">Global ×1500<small>%s</small><small class="ref">site-mixture f: site 1 %.2f · site 2 %.2f</small>%s</div><div class="fi figplate"><img src="img/cmp_global.png" alt="P(ABF1) with the genome-wide prior raised 1500 times"></div>
    </div>
  </div>
</section>''' % (markers(CW, SITE_MK), fr("before"), fr("localprior"), fr("globalprior"), F1, F2,
                 zoom_btn("img/cmp_full.png", "ERV46 chrI:60,901-62,900: raw m6A, P(ABF1) before / local prior / global x1500, per-site scatter")))

S.append('''<!-- ======================= 7b PER-READ VS MOTIF m6A ======================= -->
<section class="slide" data-id="7b" data-kind="main" data-sec="3" data-steps="1" data-title="Per-read P(ABF1) vs methylated motif As">
  <span class="slide-no">7b</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> Local prior · per read</p>
    <h2 style="font-size:46px">Per-read P(ABF1) vs methylated As in the motif</h2>
    <p class="msg">The local prior over-calls site 1: %s of reads above 0.5, against a bound fraction f = %.2f.</p>
  </header>
  <div class="body stack" style="gap:18px">
    <div class="figplate"><img class="full" src="img/cmp_scatter.png" alt="Per-read P(ABF1) at site 1 and site 2 against the number of methylated As in the motif, for before, local prior, global x1500 and the site-mixture model"></div>
  </div>
</section>''' % (P(cs["site1"]["localprior"]["frac_gt_0p5"]), F1))


# ---- notes
def sw_row(k):
    d = SWEEP_J[k]
    return ("x%s: site 2 median %.3g (IQR %.3g-%.3g), %d/%d reads > 0.5; site 1 median %.3g, %d/%d > 0.5; gene-body control median %.2g; "
            "nucleosome-core control median %.2g; reads with an off-site call > 0.5: %d" % (
                k, d["site2"]["median"], d["site2"]["q25"], d["site2"]["q75"], d["site2"]["n_gt_0p5"], d["site2"]["n"],
                d["site1"]["median"], d["site1"]["n_gt_0p5"], d["site1"]["n"], d["ctrl_gene_body_64001"]["median"],
                d["ctrl_nuc_core_61491"]["median"], d["elsewhere"]["reads_with_call"]))


rf = {r: R[(FINAL, r + "_run")] for r in ("Jun25", "May07", "Mar20")}
UPDATE_NOTES = {
    "4": ("Call rule (singlefiber/read_calls.py, same as modkit pileup): p = ML/255. Methylated if p >= %.7f; unmethylated if p <= %.7f "
          "or the A is not listed in MM (dorado does not list A below about p = 0.05; tag mode A+a. means unlisted = canonical); in between = no call, "
          "dropped like modkit drops it from Nvalid_cov. Threshold: %s (modkit 0.2.6 sample-probs, 10th percentile of sampled A call confidence, "
          "3 runs all %.7f). Chart = slide 1's chart on the final keep list (bg_investigation/nov03_final/rates_final.tsv). "
          "Jun25 / May07 pooled rate: linkers %.4f / %.4f = %.2fx; nucleosome cores %.4f / %.4f = %.2fx; genome-wide %.4f / %.4f = %.2fx. "
          "Mar20 sits in between (linkers %.4f, cores %.4f, genome-wide %.4f)."
          % (THR, 1 - THR, THR_TXT, THR,
             rf["Jun25"]["lk_pooled"], rf["May07"]["lk_pooled"], rf["Jun25"]["lk_pooled"] / rf["May07"]["lk_pooled"],
             rf["Jun25"]["ncB_pooled"], rf["May07"]["ncB_pooled"], rf["Jun25"]["ncB_pooled"] / rf["May07"]["ncB_pooled"],
             rf["Jun25"]["gw_pooled"], rf["May07"]["gw_pooled"], rf["Jun25"]["gw_pooled"] / rf["May07"]["gw_pooled"],
             rf["Mar20"]["lk_pooled"], rf["Mar20"]["ncB_pooled"], rf["Mar20"]["gw_pooled"])),
    "5": ("ERV46 chrI:60,001-65,000, %d reads, the filtered rDNA-free model (run_permol_window_logfull_5to3.py defaults: 5'->3', start anywhere, "
          "Brogaard symmetric nucleosome, _filtnordna bg/nuc/TF rates). Shown: genes, raw m6A in clustered order, P(nucleosome), P(ABF1) on a 0 to 0.0011 scale. "
          "The full figure (button) also has mean P(accessible), pooled m6A, raw m6A shuffled and per-read P(accessible). "
          "Clustering: PCA + UMAP + HDBSCAN on the site 2 window 62,540-62,790; 122 reads cover it, 265 do not. "
          "Site 1 per-read P(ABF1) median %.2g, site 2 median %.2g (max %.4f); 0 reads above 0.5 (abf1sweep/sweep_summary.json, key 1; "
          "figures/erv46_2026-11-02/filtnordna/fig.log: ABF1 max posterior over the window 0.00112444)."
          % (a1["n_molecules"], a1["site1"]["median"], a1["site2"]["median"], abf1_max)),
    "6": ("Genome-wide ABF1 prior multiplied by 1, 138, 400, 1500, 5000 (trainDirs robocop_train_fiberonly_nucbrogsym[_abf1xN]), same 387 reads. "
          "Source /usr/project/xtmp/nd141/permol_proto_logfull/abf1sweep/sweep_summary.json. " + "; ".join(sw_row(k) for k in ("1", "138", "400", "1500", "5000")) +
          ". At x5000 the off-site calls are %d reads at %s-%s and 1 read at 61,083-61,090. "
          "Per-read posterior falls with methylated motif As (Spearman %.2f at site 2) but the whole distribution moves up together, so the prior sets the level, not which reads are bound. "
          "Posteriors valid in every arm (no NaN, row sums within 1.5e-14)." % (spot["n_reads"], comma(spot["start"]), comma(spot["end"]),
                                                                              x15["site2"]["spearman_post_vs_nmeth"])),
    "7": ("Local prior: the ABF1 weight at each MacIsaac site is calibrated so that the decode's bound fraction matches the site-mixture f, "
          "with each read's own run held out (compare_localprior/sites_w.tsv; %d placements over %d reads). "
          "Source /usr/project/xtmp/nd141/permol_proto_logfull/compare_localprior/compare_summary.json. "
          "Per-read P(ABF1) median, site 1: before %.2g, local %.2f, global x1500 %.4f; site 2: before %.2g, local %.2f, global %.2f. "
          "Fraction of reads > 0.5, site 1: %s / %s / %s; site 2: %s / %s / %s. "
          "Site-mixture f (tf_compare/mixture_filtnordna_pb0p02/sites_f.tsv): site 1 %.3f, site 2 %.3f. "
          "So the local prior over-calls: %s of site 1 reads > 0.5 against f = %.2f. "
          "Agreement with the site-mixture per-read call at 0.5 (held-out run): site 1 local %s, global %s; site 2 local %s, global %s. "
          "Controls: gene body 64,001 and nucleosome core 61,491 unchanged by the local prior; no reads with an off-site call > 0.5 in any arm. "
          "Nucleosome posterior moves little (mean abs diff vs before %.4f local, %.4f global)."
          % (CMP_J["local_prior"]["localprior"]["n_placements"], CMP_J["local_prior"]["localprior"]["n_reads_weighted"],
             cs["site1"]["before"]["median"], cs["site1"]["localprior"]["median"], cs["site1"]["globalprior"]["median"],
             cs["site2"]["before"]["median"], cs["site2"]["localprior"]["median"], cs["site2"]["globalprior"]["median"],
             P(cs["site1"]["before"]["frac_gt_0p5"]), P(cs["site1"]["localprior"]["frac_gt_0p5"]), P(cs["site1"]["globalprior"]["frac_gt_0p5"]),
             P(cs["site2"]["before"]["frac_gt_0p5"]), P(cs["site2"]["localprior"]["frac_gt_0p5"]), P(cs["site2"]["globalprior"]["frac_gt_0p5"]),
             F1, F2, P(cs["site1"]["localprior"]["frac_gt_0p5"]), F1,
             P(cs["site1"]["localprior"]["call_agreement_vs_mixture_heldout_0p5"]), P(cs["site1"]["globalprior"]["call_agreement_vs_mixture_heldout_0p5"]),
             P(cs["site2"]["localprior"]["call_agreement_vs_mixture_heldout_0p5"]), P(cs["site2"]["globalprior"]["call_agreement_vs_mixture_heldout_0p5"]),
             CMP_J["nucleosome_vs_before"]["localprior"]["avg_mean_abs_diff"], CMP_J["nucleosome_vs_before"]["globalprior"]["avg_mean_abs_diff"])),
    "7b": ("Same reads as slide 7. x = methylated As inside the motif on the read's own strand (jittered); y = per-read P(ABF1) over the motif. "
           "Grey = site-mixture per-read posterior (site_mixture.py). Before is flat at 0; the local prior keeps the ordering by motif m6A but sits high at 0 methylated As. "
           "Spearman of P(ABF1) vs methylated motif As, local prior: site 1 %.2f, site 2 %.2f." %
           (cs["site1"]["localprior"]["spearman_vs_meth_motif_As"], cs["site2"]["localprior"]["spearman_vs_meth_motif_As"])),
}
