"""Write slides_fragment.html for the Oct 27 lab deck.

Every number on the four locus slides and the summary slide is read here from numbers.json
(build_payloads.py) or span_checks.out (span_checks.py); the hidden slides quote the named reports
(sources in each slide's footnote). Run after build_payloads.py, before make_template.py.
"""
import html, json, os, re
H = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(H, "numbers.json")))
L = N["loci"]
SPAN = {}
for line in open(os.path.join(H, "span_checks.out")):
    m = re.match(r"(\S+)\s+before mean ([\d.]+) max ([\d.]+) \| after mean ([\d.]+) max ([\d.]+)", line[23:])
    if m: SPAN[(line[:22].strip(), m.group(1))] = tuple(float(x) for x in m.groups()[1:])
MODELS = ["bo09", "bu01", "bv75"]
DESC = {"bo09": "9 TFs live", "bu01": "153 TFs live", "bv75": "75 TFs live"}
SW = {"bo09": "a", "bu01": "b", "bv75": "c"}


def c(v): return f"{v:,}"


def sci(x):
    if x < 100: return "%.2f" % x
    e = len(str(int(x))) - 1
    return "%.2f&nbsp;×&nbsp;10<sup>%d</sup>" % (x / 10 ** e, e)


def m(k, mod): return L[k]["models"][mod]


def broken_txt(k, mod):
    out = []
    for b in m(k, mod)["old_broken"]:
        out.append("segment %d, rows %s–%s (row sum up to %s)" % (b["segment"], c(b["rows"][0]), c(b["rows"][1]), sci(b["rs_max"])))
    return "; ".join(out)


def rows_html(post_h=52):
    r = []
    for i, mod in enumerate(MODELS):
        for side in ("before", "after"):
            run = "%s %s" % (mod, side)
            ps = " pair-start" if (i > 0 and side == "before") else ""
            rd = '<span class="rd after">after · log space</span>' if side == "after" else '<span class="rd">before · original</span>'
            r.append('      <div class="trk-lab pr m-%s%s"><span class="rn"><span class="sw %s"></span>%s</span>%s</div>\n'
                     '      <canvas class="%s%s" data-trk="post" data-run="%s" data-h="%d"></canvas>\n'
                     % (mod, ps, SW[mod], mod, rd, side, ps, run, post_h))
    return "".join(r)


def locus(sid, no, k, data_id, title, eyebrow, msg, nums, foot, aria, tf_label, color, sites=True, meth_h=50):
    leg_site = '      <span><span class="lk band"></span>MacIsaac ABF1 site</span>\n' if sites else ""
    return f'''
<!-- ======================= {sid} {html.escape(title)} ======================= -->
<section class="slide locus6" data-id="{sid}" data-kind="{'hidden' if str(no).startswith('H') else 'main'}" data-sec="1" data-steps="1" data-title="{html.escape(title)}">
  <span class="slide-no">{no}</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> {eyebrow}</p>
    <h2>{title}</h2>
    <p class="msg">{msg}</p>
  </header>
  <div class="body" style="display:flex;flex-direction:column;gap:7px">
    <div class="legend" style="font-size:16px">
      <span><span class="lk ln" style="border-color:{color}"></span>{tf_label} posterior</span>
      <span><span class="lk ar"></span>nucleosome occupancy</span>
{leg_site}      <span><span class="lk dot" style="background:var(--wat)"></span>m6A Watson</span>
      <span><span class="lk dot" style="background:var(--cri)"></span>m6A Crick</span>
      <span><span class="lk dash"></span>background 0.249</span>
      <span style="display:inline-flex;align-items:center"><span class="lk" style="width:22px;height:12px;background:color-mix(in srgb,var(--accent-soft) 55%,transparent);border:1px solid var(--rule)"></span>after (log space)</span>
      <span><span class="lk" style="width:22px;height:5px;background:var(--pending-ink)"></span>differs from before by ≥ 0.1 (any column)</span>
    </div>
    <div class="trk six" id="trk{sid}" data-locus="{data_id}" role="img" aria-label="{html.escape(aria, quote=True)}">
      <div class="trk-sites"{'' if sites else ' style="height:4px"'}></div>
{rows_html()}      <div class="trk-lab pair-start"><span class="rn">m6A per A</span><span class="rd">same reads</span></div>
      <canvas class="pair-start" data-trk="meth" data-h="{meth_h}"></canvas>
      <div class="trk-lab"><span class="rn">genes</span></div>
      <canvas data-trk="gene" data-h="50"></canvas>
      <div class="trk-lab"></div>
      <canvas data-trk="axis" data-h="28"></canvas>
      <div class="trk-hl"></div>
    </div>
    <div class="nums">{nums}</div>
    <p class="foot txt" style="font-size:14px;line-height:1.35">{foot}</p>
  </div>
</section>
'''


READER = ("Both sides read as the Run Browser does: <span class=\"m\">factor_tables</span> clipped per segment to [0, 1], overlapping segments averaged "
          "(<span class=\"m\">extract_run.py</span>). Before: <span class=\"m\">robocop_*_tw_&lt;model&gt;</span>, original <span class=\"m\">.so</span>. "
          "After: <span class=\"m\">robocop_tw3_&lt;model&gt;_logfull</span>. ")
SRC = "Numbers: <span class=\"m\">lab_oct27/numbers.json</span>; row sums: <span class=\"m\">fibercop_audit/extent/rowsums_*.tsv</span>."


def nxt(k):
    """the runners-up after the chosen factor, from the live-motif ranking in numbers.json"""
    r = L[k]["ranking"]; assert r[0][0] == L[k]["factor"]
    return ", ".join("%s %d" % (x[0].split("_")[0].upper(), x[1]) for x in r[1:4])


def diffnums(k):
    s = '<span class="k">positions where any of 160 columns differs ≥ 0.1</span>'
    for mod in MODELS:
        s += ' <span>%s <b>%s</b></span>' % (mod, c(m(k, mod)["any_ge_0p1"]))
    return s


slides = []
# ------------------------------------------------------------------ title, outline
slides.append('''<!-- ======================= T TITLE ======================= -->
<section class="slide title-slide" data-id="T" data-kind="main" data-sec="0" data-steps="1" data-title="Before and after the decoder fix">
  <p class="eyebrow">Lab meeting · 27 October 2026</p>
  <h1>Before and after the decoder fix</h1>
  <p class="sub" style="font-size:32px;max-width:52ch;line-height:1.25">Three tuned models, four loci, drawn the way our Run Browser
  draws them: the original decoder next to the log-space decoder.</p>
  <div class="ruler" aria-hidden="true"></div>
  <p class="msg" style="max-width:70ch">Models bo09, bu01 and bv75 (fiber + sequence, tuned on chrXIV + chrII). Same trained
  models, same reads; only the decoder changes.</p>
</section>

<!-- ======================= O OUTLINE ======================= -->
<section class="slide" data-id="O" data-kind="main" data-sec="0" data-steps="1" data-title="Outline">
  <span class="slide-no">1</span>
  <header class="sh">
    <p class="eyebrow">Outline</p>
    <h2>Today</h2>
  </header>
  <div class="body stack" style="gap:40px">
    <ol class="outline">
      <li><span class="k">Part 1</span><span class="t">The log-space fix</span><span class="s">What changed · four loci before and after · where they differ</span></li>
      <li><span class="k">Part 2</span><span class="t">single-fiberCOP</span><span class="s">Decoding one read at a time · four problems we found</span></li>
      <li><span class="k">Part 3</span><span class="t">Strahl lab's results</span><span class="s">&nbsp;</span></li>
      <li><span class="k">Part 4</span><span class="t">Timeline</span><span class="s">Now to March 2027</span></li>
    </ol>
    <p class="foot txt">Details are on hidden slides (press <span class="m">h</span>): the C1 bug, the DBL_MIN padding,
    the log-space review, and the guards that no longer fire.</p>
  </div>
</section>
''')
# ------------------------------------------------------------------ what changed
G = {}   # guards table numbers are quoted from things_to_revisit_before_shipping.md section 8
slides.append('''
<!-- ======================= W WHAT CHANGED ======================= -->
<section class="slide" data-id="W" data-kind="hidden" data-sec="1" data-steps="1" data-title="What changed">
  <span class="slide-no">H7</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> The decoder</p>
    <h2>What changed</h2>
    <p class="msg">Same models, same reads. Only the arithmetic inside the decoder is new.</p>
  </header>
  <div class="body stack" style="gap:20px">
    <div class="cols" style="grid-template-columns:1fr 1fr;gap:28px">
      <div class="card flag" style="border-left-color:var(--pending-line)"><h3>Before · original decoder</h3>
        <ul class="pts" style="font-size:22px;gap:10px;margin-top:8px">
          <li>Multiplies thousands of tiny probabilities, rescaling as it goes.</li>
          <li>The undo step left a leftover factor: some rows summed far above 1
            (bo09 up to <span class="bad">3.6&nbsp;×&nbsp;10<sup>60</sup></span>).</li>
          <li>The viewer clipped each value to [0, 1] and averaged overlapping segments, so
            a broken row could show up as 0.50 or 1.00.</li>
          <li>Adds a tiny number (DBL_MIN, 2.2&nbsp;×&nbsp;10<sup>-308</sup>) to every cell to avoid dividing by zero.
            Sometimes that fake mass wins.</li>
        </ul></div>
      <div class="card flag"><h3>After · log space</h3>
        <ul class="pts" style="font-size:22px;gap:10px;margin-top:8px">
          <li>Adds logarithms instead of multiplying tiny numbers. Nothing underflows.</li>
          <li>No leftover factor: every row sums to 1
            (largest row sum 1.00000000000005).</li>
          <li>No DBL_MIN padding. An impossible position stops the run with an error instead.</li>
          <li>Emissions computed exactly; no 10<sup>-30</sup> floor.</li>
        </ul></div>
    </div>
    <div class="tbl-wrap"><table class="tbl compact" style="font-size:20px">
      <thead><tr><th>chrXIV + chrII + chrIV, 783 segments</th><th class="r">bo09</th><th class="r">bu01</th><th class="r">bv75</th></tr></thead>
      <tbody>
        <tr><td>Rows summing above 1 + 10<sup>-4</sup>, before</td><td class="n r bad">12,304</td><td class="n r">0</td><td class="n r bad">84</td></tr>
        <tr class="key"><td>Rows summing above 1 + 10<sup>-4</sup>, after</td><td class="n r">0</td><td class="n r">0</td><td class="n r">0</td></tr>
      </tbody>
    </table></div>
    <p class="foot txt" style="font-size:15px">Row counts and maximum row sums: <span class="m">things_to_revisit_before_shipping.md</span> §8
    (<span class="m">posterior_fix/fibercop_audit/guards/check_guards.py</span>; a row counts if its emitting sum or any silent cell exceeds 1 + 10<sup>-4</sup>).
    DBL_MIN: <span class="m">pkg/robocop/algo.c</span> <span class="m">normalize()</span>, upstream since 2021 (<span class="m">posterior_fix/dblmin_history/REPORT.md</span>).
    Production today is fix B (Oct 6: rows divided by their sum, DBL_MIN kept); it is not shown in this deck.</p>
  </div>
</section>
''')
# ------------------------------------------------------------------ hidden a, b (mechanism)
c1 = {}
for line in open(os.path.join(H, "..", "..", "analysis", "posterior_fix", "fibercop_audit", "c1scan", "summary_c1_rows.tsv")):
    f = line.rstrip("\n").split("\t")
    c1[f[0]] = f[1:]
assert c1["total"] == ["217 / 122382", "5 / 13798", "59 / 17689"], c1["total"]
assert c1["chrXII rDNA"][:3] == ["5 / 13408 (+9100 other)", "5 / 13798", "5 / 13794"], c1["chrXII rDNA"]
slides.append('''
<!-- ======================= HA C1 ======================= -->
<section class="slide" data-id="HA" data-kind="hidden" data-sec="1" data-steps="1" data-title="C1: the backward pass builds silent cells before rescaling">
  <span class="slide-no">H1</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> Detail · the C1 bug</p>
    <h2 style="font-size:44px">C1: the backward pass builds silent cells before rescaling</h2>
    <p class="msg"><span class="m">normalize()</span> adds DBL_MIN to every cell, then divides the row by its emitting sum S.
    The original backward pass computes the silent cells first, so they are S times too small and the padding swamps them.</p>
  </header>
  <div class="body stack" style="gap:16px">
    <div class="cols" style="grid-template-columns:1fr 1.15fr;gap:36px;align-items:start">
      <div class="stack" style="gap:12px;font-size:22px">
        <p><strong>Where it bites.</strong> With the 10<sup>-30</sup> emission floor, S is 10<sup>-31</sup> to 10<sup>-99</sup>, so
        the fake mass per row is 10<sup>-277</sup> to 10<sup>-209</sup>. It does damage where a later row's true mass falls below it.</p>
        <p><strong>One example.</strong> bo09, chrXIV segment 412,001–417,000: the original and the C1-fixed backward pass differ at
        <strong>one</strong> position (413,386), by 29.96 nats — the 1.03&nbsp;×&nbsp;10<sup>13</sup> row-sum step.</p>
        <p>With MNase, S is about 10<sup>-2</sup> and the fake mass about 10<sup>-306</sup>: harmless.</p>
      </div>
      <div>
        <p class="panel-t">Rows hit by C1, whole genome (3,021 segments)</p>
        <div class="tbl-wrap"><table class="tbl compact" style="font-size:20px">
          <thead><tr><th>Model</th><th class="r">Segments / rows</th><th class="r">of which chrXII rDNA</th></tr></thead>
          <tbody>
            <tr><td class="mono">bo09</td><td class="n r bad">217 / 122,382</td><td class="n r">5 / 13,408</td></tr>
            <tr><td class="mono">bv75</td><td class="n r">59 / 17,689</td><td class="n r">5 / 13,794</td></tr>
            <tr><td class="mono">bu01</td><td class="n r">5 / 13,798</td><td class="n r">all of it</td></tr>
          </tbody>
        </table></div>
        <p class="foot txt" style="font-size:15px;margin-top:8px">A C1 row has |emitting sum − 1| &gt; 10<sup>-4</sup> in the original library
        but not in the C1-fixed one. bo09 chrXIV + chrII: 6,804 + 485 = 7,289, identical to the stored decode.</p>
      </div>
    </div>
    <p class="foot txt" style="font-size:15px">Source: <span class="m">analysis/posterior_fix/fibercop_audit/c1scan/README.md</span> and
    <span class="m">summary_c1_rows.tsv</span> (Slurm 13070845 / 13070895 / 13070896). Report: FiberCOP C1 Check,
    <span class="m">https://claude.ai/artifact/AwK7rW3xjbYVzBUP8dE8dY</span>. Caveat from the README: in rDNA the C1-fixed rows sum to 1 but are still wrong.</p>
  </div>
</section>

<!-- ======================= HB DBL_MIN ======================= -->
<section class="slide" data-id="HB" data-kind="hidden" data-sec="1" data-steps="1" data-title="DBL_MIN padding: wrong rows that still sum to 1">
  <span class="slide-no">H2</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> Detail · defect 3</p>
    <h2 style="font-size:46px">DBL_MIN padding: wrong rows that still sum to 1</h2>
    <p class="msg">DBL_MIN stops a divide-by-zero. The price: a state that should be exactly 0 gets DBL_MIN / S, and paths through
    forbidden cells exist. Upstream added it on 2021-08-31 (<span class="m">c6a8727</span>).</p>
  </header>
  <div class="body stack" style="gap:16px">
    <p class="panel-t">bo09, chrIV:1,200,001–1,205,000 (segment 300), every arm on the same captured inputs</p>
    <div class="tbl-wrap"><table class="tbl compact" style="font-size:21px">
      <thead><tr><th>Arm</th><th class="r">Log-likelihood</th><th class="r">Rows off log space by &gt; 0.01</th></tr></thead>
      <tbody>
        <tr><td>Fix B (production, with DBL_MIN)</td><td class="n r">−486,260.97</td><td class="n r bad">147 (rows 1682–1828), max 1.0</td></tr>
        <tr><td>Same C code without DBL_MIN, float64</td><td class="n r">−486,267.708292105</td><td class="n r ok">0</td></tr>
        <tr class="key"><td>Log space (floored emission)</td><td class="n r">−486,267.708292105</td><td class="n r">reference</td></tr>
      </tbody>
    </table></div>
    <ul class="pts" style="font-size:22px;gap:10px">
      <li>In those rows log space puts one nucleosome (occupancy 1.000); fix B calls background.</li>
      <li>The C1 fix alone also leaves 147 rows: it keeps the padding.</li>
      <li>Not rDNA: ordinary chrIV. How often it happens genome-wide is <strong>unverified</strong>.</li>
    </ul>
    <p class="foot txt" style="font-size:15px">Rows 1682–1828 of the segment are chrIV:1,201,683–1,201,829: the tail shown on the chrIV locus slide.
    Source: <span class="m">analysis/posterior_fix/dblmin_history/REPORT.md</span> (Q2, Q4) and <span class="m">seg300/</span>.</p>
  </div>
</section>
''')
# ------------------------------------------------------------------ locus slides
k = "rpc19"; b = m(k, "bo09"); at = N["loci"][k]["models"]["at_412655"]; sm = N["loci"][k]["models"]["site_max_pm20"]["412615-412627"]
assert abs(at["bo09 before"] - 0.5) < 1e-9 and at["bo09 after"] == 0.0
slides.append(locus(
    "L1", 2, k, "locus-rpc19", "RPC19 promoter, chrXIV",
    "Locus 1 of 4 · chrXIV:412,400–412,900 · ABF1",
    "ABF1 at the poly-A, 412,655: bo09 <strong>%.2f</strong> before, <strong>%.2f</strong> after. "
    "At the MacIsaac ABF1 site: bo09 %.2f / %.2f, bv75 %.2f / %.2f, bu01 %.2f / %.2f."
    % (at["bo09 before"], at["bo09 after"], sm["bo09 before"], sm["bo09 after"], sm["bv75 before"], sm["bv75 after"], sm["bu01 before"], sm["bu01 after"]),
    diffnums(k) + ' <span>bo09: all %d in old rows that summed above 1</span>' % b["ge_0p1_in_broken"],
    "Factor ABF1 (Abf1_murphy), fixed in advance. " + READER +
    "bo09 old rows: " + broken_txt(k, "bo09") + "; none for bu01, bv75. Site: MacIsaac c1 (merged, 20 bp slop); "
    "chip = max within ±20 bp. Poly-A: 412,654–412,665. " + SRC,
    "chrXIV 412,400 to 412,900, the RPC19 promoter. ABF1 posterior and nucleosome occupancy for bo09, bu01 and bv75, each before (original decoder) "
    "and after (log space). At 412,655 bo09 ABF1 is 0.50 before and 0.00 after. At the MacIsaac ABF1 site 412,615 to 412,627: bo09 1.00 both, "
    "bv75 0.65 both, bu01 0.00 both.", "ABF1", "#17ef13"))

k = "xiv440"; b = m(k, "bo09"); v = m(k, "bv75")
sp = b["spans_outside_broken"]
slides.append(locus(
    "L2", "H8", k, "locus-xiv440", "chrXIV 440–445 kb: RAS2 to RPS7B",
    "Locus 2 of 4 · chrXIV:440,001–445,000 · ABF1",
    "bo09's old rows here summed up to %.2f (440,001–442,954) and up to %s (444,001–444,147). "
    "ABF1, bo09: max %.2f before, %.2f after." % (b["old_broken"][0]["rs_max"], sci(b["old_broken"][1]["rs_max"]), b["factor_max_before"], b["factor_max_after"]),
    diffnums(k) + ' <span>bo09: %d in old rows above 1, %d after them (%s–%s)</span> <span>bv75: all %d at %s–%s, not ABF1</span>'
    % (b["ge_0p1_in_broken"], b["ge_0p1_outside_broken"], c(sp[0][0]), c(sp[-1][1]), v["any_ge_0p1"], c(v["spans_ge_0p1"][0][0]), c(v["spans_ge_0p1"][-1][1])),
    "Factor ABF1: the live motif with the most positions differing ≥ 0.1 (bo09 %d; next %s). "
    "bo09 nucleosome differs at %d, background at %d. " % (b["factor_ge_0p1"], nxt(k), b["nuc_ge_0p1"], b["bg_ge_0p1"]) + READER +
    "Old rows above 1: bo09 " + broken_txt(k, "bo09") + "; bv75 " + broken_txt(k, "bv75") + ". No MacIsaac ABF1 site here. " + SRC,
    "chrXIV 440,001 to 445,000. ABF1 posterior and nucleosome occupancy for bo09, bu01 and bv75, before and after. bo09 differs at %d positions, "
    "bv75 at %d, bu01 at 0." % (b["any_ge_0p1"], v["any_ge_0p1"]), "ABF1", "#17ef13", sites=False))

k = "xiv724"; b = m(k, "bo09")
rk = L[k]["ranking"][0]
slides.append(locus(
    "L3", "H9", k, "locus-xiv724", "chrXIV 724–727 kb: ESF2",
    "Locus 3 of 4 · chrXIV:724,001–727,000 · background",
    "bo09's old rows 724,001–726,207 summed up to %s. Here the change is between background and nucleosome, not a TF." % sci(b["old_broken"][0]["rs_max"]),
    diffnums(k) + ' <span>bo09: background <b>%d</b>, nucleosome <b>%d</b>, all in old rows above 1</span>' % (b["bg_ge_0p1"], b["nuc_ge_0p1"]),
    "Shown: background (purple; the viewer's grey is too close to the nucleosome grey). No live motif differs by ≥ 0.1 in any model "
    "(largest: %s, %.3f). " % (rk[0], rk[2]) + READER + "Old rows above 1: bo09 " + broken_txt(k, "bo09") + "; none for bu01, bv75. " + SRC,
    "chrXIV 724,001 to 727,000. Background posterior and nucleosome occupancy for bo09, bu01 and bv75, before and after. bo09 differs at %d positions, "
    "bu01 and bv75 at 0." % b["any_ge_0p1"], "background", "#7c4dcc", sites=False))

k = "iv1200"; b = m(k, "bo09"); sp = b["spans_outside_broken"]
bgb, bga = SPAN[("loc4 after broken rows", "background")][0], SPAN[("loc4 after broken rows", "background")][2]
nub, nua = SPAN[("loc4 after broken rows", "nucleosome")][0], SPAN[("loc4 after broken rows", "nucleosome")][2]
slides.append(locus(
    "L4", "H10", k, "locus-iv1200", "chrIV 1,200–1,202 kb: ESC2",
    "Locus 4 of 4 · chrIV:1,200,001–1,202,000 · FHL1",
    "bo09's old rows 1,200,001–1,201,697 summed up to %s. FHL1, bo09: max %.2f before, %.3f after. "
    "Just past them (1,201,698–1,201,829) nucleosome %.2f → %.2f, background %.2f → %.2f (means)."
    % (sci(b["old_broken"][0]["rs_max"]), b["factor_max_before"], b["factor_max_after"], nub, nua, bgb, bga),
    diffnums(k) + ' <span>bo09: %d in old rows above 1, %d after them (%s–%s)</span>' % (b["ge_0p1_in_broken"], b["ge_0p1_outside_broken"], c(sp[0][0]), c(sp[-1][1])),
    "Factor FHL1 (Fhl1_zhu; teal, its browser colour #42230b is unreadable on dark): the live motif with the most positions "
    "differing ≥ 0.1 (bo09 %d; next %s). " % (b["factor_ge_0p1"], nxt(k)) + READER +
    "Old rows above 1: bo09 " + broken_txt(k, "bo09") + "; none for bu01, bv75. The %d after them lie in the 147 rows where fix B also disagrees with log space; "
    "the same C code without DBL_MIN agrees (hidden H2). " % b["ge_0p1_outside_broken"] + SRC,
    "chrIV 1,200,001 to 1,202,000. FHL1 posterior and nucleosome occupancy for bo09, bu01 and bv75, before and after. bo09 differs at %d positions, "
    "bu01 and bv75 at 0. After the broken rows bo09 has a nucleosome in log space where the old run shows half background." % b["any_ge_0p1"],
    "FHL1", "#0e9f8e", sites=False))
# ------------------------------------------------------------------ summary
rows = ""
for k, lab in (("rpc19", "chrXIV:412,400–412,900"), ("xiv440", "chrXIV:440,001–445,000"), ("xiv724", "chrXIV:724,001–727,000"), ("iv1200", "chrIV:1,200,001–1,202,000")):
    cells = ""
    for mod in MODELS:
        d = m(k, mod)
        x = d["any_ge_0p1"]
        extra = (' <span style="font-size:.8em;color:var(--muted)">(%d after)</span>' % d["ge_0p1_outside_broken"]) if d["ge_0p1_outside_broken"] else ""
        cells += '<td class="n r%s">%s%s</td>' % ("" if x else " z", c(x), extra)
    rows += '        <tr><td class="mono">%s</td><td>%s</td>%s</tr>\n' % (lab, {"rpc19": "ABF1", "xiv440": "ABF1", "xiv724": "background", "iv1200": "FHL1"}[k], cells)
cc = {(r["window"], r["model"]): r for r in N["crosscheck"]}
assert all(r["match"] for r in N["crosscheck"])
wrows = ""
for w in ("chrXIV:400001-430000", "chrXIV:436001-466000", "chrXIV:715001-745000", "chrIV:1190001-1220000"):
    ch, se = w.split(":"); s0, e0 = se.split("-")
    wrows += '        <tr><td class="mono">%s:%s–%s</td>%s</tr>\n' % (ch, c(int(s0)), c(int(e0)),
             "".join('<td class="n r%s">%s</td>' % ("" if cc[(w, mod)]["tsv"] else " z", c(cc[(w, mod)]["tsv"])) for mod in MODELS))
slides.append(f'''
<!-- ======================= S SUMMARY ======================= -->
<section class="slide" data-id="S" data-kind="hidden" data-sec="1" data-steps="1" data-title="Where before and after differ">
  <span class="slide-no">H11</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> Summary</p>
    <h2>Where before and after differ</h2>
    <p class="msg">Positions where any of 160 columns differs by ≥ 0.1 between the old Run Browser view and log space.</p>
  </header>
  <div class="body stack" style="gap:18px">
    <div class="cols" style="grid-template-columns:1.25fr 1fr;gap:40px;align-items:start">
      <div>
        <p class="panel-t">The four loci in this deck</p>
        <div class="tbl-wrap"><table class="tbl compact" style="font-size:21px">
          <thead><tr><th>Window</th><th>Shown</th><th class="r">bo09</th><th class="r">bu01</th><th class="r">bv75</th></tr></thead>
          <tbody>
{rows}          </tbody>
        </table></div>
        <p class="foot txt" style="font-size:15px;margin-top:6px">"(n after)": positions past the old rows that summed above 1.</p>
      </div>
      <div>
        <p class="panel-t">The 30 kb Run Browser windows that hold them</p>
        <div class="tbl-wrap"><table class="tbl compact" style="font-size:21px">
          <thead><tr><th>Window</th><th class="r">bo09</th><th class="r">bu01</th><th class="r">bv75</th></tr></thead>
          <tbody>
{wrows}          </tbody>
        </table></div>
      </div>
    </div>
    <ul class="pts" style="font-size:24px;gap:10px">
      <li><strong>bu01</strong>: no difference at any of these positions.</li>
      <li><strong>bv75</strong>: 23 positions in these windows, all in chrXIV:436,001–466,000.</li>
      <li><strong>bo09</strong>: most differences sit in old rows that summed above 1; the rest follow right after them.</li>
    </ul>
    <p class="foot txt" style="font-size:15px">Left: <span class="m">presentation/lab_oct27/numbers.json</span>. Right:
    <span class="m">posterior_fix/fibercop_audit/logcompare/old_vs_log_windows.tsv</span> (diff_ge_0p1, both row classes summed);
    recomputed here with the same reader and identical for all 12 cells. Same reader on both sides as on the locus slides.</p>
  </div>
</section>
''')
# ------------------------------------------------------------------ hidden c, d
slides.append('''
<!-- ======================= HC LOG-SPACE REVIEW ======================= -->
<section class="slide" data-id="HC" data-kind="hidden" data-sec="1" data-steps="1" data-title="Log-space decoder review: safe to decode">
  <span class="slide-no">H3</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> Detail · review of the log-space decoder, 23 Oct</p>
    <h2>Log-space decoder review: safe to decode</h2>
    <p class="msg">Same recursion as the original C code, in the same order, minus the DBL_MIN padding. No blockers.</p>
  </header>
  <div class="body stack" style="gap:16px">
    <div class="cols" style="grid-template-columns:1fr 1fr;gap:32px;align-items:start">
      <div>
        <p class="panel-t">Brute force: 128 configurations × 30 seeds, truth by path enumeration</p>
        <div class="tbl-wrap"><table class="tbl compact" style="font-size:19px">
          <thead><tr><th></th><th class="r">Log space</th><th class="r">Fix B</th></tr></thead>
          <tbody>
            <tr><td>Emitting posterior error</td><td class="n r ok">≤ 1.7&nbsp;×&nbsp;10<sup>-12</sup></td><td class="n r">up to 1.0 at tiny emissions</td></tr>
            <tr><td>Log-likelihood error</td><td class="n r ok">≤ 3.6&nbsp;×&nbsp;10<sup>-12</sup></td><td class="n r">up to 4.1&nbsp;×&nbsp;10<sup>3</sup></td></tr>
            <tr><td>Bad return codes</td><td class="n r ok">0 of 3,840</td><td class="n r">n/a</td></tr>
          </tbody>
        </table></div>
      </div>
      <div>
        <p class="panel-t">Real bo09 segments, log space (floored emission) vs fix B</p>
        <div class="tbl-wrap"><table class="tbl compact" style="font-size:19px">
          <thead><tr><th>Segment</th><th class="r">Largest difference</th></tr></thead>
          <tbody>
            <tr><td class="mono">chrXIV:412,001–417,000</td><td class="n r">4.78&nbsp;×&nbsp;10<sup>-9</sup></td></tr>
            <tr><td class="mono">chrXIV:408,001–413,000</td><td class="n r">6.3&nbsp;×&nbsp;10<sup>-14</sup></td></tr>
          </tbody>
        </table></div>
        <p class="foot txt" style="font-size:15px;margin-top:6px">rDNA, bo09: agrees with an 80-bit reference to 9.2&nbsp;×&nbsp;10<sup>-14</sup>.</p>
      </div>
    </div>
    <p class="panel-t">Fix eventually (none can fire with these models: end_probs = 1, segments of 5,000 bp)</p>
    <div class="tbl-wrap"><table class="tbl compact" style="font-size:18px">
      <thead><tr><th>#</th><th>Item</th><th>Severity in the report</th></tr></thead>
      <tbody>
        <tr><td class="n">5</td><td>end_probs counted twice at the last position</td><td>should-fix if end_probs ≠ 1; not active</td></tr>
        <tr><td class="n">6</td><td>n_obs = 1: log-likelihood off by log(end); posterior correct</td><td>harmless</td></tr>
        <tr><td class="n">7</td><td>n_obs = 0 writes out of bounds</td><td>should-fix; unreachable from robocop.py</td></tr>
      </tbody>
    </table></div>
    <p class="foot txt" style="font-size:15px">Source: <span class="m">analysis/posterior_fix/logspace_review/REPORT.md</span>
    (<span class="m">bruteforce.txt</span>, <span class="m">compare_caps_412001.json</span>, <span class="m">replica_ld_exact_rdna2.json</span>).
    Binary md5 88f86a35… rebuilt from source; run with <span class="m">ROBOCOP_LOG_EMISSION=exact</span>.</p>
  </div>
</section>

<!-- ======================= HD GUARDS ======================= -->
<section class="slide" data-id="HD" data-kind="hidden" data-sec="1" data-steps="1" data-title="Guards for posterior above 1 no longer fire">
  <span class="slide-no">H4</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 1</span> Detail · downstream guards</p>
    <h2>Guards for posterior above 1 no longer fire</h2>
    <p class="msg">Downstream code grew guards for rows above 1. On log-space decodes none can fire. Keep them while old decodes are in use.</p>
  </header>
  <div class="body stack" style="gap:14px">
    <div class="tbl-wrap"><table class="tbl compact" style="font-size:19px">
      <thead><tr><th>Decode (chrXIV + chrII + chrIV, 783 segments)</th><th class="r">Rows where a guard can fire</th><th class="r">Max row sum</th><th class="r"><span class="m">n_invalid</span></th></tr></thead>
      <tbody>
        <tr><td>bo09, original</td><td class="n r bad">7,289 + 5,015 = 12,304</td><td class="n r bad">3.6&nbsp;×&nbsp;10<sup>60</sup></td><td class="n r">7,218 + 5,060</td></tr>
        <tr><td>bu01, original</td><td class="n r">0</td><td class="n r">1.0000000000000722</td><td class="n r">0</td></tr>
        <tr><td>bv75, original</td><td class="n r bad">63 + 21 = 84</td><td class="n r bad">7.1&nbsp;×&nbsp;10<sup>42</sup></td><td class="n r">54 + 19</td></tr>
        <tr class="key"><td>bo09, log space</td><td class="n r">0</td><td class="n r">1.000000000000048</td><td class="n r">0</td></tr>
        <tr class="key"><td>bu01, log space</td><td class="n r">0</td><td class="n r">1.0000000000000049</td><td class="n r">0</td></tr>
        <tr class="key"><td>bv75, log space</td><td class="n r">0</td><td class="n r">1.0000000000000269</td><td class="n r">0</td></tr>
      </tbody>
    </table></div>
    <div class="tbl-wrap"><table class="tbl compact" style="font-size:17px">
      <thead><tr><th>Guard that becomes unreachable</th><th>What it does</th></tr></thead>
      <tbody>
        <tr><td class="mono">write_factor_table.py:174-175</td><td>counts <span class="m">n_invalid</span>, clips every column to [0, 1] (3.84&nbsp;×&nbsp;10<sup>8</sup> → 1.0 at the RPC19 poly-A)</td></tr>
        <tr><td class="mono">count_calls.py:66, 146-170, 183</td><td>drops positions outside [0, 1 + 10<sup>-4</sup>], reports <span class="m">n_excluded</span></td></tr>
        <tr><td class="mono">abf1_stack_extract.py:50, 118-123</td><td>same mask on raw values; feeds the ABF1 stack scripts</td></tr>
        <tr><td class="mono">upstream getNucleosomes / gettfs / robocopExtras</td><td>clip to 1, skip segments with sum &gt; 10<sup>10</sup>, print a warning</td></tr>
      </tbody>
    </table></div>
    <p class="foot txt" style="font-size:15px">Keep: <span class="m">score_robocop.region_optable</span>'s per-row normalisation (shifts log-space values by ≤ 0.9%, the 10<sup>-4</sup>
    save floor). Source: <span class="m">things_to_revisit_before_shipping.md</span> §8, measured with
    <span class="m">posterior_fix/fibercop_audit/guards/check_guards.py</span>.</p>
  </div>
</section>
''')
# ================================================================== Part 2: single-fiberCOP (added Oct 27)
# Numbers on F1, F2 and F4 are quoted from the single-fiber work (SINGLEFIBER_PLAN.md and the brief of
# 2026-10-27); the histogram on F3 and its medians are computed here from the 100-read both-directions test.
import csv, math
DIRTEST = "/usr/project/xtmp/nd141/permol_proto_logfull/dirtest100"
PR = {}
with open(os.path.join(DIRTEST, "per_read.tsv")) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        PR.setdefault(r["config"], []).append(r)
SUMM = json.load(open(os.path.join(DIRTEST, "summary.json")))


def stat(cfg, key, q):
    return SUMM[cfg][key][q]


assert all(len(PR[k]) == 100 for k in ("real_motif1st", "real_any", "seqoff_any", "symps_any"))
for k in ("real_motif1st", "real_any", "seqoff_any"):   # per_read.tsv and summary.json agree
    v = sorted(float(r["max_all"]) for r in PR[k])
    assert abs(max(v) - stat(k, "max_all", "max")) < 1e-12


def f2(x): return "%.2f" % x


def sup(x):
    """6e-08 -> 6&nbsp;×&nbsp;10<sup>-8</sup>"""
    e = int(math.floor(math.log10(x))); m_ = x / 10 ** e
    return "%d&nbsp;×&nbsp;10<sup>%d</sup>" % (round(m_), e)


TODAY_MED, TODAY_MAX = stat("real_motif1st", "max_all", "median"), stat("real_motif1st", "max_all", "max")
ANY_MED, ANY_MAX = stat("real_any", "max_all", "median"), stat("real_any", "max_all", "max")
OFF_MAX = stat("seqoff_any", "max_all", "max")
SYM_MED, SYM_MAX = stat("symps_any", "max_all", "median"), stat("symps_any", "max_all", "max")
ABF_ANY = max(stat("real_any", "abf1_site1", "max"), stat("real_any", "abf1_site2", "max"))
assert ABF_ANY <= 2e-5


def hist_svg():
    """three small-multiple histograms (per-read max |difference|), shared log x axis, drawn in the deck's tokens"""
    W, X0, X1 = 1000, 235, 975
    PH, BH, TOP = 150, 104, 14           # panel height, tallest bar (100 reads), top margin
    LO, HI, STEP = -9, 0, 0.25
    def xo(lg): return X0 + (lg - LO) / (HI - LO) * (X1 - X0)
    panels = [("real_motif1st", "Today's start rule", "h-a"), ("real_any", "Start anywhere", "h-b"),
              ("seqoff_any", "Sequence layer off", "h-c")]
    nb = int((HI - LO) / STEP)
    s = ['<svg viewBox="0 0 %d %d" role="img" aria-label="Histograms of the largest posterior difference per read between the two '
         'decoding directions, 100 reads at ERV46, for three settings: today\'s start rule, start anywhere, and sequence layer off." '
         'style="width:100%%;height:auto">' % (W, TOP + 3 * PH + 50)]
    for i, (cfg, lab, cls) in enumerate(panels):
        y0 = TOP + i * PH; base = y0 + BH + 18
        cnt = [0] * nb
        for r in PR[cfg]:
            lg = math.log10(min(max(float(r["max_all"]), 10 ** LO), 1))
            cnt[min(nb - 1, int((lg - LO) / STEP))] += 1
        for g in (0, 50, 100):
            yy = base - g / 100 * BH
            s.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (X0, X1, yy, yy))
            s.append('<text class="tick" x="%d" y="%.1f" text-anchor="end" dy="5">%d</text>' % (X0 - 8, yy, g))
        for b, n in enumerate(cnt):
            if not n: continue
            xa, xb = xo(LO + b * STEP), xo(LO + (b + 1) * STEP)
            h = n / 100 * BH
            s.append('<rect class="%s" x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="1.5"/>' % (cls, xa + 1, base - h, xb - xa - 2, h))
        med = stat(cfg, "max_all", "median")
        xm = xo(math.log10(med))
        s.append('<path class="hmed" d="M%.1f %.1f l-7 -11 h14 z"/>' % (xm, base - BH - 4))
        s.append('<line class="ax" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (X0, X1, base, base))
        s.append('<text class="lbl-b" x="0" y="%.1f">%s</text>' % (y0 + 30, lab))
        s.append('<text class="lbl" x="0" y="%.1f">reads (of 100)</text>' % (y0 + 54))
    yb = TOP + 3 * PH
    for e in range(LO, HI + 1):
        xx = xo(e)
        s.append('<line class="ax" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>' % (xx, xx, yb - 12, yb - 4))
        s.append('<text class="tick" x="%.1f" y="%.1f" text-anchor="middle">%s</text>' % (xx, yb + 14, "1" if e == 0 else "10<tspan dy=\"-7\" font-size=\"12\">%d</tspan>" % e))
    xf = xo(math.log10(6e-8))
    s.append('<line class="hfl" x1="%.1f" x2="%.1f" y1="%d" y2="%.1f"/>' % (xf, xf, TOP, yb - 12))
    s.append('<text class="tick" x="%.1f" y="%d" text-anchor="start">float32 limit</text>' % (xf + 6, TOP + 4))
    s.append('<text class="lbl" x="%.1f" y="%.1f" text-anchor="middle">largest posterior difference in the read, left→right vs right→left (log scale)</text>' % ((X0 + X1) / 2, yb + 42))
    s.append("</svg>")
    return "\n".join(s)


def sec2(sid, no, title, eyebrow, inner, kind="main"):
    return '''
<!-- ======================= %s %s ======================= -->
<section class="slide" data-id="%s" data-kind="%s" data-sec="2" data-steps="1" data-title="%s">
  <span class="slide-no">%s</span>
%s</section>
''' % (sid, html.escape(title), sid, kind, html.escape(title, quote=True), no, inner.replace("%%EYEBROW%%", "%EYEBROW%").replace("%EYEBROW%", '<span class="sec-no">Part 2</span> ' + eyebrow))


slides.append(sec2("F1", 3, "How single-fiberCOP works", "single-fiberCOP · implementation", '''  <header class="sh">
    <p class="eyebrow">%%EYEBROW%%</p>
    <h2>How single-fiberCOP works</h2>
    <p class="msg">The same RoboCOP model, run on one Fiber-seq read at a time.</p>
  </header>
  <div class="body stack" style="gap:26px">
    <ol class="flow">
      <li><span class="fn">1</span><h3>Load the HMM</h3>
        <p>RoboCOP log-space decoder, <b>3,485 states</b>: background, 154 TFs × forward + reverse, and a 531-state nucleosome.</p></li>
      <li><span class="fn">2</span><h3>Read each read from the BAM</h3>
        <p>m6A probability per A, called <b>methylated</b> (p ≥ 0.81), <b>unmethylated</b> (p ≤ 0.19) or <b>no-call</b>.</p>
        <p class="fs">0.81 = modkit's automatic threshold (ML ≥ 207 of 255).</p>
        <p class="fs">Forward read: Watson A's. Reverse read: Crick A's (stored as T).</p></li>
      <li><span class="fn">3</span><h3>Place calls on the reference</h3>
        <p>Each call goes to its sacCer3 coordinate. The sequence layer uses the reference genome.</p></li>
      <li><span class="fn">4</span><h3>Emission: score each base, each state</h3>
        <p class="fx">log PWM(base)<br>+ log p(m6A) if methylated<br>+ log(1−p) if unmethylated</p>
        <p class="fs">One Bernoulli per A. No-calls add 0.</p></li>
      <li><span class="fn">5</span><h3>Forward–backward, in log space</h3>
        <p>Gives a per-base posterior for every state, on that one read.</p></li>
    </ol>
    <div class="cols" style="grid-template-columns:1fr 1fr;gap:28px">
      <div class="card flag" style="border-left-color:var(--both)"><h3><span class="ok">Works</span> · nucleosomes per read</h3>
        <p>Chereji ±1 dyads: <b>4 of 4</b> within 20 bp.</p></div>
      <div class="card flag" style="border-left-color:var(--pending-line)"><h3><span class="bad">Does not work</span> · TF calls per read</h3>
        <p>ABF1's prior is 1.8&nbsp;×&nbsp;10<sup>-7</sup> per bp, and one read gives about <b>5 informative A's</b> in the 14 bp motif.</p></div>
    </div>
  </div>
'''))

slides.append(sec2("F2", 4, "Five problems we found", "single-fiberCOP · what we checked", '''  <header class="sh">
    <p class="eyebrow">%%EYEBROW%%</p>
    <h2>Five problems we found</h2>
  </header>
  <div class="body stack" style="gap:12px">
    <div class="tbl-wrap"><table class="tbl probs" style="font-size:23px">
      <colgroup><col style="width:19%"><col style="width:25%"><col style="width:30%"><col style="width:26%"></colgroup>
      <thead><tr><th>Problem</th><th>What it is</th><th>What we found</th><th>Fix</th></tr></thead>
      <tbody>
        <tr><td class="pn"><span>1</span>Decoding direction</td>
          <td>Crick reads were decoded 3'→5'.</td>
          <td>Each read can now be decoded 5'→3' on its own strand.</td>
          <td><span class="ok">Done:</span> new driver.</td></tr>
        <tr><td class="pn"><span>2</span>Same answer both ways</td>
          <td>One read should give the same answer in either direction.</td>
          <td>"Start anywhere" fixes the read ends, but a gap of about 0.2 remains.</td>
          <td>The gap comes from the sequence layer → fix 3.</td></tr>
        <tr><td class="pn"><span>3</span>Nucleosome model asymmetry</td>
          <td>It scores a sequence and its reverse complement differently.</td>
          <td>It was trained on one strand only, and its m6A profile is 1 bp off centre.</td>
          <td><span class="warnc">Planned:</span> retrain on both strands; refit m6A 1 bp left.</td></tr>
        <tr><td class="pn"><span>4</span>Reference vs read sequence</td>
          <td>Should each read use its own bases?</td>
          <td>Read differences are mostly nanopore errors, not real variants.</td>
          <td><span class="ok">Use the reference</span> unless ≥ 10 reads and &gt; 50% agree on another base.</td></tr>
        <tr><td class="pn"><span>5</span>TF start chance too low</td>
          <td>ABF1 is entered with a chance of 1.8&nbsp;×&nbsp;10<sup>-7</sup> per bp.</td>
          <td>Even a perfectly protected fake read only reaches ABF1 ≈ 5&nbsp;×&nbsp;10<sup>-6</sup>.</td>
          <td><span class="warnc">Next:</span> a per-site prior (fraction of reads bound) instead.</td></tr>
      </tbody>
    </table></div>
    <p class="foot txt" style="font-size:15px">Details on hidden slides H5 (numbers, incl. the fake-read test) and H6 (the 1 bp offset). Code in <span class="m">analysis/singlefiber/</span>.</p>
  </div>
'''))

slides.append(sec2("HF2", "H5", "Five problems we found (detail)", "single-fiberCOP · detail", '''  <header class="sh">
    <p class="eyebrow">%%EYEBROW%%</p>
    <h2>Five problems we found</h2>
  </header>
  <div class="body stack" style="gap:12px">
    <div class="tbl-wrap"><table class="tbl compact probs" style="font-size:17.5px">
      <colgroup><col style="width:14%%"><col style="width:22%%"><col style="width:45%%"><col style="width:19%%"></colgroup>
      <thead><tr><th>Problem</th><th>What it is</th><th>What we found</th><th>Status / fix</th></tr></thead>
      <tbody>
        <tr><td class="pn"><span>1</span>Decoding direction</td>
          <td>Every read was decoded left→right on the reference, so Crick reads ran 3'→5'.</td>
          <td>New driver decodes each read 5'→3' on its own strand: Crick reads are reverse-complemented, decoded, then mapped back.
            <span class="m">--direction ref</span> reproduces the old driver exactly (0 of 44 outputs differ).</td>
          <td><span class="ok">Done.</span> <span class="m">run_permol_window_logfull_5to3.py</span></td></tr>
        <tr><td class="pn"><span>2</span>Same answer both ways</td>
          <td>One read should give the same answer decoded in either direction.</td>
          <td>100 reads at ERV46, largest difference per read (median / max):
            today's start rule <b>%(tm)s / %(tx)s</b>; start anywhere <b>%(am)s / %(ax)s</b>; sequence layer off <b>%(off)s</b> (exact).
            Start anywhere fixes the read ends. The rest comes from the sequence layer, most likely the nucleosome model (not yet isolated).</td>
          <td><span class="ok">Built</span> (<span class="m">--start any</span>). Remaining gap → problem 3.</td></tr>
        <tr><td class="pn"><span>3</span>Nucleosome model is not symmetric</td>
          <td>It scores a sequence and its reverse complement differently: 0.40 log10 on average.</td>
          <td>Its sequence part was trained on a random, unseeded draw of 5,000 of 9,271 Chereji dyads, Watson strand only
            (<span class="m">calc_dinucleotide.py:135, 147-149</span>). All 9,271 + reverse complements cuts it to <b>0.04</b>.
            Also, the m6A profile was fit 1 bp off centre (next slides).</td>
          <td><span class="warnc">Planned:</span> symmetric retrain + recentred m6A fit.</td></tr>
        <tr><td class="pn"><span>4</span>Reference or read sequence?</td>
          <td>Should the sequence layer use each read's own bases?</td>
          <td>Read differences are mostly nanopore errors, not variants: 0 positions at ERV46 where ≥ 50%% of reads disagree; strand-specific hotspots;
            a systematic G error in "CGA". Using read sequence moved nucleosome calls by up to 0.65; one error in a motif cut ABF1 100–1000×.</td>
          <td><span class="ok">Use the reference</span>, unless ≥ 10 reads cover a base and &gt; 50%% agree on another (<span class="m">consensus_seq.py</span>; 0 changes at ERV46).</td></tr>
        <tr><td class="pn"><span>5</span>TF start chance too low</td>
          <td>ABF1's start chance is 1.8&nbsp;×&nbsp;10<sup>-7</sup> per bp; one read has only 4–5 informative A's in the motif.</td>
          <td>Fake reads at ERV46 site 1, real decoder: no calls 3.3&nbsp;×&nbsp;10<sup>-6</sup>; motif fully protected <b>4.9&nbsp;×&nbsp;10<sup>-6</sup></b> (best);
            motif methylated 1.2&nbsp;×&nbsp;10<sup>-7</sup>. 0.5 needs a ~10<sup>5</sup>× higher start chance.</td>
          <td><span class="warnc">Next:</span> per-site mixture model; fraction bound f from all reads replaces the 1.8&nbsp;×&nbsp;10<sup>-7</sup>.</td></tr>
      </tbody>
    </table></div>
    <p class="foot txt" style="font-size:15px">ERV46 window: chrI:60,001–65,000. Code in <span class="m">analysis/singlefiber/</span>.
    Problem 2 numbers: <span class="m">permol_proto_logfull/dirtest100/summary.json</span> (next slide). Problem 5: fake-read test, 26 Oct (its scratch files were lost in a /tmp wipe; numbers recorded in this session, rerun to regenerate).</p>
  </div>
''' % dict(tm=f2(TODAY_MED), tx=f2(TODAY_MAX), am=f2(ANY_MED), ax=f2(ANY_MAX), off=sup(OFF_MAX)), kind="hidden"))

slides.append(sec2("F3", 5, "Same read, both directions", "single-fiberCOP · problem 2 in numbers", '''  <header class="sh">
    <p class="eyebrow">%%EYEBROW%%</p>
    <h2>Same read, both directions</h2>
    <p class="msg">Each read decoded left→right and right→left. Bars count reads by their largest posterior difference; ▼ marks the median.</p>
  </header>
  <div class="body cols" style="grid-template-columns:1.75fr 1fr;gap:44px;align-items:start">
    <figure>%(svg)s</figure>
    <div class="stack" style="gap:18px">
      <div class="tbl-wrap"><table class="tbl compact" style="font-size:21px">
        <thead><tr><th>Setting</th><th class="r">Median</th><th class="r">Max</th></tr></thead>
        <tbody>
          <tr><td><span class="sw a"></span>Today's start rule</td><td class="n r">%(tm)s</td><td class="n r">%(tx)s</td></tr>
          <tr class="key"><td><span class="sw b"></span>Start anywhere</td><td class="n r">%(am)s</td><td class="n r">%(ax)s</td></tr>
          <tr><td><span class="sw" style="background:var(--both)"></span>Sequence layer off</td><td class="n r">%(off)s</td><td class="n r">%(off)s</td></tr>
        </tbody>
      </table></div>
      <ul class="pts" style="font-size:22px;gap:12px">
        <li>Start anywhere: each state starts at its long-run share, matching "end anywhere".</li>
        <li>With the sequence layer off, both directions agree to float32 precision.</li>
        <li>ABF1 at the two sites moves at most %(abf)s (start anywhere).</li>
      </ul>
      <p class="foot txt" style="font-size:15px">A symmetric m6A profile with start anywhere gives %(sm)s / %(sx)s: the m6A profile is not the main cause.
      Source: <span class="m">/usr/project/xtmp/nd141/permol_proto_logfull/dirtest100/</span> (<span class="m">per_read.tsv</span>, <span class="m">summary.json</span>);
      difference = max over positions and all state columns.</p>
    </div>
  </div>
''' % dict(svg=hist_svg(), tm=f2(TODAY_MED), tx=f2(TODAY_MAX), am=f2(ANY_MED), ax=f2(ANY_MAX), off=sup(OFF_MAX),
           abf=sup(ABF_ANY), sm=f2(SYM_MED), sx=f2(SYM_MAX))))

slides.append(sec2("F4", "H6", "The m6A nucleosome profile is 1 bp off", "single-fiberCOP · problem 3 in detail", '''  <header class="sh">
    <p class="eyebrow">%%EYEBROW%%</p>
    <h2>The m6A nucleosome profile is 1 bp off</h2>
    <p class="msg">Both nucleosome fits read the same Chereji dyads but put the centre on different bases.</p>
  </header>
  <div class="body stack" style="gap:22px">
    <div class="cols" style="grid-template-columns:1fr 1fr;gap:32px;align-items:start">
      <div class="card"><h3>Two centres for one dyad</h3>
        <p>A Chereji row is 1 bp wide, e.g. <span class="m">chrI 2386 2387</span>.</p>
        <ul class="pts" style="font-size:21px;gap:8px;margin-top:10px">
          <li>Sequence model: centred on base <b>2386</b> (<span class="m">calc_dinucleotide.py:131, 147-148</span>).</li>
          <li>m6A fit: centred on base <b>2387</b> (<span class="m">abf1_reb1_dms_parameter_Fiber-seq_w_binom.py:951-953</span>).</li>
        </ul></div>
      <div class="card"><h3>The m6A fit is the one that is off</h3>
        <p>A nucleosome's Watson m6A profile should mirror its Crick profile about the true centre.</p>
        <div class="tbl-wrap" style="margin-top:10px"><table class="tbl compact" style="font-size:20px">
          <thead><tr><th>Centre</th><th class="r">Positions |z| &gt; 3 (of 147)</th><th class="r">p</th></tr></thead>
          <tbody>
            <tr><td>As fit</td><td class="n r bad">20</td><td class="n r">5&nbsp;×&nbsp;10<sup>-75</sup></td></tr>
            <tr class="key"><td>Shifted 1 bp left</td><td class="n r">1</td><td class="n r">0.12</td></tr>
          </tbody>
        </table></div>
        <p style="font-size:19px;margin-top:8px">The sequence model's centre is the most symmetric of 7 shifts tested.</p></div>
    </div>
    <div>
      <p class="panel-t">Planned fix</p>
      <ul class="pts" style="font-size:21px;gap:8px;margin-top:6px">
        <li>A new copy of the fitting step with the window 1 bp left (<span class="m">dyad-74 … dyad+73</span>), writing a new file
          <span class="m">inputs/nucleosome_params_recentred.pkl</span>. <span class="m">nucleosome_params.pkl</span> is never overwritten.</li>
        <li>Point the per-read code at it: <span class="m">singlefiber/permol_params.py:56</span>, and in a <b>new</b> pkgvar copy of
          <span class="m">permol_seq_maskoff_logfull</span>, <span class="m">robocop.py</span> <span class="m">build_fiber_ps</span> (line 1432) and line 639.</li>
      </ul>
    </div>
    <p class="caveats"><span class="tag caveat">unverified</span> Whether the TF footprint fits share the same 1 bp offset.</p>
  </div>
''', kind="hidden"))

slides.append('''
<!-- ======================= R STRAHL ======================= -->
<section class="slide" data-id="R" data-kind="main" data-sec="3" data-steps="1" data-title="Strahl lab's results">
  <span class="slide-no">6</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span></p>
    <h2>Strahl lab's results</h2>
  </header>
</section>
''')

# timeline: one line from 27 Oct 2026 to 31 Mar 2027, positions by day
import datetime as _dt
T0, T1 = _dt.date(2026, 10, 27), _dt.date(2027, 3, 31)
def pct(d): return 100.0 * (d - T0).days / (T1 - T0).days
months = [(_dt.date(2026, 11, 1), "Nov"), (_dt.date(2026, 12, 1), "Dec"), (_dt.date(2027, 1, 1), "Jan 2027"),
          (_dt.date(2027, 2, 1), "Feb"), (_dt.date(2027, 3, 1), "Mar")]
phases = [(T0, _dt.date(2026, 11, 30), "Finish single-fiberCOP", "now → November", "p1", 0),
          (_dt.date(2026, 12, 1), _dt.date(2027, 1, 31), "Find biology, tweak the model", "December → January", "p2", 1),
          (_dt.date(2027, 1, 1), T1, "Find biology and publish the paper", "January → March", "p3", 2)]
tl = ['<div class="tl" role="img" aria-label="Timeline from late October 2026 to March 2027: finish single-fiberCOP by November; '
      'December to January find biology and tweak the model; January to March find biology and publish the paper; defend in March.">']
for a, b, txt, when, cls, lane in phases:
    tl.append('  <div class="tl-bar %s" style="left:%.2f%%;width:%.2f%%;--lane:%d"><b>%s</b><span>%s</span></div>'
              % (cls, pct(a), pct(b) - pct(a), lane, when, txt))
tl.append('  <div class="tl-line"></div>')
tl.append('  <div class="tl-tick now" style="left:0%"><span>now<br>27 Oct</span></div>')
for d, lab in months:
    tl.append('  <div class="tl-tick" style="left:%.2f%%"><span>%s</span></div>' % (pct(d), lab))
dm = _dt.date(2027, 3, 1)
tl.append('  <div class="tl-defend" style="left:%.2f%%;width:%.2f%%"><b>March</b><span>Defend</span></div>' % (pct(dm), 100 - pct(dm)))
tl.append('</div>')
slides.append('''
<!-- ======================= TL TIMELINE ======================= -->
<section class="slide" data-id="TL" data-kind="main" data-sec="4" data-steps="1" data-title="Timeline">
  <span class="slide-no">7</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 4</span> Plan</p>
    <h2>Timeline</h2>
    <p class="msg">From now to the defence in March 2027.</p>
  </header>
  <div class="body" style="display:flex;flex-direction:column;justify-content:center">
''' + "\n".join(tl) + '''
  </div>
</section>
''')

out = "".join(slides)
open(os.path.join(H, "slides_fragment.html"), "w", encoding="utf-8").write(out)
print("wrote slides_fragment.html", len(out), "sections", out.count("<section"))
