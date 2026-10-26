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
<section class="slide locus6" data-id="{sid}" data-kind="main" data-sec="2" data-steps="1" data-title="{html.escape(title)}">
  <span class="slide-no">{no}</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 2</span> {eyebrow}</p>
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
  <div class="body stack">
    <ul class="pts" style="font-size:34px;gap:28px">
      <li>What changed in the decoder</li>
      <li>Four loci, six rows each: every model before and after</li>
      <li>Where the two disagree, by model</li>
    </ul>
    <p class="foot txt">Details are on hidden slides (press <span class="m">h</span>): the C1 bug, the DBL_MIN padding,
    the log-space review, and the guards that no longer fire.</p>
  </div>
</section>
''')
# ------------------------------------------------------------------ what changed
G = {}   # guards table numbers are quoted from things_to_revisit_before_shipping.md section 8
slides.append('''
<!-- ======================= W WHAT CHANGED ======================= -->
<section class="slide" data-id="W" data-kind="main" data-sec="1" data-steps="1" data-title="What changed">
  <span class="slide-no">2</span>
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
    "L1", 3, k, "locus-rpc19", "RPC19 promoter, chrXIV",
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
    "L2", 4, k, "locus-xiv440", "chrXIV 440–445 kb: RAS2 to RPS7B",
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
    "L3", 5, k, "locus-xiv724", "chrXIV 724–727 kb: ESF2",
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
    "L4", 6, k, "locus-iv1200", "chrIV 1,200–1,202 kb: ESC2",
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
<section class="slide" data-id="S" data-kind="main" data-sec="3" data-steps="1" data-title="Where before and after differ">
  <span class="slide-no">7</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> Summary</p>
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
<section class="slide" data-id="HC" data-kind="hidden" data-sec="3" data-steps="1" data-title="Log-space decoder review: safe to decode">
  <span class="slide-no">H3</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> Detail · review of the log-space decoder, 23 Oct</p>
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
<section class="slide" data-id="HD" data-kind="hidden" data-sec="3" data-steps="1" data-title="Guards for posterior above 1 no longer fire">
  <span class="slide-no">H4</span>
  <header class="sh">
    <p class="eyebrow"><span class="sec-no">Part 3</span> Detail · downstream guards</p>
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
out = "".join(slides)
open(os.path.join(H, "slides_fragment.html"), "w", encoding="utf-8").write(out)
print("wrote slides_fragment.html", len(out), "sections", out.count("<section"))
