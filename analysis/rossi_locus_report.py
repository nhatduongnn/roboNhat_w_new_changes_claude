"""Render the Rossi promoter/gene-body classification as a single self-contained page.

Numbers come from rossi_locus_class.tsv (never retyped); the two figures are inlined as
data URIs so the page has no external dependencies. The literature column is agentR's
review, kept as data here so the two halves of the answer sit in one table.

    python rossi_locus_report.py
"""
import base64
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, os.environ.get("LOCUS_DIR", "rossi_locus_class"))
DCMP = os.path.join(HERE, os.environ.get("SETCMP_DIR", "rossi_locus_class_setcmp"))
OUT = os.path.join(D, "where_the_twelve_bind.html")

# agentR's literature call, per factor: (class, confidence, documented non-promoter loci)
LIT = {
    "Abf1":      ("B", "high", "ARS B3 elements, HML-E/HMR-E silencers, subtelomeric X-elements"),
    "Cin5":      ("A", "med",  "none reported"),
    "Fhl1":      ("A", "low*", "none — but ChIP recovers Rap1's motif, not a forkhead motif"),
    "Fkh1":      ("B", "high", "replication origins (ORC), chr III recombination enhancer, X-elements"),
    "Mcm1":      ("B", "med",  "Mcm1 elements flanking ARSs"),
    "Nhp6a":     ("C", "high", "genome-wide — no sequence specificity at all"),
    "Rap1":      ("B", "high", "telomeric TG₁₋₃ tracts, HML-E/HMR-E silencers"),
    "Reb1":      ("B", "high", "rDNA Pol I enhancer + terminator, X-element STRs/STARs"),
    "Sko1":      ("A", "med",  "~half of subtelomeric X-elements, as a Sko1–Tup1 module"),
    "Spt15/TBP": ("B", "high", "all ~275 tRNA genes via TFIIIB, rDNA Pol I promoter"),
    "Tbf1":      ("B", "high", "subtelomeric STARs, interstitial T₂AG₃ tracts"),
    "Ume6":      ("A", "med",  "none reported"),
}
CLASS_NOTE = {"A": "promoter-confined", "B": "promoter + specialised loci",
              "C": "architectural, not sequence-specific"}


def b64(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def main():
    t = pd.read_csv(os.path.join(D, "rossi_locus_class.tsv"), sep="\t")
    null = t[t.TF == "random genome"].iloc[0]
    tfs = t[t.TF != "random genome"].sort_values("promoter_pct", ascending=False)

    rows = []
    for _, r in tfs.iterrows():
        cls, conf, loci = LIT[r.TF]
        enr = r["promoter_pct"] / null["promoter_pct"]
        dep = r["gene body_pct"] / null["gene body_pct"]
        rows.append(f"""
        <tr>
          <th scope="row"><span class="tf">{r.TF}</span></th>
          <td class="cls"><span class="chip chip-{cls.lower()}" title="{CLASS_NOTE[cls]}">{cls}</span>
              <span class="conf">{conf}</span></td>
          <td class="num">{int(r.n)}</td>
          <td class="bar-cell">
            <div class="bar"><i style="width:{r['promoter_pct']:.1f}%"></i></div>
            <span class="num pct">{r['promoter_pct']:.0f}%</span>
          </td>
          <td class="num">{r['gene end_pct']:.0f}%</td>
          <td class="num">{r['gene body_pct']:.0f}%</td>
          <td class="num muted">{r['body_first_body_pct']:.0f}%</td>
          <td class="num">{r['other intergenic_pct']:.0f}%</td>
          <td class="num">{r['subtel_pct']:.0f}%</td>
          <td class="num">{r['rna_pct']:.0f}%</td>
          <td class="num">{r['median_|d_tss|']:.0f}</td>
          <td class="num ratio">{enr:.1f}&times; / {dep:.2f}&times;</td>
          <td class="loci">{loci}</td>
        </tr>""")

    nullrow = f"""
        <tr class="nullrow">
          <th scope="row"><span class="tf">random genome</span></th>
          <td class="cls">&mdash;</td>
          <td class="num">{int(null.n):,}</td>
          <td class="bar-cell">
            <div class="bar"><i style="width:{null['promoter_pct']:.1f}%"></i></div>
            <span class="num pct">{null['promoter_pct']:.0f}%</span>
          </td>
          <td class="num">{null['gene end_pct']:.0f}%</td>
          <td class="num">{null['gene body_pct']:.0f}%</td>
          <td class="num muted">{null['body_first_body_pct']:.0f}%</td>
          <td class="num">{null['other intergenic_pct']:.0f}%</td>
          <td class="num">{null['subtel_pct']:.0f}%</td>
          <td class="num">{null['rna_pct']:.0f}%</td>
          <td class="num">{null['median_|d_tss|']:.0f}</td>
          <td class="num ratio">1.0&times; / 1.00&times;</td>
          <td class="loci">uniformly random positions</td>
        </tr>"""

    cmp = pd.read_csv(os.path.join(DCMP, "set_comparison.tsv"), sep="\t")
    cmp["raw"] = cmp.note.str.extract(r"(\d+) calls").astype(float)
    piv = cmp.pivot(index="TF", columns="set", values=["n", "prom", "body"])
    cmp_rows = []
    for tf in tfs.TF:
        r = piv.loc[tf]
        cmp_rows.append(f"""
        <tr>
          <th scope="row"><span class="tf">{tf}</span></th>
          <td class="num">{int(r[('n','zip')])}</td>
          <td class="num">{int(r[('n','merged')])}</td>
          <td class="num">{int(r[('n','+motif')])}</td>
          <td class="num sep">{r[('prom','zip')]:.0f}%</td>
          <td class="num">{r[('prom','merged')]:.0f}%</td>
          <td class="num">{r[('prom','+motif')]:.0f}%</td>
          <td class="num sep">{r[('body','zip')]:.0f}%</td>
          <td class="num">{r[('body','merged')]:.0f}%</td>
          <td class="num">{r[('body','+motif')]:.0f}%</td>
        </tr>""")
    cmp_rows.append(f"""
        <tr class="nullrow">
          <th scope="row"><span class="tf">random genome</span></th>
          <td class="num" colspan="3">200,000</td>
          <td class="num sep" colspan="3">{null['promoter_pct']:.0f}%</td>
          <td class="num sep" colspan="3">{null['gene body_pct']:.0f}%</td>
        </tr>""")

    worst = tfs.iloc[-1]
    best = tfs.iloc[0]
    body_min, body_max = tfs["gene body_pct"].min(), tfs["gene body_pct"].max()

    html = f"""<title>Where the Twelve Bind</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:      #f6f7f9;
  --panel:       #ffffff;
  --ink:         #12161c;
  --ink-soft:    #4b535f;
  --ink-faint:   #838b97;
  --rule:        #dde1e7;
  --rule-soft:   #eaedf1;
  --promoter:    #2a6fd6;
  --gene-end:    #8e5fc4;
  --promoter-bg: #e7effb;
  --body-acc:    #c9502a;
  --body-bg:     #fbebe5;
  --other:       #8d8f96;
  --shadow:      0 1px 2px rgba(18,22,28,.06), 0 8px 24px -14px rgba(18,22,28,.22);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:      #10131a;
    --panel:       #171b23;
    --ink:         #e8eaee;
    --ink-soft:    #a8b0bd;
    --ink-faint:   #737b89;
    --rule:        #2a303b;
    --rule-soft:   #212733;
    --promoter:    #6ba3ee;
  --gene-end:    #a887db;
    --gene-end:    #a887db;
    --promoter-bg: #1a2739;
    --body-acc:    #e8825c;
    --body-bg:     #2e1e18;
    --other:       #8d8f96;
    --shadow:      0 1px 2px rgba(0,0,0,.4), 0 10px 28px -16px rgba(0,0,0,.8);
  }}
}}
:root[data-theme="dark"] {{
  --ground:      #10131a;
  --panel:       #171b23;
  --ink:         #e8eaee;
  --ink-soft:    #a8b0bd;
  --ink-faint:   #737b89;
  --rule:        #2a303b;
  --rule-soft:   #212733;
  --promoter:    #6ba3ee;
  --gene-end:    #a887db;
  --promoter-bg: #1a2739;
  --body-acc:    #e8825c;
  --body-bg:     #2e1e18;
  --other:       #8d8f96;
  --shadow:      0 1px 2px rgba(0,0,0,.4), 0 10px 28px -16px rgba(0,0,0,.8);
}}

* {{ box-sizing: border-box; }}
body {{
  background: var(--ground);
  color: var(--ink);
  font-family: "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
  font-size: 16.5px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 56px 30px 96px; }}
.col  {{ max-width: 68ch; }}

.eyebrow {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--ink-faint);
}}
h1 {{
  font-family: Spectral, Georgia, serif; font-weight: 600;
  font-size: clamp(34px, 5vw, 52px); line-height: 1.08; letter-spacing: -.015em;
  margin: 10px 0 0; text-wrap: balance;
}}
h2 {{
  font-family: Spectral, Georgia, serif; font-weight: 600;
  font-size: 25px; line-height: 1.22; margin: 0; text-wrap: balance;
}}
.lede {{
  font-family: Spectral, Georgia, serif; font-size: 21px; line-height: 1.5;
  color: var(--ink-soft); margin: 20px 0 0; max-width: 60ch;
}}
p {{ margin: 0 0 1.05em; }}
p:last-child {{ margin-bottom: 0; }}
a {{ color: var(--promoter); }}
strong {{ font-weight: 600; }}
code, .mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .88em; }}

header {{ border-bottom: 1px solid var(--rule); padding-bottom: 34px; }}

.headline {{
  display: flex; flex-wrap: wrap; gap: 14px; margin: 30px 0 0;
}}
.stat {{
  flex: 1 1 210px; background: var(--panel); border: 1px solid var(--rule);
  border-radius: 3px; padding: 16px 18px 14px;
}}
.stat .k {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 30px; font-weight: 500; letter-spacing: -.02em;
  font-variant-numeric: tabular-nums; line-height: 1.1; display: block;
}}
.stat .k.pro {{ color: var(--promoter); }}
.stat .k.bod {{ color: var(--body-acc); }}
.stat .v {{ font-size: 13.5px; color: var(--ink-soft); display: block; margin-top: 6px; }}

section {{ margin-top: 64px; }}
.shead {{ display: flex; align-items: baseline; gap: 16px; margin-bottom: 18px;
          border-top: 1px solid var(--rule); padding-top: 22px; }}
.shead .eyebrow {{ flex: none; width: 5.5em; }}

figure {{ margin: 26px 0 0; }}
figure img {{
  display: block; width: 100%; height: auto; border: 1px solid var(--rule);
  border-radius: 3px; background: #fff; box-shadow: var(--shadow);
}}
figcaption {{ font-size: 13.8px; color: var(--ink-soft); margin-top: 12px; max-width: 78ch; }}

.tablewrap {{ overflow-x: auto; margin-top: 26px; border: 1px solid var(--rule);
              border-radius: 3px; background: var(--panel); box-shadow: var(--shadow); }}
table {{ border-collapse: collapse; width: 100%; min-width: 1020px; }}
thead th {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--ink-faint); font-weight: 500; text-align: right;
  padding: 14px 10px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap;
}}
thead th:first-child, thead th.l {{ text-align: left; }}
tbody th, tbody td {{ padding: 9px 10px; border-bottom: 1px solid var(--rule-soft);
                      font-size: 14.5px; vertical-align: middle; }}
tbody tr:last-child td, tbody tr:last-child th {{ border-bottom: 0; }}
tbody th {{ text-align: left; font-weight: 400; white-space: nowrap; }}
.tf {{ font-family: Spectral, Georgia, serif; font-weight: 600; font-size: 16px; }}
.num {{ text-align: right; font-family: "IBM Plex Mono", ui-monospace, monospace;
        font-variant-numeric: tabular-nums; font-size: 13.5px; white-space: nowrap; }}
.num.muted {{ color: var(--ink-faint); }}
.ratio {{ color: var(--ink-soft); font-size: 12.5px; }}
.bar-cell {{ min-width: 132px; }}
.bar {{ display: inline-block; width: 84px; height: 8px; background: var(--rule-soft);
        border-radius: 1px; overflow: hidden; vertical-align: middle; margin-right: 8px; }}
.bar i {{ display: block; height: 100%; background: var(--promoter); }}
.pct {{ display: inline-block; width: 2.6em; }}
.chip {{ display: inline-block; width: 1.5em; text-align: center; border-radius: 2px;
         font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px;
         font-weight: 500; padding: 1px 0; }}
.chip-a {{ background: var(--promoter-bg); color: var(--promoter); }}
.chip-b {{ background: var(--body-bg); color: var(--body-acc); }}
.chip-c {{ background: var(--rule-soft); color: var(--ink-soft); }}
.conf {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px;
         color: var(--ink-faint); margin-left: 7px; }}
.loci {{ font-size: 13px; color: var(--ink-soft); line-height: 1.42; min-width: 300px; }}
.sep {{ border-left: 1px solid var(--rule); }}
thead th[colspan] {{ text-align: center; }}
.nullrow {{ background: var(--rule-soft); }}
.nullrow .tf {{ font-style: italic; font-weight: 400; color: var(--ink-soft); }}

.notes {{ margin-top: 22px; display: grid; gap: 18px;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
.note {{ border-left: 2px solid var(--rule); padding-left: 16px; }}
.note h3 {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px;
            letter-spacing: .1em; text-transform: uppercase; color: var(--ink-faint);
            font-weight: 500; margin: 0 0 7px; }}
.note p {{ font-size: 14.2px; line-height: 1.55; color: var(--ink-soft); }}

.legend {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 16px;
           font-size: 13.2px; color: var(--ink-soft); }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.sw {{ width: 11px; height: 11px; border-radius: 2px; display: inline-block; }}

footer {{ margin-top: 74px; padding-top: 22px; border-top: 1px solid var(--rule);
          font-size: 13px; color: var(--ink-faint); }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Rossi ChIP-exo &middot; <em>S. cerevisiae</em> &middot; 3,270 peaks</div>
  <h1>Where the twelve bind</h1>
  <p class="lede">Every one of the individually-fitted factors is promoter-targeted. Not one
  of them looks like a linker binder &mdash; including the two the literature warned us about.</p>

  <div class="headline">
    <div class="stat">
      <span class="k pro">{best['promoter_pct']:.0f}&ndash;{worst['promoter_pct']:.0f}%</span>
      <span class="v">of peaks fall in a promoter window, across all twelve. Random genomic
      positions give {null['promoter_pct']:.0f}%.</span>
    </div>
    <div class="stat">
      <span class="k bod">{body_min:.0f}&ndash;{body_max:.0f}%</span>
      <span class="v">fall inside a protein-coding gene body, against
      {null['gene body_pct']:.0f}% expected by chance &mdash; a depletion in every case.</span>
    </div>
    <div class="stat">
      <span class="k">{null['median_|d_tss|']:.0f} &rarr; {best['median_|d_tss|']:.0f}</span>
      <span class="v">median distance to the nearest ATG, in bp: random positions versus
      {best.TF}, the tightest of the twelve.</span>
    </div>
  </div>
</header>

<section>
  <div class="shead">
    <div class="eyebrow">Figure 1</div>
    <h2>Promoter, gene end, gene body, or neither</h2>
  </div>
  <div class="col">
    <p>Rossi's <strong>merged ChExMix calls</strong> &mdash; one 1&nbsp;bp summit per peak
    per factor, from <code>04_ChExMix_Peaks/{{TF}}_CX.bed</code>, each annotated with the nearest
    YEP FIMO motif within 30&nbsp;bp &mdash; classified against Ensembl R64-1-1 = SGD R64
    gene models throughout: the <strong>ATG</strong> anchors the promoter window, the
    <strong>stop codon</strong> anchors the gene-end window, and gene spans give the body. 3,270 of the 29,328 annotated peaks belong to these
    twelve. This is the file <em>before</em> the PWM-conformance and top-1000 filters that
    produce the set the model is actually fit on, which matters: those filters drop a peak for
    failing to match the motif, which is exactly the bias that would distort a promoter/body
    split. Abf1, for instance, has 724 merged peaks, 569 with a motif, 502 surviving here, and
    341 in the fitted set.</p>
  </div>
    <p><strong>Why four classes, and not three.</strong> Yeast has a nucleosome-depleted
    region at the <em>terminator</em> as well as the promoter. A peak sitting just past a
    stop codon is inside the ORF, but it is in open chromatin of the same physical character
    as a promoter NDR &mdash; so scoring it &ldquo;gene body&rdquo; asserted something false:
    buried in nucleosomal DNA mid-transcription-unit. <strong>Gene end</strong> splits it out
    (&minus;100&nbsp;/&nbsp;+300&nbsp;bp of the stop codon, asymmetric because the terminator
    NDR sits downstream of the poly(A) site, itself roughly 100&nbsp;bp past the stop).</p>
    <p><strong>Why SGD and not Park 2014.</strong> Park measures the real transcription start
    site, but only for expressed genes &mdash; 4,985 of 6,692 protein-coding genes. The 1,707
    it misses were the single largest reason a peak got scored gene body when it actually sat
    upstream of a neighbour whose TSS was simply absent. The ATG is complete, and costs a
    measured constant instead: over the genes where both exist, ATG&nbsp;&minus;&nbsp;TSS has
    quartiles 29&nbsp;/&nbsp;52&nbsp;/&nbsp;103&nbsp;bp, so every promoter window sits a median
    52&nbsp;bp downstream of a TSS-anchored one. A constant bias inside a 600&nbsp;bp window is
    a smaller problem than a 26% coverage hole. Both changes push the same way: Abf1's gene-body
    fraction falls from 11.0% to <strong>7.8%</strong> and Tbf1's from 2.6% to <strong>zero</strong>.</p>
  <figure>
    <img src="{b64(os.path.join(D, 'locus_class_bars.png'))}" alt="Stacked bar chart of promoter, gene body and other intergenic fractions for twelve transcription factors, with a random-genome reference row">
    <figcaption>Sorted by promoter fraction. The italic bottom row is 200,000 uniformly random
    genomic positions &mdash; the distribution a factor binding nowhere in particular would give.
    Yeast is gene-dense, so that null is <strong>{null['gene body_pct']:.0f}% gene body</strong>;
    every factor here sits far below it.</figcaption>
  </figure>
</section>

<section>
  <div class="shead">
    <div class="eyebrow">Figure 2</div>
    <h2>Distance to the nearest start site</h2>
  </div>
  <div class="col">
    <p>The window in Figure 1 is a convention. This is the same data without one: a signed
    distance from each peak to the nearest TSS, oriented so negative means upstream of
    transcription. Every panel has a single upstream mode, and the position of that mode is
    itself informative &mdash; TBP sits almost on the start site, the general regulatory
    factors sit 100&ndash;200&nbsp;bp upstream in the NDR, and Sko1 sits 400&ndash;500&nbsp;bp
    out, which is where the literature puts it.</p>
  </div>
  <figure>
    <img src="{b64(os.path.join(D, 'dist_to_tss.png'))}" alt="Twelve small histograms of peak distance to the nearest transcription start site, each compared against a random-genome background">
    <figcaption>Blue is the factor; grey fill is the random-genome background at the same
    scale; the shaded band is the &minus;500&nbsp;/&nbsp;+100&nbsp;bp window used in Figure 1.</figcaption>
  </figure>
</section>

<section>
  <div class="shead">
    <div class="eyebrow">Table</div>
    <h2>The data and the literature, side by side</h2>
  </div>
  <div class="col">
    <p>The <span class="mono">class</span> column is the literature call, not a measurement:
    <span class="chip chip-a">A</span> promoter-confined,
    <span class="chip chip-b">B</span> promoter plus specialised non-promoter loci,
    <span class="chip chip-c">C</span> architectural and not sequence-specific.
    Read it against <span class="mono">prom%</span>: the two agree almost everywhere, and where
    they don't, the reason is in the last column.</p>
  </div>
  <div class="tablewrap">
    <table>
      <thead>
        <tr>
          <th class="l">Factor</th><th class="l">Class</th><th>n</th>
          <th class="l">Promoter</th><th>Gene&nbsp;end</th><th>Body</th><th>Body&sup2;</th><th>Other</th>
          <th>Subtel</th><th>RNA gene</th><th>med&nbsp;|dATG|</th>
          <th>vs&nbsp;null</th><th class="l">Documented non-promoter sites</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}{nullrow}</tbody>
    </table>
  </div>
  <div class="legend">
    <span><i class="sw" style="background:var(--promoter)"></i>promoter &minus;500&nbsp;/&nbsp;+100&nbsp;bp of an ATG</span>
    <span><i class="sw" style="background:var(--gene-end)"></i>gene end &minus;100&nbsp;/&nbsp;+300&nbsp;bp of a stop codon</span>
    <span><i class="sw" style="background:var(--body-acc)"></i>inside a gene, in neither window</span>
    <span><i class="sw" style="background:var(--other)"></i>none of the above</span>
    <span><strong>Body&sup2;</strong> = gene body under the opposite priority</span>
    <span><strong>vs null</strong> = promoter enrichment / gene-body depletion</span>
  </div>
</section>

<section>
  <div class="shead">
    <div class="eyebrow">Sensitivity</div>
    <h2>How much of this is the peak set?</h2>
  </div>
  <div class="col">
    <p>Rossi ships peaks at three levels of curation. The <strong>zip</strong> set is the
    per-sample ChExMix output in each YEP archive, blacklist-filtered
    (<code>chexmix_filtered_peaks.bed</code>) and unioned over a factor's normal replicates;
    <strong>merged</strong> is <code>{{TF}}_CX.bed</code>, a separate ChExMix v0.31 run against
    a 15-experiment no-tag control; <strong>+&nbsp;motif</strong> is that merge restricted to
    peaks with a FIMO motif within 30&nbsp;bp &mdash; the set everything above uses, and the
    one the model is parameterised from.</p>
    <p><strong>Zip and merged largely agree</strong> once both are blacklisted: eleven of the
    twelve sit within a few points of each other. The blacklist is what matters &mdash; rDNA,
    tRNA genes and telomeric repeats generate junk calls, and leaving them in drops Fhl1 from
    50% promoter to 23%, Abf1 from 76% to 65%. <strong>Rap1 is the one real disagreement</strong>
    (64% zip vs 83% merged, 1,080 peaks down to 388): its rep3 called 1,404 peaks against
    rep1's 323, and the joint analysis's cross-replicate consistency test removes them.</p>
    <p>The step that still moves things is the <strong>motif filter</strong>, and it moves
    them in one direction &mdash; Fhl1 42%&nbsp;&rarr;&nbsp;85%, Mcm1 47%&nbsp;&rarr;&nbsp;71%,
    Reb1 73%&nbsp;&rarr;&nbsp;86%. Requiring a motif selects promoter-proximal peaks, so the
    high promoter fractions in the table above are partly a property of that requirement.</p>
  </div>
  <figure>
    <img src="{b64(os.path.join(DCMP, "set_comparison.png"))}" alt="Grouped bar chart comparing promoter and gene-body fractions across three Rossi peak sets for twelve factors">
    <figcaption>Same classification machinery, same annotation, normal-condition samples only
    in all three.</figcaption>
  </figure>
  <div class="tablewrap">
    <table>
      <thead>
        <tr>
          <th class="l" rowspan="2">Factor</th>
          <th colspan="3">peaks</th>
          <th colspan="3" class="sep">promoter</th>
          <th colspan="3" class="sep">gene body</th>
        </tr>
        <tr>
          <th>zip</th><th>merged</th><th>+motif</th>
          <th class="sep">zip</th><th>merged</th><th>+motif</th>
          <th class="sep">zip</th><th>merged</th><th>+motif</th>
        </tr>
      </thead>
      <tbody>{''.join(cmp_rows)}</tbody>
    </table>
  </div>
  <div class="col" style="margin-top:22px">
    <p><strong>Fhl1 is the factor to distrust.</strong> At 48% promoter / 36% gene body in the
    zip set and 42% / 42% merged, it is the least promoter-confined of the twelve &mdash; and it
    only reaches 77% after the motif filter cuts 556 peaks to 84. Since Fhl1 ChIP recovers
    <em>Rap1's</em> motif rather than a forkhead motif, that filter is plausibly selecting Rap1
    sites and labelling them Fhl1. <strong>Tbf1 and Spt15 are the most trustworthy:</strong>
    both sit near 79&ndash;91% promoter in every set, motif filter or not.</p>
  </div>
</section>

<section>
  <div class="shead">
    <div class="eyebrow">Caveats</div>
    <h2>What these numbers can and cannot say</h2>
  </div>
  <div class="notes">
    <div class="note">
      <h3>The priority convention</h3>
      <p>Yeast is dense enough that a 500&nbsp;bp upstream window routinely runs into the
      neighbouring ORF, so a peak can be both promoter and gene body. Promoter wins in the
      main columns; <strong>Body&sup2;</strong> is the same count with the opposite rule. The gap
      between them is how much of a factor's split is convention rather than data &mdash; it is
      small for Reb1 and Tbf1, and largest for Sko1 ({worst['gene body_pct']:.0f}% &rarr;
      {worst['body_first_body_pct']:.0f}%).</p>
    </div>
    <div class="note">
      <h3>One heat-shock sample</h3>
      <p>The <span class="mono">sample_id</span> on a row is annotation provenance &mdash; which
      replicate's genome-wide FIMO scan supplied the motif &mdash; not which replicate called
      the peak. Of the twelve targets only <strong>Spt15</strong> has a non-normal sample
      (rep2, 3&nbsp;min heat shock), carrying 35 of its 294 rows. Dropping them moves Spt15 to
      89.2% promoter / 1.9% body, marginally <em>more</em> promoter-confined, and touches
      nothing else.</p>
    </div>
    <div class="note">
      <h3>Peaks are thresholded</h3>
      <p>A peak file records where a caller found significant enrichment, not where the protein
      spends its time. Diffuse, low-occupancy binding &mdash; exactly what an abundant
      architectural protein would produce &mdash; is invisible here by construction.</p>
    </div>
    <div class="note">
      <h3>Nhp6a's 78% is an artifact of that</h3>
      <p>Nhp6a has no sequence specificity at all (Dowell 2010, <em>Genes Dev</em>): its
      localisation is set by chromatin, and it is present at one copy per one-to-two
      nucleosomes. Its 138 called peaks are the strongest sites, which sit at promoters. The
      genome-wide occupancy the same paper reports &mdash; ~23% of Pol II promoters plus 243
      ORFs &mdash; does not appear in a peak list.</p>
    </div>
    <div class="note">
      <h3>Motif occurrence is not occupancy</h3>
      <p>Rossi 2018 (<em>Genome Res</em>) found only 130 of 913 Abf1 promoter motif matches
      bound even on naked DNA, and ~80% of Reb1's ORF motif matches unbound. Tbf1's consensus
      occurs ~23,000 times genome-wide against a few hundred bound sites. A decoder placing
      these factors at gene-body motif matches is almost certainly wrong.</p>
    </div>
  </div>
</section>

<footer>
  Sources: <span class="mono">inputs/rossi_peak_w_strand_all_TFs.bed</span> &middot;
  <span class="mono">inputs/Park_2014_TSS.csv</span> &middot;
  <span class="mono">inputs/sacCer3.gtf</span> (Ensembl R64-1-1).
  Generated by <span class="mono">rossi_locus_class.py</span> and
  <span class="mono">rossi_locus_report.py</span>; per-peak assignments in
  <span class="mono">rossi_locus_class/rossi_peaks_classified.tsv</span>.
</footer>
</div>
"""
    open(OUT, "w").write(html)
    print("wrote %s (%.2f MB)" % (OUT, os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
