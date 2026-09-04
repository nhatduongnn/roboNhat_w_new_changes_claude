"""Render the genic/intergenic report from rossi_genic.py's outputs."""
import base64, os
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rossi_genic")
OUT = os.path.join(D, "where_the_twelve_bind.html")

def b64(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()

t = pd.read_csv(os.path.join(D, "rossi_genic.tsv"), sep="\t")
c = pd.read_csv(os.path.join(D, "set_comparison.tsv"), sep="\t")
null = t[t.TF == "random genome"].iloc[0]
tfs = t[t.TF != "random genome"].sort_values("genic_pct")
lo, hi = tfs.iloc[0], tfs.iloc[-1]

rows = "".join(
    '<tr><th scope="row">{r.TF}</th><td class="n">{r.n}</td><td class="n">{r.genic}</td>'
    '<td class="n">{r.intergenic}</td>'
    '<td class="bar-cell"><div class="bar"><i style="width:{r.genic_pct:.1f}%"></i></div>'
    '<span class="n pct">{r.genic_pct:.1f}%</span></td>'
    '<td class="n">{r.intergenic_pct:.1f}%</td>'
    '<td class="n ratio">{r.genic_vs_null:.2f}&times;</td></tr>'.format(r=r)
    for _, r in tfs.iterrows())

nullrow = (
    '<tr class="nullrow"><th scope="row">random genome</th><td class="n">{n:,}</td>'
    '<td class="n">{g:,}</td><td class="n">{i:,}</td>'
    '<td class="bar-cell"><div class="bar"><i style="width:{gp:.1f}%"></i></div>'
    '<span class="n pct">{gp:.1f}%</span></td>'
    '<td class="n">{ip:.1f}%</td><td class="n ratio">1.00&times;</td></tr>'.format(
        n=int(null.n), g=int(null.genic), i=int(null.intergenic),
        gp=null.genic_pct, ip=null.intergenic_pct))

cmp_rows = "".join(
    '<tr><th scope="row">{r.TF}</th>'
    '<td class="n">{r.zip_n}</td><td class="n em">{r.zip_genic:.1f}%</td>'
    '<td class="n sep">{r.cx_n}</td><td class="n em">{r.cx_genic:.1f}%</td>'
    '<td class="n sep">{r.mot_n}</td><td class="n em">{r.mot_genic:.1f}%</td>'
    '<td class="n drop">{d:+.1f}</td></tr>'.format(r=r, d=r.mot_genic - r.cx_genic)
    for _, r in c.sort_values("cx_genic").iterrows())

html = """<title>Where the Twelve Bind</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Faustina:wght@500;600;700&family=Archivo:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#edeff1; --panel:#fff; --panel-2:#f5f7f8;
  --ink:#14181d; --ink-soft:#4d565f; --ink-faint:#818b95;
  --rule:#d6dbe0; --rule-soft:#e8ecef;
  --genic:#b4531f; --genic-wash:#f7e8de;
  --inter:#2b6ca8; --inter-wash:#e0ebf5;
  --shadow:0 1px 2px rgba(20,24,29,.05),0 10px 28px -18px rgba(20,24,29,.28);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0f1317; --panel:#171c22; --panel-2:#1d242b;
  --ink:#e8ecef; --ink-soft:#a5aeb7; --ink-faint:#767f89;
  --rule:#2b333c; --rule-soft:#212831;
  --genic:#e0834d; --genic-wash:#331e10;
  --inter:#6ba6dd; --inter-wash:#132635;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 12px 32px -20px rgba(0,0,0,.85);
}}
:root[data-theme="dark"]{
  --ground:#0f1317; --panel:#171c22; --panel-2:#1d242b;
  --ink:#e8ecef; --ink-soft:#a5aeb7; --ink-faint:#767f89;
  --rule:#2b333c; --rule-soft:#212831;
  --genic:#e0834d; --genic-wash:#331e10;
  --inter:#6ba6dd; --inter-wash:#132635;
  --shadow:0 1px 2px rgba(0,0,0,.45),0 12px 32px -20px rgba(0,0,0,.85);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:400 16px/1.62 "Archivo","Segoe UI",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:44px 24px 76px}
.col{max-width:680px}
.eyebrow{font:500 11.5px/1 "IBM Plex Mono",monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink-faint);margin-bottom:14px}
h1{font:700 42px/1.06 "Faustina",Georgia,serif;letter-spacing:-.015em;margin:0 0 16px;text-wrap:balance}
.lede{font-size:19px;line-height:1.55;color:var(--ink-soft);margin:0;max-width:660px}
.lede strong{color:var(--ink);font-weight:600}
h2{font:600 26px/1.2 "Faustina",Georgia,serif;margin:0 0 10px;letter-spacing:-.01em;text-wrap:balance}
p{margin:0 0 15px} p:last-child{margin-bottom:0} strong{font-weight:600}
code,.mono{font-family:"IBM Plex Mono",monospace;font-size:.9em}
section{margin-top:48px} .shead{margin-bottom:18px}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
  box-shadow:var(--shadow);padding:22px 20px 18px}
.scroller{overflow-x:auto} figure{margin:0}
figcaption{margin-top:16px;padding-top:14px;border-top:1px solid var(--rule-soft);
  font-size:14.5px;line-height:1.55;color:var(--ink-soft);max-width:780px}
img{display:block;width:100%;height:auto;border-radius:2px}
.rulebox{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--genic);
  border-radius:3px;padding:20px 22px;margin-top:22px}
.rulebox dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:10px 18px;align-items:baseline}
.rulebox dt{font:600 13px "IBM Plex Mono",monospace;letter-spacing:.06em;text-transform:uppercase}
.rulebox dt.g{color:var(--genic)} .rulebox dt.i{color:var(--inter)}
.rulebox dd{margin:0;font-size:15.5px;color:var(--ink-soft)}
pre{margin:18px 0 0;padding:16px 18px;background:var(--panel-2);border:1px solid var(--rule-soft);
  border-radius:3px;overflow-x:auto;font:400 12.5px/1.75 "IBM Plex Mono",monospace;color:var(--ink-soft)}
pre b{color:var(--ink);font-weight:500}
table{border-collapse:collapse;width:100%;font-size:14.5px;font-variant-numeric:tabular-nums}
th,td{padding:9px 11px;text-align:right;border-bottom:1px solid var(--rule-soft)}
thead th{font:500 10.5px "IBM Plex Mono",monospace;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ink-faint);border-bottom:1px solid var(--rule)}
tbody th[scope=row]{text-align:left;font-weight:600;color:var(--ink)}
td.n{font-family:"IBM Plex Mono",monospace}
td.ratio{color:var(--ink-faint)}
td.sep{border-left:1px solid var(--rule-soft)}
td.em{font-weight:600}
td.drop{color:var(--ink-faint)}
tr.nullrow th,tr.nullrow td{border-top:1px solid var(--rule);border-bottom:none;
  color:var(--ink-faint);font-style:italic;padding-top:14px}
.bar-cell{display:flex;align-items:center;gap:10px;justify-content:flex-end;min-width:150px}
.bar{flex:1;height:9px;background:var(--inter-wash);border-radius:2px;overflow:hidden;min-width:66px}
.bar i{display:block;height:100%;background:var(--genic)}
.pct{min-width:46px;text-align:right}
.note{border-left:2px solid var(--rule);padding-left:16px;margin-top:24px;
  font-size:14.5px;color:var(--ink-soft);max-width:680px}
.note strong{color:var(--ink)}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:24px}
@media(max-width:700px){.kpis{grid-template-columns:1fr}}
.kpi{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:16px 18px}
.kpi .k{font:500 30px/1 "IBM Plex Mono",monospace;display:block;margin-bottom:8px}
.kpi .k.g{color:var(--genic)} .kpi .k.i{color:var(--inter)}
.kpi .v{font-size:14px;line-height:1.5;color:var(--ink-soft)}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Rossi ChIP-exo &middot; <em>S. cerevisiae</em> &middot; Ensembl R64-1-1 = SGD R64</div>
  <h1>Where the twelve bind</h1>
  <p class="lede">One question, asked of every reported summit: <strong>is it between a gene&rsquo;s
  ATG and its stop codon?</strong> No windows, no priority order, nothing to arbitrate. All twelve
  factors come out strongly depleted inside genes &mdash; the weakest by a factor of 1.6, the
  strongest by 14.</p>

  <div class="kpis">
    <div class="kpi"><span class="k g">__NULLG__%</span><span class="v">of the genome is inside an
      ORF, and __NULLG2__% of random positions land there. That is the bar every factor has to clear.</span></div>
    <div class="kpi"><span class="k i">__LOG__&ndash;__HIG__%</span><span class="v">genic, across the twelve
      &mdash; __LONAME__ lowest, __HINAME__ highest. Every one below the null.</span></div>
    <div class="kpi"><span class="k">6,692</span><span class="v">protein-coding ORFs, median
      1,086&nbsp;bp, spanning 8.90&nbsp;Mb of the 12.16&nbsp;Mb genome.</span></div>
  </div>
</header>

<section>
  <div class="col shead"><h2>The rule</h2></div>
  <div class="rulebox">
    <dl>
      <dt class="g">genic</dt><dd>the summit lies between some gene&rsquo;s ATG and its stop codon</dd>
      <dt class="i">intergenic</dt><dd>it does not</dd>
    </dl>
  </div>
  <div class="col" style="margin-top:20px">
    <p>Ensembl R64-1-1 is SGD R64, and its <code>gene</code> feature for a protein-coding gene runs
    ATG&nbsp;&rarr;&nbsp;stop codon and nothing else. That is checked on every run rather than
    assumed, across all 6,516 genes carrying both a CDS and a stop codon:</p>
  </div>
  <pre>gene.start - CDS.start                  <b>[-3  0]</b>
gene.end   - CDS.end                    <b>[ 0  3]</b>
+ strand   gene.end   - stop_codon.end  <b>[ 0]</b>
- strand   gene.start - stop_codon.start <b>[ 0]</b></pre>
  <div class="col" style="margin-top:16px">
    <p>The 3&nbsp;bp is only whether the stop codon is counted inside the CDS. No UTRs are
    annotated, so the gene span <em>is</em> the ORF with no interpretation applied.</p>
    <p>Strand never enters into it. &ldquo;Between the ATG and the stop&rdquo; is the same interval
    whether the gene reads left-to-right or right-to-left, so unlike anything anchored on one end
    this needs no direction-of-transcription bookkeeping and cannot be silently mis-registered.</p>
  </div>
</section>

<section>
  <div class="col shead"><div class="eyebrow">Figure</div><h2>Genic or intergenic</h2></div>
  <figure class="panel scroller" style="margin-top:20px">
    <img src="__FIG__" alt="Horizontal stacked bar chart of genic versus intergenic fractions for twelve transcription factors, with a random-genome reference row showing 73 percent genic">
    <figcaption>Sorted by genic fraction. The italic bottom row is 200,000 uniformly random genomic
    positions, and the dashed line marks it. Every factor sits far to the left of that line: the
    yeast genome is __NULLG__% coding, and none of these factors reports anything like __NULLG__%
    of its peaks there.</figcaption>
  </figure>

  <div class="panel scroller" style="margin-top:18px">
    <table>
      <thead><tr><th style="text-align:left">Factor</th><th>peaks</th><th>genic</th>
        <th>intergenic</th><th style="text-align:right">genic share</th><th>intergenic</th>
        <th>vs null</th></tr></thead>
      <tbody>__ROWS____NULLROW__</tbody>
    </table>
  </div>

  <div class="note"><strong>Sko1 is the one to look at twice.</strong> At 45.1% genic it is only
  1.6&times; depleted, where the other eleven run 2.4&times; to 14&times;. Cin5 at 30.3% is second.
  Both are bZIP stress factors, and both were reported earlier as the least promoter-confined of
  the twelve; this is the same fact measured without any window.</div>
</section>

<section>
  <div class="col shead"><div class="eyebrow">Sensitivity</div><h2>Does the peak file change the answer?</h2></div>
  <div class="col">
    <p>Three peak sets for the same twelve factors, all normal-condition. <strong>zip</strong> is the
    per-sample ChExMix calls out of the YEP zips, blacklist applied, unioned over replicates within
    10&nbsp;bp. <strong>merged CX</strong> is Rossi&rsquo;s own cross-replicate release,
    <code>04_ChExMix_Peaks/{TF}_CX.bed</code>. <strong>+motif</strong> restricts that to peaks
    carrying a FIMO motif within 30&nbsp;bp &mdash; the file the model is parameterised from.</p>
  </div>
  <div class="panel scroller" style="margin-top:20px">
    <table>
      <thead><tr><th style="text-align:left" rowspan="2">Factor</th>
        <th colspan="2">zip (blacklisted)</th><th colspan="2" class="sep">merged CX</th>
        <th colspan="2" class="sep">merged + motif</th><th rowspan="2">motif<br>effect</th></tr>
      <tr><th>n</th><th>genic%</th><th class="sep">n</th><th>genic%</th>
        <th class="sep">n</th><th>genic%</th></tr></thead>
      <tbody>__CMP__</tbody>
    </table>
  </div>
  <div class="note"><strong>zip and merged agree closely</strong> for ten of the twelve &mdash;
  within about 3 points. Rap1 is the exception (26.0% vs 10.1%), traced to its replicate 3 calling
  1,404 peaks against replicate 1&rsquo;s 323; the merged run&rsquo;s cross-replicate consistency
  test removes them. <strong>The motif filter is what really moves things</strong>, and always
  downward: Fhl1 53.2&nbsp;&rarr;&nbsp;17.9%, Reb1 21.2&nbsp;&rarr;&nbsp;9.3%, Tbf1
  20.4&nbsp;&rarr;&nbsp;5.2%. Requiring a motif preferentially removes genic peaks, so the
  headline figures above &mdash; which use the +motif set &mdash; are the <em>lowest</em> genic
  fractions of the three, and the least conservative choice for anyone arguing these factors avoid
  gene bodies.</div>
</section>

<section>
  <div class="col shead"><h2>What this does and does not say</h2></div>
  <div class="col">
    <p><strong>It says:</strong> every one of the twelve is depleted inside ORFs relative to what
    chance would give, and the ranking is stable across peak sets. Tbf1, Reb1, Spt15 and Rap1 are
    almost entirely intergenic; Sko1 and Cin5 much less so.</p>
    <p><strong>It does not say</strong> where in the intergenic space a peak sits. Promoters,
    terminators, telomeric tracts and origins are all intergenic, and this rule does not separate
    them. It also says nothing about occupancy: these are reported peak positions, not dwell times,
    and a factor with few genic peaks may still visit gene bodies transiently.</p>
    <p><strong>One caveat on the null.</strong> 73.0% of random positions are genic because yeast is
    gene-dense. That is the right comparison for &ldquo;more or less than chance&rdquo;, but it is
    not a claim about accessible DNA &mdash; most of that 73% is wrapped in nucleosomes, and no
    sequence-specific factor could bind much of it regardless of preference.</p>
  </div>
</section>
</div>
"""

html = (html.replace("__FIG__", b64(os.path.join(D, "genic_bars.png")))
            .replace("__ROWS__", rows).replace("__NULLROW__", nullrow)
            .replace("__CMP__", cmp_rows)
            .replace("__NULLG2__", "%.1f" % null.genic_pct)
            .replace("__NULLG__", "%.0f" % null.genic_pct)
            .replace("__LOG__", "%.0f" % lo.genic_pct).replace("__HIG__", "%.0f" % hi.genic_pct)
            .replace("__LONAME__", lo.TF).replace("__HINAME__", hi.TF))
open(OUT, "w").write(html)
print("wrote %s (%.2f MB)" % (OUT, os.path.getsize(OUT) / 1e6))
