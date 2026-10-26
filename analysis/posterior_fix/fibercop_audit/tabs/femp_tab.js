/* ================= fiberCOP: empirical results (clone of the EMP tab on FEMP / FDEP / FREAL) ================= */
const FEMP = __FEMP__;
const FDEP = __FDEP__;
const FREAL = __FREAL__;
const FSEG = __FSEG__;   // real 3,485-state rDNA segment run, or null when realseg/results.json is absent
/* shared helpers for the three fiberCOP tabs */
const FX = (function () {
  const TOL = 1e-9;
  const ME = [["current", "original RoboCOP"], ["fixB", "fix B (production)"], ["fixed", "C1 fix (audit)"], ["logD0", "log space, floored emission"], ["logD1", "log space, exact emission"]];
  const TRUTHS = [["brute", "truth: floored model"], ["brute_exact", "truth: exact model"]];
  const FLOORED = ["current", "fixB", "fixed", "logD0"];
  const SERSTYLE = { current: ["var(--s1)", "circle"], fixB: ["var(--s4)", "triangle"], fixed: ["var(--s2)", "square"], logD0: ["var(--s5)", "ring"], logD1: ["var(--s3)", "diamond"] };
  const nfmt = v => (typeof v === "number" ? v.toLocaleString("en-US") : String(v));
  const bad = e => e === null || e === undefined || e > TOL;
  const sc = key => FEMP.scenarios.find(s => s.key === key);
  const reg = FREAL.regions;
  const G = reg.genome_ex_rdna, R = Object.entries(reg).find(([k]) => k.startsWith("rdna"))[1], X14 = reg.chrXIV;
  const rdnaKey = Object.keys(reg).find(k => k.startsWith("rdna"));
  const rdnaSpan = rdnaKey.replace("rdna_", "").split("_");   // ["chrXII", start, end]
  const mn = FREAL.mnase_reference.reads;
  const mnaseMax = Math.max(...Object.values(mn.chroms).map(c => c.long.adjmax), mn.segments.long.adjmax);   // same rule as the original PI tab
  const series = f => FDEP.series[String(f.toFixed ? f.toFixed(1) : f)] || FDEP.series[String(f)];
  const fails = (rows, key) => rows.filter(r => bad(r[key])).map(r => r.n);
  const worstFloored = (r, suffix) => Math.max(...FLOORED.map(k => (r[k + suffix] === null ? Infinity : r[k + suffix])));
  const onsetExact = rows => { const r = rows.find(r => worstFloored(r, "_exact") > TOL); return r ? r.n : null; };
  const fracKey = f => Object.keys(FDEP.series).find(k => +k === +f);
  const chromUnder = Object.entries(FREAL.per_chrom).filter(([c, v]) => v.underflow_bases > 0).sort((a, b) => b[1].underflow_bases - a[1].underflow_bases);
  const kn08 = r => r.runs["windows_min_n_ge_and_all_kn_ge_0.8"]["1"];   // windows (min n >= 1) whose four k/n are all >= 0.8
  const mark = (shape, x, y, col) => shape === "circle" ? `<circle cx="${x}" cy="${y}" r="4" fill="${col}" stroke="var(--surface)" stroke-width="2"/>`
    : shape === "square" ? `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" rx="1" fill="${col}" stroke="var(--surface)" stroke-width="2"/>`
    : shape === "triangle" ? `<path d="M${x},${y - 5} L${x + 5},${y + 4} L${x - 5},${y + 4} z" fill="${col}" stroke="var(--surface)" stroke-width="1.5"/>`
    : shape === "ring" ? `<circle cx="${x}" cy="${y}" r="3.6" fill="var(--surface)" stroke="${col}" stroke-width="2.2"/>`
    : `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" rx="1" fill="${col}" stroke="var(--surface)" stroke-width="2" transform="rotate(45 ${x} ${y})"/>`;
  const legend = SER => SER.map(s => `<span><svg width="26" height="14" viewBox="0 0 26 14" aria-hidden="true"><line x1="1" x2="25" y1="7" y2="7" stroke="${s.col}" stroke-width="2"${s.dash ? ' stroke-dasharray="5 3"' : ""}/>${mark(s.shape, 13, 7, s.col)}</svg>${s.label}</span>`).join("");
  /* error-vs-coverage chart: log x (reads), log y (largest posterior error), vertical markers for real data */
  function depChart(svgId, tipId, rows, SER, marks, opt) {
    const $ = id => document.getElementById(id);
    const W = 760, L = 70, Rm = 24, Bm = 52, xmin = 1, xmax = 10000, ymin = -16, ymax = 6;
    // marker labels: greedy rows so labels do not overlap
    const X = k => L + (Math.log10(k) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)) * (W - L - Rm);
    const placed = [], rowEnd = [];
    [...marks].sort((a, b) => a.x - b.x).forEach(m => {
      const x = X(m.x), wtxt = 6.2 * m.label.length + 8, anchorEnd = x + wtxt > W - 4;
      const x0 = anchorEnd ? x - wtxt : x, x1 = anchorEnd ? x : x + wtxt;
      let r = 0; while (rowEnd[r] !== undefined && rowEnd[r] > x0) r++;
      rowEnd[r] = x1; placed.push({ ...m, px: x, row: r, anchorEnd });
    });
    const T = 16 + 14 * Math.max(1, rowEnd.length), H = T + 300 + Bm;
    const Y = v => T + (ymax - v) / (ymax - ymin) * (H - T - Bm);
    const ly = e => (e === null || e === undefined ? ymax : Math.max(ymin, Math.min(ymax, Math.log10(Math.max(e, 1e-300)))));
    let g = "";
    [[-15, "1e-15"], [-10, "1e-10"], [-5, "1e-5"], [0, "1"], [5, "1e5"]].forEach(([v, lab]) => { g += `<line x1="${L}" x2="${W - Rm}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end" class="vax">${lab}</text>`; });
    g += `<line x1="${L}" x2="${W - Rm}" y1="${Y(-9)}" y2="${Y(-9)}" stroke="var(--muted)" stroke-dasharray="2 3"/><text x="${L + 4}" y="${Y(-9) - 4}" class="vax">correct below this line (1e-9)</text>`;
    [1, 10, 100, 1000, 10000].forEach(k => { g += `<line x1="${X(k)}" x2="${X(k)}" y1="${H - Bm}" y2="${H - Bm + 5}" stroke="var(--muted)"/><text x="${X(k)}" y="${H - Bm + 18}" text-anchor="middle" class="vax">${k.toLocaleString("en-US")}</text>`; });
    g += `<line x1="${L}" x2="${W - Rm}" y1="${H - Bm}" y2="${H - Bm}" stroke="var(--muted)"/>`;
    g += `<text x="${(L + W - Rm) / 2}" y="${H - 12}" text-anchor="middle" class="vax">${opt.xlabel}</text>`;
    g += `<text x="16" y="${(T + H - Bm) / 2}" text-anchor="middle" class="vax" transform="rotate(-90 16 ${(T + H - Bm) / 2})">${opt.ylabel}</text>`;
    placed.forEach(m => {
      const ylab = 14 + 14 * m.row;
      g += `<line x1="${m.px}" x2="${m.px}" y1="${ylab + 3}" y2="${H - Bm}" stroke="${m.col}" stroke-dasharray="3 3" stroke-width="1.3"/>`;
      g += `<text x="${m.px + (m.anchorEnd ? -3 : 3)}" y="${ylab}" text-anchor="${m.anchorEnd ? "end" : "start"}" class="vlab2" style="fill:${m.col}">${m.label}</text>`;
    });
    SER.forEach(s => {
      const pts = rows.map(r => [X(r.n), Y(ly(s.val(r)))]);
      g += `<polyline points="${pts.map(p => p.join(",")).join(" ")}" fill="none" stroke="${s.col}" stroke-width="2"${s.dash ? ' stroke-dasharray="6 4"' : ""}/>`;
      pts.forEach(([x, y]) => { g += mark(s.shape, x, y, s.col); });
    });
    rows.forEach((r, i) => { g += `<rect class="hit" data-i="${i}" x="${X(r.n) - 7}" y="${T}" width="14" height="${H - T - Bm}" fill="transparent"/>`; });
    const svg = $(svgId);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = g;
    const tip = $(tipId), f = e => (e === null || e === undefined ? "NaN" : fmt(e));
    if (!svg._fwired) {
      svg._fwired = true;
      svg.addEventListener("mousemove", e => {
        const t = e.target.closest ? e.target.closest(".hit") : null;
        if (!t) { tip.hidden = true; return; }
        const cur = svg._fstate, r = cur.rows[+t.dataset.i];
        tip.innerHTML = `<b>${r.n.toLocaleString("en-US")} reads</b>, ${r.k.toLocaleString("en-US")} methylated<br>` + cur.SER.map(s => `${s.label}: ${f(s.val(r))}`).join("<br>");
        tip.hidden = false;
        const box = svg.getBoundingClientRect(), x = (+t.getAttribute("x") + 7) / W * box.width;
        tip.style.left = Math.min(box.width - 260, x + 10) + "px"; tip.style.top = "10px";
      });
      svg.addEventListener("mouseleave", () => { tip.hidden = true; });
    }
    svg._fstate = { rows, SER };
  }
  /* real-data markers for the depth charts */
  const MARKS = [
    { x: mnaseMax, label: `MNase busiest pair ${mnaseMax}`, col: "var(--muted)" },
    { x: G.coverage_n.p50, label: `genome median ${nfmt(G.coverage_n.p50)}`, col: "var(--accent)" },
    { x: X14.coverage_n.max, label: `chrXIV max ${nfmt(X14.coverage_n.max)}`, col: "var(--accent)" },
    { x: G.coverage_n.p99, label: `genome 99th pct ${nfmt(G.coverage_n.p99)}`, col: "var(--accent)" },
    { x: FREAL.per_chrom.chrM.max_n, label: `chrM max ${nfmt(FREAL.per_chrom.chrM.max_n)}`, col: "var(--bad)" },
    { x: R.coverage_n.p50, label: `rDNA median ${nfmt(R.coverage_n.p50)}`, col: "var(--bad)" },
    { x: R.coverage_n.max, label: `rDNA max ${nfmt(R.coverage_n.max)}`, col: "var(--bad)" },
  ];
  return { TOL, ME, TRUTHS, FLOORED, SERSTYLE, nfmt, bad, sc, G, R, X14, rdnaSpan, mnaseMax, fails, worstFloored, onsetExact, fracKey, chromUnder, kn08, mark, legend, depChart, MARKS };
})();

(function () {
  const $ = id => document.getElementById(id);
  const ALL = [...FX.ME, ...FX.TRUTHS];
  const S = FEMP.S, n = FEMP.n, SHN = FD.short;
  const fv = x => (x === null || x === undefined ? "NaN" : fmt(x));
  const differs = (a, b, tolRel = 1e-6, tolAbs = 1e-9) => {
    if (a === null || b === null) return a !== b;
    return Math.abs(a - b) > tolAbs && Math.abs(a - b) > tolRel * Math.max(Math.abs(b), 1e-300);
  };
  let cur = FEMP.scenarios.findIndex(s => s.key === "c1"), truth = "brute_exact";
  if (cur < 0) cur = 0;
  $("feScen").innerHTML = FEMP.scenarios.map((s, i) => `<button type="button" role="tab" data-i="${i}">${s.title}</button>`).join("");
  $("feScen").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { cur = +b.dataset.i; render(); } });
  $("feTruth").innerHTML = FX.TRUTHS.map(([k, label]) => `<button type="button" role="tab" data-k="${k}">compare with ${label.replace("truth: ", "the ")} truth</button>`).join("");
  $("feTruth").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { truth = b.dataset.k; render(); } });
  const llok = (m, b) => m.ll !== null && b.ll !== null && Math.abs(m.ll - b.ll) < 1e-6 * Math.max(1, Math.abs(b.ll));
  function verdict(sc, k) {
    if (k === "brute" || k === "brute_exact") return `<span class="pill ref">reference</span>`;
    const m = sc.methods[k];
    const okE = m.max_err_exact !== null && m.max_err_exact < FX.TOL && llok(m, sc.methods.brute_exact);
    const okF = m.max_err !== null && m.max_err < FX.TOL && llok(m, sc.methods.brute);
    return okE ? `<span class="pill ok">right</span>` : okF ? `<span class="pill fhalf">right for the floored model only</span>` : `<span class="pill no">wrong</span>`;
  }
  function render() {
    const sc = FEMP.scenarios[cur], B = sc.methods[truth], tname = truth === "brute" ? "the floored-model truth" : "the exact truth";
    [...$("feScen").children].forEach((b, i) => b.setAttribute("aria-selected", String(i === cur)));
    [...$("feTruth").children].forEach(b => b.setAttribute("aria-selected", String(b.dataset.k === truth)));
    const nn = t => (sc.scored[t] === "W" ? sc.n_w[t] : sc.n_c[t]), kk = t => (sc.scored[t] === "W" ? sc.k_w[t] : sc.k_c[t]);
    const zero = sc.zero_emission_rows.length ? `<p class="bad" style="margin-top:6px">In ordinary floating point, every state's emission at position${sc.zero_emission_rows.length > 1 ? "s" : ""} ${sc.zero_emission_rows.join(" and ")} is exactly 0.</p>` : "";
    $("feDesc").innerHTML = `<h3 class="sub3">${sc.title}</h3><p>${sc.desc}</p>
      <div class="gridbox"><table class="mini"><thead><tr><th>position</th>${[...sc.seq].map((c, t) => `<th>${t}</th>`).join("")}</tr></thead><tbody>
      <tr><th>base</th>${[...sc.seq].map(c => `<td>${c}</td>`).join("")}</tr>
      <tr><th>strand scored</th>${sc.scored.map(c => `<td>${c === "W" ? "Watson" : "Crick"}</td>`).join("")}</tr>
      <tr><th>reads n</th>${sc.seq.split("").map((c, t) => `<td>${FX.nfmt(nn(t))}</td>`).join("")}</tr>
      <tr><th>methylated k</th>${sc.seq.split("").map((c, t) => `<td>${FX.nfmt(kk(t))}</td>`).join("")}</tr>
      <tr><th>largest emission, exact (log₁₀)</th>${sc.maxemit_log10.map(v => `<td class="${v < -307 ? "bad" : (v < -100 ? "em" : "")}">${v.toFixed(0)}</td>`).join("")}</tr>
      <tr><th>largest emission, floored (log₁₀)</th>${sc.maxemit_log10_floored.map(v => `<td class="${v < -307 ? "bad" : (v < -100 ? "em" : "")}">${v.toFixed(0)}</td>`).join("")}</tr>
      </tbody></table></div>
      <p>${sc.floored_cells ? `<b class="bad">${sc.floored_cells}</b> (position, state) emissions underflowed to 0 and were set to the floor (${fmt(FD.fiber.floor)}).` : "No emission underflowed: the floor is never used."} Brute force adds up all ${sc.npaths} paths.</p>${zero}`;
    // scoreboard: both truths
    let h = `<thead><tr><th>method</th><th>log-likelihood</th><th>largest error vs floored-model truth</th><th>largest error vs exact truth</th><th>emitting rows sum to 1?</th><th>verdict</th></tr></thead><tbody>`;
    ALL.forEach(([k, label]) => {
      const m = sc.methods[k];
      const rs = m.rowsum.every(x => x !== null && Math.abs(x - 1) < 1e-9);
      const ce = (e, self) => self ? `<td class="mono">—</td>` : `<td class="mono${FX.bad(e) ? " bad" : ""}">${e === null ? "NaN" : fmt(e)}</td>`;
      h += `<tr><th>${label}</th><td class="mono">${m.ll === null ? "NaN" : m.ll.toFixed(4)}</td>${ce(m.max_err, k === "brute")}${ce(m.max_err_exact, k === "brute_exact")}
        <td class="mono${rs ? "" : " bad"}">${rs ? "yes" : m.rowsum.map(fv).join(", ")}</td><td>${verdict(sc, k)}</td></tr>`;
    });
    $("feScore").innerHTML = h + "</tbody>";
    $("feGridNote").innerHTML = `Red = emitting cells that differ from ${tname}. The silent columns (CENTRAL, TF-entry) are shown but not compared: the methods fill them with different conventions.`;
    $("feGrids").innerHTML = ALL.map(([k, label]) => {
      const m = sc.methods[k];
      let g = `<div class="egrid"><h4>${label} <span class="muted">· ${verdict(sc, k)}</span></h4><div class="gridbox"><table class="grid"><thead><tr><th class="st">t</th>` +
        [...Array(12).keys()].map(j => `<th class="${j === S ? "sil1" : ""}${j >= S ? " silh" : ""}" title="${FNAMES[j]}">${SHN[j]}</th>`).join("") + `<th class="sil1">row sum</th></tr></thead><tbody>`;
      for (let t = 0; t < n; t++) {
        g += `<tr><th class="st">${t} <span class="muted">${sc.seq[t]}</span></th>`;
        for (let j = 0; j < 12; j++) {
          const v = m.post[t][j], bad = k !== truth && j < S && differs(v, B.post[t][j]);
          g += `<td class="${j === S ? "sil1" : ""}${bad ? " badc" : ""}" title="${FNAMES[j]}, t=${t}${k !== truth ? " · " + tname + ": " + fv(B.post[t][j]) : ""}">${fv(v)}</td>`;
        }
        const rs = m.rowsum[t];
        g += `<td class="sil1 rsum${rs === null || Math.abs(rs - 1) > 1e-9 ? " badc" : ""}">${fv(rs)}</td></tr>`;
      }
      return g + "</tbody></table></div></div>";
    }).join("");
    // downstream occupancy (columns of sum_for_dbf_probs: 0 background, 1 TF1, 3 nucleosome, 5 nuc_start)
    const DC = [[0, "background"], [1, "TF1"], [3, "nucleosome"], [5, "nuc_start"]];
    $("feDbf").innerHTML = ALL.map(([k, label]) => {
      const m = sc.methods[k];
      let g = `<div class="egrid"><h4>${label}</h4><div class="gridbox"><table class="grid"><thead><tr><th class="st">t</th>` + DC.map(([c, nm]) => `<th>${nm}</th>`).join("") + `</tr></thead><tbody>`;
      for (let t = 0; t < n; t++) {
        g += `<tr><th class="st">${t} <span class="muted">${sc.seq[t]}</span></th>`;
        DC.forEach(([c]) => { const v = m.dbf[t][c], bad = k !== truth && differs(v, B.dbf[t][c], 1e-4, 1e-4); g += `<td class="${bad ? "badc" : ""}">${fv(v)}</td>`; });
        g += "</tr>";
      }
      return g + "</tbody></table></div></div>";
    }).join("");
    h = `<thead><tr><th>method</th><th>background</th><th>TF1</th><th>nucleosome</th></tr></thead><tbody>`;
    ALL.forEach(([k, label]) => {
      const m = sc.methods[k];
      h += `<tr><th>${label}</th>` + m.em.map((v, i) => `<td class="mono${k !== truth && differs(v, B.em[i], 1e-4, 1e-4) ? " bad" : ""}">${fv(v)}</td>`).join("") + "</tr>";
    });
    $("feEM").innerHTML = h + "</tbody>";
  }
  render();
})();

/* ----- error vs coverage chart, real-data tables, conclusion ----- */
(function () {
  const $ = id => document.getElementById(id);
  let frac = FDEP.fracs[0], truth = "_exact";
  const f = e => (e === null || e === undefined ? "NaN" : fmt(e));
  $("fdFrac").innerHTML = FDEP.fracs.map(x => `<button type="button" role="tab" data-f="${x}">${Math.round(x * 100)}% methylated</button>`).join("");
  $("fdFrac").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { frac = +b.dataset.f; draw(); } });
  $("fdTruth").innerHTML = [["_exact", "error vs exact truth"], ["", "error vs floored-model truth"]].map(([k, l]) => `<button type="button" role="tab" data-t="${k}">${l}</button>`).join("");
  $("fdTruth").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { truth = b.dataset.t; draw(); } });
  function draw() {
    const rows = FDEP.series[FX.fracKey(frac)];
    [...$("fdFrac").children].forEach(b => b.setAttribute("aria-selected", String(+b.dataset.f === frac)));
    [...$("fdTruth").children].forEach(b => b.setAttribute("aria-selected", String(b.dataset.t === truth)));
    const SER = FX.ME.map(([k, label]) => ({ key: k, label, col: FX.SERSTYLE[k][0], shape: FX.SERSTYLE[k][1], dash: false, val: r => r[k + truth] }));
    FX.depChart("fdChart", "fdTip", rows, SER, FX.MARKS, { xlabel: `reads at each of positions ${FDEP.positions.join(", ")}, ${Math.round(frac * 100)}% methylated (log scale)`, ylabel: `largest posterior error vs ${truth ? "exact" : "floored-model"} truth` });
    $("fdLegend").innerHTML = FX.legend(SER);
    let h = `<thead><tr><th>reads</th><th>methylated</th><th>largest emission, exact (log₁₀)</th><th>floored cells</th>${FX.ME.map(([k, l]) => `<th>${l}</th>`).join("")}</tr></thead><tbody>`;
    rows.forEach(r => { h += `<tr><td>${r.n.toLocaleString("en-US")}</td><td>${r.k.toLocaleString("en-US")}</td><td>${r.maxemit_log10.toFixed(0)}</td><td>${r.floored_cells}</td>` + FX.ME.map(([k]) => `<td class="${FX.bad(r[k + truth]) ? "bad" : ""}">${f(r[k + truth])}</td>`).join("") + `</tr>`; });
    $("fdTable").innerHTML = h + "</tbody>";
    const failF = FX.fails(rows, "current"), on = FX.onsetExact(rows);
    $("fdNote").innerHTML = `${Math.round(frac * 100)}% methylated: against the floored model, original RoboCOP fails at ${failF.length ? failF.map(x => x.toLocaleString("en-US")).join(", ") + " reads" : "no coverage tested"}. Against the exact model, every floored method fails from ${on === null ? "no coverage tested" : on.toLocaleString("en-US") + " reads"} on; log space with exact emission stays at or below ${fmt(Math.max(...rows.map(r => r.logD1_exact)))}.`;
  }
  draw();
  // real-data table: regions x key stats
  const regName = { genome_ex_rdna: "genome outside rDNA", chrXIV: "chrXIV" };
  const rk = Object.keys(FREAL.regions);
  const label = k => regName[k] || `rDNA (${FX.rdnaSpan[0]}:${(+FX.rdnaSpan[1]).toLocaleString("en-US")}–${(+FX.rdnaSpan[2]).toLocaleString("en-US")})`;
  let h = `<thead><tr><th>region</th><th>scored bases</th><th>median reads</th><th>99th pct reads</th><th>max reads</th><th>four-base windows</th><th>windows with all k/n ≥ 0.8</th><th>bases where binom.pmf underflows</th></tr></thead><tbody>`;
  rk.forEach(k => {
    const r = FREAL.regions[k], c = r.coverage_n, w = FX.kn08(r);
    h += `<tr><th>${label(k)}</th><td>${FX.nfmt(r.n_scored_bases)}</td><td>${FX.nfmt(c.p50)}</td><td>${FX.nfmt(c.p99)}</td><td>${FX.nfmt(c.max)}</td><td>${FX.nfmt(r.runs.n_4windows)}</td><td class="${w ? "bad" : ""}">${FX.nfmt(w)}</td><td class="${r.underflow_bases ? "bad" : ""}">${FX.nfmt(r.underflow_bases)}</td></tr>`;
  });
  $("fdReal").innerHTML = h + "</tbody>";
  h = `<thead><tr><th>chromosome</th><th>scored bases</th><th>max reads</th><th>busiest four-base window (its lowest read count)</th><th>busiest such window with all k/n ≥ 0.8 (0 = none)</th><th>underflow bases</th></tr></thead><tbody>`;
  Object.entries(FREAL.per_chrom).forEach(([c, v]) => { h += `<tr><th>${c}</th><td>${FX.nfmt(v.n_scored_bases)}</td><td>${FX.nfmt(v.max_n)}</td><td>${FX.nfmt(v.max_4window_min_n)}</td><td class="${v["max_4window_min_n_all_kn_ge_0.8"] ? "bad" : ""}">${FX.nfmt(v["max_4window_min_n_all_kn_ge_0.8"])}</td><td class="${v.underflow_bases ? "bad" : ""}">${FX.nfmt(v.underflow_bases)}</td></tr>`; });
  $("fdChrom").innerHTML = h + "</tbody>";
  $("fdMethod").textContent = FREAL.method;
  // conclusion, from the data
  const c1rows = FDEP.series[FX.fracKey(1)], failC = FX.fails(c1rows, "current"), rd = FX.sc("rdna");
  const windows = FX.kn08(FX.G) + FX.kn08(FX.R), nwin = FX.G.runs.n_4windows + FX.R.runs.n_4windows;
  const worstRd = Math.max(...FX.FLOORED.map(k => rd.methods[k].max_err_exact));
  $("fdConclusion").innerHTML = `<b>Conclusion.</b> C1 needs ${FDEP.positions.length} neighbouring bases with about ${failC.map(x => x.toLocaleString("en-US")).join("–")} reads, all methylated. Real Fiber-seq data has ${windows.toLocaleString("en-US")} such windows out of ${nwin.toLocaleString("en-US")} (k/n ≥ 0.8 at all four bases), so the toy's C1 trigger never occurs in our data. ` +
    (FSEG ? `In the real 3,485-state model C1 still plays a small part: on one real rDNA segment the C1 fix changes ${FSEG.pairwise_rows_gt_0p01["FIXB vs C1FIX+B"].toLocaleString("en-US")} positions, but ${FSEG.variants["C1FIX+B"].rows_gt_0p01.toLocaleString("en-US")} stay wrong, mostly from normalize()'s DBL_MIN padding. ` : ``) +
    `The emission floor reaches real data too: binom.pmf underflows at ${FX.nfmt(FX.R.underflow_bases)} rDNA bases, ${FX.nfmt(FREAL.per_chrom.chrM.underflow_bases)} on chrM and ${FX.nfmt(FX.G.underflow_bases - FREAL.per_chrom.chrM.underflow_bases)} elsewhere (chrXIV: ${FX.nfmt(FX.X14.underflow_bases)}). In the rDNA-like toy, every method fed floored emissions is off by up to ${fmt(worstRd)} against the exact model; log space with exact emission is off by ${fmt(rd.methods.logD1.max_err_exact)}.`;
})();

/* ----- real model, one rDNA segment (only when realseg/results.json was present at build time) ----- */
const FSEGX = (function () {
  if (!FSEG) return null;
  const V = FSEG.variants, REF = "LOG-D0";
  const ROWS = [["ORIG", "original RoboCOP"], ["ORIG+B", "original, rows rescaled to sum 1 (fix B's idea, in Python)"], ["FIXB", "fix B (production)"], ["C1FIX", "C1 fix (audit)"], ["C1FIX+B", "C1 fix + fix B"]];
  const seg = FSEG.segment, tree = FSEG.tree.split("/").filter(Boolean).pop();
  const nf = v => v.toLocaleString("en-US");
  const where = seg ? `${seg.chrom}:${nf(seg.start)}–${nf(seg.end)}` : "one chrXII rDNA segment";
  const table = () => {
    let h = `<thead><tr><th>decoder</th><th>largest error vs ${REF}</th><th>positions off by &gt; 0.01</th><th>largest row sum</th><th>largest silent value</th><th>log-likelihood − ${REF}</th></tr></thead><tbody>`;
    ROWS.forEach(([k, l]) => {
      const v = V[k];
      h += `<tr><th>${l} <span class="muted mono">${k}</span></th><td class="mono${v.max_abs_vs_logD0 > 0.01 ? " bad" : ""}">${fmt(v.max_abs_vs_logD0)}</td><td class="mono${v.rows_gt_0p01 ? " bad" : ""}">${nf(v.rows_gt_0p01)} of ${nf(FSEG.n_obs)}</td><td class="mono${v.rowsum_max > 1 + 1e-9 ? " bad" : ""}">${fmt(v.rowsum_max)}</td><td class="mono${v.silent_max > 1 ? " bad" : ""}">${fmt(v.silent_max)}</td><td class="mono${Math.abs(v.loglik_minus_logD0) > 1e-6 ? " bad" : ""}">${v.loglik_minus_logD0.toFixed(2)}</td></tr>`;
    });
    const r = V[REF];
    h += `<tr><th>log space, floored emission <span class="muted mono">${REF}</span></th><td class="mono">reference</td><td class="mono">${nf(r.rows_gt_0p01)}</td><td class="mono">${fmt(r.rowsum_max)}</td><td class="mono">${fmt(r.silent_max)}</td><td class="mono">0</td></tr>`;
    return h + "</tbody>";
  };
  const pw = FSEG.pairwise_rows_gt_0p01["FIXB vs C1FIX+B"];
  const intro = `A real chrXII rDNA segment, ${where}: ${nf(FSEG.n_obs)} positions × ${nf(FSEG.n_states)} states, ${FSEG.n_vars} data layers, production "both" tree <span class="mono">${tree}</span>. Every decoder gets the same emission array, floored as production builds it (${nf(FSEG.emission_floor_cells)} cells lifted to the floor). Reference = <b>${REF}</b>: log space on that same floored emission (an 80-bit replica showed it is the exact answer for that emission).`;
  const reading = [
    `The original RoboCOP decoder gives rows that sum to <b class="bad">${fmt(V.ORIG.rowsum_max)}</b> instead of 1: the blow-up the toy shows, on real data.`,
    `Fix B makes every row sum to 1, but <b class="bad">${nf(V.FIXB.rows_gt_0p01)}</b> positions are still wrong by up to ${fmt(V.FIXB.max_abs_vs_logD0)}.`,
    `The audit's C1 fix changes only ${nf(pw)} positions; ${nf(V["C1FIX+B"].rows_gt_0p01)} stay wrong. The log-likelihood gap (${V["C1FIX+B"].loglik_minus_logD0.toFixed(0)} nats, a log-probability unit) does not move, because it comes from the forward pass, which C1 does not touch. So on real rDNA most of the damage is normalize()'s DBL_MIN padding (fork defect 3, audit C2), which hits both passes. Only log space is right.`,
    `This is all <b>before</b> the emission problem: against the exact emission, the floor adds more error on top (see the toy above).`,
  ];
  const takeaway = `On one real rDNA segment (${where}) the original decoder's rows sum to ${fmt(V.ORIG.rowsum_max)}; fix B still gets ${nf(V.FIXB.rows_gt_0p01)} of ${nf(FSEG.n_obs)} positions wrong by up to ${fmt(V.FIXB.max_abs_vs_logD0)}; the C1 fix changes only ${nf(pw)} of them (${nf(V["C1FIX+B"].rows_gt_0p01)} stay wrong) and leaves the ${V["C1FIX+B"].loglik_minus_logD0.toFixed(0)}-nat log-likelihood gap untouched. Most of that damage is normalize()'s DBL_MIN padding, and it comes before the emission floor's own error.`;
  function fill(prefix) {
    const $ = id => document.getElementById(id);
    $(prefix + "Intro").innerHTML = intro;
    $(prefix + "Table").innerHTML = table();
    $(prefix + "Read").innerHTML = reading.map(x => `<li>${x}</li>`).join("");
  }
  return { fill, takeaway };
})();
if (FSEGX) FSEGX.fill("feSeg");
