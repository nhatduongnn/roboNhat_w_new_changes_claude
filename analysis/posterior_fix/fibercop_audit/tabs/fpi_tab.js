/* ================= fiberCOP: PI report tab ================= */
const fpNow = fMakeStepper("fpl", FLB.steps, "now", "tab-fpi");
const fpFix = fMakeStepper("fpk", FLB.steps_fix, "fix", "tab-fpi");
const fpSync = fMakeSync("fps", fpNow, fpFix, "tab-fpi");
window.fpiStop = () => { fpNow.stop(); fpFix.stop(); fpSync.stop(); };
document.addEventListener("keydown", e => {
  if (document.getElementById("tab-fpi").hidden) return;
  if (e.target.closest && e.target.closest("input, textarea, select")) return;
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  e.preventDefault();
  const d = e.key === "ArrowRight" ? 1 : -1;
  if (fpSync.linked()) fpSync.step(d); else (e.shiftKey ? fpFix : fpNow).step(d);
});
(function () {
  const $ = id => document.getElementById(id);
  const S = FEMP.S, n = FEMP.n, SH = FD.short, FB = FD.fiber;
  const c1 = FX.sc("c1"), rd = FX.sc("rdna");
  const POS = new Set(FDEP.positions);
  const DMIN = 2.2250738585072014e-308;
  const nf = FX.nfmt;
  const nn = t => (FB.scored[t] === "W" ? FB.n_w[t] : FB.n_c[t]), kk = t => (FB.scored[t] === "W" ? FB.k_w[t] : FB.k_c[t]);
  const sweep1 = FDEP.series[FX.fracKey(1)];
  const failC = FX.fails(sweep1, "current"), failB = FX.fails(sweep1, "fixB");
  const onset1 = FX.onsetExact(sweep1), onset0 = FX.onsetExact(FDEP.series[FX.fracKey(0)]), onset6 = FX.onsetExact(FDEP.series[FX.fracKey(0.6)]);
  const G = FX.G, R = FX.R, X14 = FX.X14;
  const winG = FX.kn08(G), winR = FX.kn08(R);
  const under = FX.chromUnder, chrM = FREAL.per_chrom.chrM;
  const otherUnder = G.underflow_bases - chrM.underflow_bases;
  const worstRd = Math.max(...FX.FLOORED.map(k => rd.methods[k].max_err_exact));
  const minUnder = Math.min(...Object.values(FREAL.underflow_min_n_seen));
  const range = a => (a.length ? (a.length > 1 ? `${nf(a[0])}–${nf(a[a.length - 1])}` : nf(a[0])) : "none");

  // ---- takeaway, (a)-(f), generated from the data
  const items = [
    `<b>The C1 code is still in fiberCOP.</b> Fix B changed only the posterior step. The backward pass (<span class="mono">bward</span>) and <span class="mono">normalize</span> are byte-identical to original RoboCOP, so production has exactly the backward table shown below.`,
    `<b>In this toy, with the real fitted methylation probabilities, C1 fires when ${FDEP.positions.length} neighbouring bases each have about ${range(failC)} reads, all methylated.</b> Against the floored model: original code off by <b class="bad">${fmt(c1.methods.current.max_err)}</b>, fix B by <b class="bad">${fmt(c1.methods.fixB.max_err)}</b>, the audit's C1 fix by ${fmt(c1.methods.fixed.max_err)} (exact).`,
    `<b>Real Fiber-seq data never forms the toy's trigger run.</b> ${nf(winG)} of ${nf(G.runs.n_4windows)} four-base windows outside rDNA, and ${nf(winR)} of ${nf(R.runs.n_4windows)} in rDNA, have k/n ≥ 0.8 at all four bases (k/n = methylated reads ÷ reads).`,
    `<b>The underflow that does reach real data is in the emission itself.</b> At high coverage <span class="mono">binom.pmf</span> underflows to 0 (in our data from ${nf(minUnder)} reads at one base). The floor (${fmt(FB.floor)}) then lifts it <i>above</i> the right value and flips the evidence. Every method fed the floored emission (original, fix B, C1 fix, log space with floored emission) is then wrong: off by up to <b class="bad">${fmt(worstRd)}</b> against the exact model in the rDNA-like toy. Only log space with exact emission is right (${fmt(rd.methods.logD1.max_err_exact)}). Real data: <b class="bad">${nf(R.underflow_bases)}</b> underflow bases in rDNA, <b class="bad">${nf(chrM.underflow_bases)}</b> in chrM, ${nf(otherUnder)} elsewhere; chrXIV ${nf(X14.underflow_bases)}.`,
    `<b>Coverage is much higher than MNase.</b> Fiber-seq: median ${nf(G.coverage_n.p50)} reads per base, 99th percentile ${nf(G.coverage_n.p99)} (genome outside rDNA); rDNA up to ${nf(R.coverage_n.max)}. MNase: the busiest pair of neighbouring positions had ${nf(FX.mnaseMax)}.`,
  ];
  if (FSEGX) items.push(`<b>Real model, one rDNA segment.</b> ${FSEGX.takeaway}`);
  $("fpiTake").innerHTML = items.map(x => `<li>${x}</li>`).join("");

  // ---- background: the storage limit (clone of the original chart)
  (function () {
    const x0 = 30, x1 = 730, ya = 70;
    const X = v => x0 + (0 - v) / 335 * (x1 - x0), cut = Math.log10(DMIN), zero = Math.log10(4.9406564584124654e-324);
    let g = `<rect x="${X(0)}" y="${ya - 14}" width="${X(cut) - X(0)}" height="28" fill="var(--good-soft)"/>`;
    g += `<rect x="${X(cut)}" y="${ya - 14}" width="${X(zero) - X(cut)}" height="28" fill="var(--warn-soft)" stroke="var(--warn)"/>`;
    g += `<rect x="${X(zero)}" y="${ya - 14}" width="${X(-335) - X(zero)}" height="28" fill="var(--bad-soft)" stroke="var(--bad)"/>`;
    g += `<text x="${(X(0) + X(cut)) / 2}" y="${ya + 5}" text-anchor="middle" class="vdl">full precision (~16 digits)</text>`;
    const lead = (xz, xt, y, txt, col, anchor) =>
      `<polyline points="${xz},${ya - 15} ${xz},${y + 6} ${xt},${y + 6}" fill="none" stroke="${col}" stroke-width="1.3"/>` +
      `<text x="${xt + (anchor === "end" ? -4 : 4)}" y="${y + 10}" text-anchor="${anchor}" class="vdl" style="fill:${col}">${txt}</text>`;
    g += lead((X(cut) + X(zero)) / 2, X(cut) - 40, 14, "reduced precision: 2.2 × 10⁻³⁰⁸ down to 4.9 × 10⁻³²⁴ (fewer digits kept)", "var(--warn)", "end");
    g += lead((X(zero) + X(-335)) / 2, X(zero) - 40, 34, "stored as 0: below 4.9 × 10⁻³²⁴", "var(--bad)", "end");
    [0, -50, -100, -150, -200, -250, -300].forEach(v => { g += `<line x1="${X(v)}" x2="${X(v)}" y1="${ya + 14}" y2="${ya + 20}" stroke="var(--muted)"/><text x="${X(v)}" y="${ya + 33}" text-anchor="middle" class="vax">${v === 0 ? "1" : "10^" + v}</text>`; });
    $("fpiLine").innerHTML = g;
  })();

  // ---- model diagram (clone; transition probabilities read from FD.trans)
  (function () {
    const N = 12, T = Array.from({ length: N }, () => Array(N).fill(0));
    FD.trans.forEach(([i, j, v]) => { T[i][j] = v; });
    const lab = v => Number(v.toPrecision(3)).toString();
    const R = { w: 74, h: 36 };
    const nodes = {
      0: { x: 460, y: 62, kind: "e", name: "BG", sub: "background" },
      10: { x: 460, y: 220, kind: "s", r: 38, name: "CENTRAL", sub: "silent" },
      11: { x: 320, y: 220, kind: "s", r: 38, name: "TF1-entry", sub: "silent" },
      1: { x: 215, y: 140, kind: "e", name: "F1" }, 2: { x: 95, y: 140, kind: "e", name: "F2" },
      3: { x: 215, y: 300, kind: "e", name: "R1" }, 4: { x: 95, y: 300, kind: "e", name: "R2" },
      5: { x: 590, y: 220, kind: "e", name: "N1" }, 6: { x: 700, y: 220, kind: "e", name: "N2" },
      7: { x: 810, y: 160, kind: "e", name: "N3a" }, 8: { x: 810, y: 280, kind: "e", name: "N3b" },
      9: { x: 920, y: 220, kind: "e", name: "N4" },
    };
    const edgePt = (nd, dx, dy) => {
      const L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
      if (nd.kind === "s") return [nd.x + ux * (nd.r + 3), nd.y + uy * (nd.r + 3)];
      const tx = (R.w / 2 + 3) / Math.abs(ux || 1e-9), ty = (R.h / 2 + 3) / Math.abs(uy || 1e-9), t = Math.min(tx, ty);
      return [nd.x + ux * t, nd.y + uy * t];
    };
    let g = `<defs><marker id="fpiArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>`;
    const label = (x, y, txt) => `<text x="${x}" y="${y}" text-anchor="middle" class="hlab">${txt}</text>`;
    const straight = (a, b, off = 0, lx = 0, ly = -8) => {
      const A = nodes[a], B2 = nodes[b];
      const dx = B2.x - A.x, dy = B2.y - A.y, L = Math.hypot(dx, dy), px = -dy / L * off, py = dx / L * off;
      const p0 = edgePt({ ...A, x: A.x + px, y: A.y + py }, dx, dy), p1 = edgePt({ ...B2, x: B2.x + px, y: B2.y + py }, -dx, -dy);
      const mx = (p0[0] + p1[0]) / 2 + lx, my = (p0[1] + p1[1]) / 2 + ly;
      return `<line x1="${p0[0]}" y1="${p0[1]}" x2="${p1[0]}" y2="${p1[1]}" class="hedge" marker-end="url(#fpiArrow)"/>` + label(mx, my, lab(T[a][b]));
    };
    g += straight(10, 0, 14, 22, 4); g += straight(0, 10, 14, -22, 4); g += straight(10, 11, 0, 0, -8);
    g += straight(11, 1, 0, 6, -6); g += straight(11, 3, 0, 6, 16);
    g += straight(1, 2, 0, 0, -8); g += straight(3, 4, 0, 0, -8);
    g += straight(10, 5, 0, 0, -8); g += straight(5, 6, 0, 0, -8);
    g += straight(6, 7, 0, -10, -6); g += straight(6, 8, 0, -10, 16);
    g += straight(7, 9, 0, 10, -6); g += straight(8, 9, 0, 10, 16);
    const ret = (a, pts, lx, ly) => `<polyline points="${pts.map(p => p.join(",")).join(" ")}" class="hedge hret" fill="none" marker-end="url(#fpiArrow)"/>` + label(lx, ly, lab(T[a][10]));
    g += ret(2, [[95, 122], [95, 100], [395, 100], [432, 190]], 245, 94);
    g += ret(4, [[95, 318], [95, 360], [395, 360], [432, 250]], 245, 354);
    g += ret(9, [[920, 238], [920, 380], [525, 380], [488, 250]], 720, 374);
    g += `<line x1="460" y1="420" x2="460" y2="${220 + 38 + 3}" class="hedge" marker-end="url(#fpiArrow)"/>` + `<text x="472" y="412" class="hlab">start</text>`;
    g += `<text x="155" y="182" text-anchor="middle" class="hgrp">TF1 forward strand</text>`;
    g += `<text x="155" y="342" text-anchor="middle" class="hgrp">TF1 reverse strand</text>`;
    g += `<text x="755" y="120" text-anchor="middle" class="hgrp">nucleosome (147 positions in RoboCOP; 4 here)</text>`;
    Object.values(nodes).forEach(nd => {
      if (nd.kind === "e") g += `<rect x="${nd.x - R.w / 2}" y="${nd.y - R.h / 2}" width="${R.w}" height="${R.h}" rx="7" class="hnode"/><text x="${nd.x}" y="${nd.y + 4.5}" text-anchor="middle" class="hname">${nd.name}</text>`;
      else g += `<circle cx="${nd.x}" cy="${nd.y}" r="${nd.r}" class="hnode hsil"/><text x="${nd.x}" y="${nd.y + 2}" text-anchor="middle" class="hname hsiln">${nd.name}</text><text x="${nd.x}" y="${nd.y + 15}" text-anchor="middle" class="hsub">${nd.sub}</text>`;
      if (nd.name === "BG") g += `<text x="${nd.x + R.w / 2 + 8}" y="${nd.y + 4}" class="hsub" text-anchor="start">background</text>`;
    });
    g += `<rect x="20" y="398" width="26" height="18" rx="4" class="hnode"/><text x="52" y="411" class="hsub" text-anchor="start">emitting state (produces a base)</text>`;
    g += `<circle cx="273" cy="407" r="9" class="hnode hsil"/><text x="288" y="411" class="hsub" text-anchor="start">silent state (produces nothing)</text>`;
    const svg = $("fpiHmm");
    svg.innerHTML = g;
    svg.setAttribute("aria-label", `HMM design: every feature returns to the CENTRAL silent state, which chooses the next feature: background ${lab(T[10][0])}, nucleosome ${lab(T[10][5])}, TF1 ${lab(T[10][11])}`);
  })();

  // ---- Fiber-seq data
  (function () {
    const cls = t => (POS.has(t) ? "picol" : "");
    const pos = [...POS];
    $("fpiDataNote").innerHTML = `The same 6-base model, now with Fiber-seq counts. At an A the Watson strand is scored, at a T the Crick strand (an m6A call needs an A on that strand). n = reads covering the base, k = reads with that A methylated. Positions ${pos[0]}–${pos[pos.length - 1]} get ${nf(nn(pos[0]))} reads, all methylated; the others ${nf(nn(0))} reads, ${nf(kk(0))} methylated.`;
    let h = `<thead><tr><th>position</th>${[...FB.seq].map((c, t) => `<th class="${cls(t)}">${t}</th>`).join("")}</tr></thead><tbody>`;
    const r = (label, arr) => `<tr><th>${label}</th>${arr.map((v, t) => `<td class="${cls(t)}">${v}</td>`).join("")}</tr>`;
    const T6 = [...FB.seq].map((c, t) => t);
    h += r("base", [...FB.seq]) + r("strand scored", FB.scored.map(s => (s === "W" ? "Watson" : "Crick"))) + r("reads n", T6.map(t => nf(nn(t)))) + r("methylated k", T6.map(t => nf(kk(t)))) + r("k/n", T6.map(t => (kk(t) / nn(t)).toFixed(2)));
    $("fpiData").innerHTML = h + "</tbody>";
    h = `<thead><tr><th>strand</th>${SH.slice(0, S).map(s => `<th>${s}</th>`).join("")}</tr></thead><tbody>`;
    h += `<tr><th>Watson</th>${FB.p_watson.map(v => `<td>${v.toFixed(3)}</td>`).join("")}</tr><tr><th>Crick</th>${FB.p_crick.map(v => `<td>${v.toFixed(3)}</td>`).join("")}</tr>`;
    $("fpiP").innerHTML = h + "</tbody>";
  })();

  // ---- transition matrix (with start probabilities) and emission matrix
  (function () {
    const N = 12, T = Array.from({ length: N }, () => Array(N).fill(0));
    FD.trans.forEach(([i, j, v]) => { T[i][j] = v; });
    const cell = v => (v === 0 ? `<td class="z">·</td>` : `<td>${Number(v.toPrecision(3))}</td>`);
    let h = `<thead><tr><th>from \\ to</th>${SH.map((s, j) => `<th class="${j >= S ? "silh" : ""}">${s}</th>`).join("")}</tr></thead><tbody>`;
    h += `<tr class="pistart"><th>start</th>${FD.init.map(v => cell(v)).join("")}</tr>`;
    for (let i = 0; i < N; i++) h += `<tr><th class="${i >= S ? "silh" : ""}">${SH[i]}</th>${T[i].map(cell).join("")}</tr>`;
    $("fpiT").innerHTML = h + "</tbody>";
    $("fpiENote").innerHTML = `What C receives: the product over all ${FD.n_vars} data layers (the other ${FD.n_vars - FB.layer_names.length} layers, MNase, are switched off and equal 1). <span class="fpifl">Outlined</span> cells: <span class="mono">binom.pmf</span> underflowed to 0 and was set to the floor, ${fmt(FB.floor)} (${FB.n_floored} cells). Where the product itself is too small for a double it shows as 0, with its true size (log₁₀) in grey. Every cell also carries one factor of ${fmt(FB.floor)} from the strand that is not scored at that base; that factor is the same for every state, so it does no harm.`;
    h = `<thead><tr><th>position</th>${SH.slice(0, S).map(s => `<th>${s}</th>`).join("")}</tr></thead><tbody>`;
    for (let t = 0; t < n; t++) h += `<tr class="${POS.has(t) ? "picolr" : ""}"><th>t=${t} ${FB.seq[t]}</th>${FD.emit[t].map((v, j) => {
      const l = FB.emit_log10_floored[t][j], fl = FB.floored[t][j];
      const txt = v !== 0 ? fmt(v) : (l === null ? "0" : `0 <span class="muted">(10^${Math.round(l)})</span>`);
      return `<td class="${fl ? "fpifl" : ""}" title="${FNAMES[j]}, t=${t}${l === null ? "" : ", log10 = " + l.toFixed(1)}${fl ? ", floored" : ""}">${txt}</td>`;
    }).join("")}</tr>`;
    $("fpiE").innerHTML = h + "</tbody>";
  })();

  // ---- output: occupancy, exact truth vs floored truth vs fix B
  (function () {
    const F = [[0, "background", "var(--s1)"], [1, "TF1", "var(--s2)"], [3, "nucleosome", "var(--s3)"]];
    const RW = [["exact truth", c1.methods.brute_exact], ["floored-model truth", c1.methods.brute], ["fix B (production)", c1.methods.fixB]];
    const W = 760, H = 330, L = 140, top = 16, bh = 70, gap = 26, bw = 70, step = (W - L - 20) / n;
    let g = "";
    RW.forEach(([label, m], ri) => {
      const y0 = top + ri * (bh + gap);
      g += `<text x="${L - 10}" y="${y0 + bh / 2 + 4}" text-anchor="end" class="vlab2">${label}</text>`;
      for (let t = 0; t < n; t++) {
        const x = L + t * step + (step - bw) / 2, tot = F.reduce((a, [c]) => a + (m.dbf[t][c] || 0), 0);
        if (tot > 1.0001) {
          g += `<rect x="${x}" y="${y0}" width="${bw}" height="${bh}" rx="3" fill="var(--bad-soft)" stroke="var(--bad)" stroke-dasharray="4 3"/>`;
          g += `<text x="${x + bw / 2}" y="${y0 + bh / 2 - 2}" text-anchor="middle" class="vdl" style="fill:var(--bad)">sum ${fmt(tot)}</text><text x="${x + bw / 2}" y="${y0 + bh / 2 + 13}" text-anchor="middle" class="vax">off scale</text>`;
        } else {
          let yy = y0 + bh;
          F.forEach(([c, nm, col]) => { const hgt = (m.dbf[t][c] || 0) * bh; if (hgt > 0.3) g += `<rect x="${x}" y="${yy - hgt}" width="${bw}" height="${Math.max(0, hgt - 1)}" fill="${col}"><title>${nm}, t=${t}: ${fmt(m.dbf[t][c])}</title></rect>`; yy -= hgt; });
        }
        if (ri === RW.length - 1) g += `<text x="${x + bw / 2}" y="${H - 6}" text-anchor="middle" class="vax">t=${t} ${c1.seq[t]}</text>`;
      }
    });
    $("fpiOcc").innerHTML = g;
    $("fpiOccLeg").innerHTML = F.map(([c, nm, col]) => `<span><svg width="12" height="12" aria-hidden="true"><rect width="12" height="12" rx="2" fill="${col}"/></svg>${nm}</span>`).join("");
    const call = (m, t) => F.reduce((b, f) => ((m.dbf[t][f[0]] || 0) > (m.dbf[t][b[0]] || 0) ? f : b), F[0])[1];
    const E = c1.methods.brute_exact, Bf = c1.methods.brute, FBm = c1.methods.fixB;
    const flip = [...Array(n).keys()].filter(t => call(E, t) !== call(Bf, t));
    const groups = {};
    flip.forEach(t => { const k = `${call(E, t)} → ${call(Bf, t)}`; (groups[k] = groups[k] || []).push(t); });
    const desc = Object.entries(groups).map(([k, ts]) => `${ts.length === n ? "every position" : (ts.length > 1 ? "positions " : "position ") + ts.join(", ")}: ${k}`).join("; ");
    $("fpiOccNote").innerHTML = `<b>The floored truth is already wrong.</b> ${flip.length ? `The floor alone changes the main call, before any decoder runs (${desc}).` : "The floor does not change the main call here."} ` +
      `Fix B is off by up to <b class="bad">${fmt(FBm.max_err)}</b> from the floored truth (the C1 error) and <b class="bad">${fmt(FBm.max_err_exact)}</b> from the exact truth. Log-likelihood: ${FBm.ll.toFixed(1)} for fix B and the floored truth, ${E.ll.toFixed(1)} for the exact model.`;
  })();

  // ---- five decoders vs brute force, grouped by scenario
  (function () {
    $("fpiMatNote").innerHTML = `Largest posterior error against each truth (brute force, all ${c1.npaths} paths). ✓ = below 1e-9. Log-likelihood ✓ = matches the truth of the model that method decodes (floored model, or exact model for log space with exact emission).`;
    let h = `<thead><tr><th>data / method</th><th>error vs floored-model truth</th><th>error vs exact truth</th><th>log-likelihood</th></tr></thead><tbody>`;
    const ck = e => `<td class="${FX.bad(e) ? "pino" : "piok"}">${FX.bad(e) ? "✗" : "✓"} ${e === null ? "NaN" : fmt(e)}</td>`;
    FEMP.scenarios.forEach(s => {
      h += `<tr class="pistart"><th colspan="4">${s.title}</th></tr>`;
      FX.ME.forEach(([k, label]) => {
        const m = s.methods[k], ref = k === "logD1" ? s.methods.brute_exact : s.methods.brute;
        const ok = m.ll !== null && Math.abs(m.ll - ref.ll) < 1e-6 * Math.max(1, Math.abs(ref.ll));
        h += `<tr><th>${label}</th>${ck(m.max_err)}${ck(m.max_err_exact)}<td class="${ok ? "piok" : "pino"}">${ok ? "✓" : "✗"} ${m.ll === null ? "NaN" : m.ll.toFixed(2)}</td></tr>`;
      });
    });
    $("fpiMat").innerHTML = h + "</tbody>";
  })();

  // ---- where our Fiber-seq data lies (frac 1.0, both truths)
  (function () {
    const st = k => FX.SERSTYLE[k];
    const SER = [
      { key: "current", label: "original vs floored truth", col: st("current")[0], shape: st("current")[1], dash: false, val: r => r.current },
      { key: "fixB", label: "fix B vs floored truth", col: st("fixB")[0], shape: st("fixB")[1], dash: false, val: r => r.fixB },
      { key: "fixed", label: "C1 fix vs floored truth", col: st("fixed")[0], shape: st("fixed")[1], dash: false, val: r => r.fixed },
      { key: "flooredx", label: "worst floored method vs exact truth", col: "var(--bad)", shape: "ring", dash: true, val: r => FX.worstFloored(r, "_exact") },
      { key: "logD1", label: "log space, exact emission, vs exact truth", col: st("logD1")[0], shape: st("logD1")[1], dash: true, val: r => r.logD1_exact },
    ];
    FX.depChart("fpiDep", "fpiTip", sweep1, SER, FX.MARKS, { xlabel: `reads at each of positions ${FDEP.positions.join(", ")}, all methylated (log scale)`, ylabel: "largest posterior error" });
    $("fpiDepLeg").innerHTML = FX.legend(SER);
    const logMax = Math.max(...Object.values(FDEP.series).flat().map(r => r.logD1_exact));
    $("fpiConc").innerHTML = `<b>C1 (solid lines) needs ${range(failC)} reads at ${FDEP.positions.length} neighbouring bases, all methylated; our data has none (${nf(winG + winR)} such windows).</b> Above that the floor takes over and the linear codes look "right" again, but only for the floored model. Against the exact model (dashed) every floored method is wrong from ${onset1 === null ? "—" : nf(onset1)} reads on, while log space with exact emission stays at or below ${fmt(logMax)} in every sweep. Our genome median (${nf(G.coverage_n.p50)}) is safe; rDNA (${nf(R.coverage_n.p50)}–${nf(R.coverage_n.max)}) and chrM (up to ${nf(chrM.max_n)}) are not.`;
    $("fpiCap").innerHTML = `Toy model: the failure points come from its parameters. With other methylated shares the exact-model failure starts later: ${onset0 === null ? "never" : "from " + nf(onset0)} reads with none methylated, ${onset6 === null ? "never" : "from " + nf(onset6)} with 60% methylated. Real coverage: merged Fiber-seq pileup (scored bases only). MNase: busiest neighbouring pair in the original audit's BAM.`;
  })();
  if (FSEGX) FSEGX.fill("fpiSeg");
})();
