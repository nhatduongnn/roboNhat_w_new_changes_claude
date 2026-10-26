/* ================= fiberCOP: line-by-line backward (clone of makeStepper/makeSync on FLB) ================= */
const FD = __FD__;
const FLB = __FLB__;
/* display names: the data's long labels, tidied the way the original page's NAMES are (minus sign, no "(silent)") */
const FNAMES = FD.names.map(s => s.replace("(silent)", "").replace(/\((A\/T|C\/G)\)/, " $1").replace(/^TF1- /, "TF1− ").trim());
/* fork line numbers: FLB.line_map maps every upstream line number the original tab hardcodes to the fork's */
const FM = u => { const v = FLB.line_map[String(u)]; if (v === undefined) throw new Error("no fork line for upstream " + u); return String(v); };
const FC1 = (function () {
  const cr = FLB.c1_rows;
  const cells = cr.subnormal_silent_cells;                       // [[row, j], ...] the C1 cells proper
  const rows = [...new Set(cells.map(c => c[0]))].sort((a, b) => a - b);
  const stepKeys = new Set(cr.subnormal_silent_steps.map(i => { const s = FLB.steps[i]; return `${s.row},${s.j},${s.child}`; }));
  const cellKey = new Set(cells.map(c => c.join(",")));
  return { rows, rowSet: new Set(rows), stepKeys, cellKey, cells, wrongRows: cr.rows, lastC1: rows[rows.length - 1] };
})();
const fRowLabel = r => `row ${r} · t=${FLB.n - 1 - r}${FC1.rowSet.has(+r) ? " · C1" : ""}`;

function fMakeStepper(P, ST, side, tabId) {
  const $ = id => document.getElementById(id);
  const n = FLB.n, S = FLB.S, NM = FNAMES, SH = FD.short, DMIN = FLB.dbl_min;
  const sub = k => String(k).split("").map(c => "₀₁₂₃₄₅₆₇₈₉"[+c]).join("");
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/\t/g, "    ");
  const other = side === "now" ? FLB.steps_fix : FLB.steps;
  const otherVal = {};
  other.forEach(s => { if (s.child !== undefined) otherVal[`${s.row},${s.j},${s.child}`] = s.val; });
  const code = side === "now" ? FLB.code_b.map(([ln, txt]) => [String(ln), String(ln), txt]) : FLB.code_fix;
  let cur = 0, timer = null;

  const firstOfRow = {};
  ST.forEach((s, i) => { if (!(s.row in firstOfRow)) firstOfRow[s.row] = i; });
  $(P + "Jumps").innerHTML = Object.keys(firstOfRow).map(r => `<button type="button" data-i="${firstOfRow[r]}" data-r="${r}"${FC1.rowSet.has(+r) ? ' class="c1j"' : ""}>${fRowLabel(r)}</button>`).join("");
  $(P + "Jumps").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { stop(); go(+b.dataset.i); } });

  const MM = a => a.map(FM);
  const HL_NOW = { init: [MM([149, 150]), []], silent0: [MM([159, 160]), MM([154, 155, 157, 158])], emit: [MM([175, 176, 177, 178, 179]), MM([169, 170, 171, 173, 174])],
                   silent: [MM([190, 192, 193, 194, 195]), MM([185, 186, 188, 189])] };
  const HL_FIX = { init: [MM([149, 150]), []], emit: [MM([175, 176]), MM([169, 170, 171, 173])],
                   silent0: [MM([159, 160]), MM([154, 155, 157])], silent: [MM([190, 192]), MM([185, 186, 188])] };
  const lastLine = row => FM(row === 0 ? 166 : 201);
  function hl(s) {   // [active keys, context keys]
    if (side === "now") {
      if (s.kind === "nsum" || s.kind === "ndiv" || s.kind === "store") return [[lastLine(s.row)], []];
      return HL_NOW[s.kind];
    }
    const nk = s.row === 0 ? "NEW0" : "NEW";
    if (s.kind === "nsum" || s.kind === "ndiv" || s.kind === "store") return [[nk], []];
    if (s.kind === "silent") return s.row === 0 ? HL_FIX.silent0 : HL_FIX.silent;
    return HL_FIX[s.kind];
  }
  function codeHTML(lines, act, ctx) {
    return lines.map(([key, disp, txt]) => `<div class="${act.includes(key) ? "hl" : (ctx.includes(key) ? "ctx" : "")}${key.startsWith("NEW") ? " newl" : ""}${key.startsWith("X") ? " gone" : ""}"><span class="ln">${disp}</span>${esc(txt) || " "}</div>`).join("");
  }
  function replay(k) {
    const g = [...Array(n)].map(() => Array(12).fill(null)), post = [...Array(n)].map(() => Array(12).fill(false));
    for (let i = 0; i <= k; i++) {
      const s = ST[i];
      if (["init", "emit", "silent", "silent0"].includes(s.kind)) {
        g[s.row][s.j] = s.val;
        post[s.row][s.j] = side === "fix" && s.kind === "silent";
      }
      if (s.kind === "ndiv") s.after.forEach((v, j) => { g[s.row][j] = v; post[s.row][j] = true; });
    }
    return { g, post };
  }
  const isWrong = (row, j, v) => { const c = FLB.fix[row][j]; return Math.abs(v - c) > 1e-6 * Math.max(Math.abs(c), 1e-300) && Math.abs(v - c) > 1e-250; };

  function explain(s) {
    const nm = j => NM[j];
    let h = "", flag = "";
    if (s.kind === "init") {
      h = `<h3>Line ${FM(150)} · start row 0 with the end probabilities</h3><p>Row 0 holds the <b>last</b> base (t=${s.t}). Nothing comes after it, so each emitting state starts at its end probability.</p>
      <div class="eqbox mono">btable[0][${s.j}] = end_probs[${s.j}] = <b>${fmt(s.val)}</b>     (${nm(s.j)})</div>`;
    } else if (s.kind === "emit") {
      h = `<h3>Lines ${FM(175)}-${FM(179)} · emitting state ${nm(s.j)}, one child</h3>
      <p>${nm(s.j)} produced base ${s.t}. It looks ahead to its child ${nm(s.child)}${s.child < S ? `, which produces base ${s.epos}` : ", a junction just before base " + (s.t + 1)}. The child's value comes from row ${s.readrow} (base ${s.t + 1}), already rescaled.</p>
      <div class="eqbox mono">emissionValues = ${s.child < S ? `e${sub(s.epos)}(${nm(s.child)}) = ` : "1 (silent child)  "}${fmt(s.ev)}
btable[${s.row}][${s.j}] += btable[${s.readrow}][${s.child}] × T(${nm(s.j)}→${nm(s.child)}) × emissionValues
                += ${fmt(s.bchild)} × ${s.T} × ${fmt(s.ev)}
                 = ${fmt(s.old)} + ${fmt(s.val - s.old)} = <b>${fmt(s.val)}</b></div>`;
    } else if (s.kind === "silent0" || s.kind === "silent") {
      const below = s.val > 0 && s.val < DMIN;
      const rescaled = side === "fix";
      const ln = s.row === 0 ? `${FM(157)}-${FM(160)}` : `${FM(188)}-${FM(195)}`;
      const isC1 = FC1.stepKeys.has(`${s.row},${s.j},${s.child}`);
      h = `<h3>Lines ${ln} · silent state ${nm(s.j)}, one child</h3>
      <p>${nm(s.j)} sits just <b>before</b> base ${s.t}. Its child ${nm(s.child)} ${s.child < S ? `produces base ${s.t}, so the child's emission e${sub(s.t)}(${nm(s.child)}) is computed` : "is silent too, so there is no emission"}. The child's value comes from the <b>same row</b>, which ${rescaled ? "<b>has already been rescaled</b>" : "has <b>not</b> been rescaled yet"}.</p>
      <div class="eqbox mono">emissionValues = ${s.child < S ? `e${sub(s.t)}(${nm(s.child)}) = ` : ""}${fmt(s.ev)}
btable[${s.row}][${s.j}] += btable[${s.row}][${s.child}] × T(${nm(s.j)}→${nm(s.child)}) × emissionValues
                += ${fmt(s.bchild)} × ${s.T} × ${fmt(s.ev)}
                 = ${fmt(s.old)} + ${fmt(s.val - s.old)} = <b class="${below ? "bad" : (rescaled && FC1.rowSet.has(s.row) ? "good" : "")}">${fmt(s.val)}</b></div>`;
      const ov = otherVal[`${s.row},${s.j},${s.child}`];
      if (side === "now" && isC1 && s.child < S)
        flag = `<div class="c1flag"><b>C1 is happening here.</b> btable[${s.row}][${s.child}] = ${fmt(s.bchild)} is not rescaled yet: it still carries the tiny emission at base ${s.t + 1}. Multiplying by e${sub(s.t)} = ${fmt(s.ev)} puts two tiny emissions in one number${below ? `: <b>${fmt(s.val)}</b>, below DBL_MIN (${fmt(DMIN)})` : ""}. Rescaled first, this child's value would be ${fmt(FLB.fix[s.row][s.child])} (see the right panel, same step).</div>`;
      else if (side === "now" && isC1)
        flag = `<div class="c1flag"><b>C1 carries on here.</b> The child ${nm(s.child)} is silent and already below DBL_MIN (${fmt(s.bchild)}), so this cell inherits the damage${below ? `: <b>${fmt(s.val)}</b>` : ""}.</div>`;
      else if (side === "now" && below) flag = `<div class="c1flag"><b>Below DBL_MIN.</b> ${fmt(s.val)} is smaller than the smallest normal double. When normalize adds DBL_MIN, the padding will swamp it. The right panel gets ${fmt(ov)} for this cell.</div>`;
      else if (side === "fix" && FC1.rowSet.has(s.row) && ov !== undefined)
        flag = `<div class="okflag"><b>Compare with the left panel.</b> The child here is ${fmt(s.bchild)} (rescaled), so the result stays a normal number: <b>${fmt(s.val)}</b>. The current code gets ${fmt(ov)} for this same step.</div>`;
    } else if (s.kind === "nsum") {
      h = `<h3>${side === "now" ? `Line ${lastLine(s.row)} → normalize, lines ${FM(19)}-${FM(20)}` : "Rescale, part 1"} · add up the emitting states</h3>
      <p>${side === "now" ? `Row ${s.row} (base ${s.t}) is complete, silent states included.` : `The emitting cells of row ${s.row} (base ${s.t}) are done. The silent cells are not computed yet.`} Add up the <b>emitting</b> states 0–${S - 1}, each + DBL_MIN.</p>
      <div class="eqbox mono">sum = ${s.terms.map(x => fmt(x)).join(" + ")}
    = <b>${fmt(s.sum)}</b></div>`;
    } else if (s.kind === "ndiv") {
      const ncols = side === "now" ? 12 : S;
      const rows = s.before.slice(0, ncols).map((b, j) => `<tr><td>${j} ${SH[j]}</td><td class="mono">${fmt(b)}</td><td class="mono${isWrong(s.row, j, s.after[j]) ? " bad" : ""}">${fmt(s.after[j])}</td><td class="mono">${fmt(FLB.fix[s.row][j])}</td></tr>`).join("");
      h = `<h3>${side === "now" ? `normalize, lines ${FM(23)}-${FM(24)}` : "Rescale, part 2"} · divide by the sum</h3>
      <p>${side === "now" ? `Every entry of row ${s.row}, <b>silent states included</b>, becomes (value + DBL_MIN) ÷ ${fmt(s.sum)}.` : `Only the emitting cells 0–${S - 1} are divided by ${fmt(s.sum)}. The silent cells will be computed next, from these rescaled values, so they come out on the right scale with no division needed.`}</p>
      <div class="gridbox"><table class="calc"><thead><tr><th>state</th><th>before</th><th>after</th><th>correct</th></tr></thead><tbody>${rows}</tbody></table></div>`;
      const c1here = FC1.cells.filter(c => c[0] === s.row).map(c => c[1]);
      if (side === "now" && c1here.length)
        flag = `<div class="c1flag"><b>The damage becomes visible.</b> ` + c1here.map(j => `${SH[j]} was ${fmt(s.before[j])}; (${fmt(s.before[j])} + DBL_MIN) ÷ ${fmt(s.sum)} = <b>${fmt(s.after[j])}</b>, but the correct value is ${fmt(FLB.fix[s.row][j])}.`).join(" ") + ` DBL_MIN is far larger than the value itself, so the padding wins.</div>`;
      else if (side === "now" && s.row > FC1.lastC1 && FC1.wrongRows.includes(s.row)) flag = `<div class="c1flag">This row read row ${s.row - 1}'s inflated values, so its results (red) differ from the correct ones.</div>`;
    } else if (s.kind === "store") {
      h = `<h3>${side === "now" ? `Line ${lastLine(s.row)}` : "Rescale, part 3"} · store the scaling factor</h3><p>The sum is kept as the backward scaling factor for base ${s.t}.${side === "fix" ? " Same value as the left panel: moving the rescale doesn't change the sum, because the sum only uses the emitting cells." : ""}</p>
      <div class="eqbox mono">scaling_factors[${s.t}] = sb[${s.t}] = <b>${fmt(s.sb !== undefined ? s.sb : s.sum)}</b></div>`;
    }
    return h + flag;
  }

  // ----- animated equation: which term of the backward sum this step computes
  const grp = [];
  for (let i = 0; i < ST.length; i++) {
    const s = ST[i];
    if (["emit", "silent", "silent0"].includes(s.kind)) {
      let a = i; while (a > 0 && ST[a - 1].kind === s.kind && ST[a - 1].row === s.row && ST[a - 1].j === s.j) a--;
      let z = i; while (z < ST.length - 1 && ST[z + 1].kind === s.kind && ST[z + 1].row === s.row && ST[z + 1].j === s.j) z++;
      grp[i] = [a, z];
    }
  }
  function eqPanel(i) {
    const s = ST[i], j = s.j, t = s.t;
    const fT = x => `<span class="fT">${x}</span>`, fE = x => `<span class="fE">${x}</span>`, fB = (x, bad) => `<span class="fB${bad ? " bad" : ""}">${x}</span>`;
    let gen = "", inst = "", foot = "";
    if (s.kind === "init") {
      gen = `b[t, j] = <span class="tcur">end(j)</span> &nbsp;&nbsp;<span class="muted">at the last base: nothing comes after it</span>`;
      inst = `b[${t}, ${SH[j]}] = end(${SH[j]}) = <b>${fmt(s.val)}</b>`;
    } else if (s.kind === "emit" || s.kind === "silent" || s.kind === "silent0") {
      const silent = s.kind !== "emit";
      const bnext = silent ? "t" : "t+1", epos = silent ? "t" : "t+1";
      const hatSum = silent ? (side === "fix" ? "b̂" : "b") : "b̂";
      gen = `b[t, ${silent ? "s" : "j"}] = Σ<sub>c</sub> <span class="tcur">${fT(`T(${silent ? "s" : "j"}→c)`)} · ${fE(`e<sub>${epos}</sub>(c)`)} · ${fB(`${hatSum}[${bnext}, c]`)}</span>` +
            ` &nbsp;<span class="muted">${silent ? `silent: a junction just before base t, so the child emits base t${side === "fix" ? "; the child is already rescaled" : "; the child is <b>not</b> rescaled yet"}` : "c = each state j can move to; c emits base t+1"}</span>`;
      const [a, z] = grp[i];
      const terms = [];
      for (let k = a; k <= z; k++) {
        const q = ST[k], c = q.child, cs = c >= S;
        const tb = silent ? t : t + 1;
        const badB = side === "now" && silent && !cs && FC1.stepKeys.has(`${q.row},${q.j},${q.child}`);
        const sym = `${fT(`T(${SH[j]}→${SH[c]})`)}${cs ? "" : " · " + fE(`e<sub>${tb}</sub>(${SH[c]})`)} · ${fB(`${hatSum}[${tb}, ${SH[c]}]`, badB)}`;
        const num = `${fT(q.T)}${cs ? "" : " · " + fE(fmt(q.ev))} · ${fB(fmt(q.bchild), badB)}`;
        const cls = k < i ? "tdone" : (k === i ? "tcur" : "tnext");
        terms.push(`<span class="term ${cls}"><span class="sym">${sym}</span>${k <= i ? `<span class="num">${num} = <b>${fmt(q.val - q.old)}</b></span>` : `<span class="num">not computed yet</span>`}</span>`);
      }
      inst = `b[${t}, ${SH[j]}] = ` + terms.join(`<span class="plus">+</span>`);
      foot = `running total after this step: ${fmt(s.old)} + ${fmt(s.val - s.old)} = <b>${fmt(s.val)}</b>` +
             (z > i ? ` &nbsp;<span class="muted">(${z - i} more term${z - i > 1 ? "s" : ""} to add)</span>` : ` &nbsp;<span class="muted">(last term: this cell is complete)</span>`);
    } else {
      gen = `b̂[t, j] = <span class="tcur">b[t, j] ÷ Σ<sub>emitting k</sub> b[t, k]</span> &nbsp;<span class="muted">rescale ${side === "fix" ? "the emitting cells only" : "the whole row, silent cells included"}</span>`;
      if (s.kind === "nsum") inst = `Σ<sub>emitting k</sub> b[${t}, k] = <b>${fmt(s.sum)}</b>`;
      else if (s.kind === "ndiv") inst = `every ${side === "fix" ? "emitting " : ""}cell of row ${s.row} ÷ ${fmt(s.sum)}`;
      else inst = `sb[${t}] = ${fmt(s.sb)} &nbsp;<span class="muted">kept so the scaling can be undone later (log-likelihood, posterior)</span>`;
    }
    return `<div class="eqanim"><div class="gen">${gen}</div><div class="inst">${inst}</div>${foot ? `<div class="foot">${foot}</div>` : ""}</div>`;
  }

  const NSUM = MM([17, 19, 20]), NDIV = MM([23, 24]), NSTORE = MM([27]);
  function render() {
    const s = ST[cur], st = replay(cur);
    const [act, ctx] = hl(s);
    if ($(P + "CodeB")) $(P + "CodeB").innerHTML = codeHTML(code, act, ctx);
    if (side === "now" && $(P + "CodeN")) $(P + "CodeN").innerHTML = codeHTML(FLB.code_n.map(([ln, txt]) => [String(ln), String(ln), txt]), (s.kind === "nsum" ? NSUM : s.kind === "ndiv" ? NDIV : s.kind === "store" ? NSTORE : []), []);
    const box = $(P + "CodeB"); const el = box && typeof box.querySelector === "function" ? box.querySelector(".hl") : null;
    if (el && !$(tabId).hidden && typeof box.scrollTo === "function") box.scrollTo({ top: Math.max(0, el.offsetTop - box.offsetTop - 60) });
    const slot = (label, val) => `<div class="vslot"><span class="vlab">${label}</span><span class="vval">${val === undefined ? "—" : val}</span></div>`;
    $(P + "Vars").innerHTML =
      slot("i (row)", s.row) +
      slot("t (base)", `${s.t} ${FLB.seq[s.t] || ""}`) +
      slot("j", s.j !== undefined ? `${s.j} ${SH[s.j]}` : undefined) +
      slot("child", s.child !== undefined ? `${s.child} ${SH[s.child]}` : undefined) +
      slot("emissionValues", s.ev !== undefined ? fmt(s.ev) : undefined);
    let ex = explain(s);
    if (tabId !== "tab-flbl") ex = ex.replace(/<h3>[^<·]*(?:[Ll]ines?|normalize)[^<·]*· /, "<h3>");   // no code references outside the code tab
    $(P + "Exp").innerHTML = eqPanel(cur) + ex;
    let gh = `<thead><tr><th class="st">btable row<br>(= base t)</th>` + [...Array(12).keys()].map(j => `<th class="${j === S ? "sil1" : ""}${j >= S ? " silh" : ""}" title="${NM[j]}">j=${j}<br>${SH[j]}</th>`).join("") + "</tr></thead><tbody>";
    for (let r = 0; r < n; r++) {
      gh += `<tr><th class="st">row ${r} <span class="muted">t=${n - 1 - r} ${FLB.seq[n - 1 - r]}</span></th>`;
      for (let j = 0; j < 12; j++) {
        const val = st.g[r][j];
        let cls = j === S ? "sil1" : "";
        const target = (s.row === r && s.j === j) || ((s.kind === "ndiv" || s.kind === "nsum") && s.row === r && (j < S || (side === "now" && s.kind === "ndiv")));
        const read = s.child !== undefined && s.readrow === r && s.child === j;
        if (target) cls += " active";
        if (read) cls += " readc";
        if (val === null) cls += " empty";
        else if (!st.post[r][j]) cls += " pre";
        if (val !== null && st.post[r][j] && isWrong(r, j, val)) cls += " badc";
        if (val !== null && !st.post[r][j] && val > 0 && val < DMIN) cls += " badc";
        gh += `<td class="${cls}${(s.row === r && s.j === j) ? " flash" : ""}" title="${NM[j]}, t=${n - 1 - r}">${val === null ? "·" : fmt(val)}</td>`;
      }
      gh += "</tr>";
    }
    $(P + "Grid").innerHTML = gh + "</tbody>";
    $(P + "Count").textContent = `step ${cur + 1} / ${ST.length}`;
    [...$(P + "Jumps").children].forEach(b => b.classList.toggle("on", +b.dataset.r === s.row));
  }
  function go(i) { cur = Math.max(0, Math.min(ST.length - 1, i)); render(); }
  function stop() { if (timer) { clearInterval(timer); timer = null; } $(P + "Play").textContent = "▶ Play"; }
  function play() {
    if (timer) { stop(); return; }
    if (cur >= ST.length - 1) go(0);
    $(P + "Play").textContent = "❚❚ Pause";
    timer = setInterval(() => { if (cur >= ST.length - 1) { stop(); return; } go(cur + 1); }, +$(P + "Speed").value);
  }
  $(P + "Prev").addEventListener("click", () => { stop(); go(cur - 1); });
  $(P + "Next").addEventListener("click", () => { stop(); go(cur + 1); });
  $(P + "Play").addEventListener("click", play);
  $(P + "Speed").addEventListener("change", () => { if (timer) { stop(); play(); } });
  go(0);
  function setWait(msg) { const w = $(P + "Wait"); if (!w) return; w.textContent = msg || ""; w.className = "wait" + (msg ? " on" : ""); }
  return { go, stop, play, step: d => { stop(); go(cur + d); }, count: () => ST.length, cur: () => cur, setWait, steps: ST };
}
/* ----- linked mode: merge the two step lists into one timeline ----- */
function fMakeSync(Q, now, fix, tabId) {
  const $ = id => document.getElementById(id);
  const A = FLB.steps, B = FLB.steps_fix;
  const isRes = s => s.kind === "nsum" || s.kind === "ndiv" || s.kind === "store";
  const key = s => s.kind === "init" ? `i${s.row},${s.j}` : (s.kind === "emit" ? `e${s.row},${s.j},${s.child}` : (isRes(s) ? `r${s.kind}${s.row}` : `s${s.row},${s.j},${s.child}`));
  const frames = [[0, 0, "both"]];
  let a = 0, b = 0, mismatches = 0;
  while (a < A.length - 1 || b < B.length - 1) {
    const na = a + 1 < A.length ? A[a + 1] : null, nb = b + 1 < B.length ? B[b + 1] : null;
    if (na && nb && key(na) === key(nb)) { a++; b++; frames.push([a, b, "both"]); }
    else if (na && (isRes(na) || !nb)) { a++; frames.push([a, b, "now"]); }
    else if (nb && (isRes(nb) || !na)) { b++; frames.push([a, b, "fix"]); }
    else { a++; b++; mismatches++; frames.push([a, b, "both"]); }
  }
  let cur = 0, timer = null;
  const firstOfRow = {};
  frames.forEach(([ia, ib], k) => { const r = Math.min(A[ia].row, B[ib].row); if (!(r in firstOfRow) && A[ia].row === r && B[ib].row === r) firstOfRow[r] = k; });
  $(Q + "Jumps").innerHTML = Object.keys(firstOfRow).map(r => `<button type="button" data-k="${firstOfRow[r]}" data-r="${r}"${FC1.rowSet.has(+r) ? ' class="c1j"' : ""}>${fRowLabel(r)}</button>`).join("");
  function show() {
    const [ia, ib, mv] = frames[cur];
    now.go(ia); fix.go(ib);
    const waitMsg = side => {
      if (cur === 0 || mv === "both" || mv === side) return "";
      const other = side === "now" ? B[ib] : A[ia];
      return `⏸ waiting while the other side ${other.kind === "nsum" ? "adds up the row" : other.kind === "ndiv" ? "divides the row" : "stores the scaling factor"}`;
    };
    now.setWait(waitMsg("now")); fix.setWait(waitMsg("fix"));
    $(Q + "Count").textContent = `linked step ${cur + 1} / ${frames.length}`;
    [...$(Q + "Jumps").children].forEach(btn => btn.classList.toggle("on", +btn.dataset.r === Math.min(A[ia].row, B[ib].row)));
  }
  function go(k) { cur = Math.max(0, Math.min(frames.length - 1, k)); show(); }
  function stop() { if (timer) { clearInterval(timer); timer = null; } $(Q + "Play").textContent = "▶ Play"; }
  function play() {
    if (timer) { stop(); return; }
    if (cur >= frames.length - 1) go(0);
    $(Q + "Play").textContent = "❚❚ Pause";
    timer = setInterval(() => { if (cur >= frames.length - 1) { stop(); return; } go(cur + 1); }, +$(Q + "Speed").value);
  }
  function setLinked(on) {
    $(tabId).classList.toggle("linked", on);
    $(Q + "Ctl").hidden = !on; $(Q + "Jumps").hidden = !on; $(Q + "Note").hidden = !on;
    stop(); now.stop(); fix.stop();
    if (on) { const ia = now.cur(); let k = frames.findIndex(f => f[0] === ia); go(k < 0 ? 0 : k); }
    else { now.setWait(""); fix.setWait(""); }
  }
  $(Q + "Prev").addEventListener("click", () => { stop(); go(cur - 1); });
  $(Q + "Next").addEventListener("click", () => { stop(); go(cur + 1); });
  $(Q + "Play").addEventListener("click", play);
  $(Q + "Speed").addEventListener("change", () => { if (timer) { stop(); play(); } });
  $(Q + "Jumps").addEventListener("click", e => { const btn = e.target.closest("button"); if (btn) { stop(); go(+btn.dataset.k); } });
  $(Q + "Link").addEventListener("change", e => setLinked(e.target.checked));
  setLinked(true);
  return { linked: () => $(Q + "Link").checked, step: d => { stop(); go(cur + d); }, stop, frames, mismatches, go };
}
const fblNow = fMakeStepper("fbl", FLB.steps, "now", "tab-flbl");
const fblFix = fMakeStepper("fbk", FLB.steps_fix, "fix", "tab-flbl");
const fblSync = fMakeSync("fbs", fblNow, fblFix, "tab-flbl");
window.flblStop = () => { fblNow.stop(); fblFix.stop(); fblSync.stop(); };
document.addEventListener("keydown", e => {
  if (document.getElementById("tab-flbl").hidden) return;
  if (e.target.closest && e.target.closest("input, textarea, select")) return;
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  e.preventDefault();
  const d = e.key === "ArrowRight" ? 1 : -1;
  if (fblSync.linked()) fblSync.step(d); else (e.shiftKey ? fblFix : fblNow).step(d);
});
/* intro text for the line-by-line tab, from the data */
(function () {
  const sc = FLB.scenario, pos = sc.n_w.map((v, t) => t).filter(t => sc.n_w[t] !== sc.n_w[0] || sc.k_w[t] !== sc.k_w[0]);
  const el = document.getElementById("fblIntro");
  el.innerHTML = `The backward pass replayed one statement at a time, on the Fiber-seq C1 example: ${sc.n_w[pos[0]]} reads at positions ${pos[0]}–${pos[pos.length - 1]}, all ${sc.k_w[pos[0]]} methylated (${sc.n_w[0]} reads, ${sc.k_w[0]} methylated elsewhere). ` +
    `<b>Left:</b> fiberCOP's <span class="mono">bward</span> as written. This one code is used by both original RoboCOP and production fix B (fix B only changed the posterior step), so both get exactly this table. Line numbers are the fork's <span class="mono">algo.c</span> (upstream + ${FLB.line_offset}). ` +
    `<b>Right:</b> the same loops with one change: each row's emitting cells are rescaled <i>before</i> the silent states are computed (the audit's C1 fix). ` +
    `C1 starts at row ${FLB.c1_rows.first_row} (t=${FLB.c1_rows.first_row_t}), one row earlier than in the MNase example; the red "C1" buttons jump there. With the panels linked, one set of controls drives both (←/→ also works). Unlink them to step each side on its own (←/→ left, Shift+←/→ right).`;
})();
