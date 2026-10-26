// Smoke test for build/c1_underflow_steps_fibercop.html (minimal DOM shim, as the audit's run*.js).
// usage: node tests/smoke.js <base.html> <fibercop.html>
const fs = require("fs");
function load(file) {
  const html = fs.readFileSync(file, "utf8");
  const m = html.match(/<script>([\s\S]*)<\/script>/); if (!m) throw new Error("no script");
  const els = {}, store = {};
  const mk = id => ({ id, innerHTML: "", textContent: "", value: "1300", checked: true, hidden: false, offsetTop: 0, className: "",
    style: { setProperty() {} }, children: [], dataset: {}, L: {}, attrs: {},
    addEventListener(e, f) { (this.L[e] = this.L[e] || []).push(f); }, setAttribute(k, v) { this.attrs[k] = v; }, getAttribute() { return "0"; },
    getBoundingClientRect() { return { width: 760 }; }, scrollIntoView() {}, scrollTo() {}, querySelector() { return null; },
    closest() { return mk("c"); }, classList: { toggle() {} } });
  const el = id => (els[id] = els[id] || mk(id));
  const sandbox = {
    document: { documentElement: { style: { setProperty() {} } }, getElementById: el, addEventListener() {}, querySelectorAll: () => [], querySelector: () => mk("q") },
    window: { innerWidth: 1600, scrollTo() {}, addEventListener() {} }, matchMedia: () => ({ matches: false }), location: { hash: "" }, history: { replaceState() {} },
    setInterval: () => 1, clearInterval: () => {}, localStorage: { getItem: k => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = v; } },
  };
  const fn = new Function(...Object.keys(sandbox), m[1] + "\n;return { showTab, fblNow: typeof fblNow !== 'undefined' ? fblNow : null, fblFix: typeof fblFix !== 'undefined' ? fblFix : null, fblSync: typeof fblSync !== 'undefined' ? fblSync : null, fpNow: typeof fpNow !== 'undefined' ? fpNow : null, fpFix: typeof fpFix !== 'undefined' ? fpFix : null, fpSync: typeof fpSync !== 'undefined' ? fpSync : null, lblNow, lblFix, syncL, piNow, piFix, syncP };");
  const api = fn(...Object.values(sandbox));
  return { html, els, api };
}
const [baseF, newF] = process.argv.slice(2);
const A = load(baseF), Bp = load(newF);
let fail = 0; const say = (ok, msg) => { console.log((ok ? "PASS " : "FAIL ") + msg); if (!ok) fail++; };
const baseIds = Object.keys(A.els);
// 1. old tabs: identical DOM output after load, and after stepping every old stepper to every step
const diffIds = baseIds.filter(id => A.els[id].innerHTML !== Bp.els[id].innerHTML || A.els[id].textContent !== Bp.els[id].textContent || A.els[id].hidden !== Bp.els[id].hidden);
say(diffIds.length === 0, `old tabs after load: ${baseIds.length} elements compared, ${diffIds.length} differ ${diffIds.slice(0, 5)}`);
let oldSteps = 0, oldDiff = 0;
for (const k of ["lblNow", "lblFix", "piNow", "piFix"]) {
  const n = A.api[k].count();
  for (let i = 0; i < n; i++) { A.api[k].go(i); Bp.api[k].go(i); oldSteps++;
    for (const id of baseIds) if (A.els[id].innerHTML !== Bp.els[id].innerHTML) { oldDiff++; break; } }
}
for (const k of ["syncL", "syncP"]) { for (let i = 0; i < A.api[k].frames.length; i++) { A.api[k].go(i); Bp.api[k].go(i); oldSteps++;
  for (const id of baseIds) if (A.els[id].innerHTML !== Bp.els[id].innerHTML) { oldDiff++; break; } } }
say(oldDiff === 0, `old steppers: ${oldSteps} steps rendered in both pages, ${oldDiff} differ`);
// old tabs still switch
for (const t of ["pi", "c1", "brute", "eq", "lbl", "emp", "fpi", "flbl", "femp"]) {
  Bp.api.showTab(t);
  const panel = { pi: "tab-pi", c1: "tab-c1", brute: "tab-bf", eq: "tab-eq", lbl: "tab-lbl", emp: "tab-emp", fpi: "tab-fpi", flbl: "tab-flbl", femp: "tab-femp" };
  const shown = Object.values(panel).filter(p => !Bp.els[p].hidden);
  say(shown.length === 1 && shown[0] === panel[t], `showTab("${t}") shows only ${shown}`);
}
// 2. new tabs: every step renders, no NaN / undefined / Infinity text
const newIds = () => Object.keys(Bp.els).filter(id => !baseIds.includes(id));
const bad = () => newIds().filter(id => /NaN|undefined|Infinity|\[object/.test(Bp.els[id].innerHTML + Bp.els[id].textContent));
let steps = 0, badSteps = 0;
for (const k of ["fblNow", "fblFix", "fpNow", "fpFix"]) { const st = Bp.api[k]; for (let i = 0; i < st.count(); i++) { st.go(i); steps++; if (bad().length) badSteps++; } }
for (const k of ["fblSync", "fpSync"]) { const st = Bp.api[k]; for (let i = 0; i < st.frames.length; i++) { st.go(i); steps++; if (bad().length) badSteps++; } }
say(badSteps === 0 && steps > 0, `new steppers: ${steps} steps rendered, ${badSteps} with NaN/undefined (mismatches: ${Bp.api.fblSync.mismatches}, ${Bp.api.fpSync.mismatches})`);
// click every scenario / truth / frac button in the new empirical tab
const click = (id, ds) => Bp.els[id].L.click.forEach(f => f({ target: { closest: () => ({ dataset: ds }) } }));
let clicks = 0;
for (let i = 0; i < 4; i++) for (const t of ["brute", "brute_exact"]) { click("feScen", { i: String(i) }); click("feTruth", { k: t }); clicks++; if (bad().length) say(false, `scenario ${i} truth ${t}: ${bad()}`); }
for (const f of ["1", "0", "0.6"]) for (const t of ["", "_exact"]) { click("fdFrac", { f }); click("fdTruth", { t }); clicks++; if (bad().length) say(false, `frac ${f} truth ${t}: ${bad()}`); }
say(bad().length === 0, `new empirical tab: ${clicks} button states rendered, problems: ${bad()}`);
const empty = newIds().filter(id => /^(fpi|fe|fd|fbl|fbk|fbs|fpl|fpk|fps)/.test(id) && !/(Wait|Tip|Speed|Link|Play|Prev|Next|Ctl|CodeB|Note|Bar)$/.test(id) && !Bp.els[id].innerHTML && !Bp.els[id].textContent);
say(empty.length === 0, `new elements left empty: ${empty}`);
// print the generated prose for review
const strip = h => h.replace(/<\/(li|tr|p)>/g, "\n").replace(/<[^>]+>/g, " ").replace(/[ \t]+/g, " ").replace(/\n /g, "\n").trim();
if (process.env.SHOW) for (const id of process.env.SHOW.split(",")) console.log(`--- ${id}\n${strip(Bp.els[id].innerHTML || Bp.els[id].textContent)}`);
process.exit(fail ? 1 : 0);
