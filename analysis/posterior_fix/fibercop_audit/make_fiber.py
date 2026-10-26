"""Add three fiberCOP tabs to the C-audit page.

Reads the audit page (read only) and build/f_*.json + build/realdata.json (+ realseg/results.json if present),
inserts three tab buttons, panels and JS after the existing ones, and writes build/c1_underflow_steps_fibercop.html.
Existing tabs are not edited: new panels, ids, data constants and functions all carry new names.

    python make_fiber.py [--base PATH] [--out PATH]
"""
import argparse, csv, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/usr/project/xtmp/nd141/programs/robocop_c_code_test/robocop_c_audit/walkthrough/c1_underflow_steps.html"
B = os.path.join(HERE, "build")
T = os.path.join(HERE, "tabs")
SEG_JSON = os.path.join(HERE, "realseg", "results.json")
SEG_COORD = os.path.join(HERE, "..", "coord_rdna_chrXII_seg2.tsv")   # the segment results.json was run on (realseg/README.md)

CSS = """
/* ---- fiberCOP tabs ---- */
#tab-fpi, #tab-flbl, #tab-femp { display: grid; gap: 16px; }
#tab-fpi[hidden], #tab-flbl[hidden], #tab-femp[hidden] { display: none; }
#tab-flbl { max-width: none; gap: 14px; }
#tab-femp { gap: 18px; }
#tab-fpi .lblside { padding: 10px; }
#tab-fpi table.grid th, #tab-fpi table.grid td { padding: 2px 4px; }
#tab-fpi .lblpair { margin-top: 4px; }
.fpilist { margin: 0; padding-left: 1.4em; display: grid; gap: 8px; font-size: 15px; line-height: 1.5; max-width: 82ch; }
.fpilist li::marker { font-weight: 600; color: var(--muted); }
.pitake ol.fpilist { list-style: lower-alpha; }
table.pit td.fpifl, .fpifl { outline: 2px solid var(--warn); outline-offset: -2px; background: var(--warn-soft); }
span.fpifl { padding: 0 4px; border-radius: 3px; }
.pill.fhalf { background: var(--warn-soft); color: var(--warn); }
.fviz { --s4: #8a55c9; --s5: #6b7785; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .fviz { --s4: #a77ee0; --s5: #9aa8b5; } }
:root[data-theme="dark"] .fviz { --s4: #a77ee0; --s5: #9aa8b5; }
.fseg ul { margin: 0; padding-left: 1.3em; display: grid; gap: 6px; max-width: 82ch; }
"""

SEG_PI = """
    <section class="pisec fseg">
      <div class="pinum">Real model, one rDNA segment</div>
      <p id="fpiSegIntro"></p>
      <div class="gridbox"><table class="mini pit" id="fpiSegTable"></table></div>
      <ul id="fpiSegRead"></ul>
    </section>
"""
SEG_EMP = """
    <section class="bfsec fseg">
      <div class="phase">Real model</div>
      <h2 class="bfh">One real rDNA segment, every decoder</h2>
      <p id="feSegIntro"></p>
      <div class="gridbox"><table class="mini cmp" id="feSegTable"></table></div>
      <ul id="feSegRead"></ul>
    </section>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=os.path.join(B, "c1_underflow_steps_fibercop.html"))
    a = ap.parse_args()
    base = open(a.base, encoding="utf-8").read()
    s = base

    def rep(old, new, cnt=1):
        nonlocal s
        assert s.count(old) == cnt, (s.count(old), old[:90])
        s = s.replace(old, new)

    load = lambda p: json.load(open(p))
    FD, FLB, FEMP, FDEP, FREAL = (load(os.path.join(B, f)) for f in ("f_d.json", "f_lbl.json", "f_emp.json", "f_dep.json", "realdata.json"))
    FSEG = None
    if os.path.exists(SEG_JSON):
        FSEG = load(SEG_JSON)
        if os.path.exists(SEG_COORD):
            with open(SEG_COORD) as fh:
                row = next(csv.DictReader(fh, delimiter="\t"))
            FSEG["segment"] = {"chrom": row["chr"], "start": int(row["start"]), "end": int(row["end"]),
                               "file": os.path.relpath(os.path.realpath(SEG_COORD), os.path.realpath(os.path.join(HERE, "..", "..")))}
            assert FSEG["segment"]["end"] - FSEG["segment"]["start"] + 1 == FSEG["n_obs"], "segment length != n_obs"

    # ---- html snippets (fork line numbers filled from FLB.line_map)
    def lines(txt):
        return re.sub(r"@@L:(\d+)@@", lambda m: str(FLB["line_map"][m.group(1)]), txt)
    pi_html = open(os.path.join(T, "fpi_tab.html")).read().replace("@@SEG_PI@@", SEG_PI if FSEG else "")
    lbl_html = lines(open(os.path.join(T, "flbl_tab.html")).read())
    emp_html = open(os.path.join(T, "femp_tab.html")).read().replace("@@SEG_EMP@@", SEG_EMP if FSEG else "")
    new_html = pi_html + "\n" + lbl_html + "\n" + emp_html
    assert "@@" not in new_html

    # new ids must not exist in the base page
    base_ids = set(re.findall(r'id="([^"]+)"', base))
    new_ids = re.findall(r'id="([^"]+)"', new_html)
    assert len(new_ids) == len(set(new_ids)), "duplicate id in new panels"
    clash = base_ids & set(new_ids)
    assert not clash, clash

    # ---- buttons, panels, css
    rep("""    <button type="button" role="tab" id="tabEMP" aria-selected="false" aria-controls="tab-emp">Empirical results</button>""",
        """    <button type="button" role="tab" id="tabEMP" aria-selected="false" aria-controls="tab-emp">Empirical results</button>
    <button type="button" role="tab" id="tabFPI" aria-selected="false" aria-controls="tab-fpi">PI report (fiberCOP)</button>
    <button type="button" role="tab" id="tabFLBL" aria-selected="false" aria-controls="tab-flbl">Backward, line by line (fiberCOP)</button>
    <button type="button" role="tab" id="tabFEMP" aria-selected="false" aria-controls="tab-femp">Empirical results (fiberCOP)</button>""")
    rep("""\n  <div class="pagew" role="group" aria-label="Page width">""", "\n" + new_html + """\n  <div class="pagew" role="group" aria-label="Page width">""")
    rep(".tabs { display: flex;", CSS.lstrip() + ".tabs { display: flex;")

    # ---- js: data + code, before the tab switcher
    js = open(os.path.join(T, "flbl_tab.js")).read()
    js = js.replace("__FD__", json.dumps(FD)).replace("__FLB__", json.dumps(FLB))
    js2 = open(os.path.join(T, "femp_tab.js")).read()
    js2 = js2.replace("__FEMP__", json.dumps(FEMP)).replace("__FDEP__", json.dumps(FDEP)).replace("__FREAL__", json.dumps(FREAL)).replace("__FSEG__", json.dumps(FSEG))
    js3 = open(os.path.join(T, "fpi_tab.js")).read()
    for x in (js, js2, js3):
        assert not re.search(r"__F[A-Z]+__", x)
    rep("\n/* ================= tabs ================= */", "\n" + js + "\n" + js2 + "\n" + js3 + "\n/* ================= tabs ================= */")
    rep("""lbl: ["tabLBL", "tab-lbl"], emp: ["tabEMP", "tab-emp"] };""",
        """lbl: ["tabLBL", "tab-lbl"], emp: ["tabEMP", "tab-emp"], fpi: ["tabFPI", "tab-fpi"], flbl: ["tabFLBL", "tab-flbl"], femp: ["tabFEMP", "tab-femp"] };""")
    rep("""  if (name !== "lbl") lbl.stopL();""", """  if (name !== "lbl") lbl.stopL();
  if (name !== "fpi" && window.fpiStop) window.fpiStop();
  if (name !== "flbl" && window.flblStop) window.flblStop();""")
    rep("""document.getElementById("tabEMP").addEventListener("click", () => showTab("emp"));""",
        """document.getElementById("tabEMP").addEventListener("click", () => showTab("emp"));
document.getElementById("tabFPI").addEventListener("click", () => showTab("fpi"));
document.getElementById("tabFLBL").addEventListener("click", () => showTab("flbl"));
document.getElementById("tabFEMP").addEventListener("click", () => showTab("femp"));""")
    rep("""if (location.hash === "#emp") showTab("emp");""", """if (location.hash === "#emp") showTab("emp");
if (location.hash === "#fpi") showTab("fpi");
if (location.hash === "#flbl") showTab("flbl");
if (location.hash === "#femp") showTab("femp");""")

    assert "</script" not in js + js2 + js3, "data would close the script tag"
    open(a.out, "w", encoding="utf-8").write(s)
    body = s[:s.index("<script>")]
    print("wrote", a.out, len(s.encode()), "bytes | realseg section:", "yes" if FSEG else "no (results.json absent)",
          "| divs", len(re.findall(r"<div\b", body)), body.count("</div>"),
          "| sections", len(re.findall(r"<section\b", body)), body.count("</section>"))


if __name__ == "__main__":
    main()
