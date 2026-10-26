#!/usr/bin/env python3
"""
build/f_lbl.json -- data for the "Backward, line by line" tab, fiberCOP "c1" scenario.

Same schema as the audit's lbl.json (robocop_c_audit/walkthrough/artifact_build/gen_lbl.py +
gen_lbl_fix.py), with these changes:
  * model = fiber_model "c1" (n_vars = 7: sequence PWM, 4 MNase/ATAC layers = 1, Watson, Crick).
  * "Now" steps come from the audit's traced upstream algo.c (LIB_TRACED) run with n_vars = 7.
    fward/bward/normalize are byte-identical between upstream, the fork's original algo.c and
    production fix B, so these are the backward steps of BOTH the original code and fix B.
  * "Fix" steps are a Python replay of the C1-fixed order (rescale emitting cells, then silent),
    looping over 7 layers; checked bit-for-bit against LIB_C1FIX's btable.
  * Code listings and every line number are the FORK's algo.c (= upstream + 2: two #includes).
  * New keys: line_offset, source, line_map, scenario, c1_rows, c1_first, emit_log10.

Run:  python gen_f_lbl.py      (from anywhere; writes build/f_lbl.json and build/f_trace.tsv)
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fiber_model as FM

BUILD = os.path.join(HERE, "build"); os.makedirs(BUILD, exist_ok=True)
FORK_SRC = os.path.join(FM.FORK, "pkg/robocop/algo.c")
UP_SRC = "/usr/project/xtmp/nd141/programs/robocop_c_code_test/RoboCOP/pkg/robocop/algo.c"
OFF = 2
DM = np.finfo(float).tiny

# ---- verify the +2 offset: upstream lines 1..226 == fork lines 3..228
up = open(UP_SRC).read().split("\n"); fk = open(FORK_SRC).read().split("\n")
assert up[:226] == fk[OFF:226 + OFF], "fork algo.c is not upstream + 2 over lines 1-226"
F = lambda L: L + OFF                     # upstream line -> fork line

scen = {k: d for k, _, _, d in FM.SCENARIOS}["c1"]

# ---------------- NOW: traced upstream algo.c, n_vars = 7
tr = FM.load_lin(FM.LIB_TRACED)
m = FM.make_model(tr, **scen)
tp = os.path.join(BUILD, "f_trace.tsv")
r_now = FM.run_lin(tr, m, trace=tp)
n, S, N, T = m.T_obs, m.S, m.N, m.T
assert m.E.shape[0] == 7

# the traced run must equal the fork's original (pre fix B) library, and its bward the fix-B library's
pre = FM.run_lin(FM.load_lin(FM.LIB_PRENORM), m)
fb = FM.run_lin(FM.load_lin(FM.LIB_FIXB, v2=True), m)
for k in ("ftable", "btable", "post"):
    assert np.array_equal(r_now[k], pre[k]), f"traced != prenorm in {k}"
assert np.array_equal(r_now["btable"], fb["btable"]) and np.array_equal(r_now["sb"], fb["sb"]), "fix B bward differs"

recs = []
for ln in open(tp):
    p = ln.rstrip("\n").split("\t")
    if p[0] != "bward": continue
    recs.append(dict(line=int(p[1]), i=int(p[2]), j=int(p[3]), k=int(p[4]), a=float(p[5]), b=float(p[6]), c=float(p[7]), r=float(p[8])))
steps = []
k = 0
while k < len(recs):
    x = recs[k]; L = x["line"]
    row = x["i"]; t = n - 1 - row
    if L == 150:
        steps.append(dict(kind="init", row=row, t=t, j=x["j"], val=x["r"], lines=[F(149), F(150)])); k += 1
    elif L in (159, 175, 190):
        y = recs[k + 1]; assert y["line"] in (160, 176, 192) and y["j"] == x["j"] and y["k"] == x["k"]
        kind = {159: "silent0", 175: "emit", 190: "silent"}[L]
        readrow = row - 1 if L == 175 else row
        epos = {159: n - 1, 175: n - row, 190: n - row - 1}[L]
        lines = {159: [154, 155, 157, 158, 159, 160], 175: [169, 170, 171, 173, 174, 175, 176, 177, 178, 179],
                 190: [185, 186, 188, 189, 190, 192, 193, 194, 195]}[L]
        steps.append(dict(kind=kind, row=row, t=t, j=y["j"], child=y["k"], ev=x["r"], epos=epos, readrow=readrow,
                          old=y["a"], bchild=y["b"], T=y["c"], val=y["r"], lines=[F(v) for v in lines]))
        k += 2
    elif L == 20:
        terms = []
        while k < len(recs) and recs[k]["line"] == 20:
            terms.append(recs[k]["b"]); s_ = recs[k]["r"]; k += 1
        steps.append(dict(kind="nsum", row=row, t=t, terms=terms, sum=s_, lines=[F(166 if row == 0 else 201)], nlines=[F(17), F(19), F(20)]))
    elif L == 24:
        before, after = [], []
        while k < len(recs) and recs[k]["line"] == 24:
            before.append(recs[k]["a"]); after.append(recs[k]["r"]); s_ = recs[k]["c"]; k += 1
        steps.append(dict(kind="ndiv", row=row, t=t, before=before, after=after, sum=s_, lines=[F(166 if row == 0 else 201)], nlines=[F(23), F(24)]))
    elif L == 27:
        k += 1
    elif L in (166, 201):
        steps.append(dict(kind="store", row=row, t=x["k"], sb=x["r"], lines=[F(L)], nlines=[F(27)])); k += 1
    else:
        raise SystemExit(f"unexpected line {L}")

# trace's normalised values == the C btable
for s in steps:
    if s["kind"] == "ndiv":
        assert np.array_equal(np.array(s["after"]), r_now["btable"][s["row"]]), s["row"]

# ---------------- FIX: Python replay of the C1-fixed order, 7 layers, checked against LIB_C1FIX
Lyr = m.E
nv = Lyr.shape[0]
children = [[c for c in range(N) if T[j, c] != 0.0] for j in range(N)]   # same order as find_parents_and_children
def ev_of(pos, c):
    if c >= S: return 1.0
    v = 1.0
    for l in range(nv): v *= Lyr[l, pos, c]                                 # same order as the C loop
    return v
b = [[0.0] * N for _ in range(n)]
fixpre = [[0.0] * N for _ in range(n)]
steps_fix = []
for r in range(n):
    t = n - 1 - r
    if r == 0:
        for j in range(S):
            b[0][j] = float(m.end[j]); steps_fix.append(dict(kind="init", row=0, t=t, j=j, val=b[0][j]))
    else:
        for j in range(S):
            for c in children[j]:
                ev = ev_of(t + 1, c); old = b[r][j]
                b[r][j] += b[r - 1][c] * T[j, c] * ev
                steps_fix.append(dict(kind="emit", row=r, t=t, j=j, child=c, ev=ev, epos=t + 1, readrow=r - 1, old=old, bchild=b[r - 1][c], T=float(T[j, c]), val=b[r][j]))
    fixpre[r][:S] = b[r][:S]
    s = 0.0; terms = []
    for j in range(S):
        terms.append(b[r][j]); s += b[r][j] + DM
    steps_fix.append(dict(kind="nsum", row=r, t=t, terms=terms, sum=s))
    before = list(b[r])
    for j in range(S): b[r][j] = (b[r][j] + DM) / s
    steps_fix.append(dict(kind="ndiv", row=r, t=t, before=before[:S], after=b[r][:S], sum=s))
    steps_fix.append(dict(kind="store", row=r, t=t, sb=s))
    for j in range(N - 1, S - 1, -1):
        for c in children[j]:
            ev = ev_of(t, c); old = b[r][j]
            b[r][j] += b[r][c] * T[j, c] * ev
            steps_fix.append(dict(kind="silent", row=r, t=t, j=j, child=c, ev=ev, epos=t, readrow=r, old=old, bchild=b[r][c], T=float(T[j, c]), val=b[r][j]))
        fixpre[r][j] = b[r][j]
fix = np.array(b)
c1f = FM.run_lin(FM.load_lin(FM.LIB_C1FIX), m)
assert np.array_equal(fix, c1f["btable"]), "Python fix replay != LIB_C1FIX btable: max abs %g" % np.abs(fix - c1f["btable"]).max()
sbfix = np.array([s["sb"] for s in steps_fix if s["kind"] == "store"])[::-1]       # by position t
assert np.array_equal(sbfix, c1f["sb"])

# ---------------- C1 cells: where Now's final btable differs from Fix (same rule as lbl_tab.js isWrong)
now = r_now["btable"]
def wrong(v, c): return abs(v - c) > 1e-6 * max(abs(c), 1e-300) and abs(v - c) > 1e-250
c1_cells = [[int(rr), int(j)] for rr in range(n) for j in range(N) if wrong(now[rr, j], fix[rr, j])]
rows_bad = sorted({rr for rr, _ in c1_cells})
first = rows_bad[0] if rows_bad else None
# the C1 moment: silent steps in Now whose pre-normalise value fell below DBL_MIN, or that read an un-rescaled child
c1_steps = [i for i, s in enumerate(steps) if s["kind"] in ("silent", "silent0") and 0 < s["val"] < DM]
sub_dm = sorted({(steps[i]["row"], steps[i]["j"]) for i in c1_steps})
c1_rows = dict(
    rule="cell (row, j) of the final btable where Now differs from Fix: |now-fix| > 1e-6*max(|fix|,1e-300) and > 1e-250 (lbl_tab.js isWrong)",
    rows=rows_bad, cells=c1_cells,
    first_row=first, first_row_t=(n - 1 - first) if first is not None else None,
    first_row_cols=[j for rr, j in c1_cells if rr == first],
    subnormal_silent_cells=[[int(a), int(c)] for a, c in sub_dm],
    subnormal_silent_steps=c1_steps,
    now_vs_fix={f"{rr},{j}": [float(now[rr, j]), float(fix[rr, j])] for rr, j in c1_cells},
)

# ---------------- code listings (fork line numbers)
code_b = [[ln, fk[ln - 1]] for ln in range(F(147), F(202) + 1)]
code_n = [[ln, fk[ln - 1]] for ln in range(F(8), F(28) + 1)]
def C(u): return str(F(u))
code_fix = [
 [C(147), C(147), "    // initialization"],
 [C(149), C(149), "    for(j = 0; j < silent_states_begin; j++) {"],
 [C(150), C(150), "        btable[I(0, j, n_states)] = end_probs[j];"],
 [C(151), C(151), "    }"],
 ["NEW0", "new", "    scaling_factors[n_obs - 1] = normalize_emitting(btable, 0, ...);   // rescale the emitting cells FIRST"],
 [C(154), C(154), "    for(j = n_states - 1; j > silent_states_begin - 1; j--) {           // then the silent states"],
 [C(155), C(155), "        for(k = 0; k < n_children[j]; k++) {"],
 [C(157), C(157), "            child = children_mat[I(j,k,n_states)];"],
 [C(159), C(159), "            emissionValues = e_last(child);                 // 7 layers multiplied (fiberCOP n_vars = 7)"],
 [C(160), C(160), "            btable[I(0, j, n_states)] += btable[I(0, child, n_states)] * transition_mat[I(j, child, n_states)] * emissionValues;"],
 [C(163), C(163), "        }"],
 [C(164), C(164), "    }"],
 ["X" + C(166), "—", f"    // (line {C(166)} removed: the rescale moved up)"],
 [C(169), C(169), "    for(i = 1; i < n_obs; i++) {"],
 [C(170), C(170), "        for(j = 0; j < silent_states_begin; j++) {"],
 [C(171), C(171), "            for(k = 0; k < n_children[j]; k++) {"],
 [C(173), C(173), "                child = children_mat[I(j,k,n_states)];"],
 [C(175), C(175), "                emissionValues = e_{t+1}(child);"],
 [C(176), C(176), "                btable[I(i, j, n_states)] += btable[I(i - 1, child, n_states)] * transition_mat[I(j, child, n_states)] * emissionValues;"],
 [C(182), C(182), "            }"],
 [C(183), C(183), "        }"],
 ["NEW", "new", "        scaling_factors[n_obs - 1 - i] = normalize_emitting(btable, i, ...);   // ← rescale moved HERE"],
 [C(185), C(185), "        for(j = n_states - 1; j > silent_states_begin - 1; j--) {"],
 [C(186), C(186), "            for(k = 0; k < n_children[j]; k++) {"],
 [C(188), C(188), "                child = children_mat[I(j,k,n_states)];"],
 [C(190), C(190), "                emissionValues = e_t(child);"],
 [C(192), C(192), "                btable[I(i, j, n_states)] += btable[I(i, child, n_states)] * transition_mat[I(j, child, n_states)] * emissionValues;"],
 ["", "", "                //                          ↑ child's value is already rescaled"],
 [C(197), C(197), "            }"],
 [C(198), C(198), "        }"],
 ["X" + C(201), "—", f"        // (line {C(201)} removed: the rescale moved up)"],
 [C(202), C(202), "    }"],
]
# the listed fix lines must start with the fork's text where they claim to be unchanged (multi-line statements are joined)
for key, disp, txt in code_fix:
    if key.isdigit() and "e_" not in txt and "normalize" not in txt and "//" not in txt:
        assert txt.replace(" ", "").startswith(fk[int(key) - 1].replace(" ", "").replace("\t", "")), (key, fk[int(key) - 1], txt)

E = m.E.prod(0)
with np.errstate(divide="ignore"):
    emit_log10 = [[(float(v) / np.log(10) if np.isfinite(v) else None) for v in m.LE_fl.sum(0)[t, :S]] for t in range(n)]
out = dict(
    steps=steps, steps_fix=steps_fix, code_b=code_b, code_n=code_n, code_fix=code_fix,
    fix=[list(map(float, fix[rr])) for rr in range(n)], fixpre=[list(map(float, fixpre[rr])) for rr in range(n)],
    dbl_min=DM, n=n, S=S, seq=m.seq,
    line_offset=OFF, source="fork algo.c (identical bward in original and fix B)",
    line_map={str(u): F(u) for u in (8, 17, 19, 20, 23, 24, 27, 28, 147, 149, 150, 151, 154, 155, 157, 158, 159, 160, 163, 164, 166,
                                     169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 182, 183, 185, 186, 188, 189, 190, 192,
                                     193, 194, 195, 197, 198, 201, 202)},
    scenario=dict(key="c1", n_w=list(map(int, m.n_w)), k_w=list(map(int, m.k_w)), n_c=list(map(int, m.n_c)), k_c=list(map(int, m.k_c))),
    c1_rows=c1_rows, emit_log10=emit_log10,
)
json.dump(out, open(os.path.join(BUILD, "f_lbl.json"), "w"), allow_nan=False)
from collections import Counter
print(len(steps), "now steps", dict(Counter(s["kind"] for s in steps)), "|", len(steps_fix), "fix steps")
print("fix replay == LIB_C1FIX btable: bit-identical")
print("C1 rows (Now != Fix):", rows_bad, " first row", first, "cols", c1_rows["first_row_cols"])
print("all differing cells:", c1_cells)
print("silent cells whose Now pre-normalise value < DBL_MIN:", sub_dm)
for i in c1_steps[:12]:
    s = steps[i]; print("  step %d row %d j %d <- child %d: b %.3g x T %g x e %.3g = %.3g" % (i, s["row"], s["j"], s["child"], s["bchild"], s["T"], s["ev"], s["val"]))
print("json", os.path.getsize(os.path.join(BUILD, "f_lbl.json")), "bytes")
