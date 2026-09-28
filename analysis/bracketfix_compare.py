"""Bracket-fix reruns (2026-09-20): compare each pre-fix tuner-v2 campaign with its post-fix rerun.

Writes ../analysis/bracketfix_rerun/bracketfix_results.md from the campaign files only (no decoding,
no Slurm, no writes anywhere under conc_tuning/). Default pairs: bw01 vs bw01x and fw01 vs fw01x.

What is under test (plan okay-so-i-want-humming-hearth.md, Step 2/3)
-------------------------------------------------------------------
`tune_w.propose` keeps a bisection bracket (th_under / th_over). Pre-fix, an end was never dropped,
so a stale measurement at a delta the campaign had long left could pin a group off target and, by
halving its step every round, also block the zero-step convergence stop. Replay showed bisection only
ever fired in bw01 (62 steps, 3 groups pinned) and fw01 (261 steps, 13 pinned). The reruns bw01x /
fw01x are the same campaigns from round 0 with `bracket_max_age: 3` in state.json.

Confounds (rule 7) — stated, NOT corrected for
----------------------------------------------
- Under test: bracket expiry on (`bracket_max_age` 3) in the rerun, absent in the original.
- Also different: the reruns use deadband 1.1x from round 0 with a 25-round cap set at init; the
  originals ran 1.25x for rounds 0-7 then 1.1x, with the cap staged 8 -> 15 -> 25 (see each
  original's `settings_history`). A faithful staged replay was considered and rejected by the user.
- Identical: driver, pkgvar tree, `robocop_train_fiberonly` source, the 58-group / 61-motif mr58 set
  with ARO80/RFX1/RPH1 frozen, untuned start, w_nuc 35, w_unknown 1e-3, no phi, no nucleosome hold,
  1-decade step cap, beta seed, rho cap 0.70, 40 tasks, tune chrXIV+chrII, holdout chrIV.

Inputs (read-only; nothing is created outside --out's directory)
  conc_tuning/<run>/state.json              history, tuned set, targets, rule_state, settings_history
  conc_tuning/<run>/STOPPED                 stop reason, if the campaign has stopped
  conc_tuning/<run>/chain.log               last chain action (used to say where a running rerun is)
  conc_tuning/<run>/report_NN.tsv           per-group T/E/fold/calls/matched_mac/lambda/rho_next/why
  conc_tuning/<run>/validation.tsv          pooled P/R/F1 per round x set x ref x fitted (tw_validate)
  conc_tuning/<run>/validation_groups.tsv   the same per group
  overnight/chereji_chrXIV_tw_<run>_NN.json score_robocop.py on that round's chrXIV decode

Definitions
  within 1.1x   tuned group (of state["tuned"]) with |ln((E+1)/(T+1))| <= ln 1.1 at that round
  bisect-set    final-round report row whose `why` contains "bisect" (the step came from the bracket)
  capped_under  report flag: at the per-bp cap and still under target
  at rho cap    report `rho_next` >= 0.70 - 1e-6 (RHO_MAX, tune_w.py:97)
  wasted rounds trailing rounds sharing the final round's pooled tuning call count exactly
                (validation.tsv, set=tune ref=macisaac fitted=all) - rounds that bought no change
  P / R / F1    calls = posterior >= 0.10, greedy one-to-one within 30 bp (tw_validate.py)

Pre-registered criterion (plan "Pre-registered criterion", fixed before any result was seen)
  bw01 pair: rerun's FINAL held-out (chrIV) MacIsaac pooled F1 within +-0.003 of 0.0468
  fw01 pair: rerun's FINAL held-out MacIsaac pooled F1 inside [0.001, 0.009]
  AND the layer ordering bt10 > bt05 > sw01 > bt02 > bw01 > fw01 (held-out MacIsaac pooled F1,
  each campaign's own validation.tsv) unchanged when the rerun's number is substituted.
  Both must hold -> UPHELD; either fails -> WRONG.

Safe to run while the reruns are going: a rerun with fewer rounds is compared over what exists and
the verdict is replaced by an unmissable PENDING banner. Never blocks, never waits.

Usage
  python bracketfix_compare.py                                   # bw01:bw01x fw01:fw01x
  python bracketfix_compare.py --pairs bw01:bw01x fw01:fw01x
  python bracketfix_compare.py --pairs bw01:bw01 --out /tmp/selftest.md   # machinery self-test
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CT = os.path.join(HERE, "conc_tuning")
ON = os.path.join(HERE, "overnight")

LN11 = math.log(1.1)
RHO_MAX = 0.70                      # tune_w.py:97
MAX_ROUNDS_EXPECTED = 25            # the reruns' cap (state["max_rounds"] is printed too)
REFS = (("macisaac", "MacIsaac"), ("rossi_cx", "Rossi _CX"))
SETS = (("tune", "tuning chrXIV+chrII"), ("holdout", "held-out chrIV"))
LAYER_ORDER = ["bt10", "bt05", "sw01", "bt02", "bw01", "fw01"]

# Groups pinned off target by the stale bracket in each original (plan Step 3; cross-checked below
# against the original's final report `why` column, and any mismatch is printed).
PINNED = {"bw01": ["BAS1", "RTG3", "SKN7"],
          "fw01": ["ACE2", "CAD1", "HAP1", "MBP1", "MET31", "MSN2", "NRG1", "PDR3", "PHD1",
                   "RTG3", "SKN7", "STE12", "SUT1"]}

# Pre-registered bound per pair, keyed by the ORIGINAL run: ("abs", centre, tol) or ("band", lo, hi).
CRITERION = {"bw01": ("abs", 0.0468, 0.003),
             "fw01": ("band", 0.001, 0.009)}

INC = "**INCOMPLETE**"


# ---------------------------------------------------------------- small helpers (masked_summary.py style)
def fnum(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if v != v else "%.*f" % (nd, v)


def table(hdr, body):
    return (["| " + " | ".join(str(h) for h in hdr) + " |",
             "|" + "|".join("---" for _ in hdr) + "|"]
            + ["| " + " | ".join("–" if x is None else str(x) for x in r) + " |" for r in body])


def prf(r):
    return "%s / %s / %s" % (fnum(r["P"]), fnum(r["R"]), fnum(r["F1"])) if r else "–"


def tsv(path):
    return list(csv.DictReader(open(path), delimiter="\t")) if os.path.exists(path) else []


def load(run, root=CT):
    """Everything one campaign contributes. Returns None if the campaign does not exist."""
    d = os.path.join(root, run)
    sp = os.path.join(d, "state.json")
    if not os.path.exists(sp):
        return None
    st = json.load(open(sp))
    agg = {(int(r["round"]), r["set"], r["ref"], r["fitted"]): r
           for r in tsv(os.path.join(d, "validation.tsv")) if r["round"].isdigit()}
    grp = {(int(r["round"]), r["set"], r["ref"], r["group"]): r
           for r in tsv(os.path.join(d, "validation_groups.tsv")) if r["round"].isdigit()}
    reports = {}
    for p in glob.glob(os.path.join(d, "report_[0-9][0-9].tsv")):
        reports[int(re.search(r"report_(\d\d)\.tsv$", p).group(1))] = {r["group"]: r for r in tsv(p)}
    stp = os.path.join(d, "STOPPED")
    stopped = open(stp).read().strip() if os.path.exists(stp) else None
    chain = [l.rstrip("\n") for l in open(os.path.join(d, "chain.log"))] \
        if os.path.exists(os.path.join(d, "chain.log")) else []
    hist = {h["iter"]: h for h in st.get("history", [])}
    return dict(run=run, dir=d, st=st, agg=agg, grp=grp, reports=reports, stopped=stopped,
                chain=chain, hist=hist, last=max(hist) if hist else None,
                hold=sorted({k[0] for k in agg if k[1] == "holdout"}),
                tuned=list(st.get("tuned", [])))


def chereji(run, t):
    if t is None:
        return None
    p = os.path.join(ON, "chereji_chrXIV_tw_%s_%02d.json" % (run, t))
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["nucleosome"]
    except Exception:
        return None


def within11(c, t):
    h = c["hist"].get(t)
    if h is None:
        return None
    return sum(1 for g in c["tuned"] if g in h["gap"] and abs(h["gap"][g]) <= LN11 + 1e-12)


def report_flags(c, t):
    """(bisect-set, capped_under, at rho cap) from the round-t report, restricted to tuned groups."""
    rep = c["reports"].get(t)
    if rep is None:
        return None
    bis, cap, rho = [], [], []
    for g in c["tuned"]:
        r = rep.get(g)
        if r is None:
            continue
        if "bisect" in r["why"]:
            bis.append(g)
        if r.get("capped_under") == "1":
            cap.append(g)
        try:
            if float(r["rho_next"]) >= RHO_MAX - 1e-6:
                rho.append(g)
        except (TypeError, ValueError):
            pass
    return sorted(bis), sorted(cap), sorted(rho)


def pooled_calls(c, t):
    r = c["agg"].get((t, "tune", "macisaac", "all"))
    return int(r["calls"]) if r else None


def wasted_rounds(c):
    """Trailing rounds (beyond the first) whose pooled tuning call count equals the final round's."""
    if c["last"] is None:
        return None, None
    rounds = sorted(r for r in c["hist"] if pooled_calls(c, r) is not None)
    if not rounds:
        return None, None
    fin = pooled_calls(c, rounds[-1])
    n = 0
    for r in reversed(rounds):
        if pooled_calls(c, r) == fin:
            n += 1
        else:
            break
    return max(n - 1, 0), fin


def final_holdout_round(c):
    """Last round with a holdout (chrIV) validation row, or None."""
    return c["hold"][-1] if c["hold"] else None


def holdout_f1(run, root=CT):
    """(F1, round) of the last held-out MacIsaac pooled row of `run`, or (None, None)."""
    c = load(run, root)
    if c is None:
        return None, None
    t = final_holdout_round(c)
    r = c["agg"].get((t, "holdout", "macisaac", "all")) if t is not None else None
    if not r:
        return None, None
    try:
        return float(r["F1"]), t
    except (TypeError, ValueError):
        return None, t


def complete(c):
    return bool(c and c["stopped"] and c["hist"])


def clean_final(c):
    """(True, "") only if the campaign stopped cleanly AND its final round's chrIV holdout is
    validated, so a final held-out F1 exists. Otherwise (False, why) and no verdict is given."""
    if not complete(c):
        return False, "has not stopped"
    if "error" in c["stopped"].lower():
        return False, "stopped with an error (`%s`), so its last round is not a final result" % c["stopped"]
    t = final_holdout_round(c)
    if t is None:
        return False, "has no chrIV holdout row in validation.tsv yet"
    if t != c["last"]:
        return False, ("stopped at round %d but its latest validated chrIV holdout is round %d "
                       "(holdout rounds %s) — the final holdout has not landed" % (c["last"], t, c["hold"]))
    return True, ""


def where(c):
    """Human statement of how far a campaign has got."""
    if c is None:
        return "not started (no state.json)"
    if c["stopped"]:
        return "stopped: " + c["stopped"].replace("stopped after ", "")
    last = "round %d updated" % c["last"] if c["last"] is not None else "no round updated yet"
    tail = c["chain"][-1].strip() if c["chain"] else "no chain.log"
    return "%s running — %s; next round to update %s; last chain.log line: `%s`" % (
        INC, last, c["st"].get("iter"), tail)


# ---------------------------------------------------------------- sections
def sec_stop(o, n):
    L = ["### 1. Stop", ""]
    body = []
    for c in (o, n):
        if c is None:
            body.append(["–", "not started"] + ["–"] * 5)
            continue
        w, fin = wasted_rounds(c)
        h = c["hist"].get(c["last"]) if c["last"] is not None else None
        conv = "–"
        if h is not None:
            conv = ("converged (every tuned group zero-step)" if h["n_zero_step"] == len(c["tuned"])
                    else "round cap" if c["stopped"] and "round limit" in c["stopped"] else "not stopped")
        body.append([c["run"],
                     "r%02d" % c["last"] if c["last"] is not None else "none",
                     c["st"].get("max_rounds"),
                     (c["stopped"] or INC + " still running"),
                     conv,
                     "%d/%d" % (h["n_zero_step"], len(c["tuned"])) if h else "–",
                     "%s (pooled tuning calls %s)" % (w, fin) if w is not None else "–"])
    L += table(["run", "last updated round", "max_rounds", "stop reason", "converged / cap",
                "zero-step groups at final", "wasted trailing rounds"], body)
    L += ["", "State: **%s** — %s;  **%s** — %s." % (o["run"] if o else "?", where(o),
                                                     n["run"] if n else "?", where(n)), ""]
    return L


def sec_counts(o, n):
    L = ["### 2. Counts", ""]
    body = []
    for c in (o, n):
        if c is None or c["last"] is None:
            body.append([c["run"] if c else "–", "no round updated yet"] + ["–"] * 4)
            continue
        t = c["last"]
        fl = report_flags(c, t)
        w = within11(c, t)
        body.append([c["run"], "r%02d" % t,
                     "%s/%d" % (w if w is not None else "–", len(c["tuned"])),
                     ("%d (%s)" % (len(fl[0]), ", ".join(fl[0]) or "none")) if fl else "no report_%02d.tsv" % t,
                     ("%d (%s)" % (len(fl[1]), ", ".join(fl[1]) or "none")) if fl else "–",
                     ("%d (%s)" % (len(fl[2]), ", ".join(fl[2]) or "none")) if fl else "–"])
    L += table(["run", "round", "within 1.1x (of tuned)", "last step from bisection",
                "capped_under", "at rho cap (>= %.2f)" % RHO_MAX], body)
    L.append("")
    return L


def sec_pinned(o, n, pinned):
    L = ["### 3. Previously pinned factors (the direct test of the mechanism)", ""]
    if o is None:
        return L + ["original campaign missing.", ""]
    ot, nt = o["last"], (n["last"] if n else None)
    orep = o["reports"].get(ot, {}) if ot is not None else {}
    nrep = n["reports"].get(nt, {}) if (n and nt is not None) else {}
    derived = sorted(report_flags(o, ot)[0]) if (ot is not None and report_flags(o, ot)) else []
    if derived and sorted(pinned) != derived:
        L += ["> NOTE: the expected pinned list %s differs from the groups whose final step came from "
              "bisection in `%s/report_%02d.tsv` (%s); both are shown, the table uses the expected list."
              % (sorted(pinned), o["run"], ot, derived), ""]
    else:
        L += ["Expected pinned list confirmed by `%s/report_%02d.tsv` (`why` contains `bisect`): %s."
              % (o["run"], ot, ", ".join(derived) or "none"), ""]
    body = []
    for g in sorted(pinned):
        a, b = orep.get(g), nrep.get(g)
        mg = lambda c, t, key: (c["grp"].get((t, "tune", "macisaac", g), {}) or {}).get(key) \
            if (c and t is not None) else None
        body.append([g,
                     a["T"] if a else "–",
                     "%s → %s" % (a["E"] if a else "–", b["E"] if b else "–"),
                     "%s → %s" % (a["fold"] if a else "–", b["fold"] if b else "–"),
                     "%s → %s" % (a["lambda"] if a else "–", b["lambda"] if b else "–"),
                     "%s → %s" % (a["calls"] if a else "–", b["calls"] if b else "–"),
                     "%s → %s" % (a["matched_mac"] if a else "–", b["matched_mac"] if b else "–"),
                     "%s → %s" % (a["why"] if a else "–", b["why"] if b else "–"),
                     "%s → %s" % (mg(o, ot, "matched") or "–", mg(n, nt, "matched") or "–")])
    L += table(["group", "T", "E (%s r%s → %s r%s)" % (o["run"], "%02d" % ot if ot is not None else "–",
                                                       n["run"] if n else "?",
                                                       "%02d" % nt if nt is not None else "–"),
                "(E+1)/(T+1)", "lambda", "tuning calls", "MacIsaac matches (report)", "last step (`why`)",
                "MacIsaac matches (validation_groups)"], body)
    L += ["", "Columns read from `report_NN.tsv` (`T`, `E`, `fold`, `lambda`, `calls`, `matched_mac`, "
          "`why`) and `validation_groups.tsv` (tune / macisaac / `matched`). `–` on the right of an "
          "arrow means the rerun has not reached a comparable round.", ""]
    return L


def prf_cells(c, t, which, ref, fitted="all"):
    return c["agg"].get((t, which, ref, fitted)) if (c and t is not None) else None


def sec_accuracy(o, n, per_group=True):
    L = ["### 4. Accuracy", ""]
    oh, nh = final_holdout_round(o), final_holdout_round(n) if n else None
    ot, nt = (o["last"] if o else None), (n["last"] if n else None)
    body = []
    for which, wname in SETS:
        for ref, rname in REFS:
            a_t = ot if which == "tune" else oh
            b_t = nt if which == "tune" else nh
            a0, a1 = prf_cells(o, 0, which, ref), prf_cells(o, a_t, which, ref)
            b0, b1 = prf_cells(n, 0, which, ref), prf_cells(n, b_t, which, ref)
            dF1 = "–"
            if a1 and b1:
                try:
                    dF1 = "%+.4f" % (float(b1["F1"]) - float(a1["F1"]))
                except (TypeError, ValueError):
                    pass
            body.append([wname, rname,
                         "r%s" % ("%02d" % a_t if a_t is not None else "–"),
                         prf(a0), prf(a1), a1["calls"] if a1 else "–",
                         "r%s" % ("%02d" % b_t if b_t is not None else "–"),
                         prf(b0), prf(b1) if b1 else INC, b1["calls"] if b1 else "–", dF1])
    L += table(["set", "ref", "%s round" % (o["run"] if o else "orig"), "orig r00", "orig final",
                "orig calls", "%s round" % (n["run"] if n else "rerun"), "rerun r00", "rerun final",
                "rerun calls", "ΔF1 (rerun − orig)"], body)
    L += ["", "Rows are `validation.tsv` with `fitted == \"all\"`; the holdout round is the last round "
          "with a chrIV validation row (round 0 only, until the campaign stops).", ""]

    # nucleosomes + Chereji
    body = []
    for c in (o, n):
        if c is None or c["last"] is None:
            body.append([c["run"] if c else "–", "no round updated yet"] + ["–"] * 5)
            continue
        h = c["hist"][c["last"]]
        h0 = c["hist"].get(0)
        ho = prf_cells(c, final_holdout_round(c), "holdout", "macisaac")
        ch = chereji(c["run"], c["last"])
        body.append([c["run"], "r%02d" % c["last"],
                     "%.0f" % h0["nucleosome_copies"] if h0 else "–",
                     "%.0f" % h["nucleosome_copies"],
                     "%+.2f%%" % (100 * h["nucleosome_rel_r0"]),
                     fnum(ho["nuc_copies"], 0) if ho else "–",
                     ("%s (n_ref %s, median dyad err %s bp)" % (fnum(ch["recall"]), ch["n_ref"],
                                                                ch.get("median_dyad_err")))
                     if ch else "no overnight/chereji_chrXIV_tw_%s_%02d.json" % (c["run"], c["last"])])
    L += ["#### Nucleosomes", ""]
    L += table(["run", "round", "copies r00 (tune)", "copies final (tune)", "nucleosome_rel_r0",
                "holdout chrIV copies", "Chereji +1/-1 recall chrXIV"], body)
    L.append("")

    if per_group:
        for which, wname in SETS:
            a_t = ot if which == "tune" else oh
            b_t = nt if which == "tune" else nh
            for ref, rname in REFS:
                L += ["#### Per group — %s, %s (orig r%s vs rerun r%s)"
                      % (wname, rname, "%02d" % a_t if a_t is not None else "–",
                         "%02d" % b_t if b_t is not None else "–"), ""]
                body = []
                for g in sorted(o["st"]["groups"]) if o else []:
                    a = o["grp"].get((a_t, which, ref, g)) if a_t is not None else None
                    b = n["grp"].get((b_t, which, ref, g)) if (n and b_t is not None) else None
                    body.append([g, (a or b or {}).get("sites", "–"),
                                 a["calls"] if a else "–", a["matched"] if a else "–", prf(a),
                                 b["calls"] if b else "–", b["matched"] if b else "–", prf(b)])
                L += table(["group", "sites", "orig calls", "orig matched", "orig P/R/F1",
                            "rerun calls", "rerun matched", "rerun P/R/F1"], body)
                L.append("")
    return L


def criterion_text(kind, a, b):
    return ("within ±%.3f of %.4f (i.e. [%.4f, %.4f])" % (b, a, a - b, a + b)) if kind == "abs" \
        else ("inside [%.3f, %.3f]" % (a, b))


def criterion_ok(kind, a, b, f1):
    return abs(f1 - a) <= b + 1e-12 if kind == "abs" else (a - 1e-12 <= f1 <= b + 1e-12)


def layer_table(subs, root=CT):
    """(rows, ordering_holds, missing) for the layer ordering with `subs` {run: (F1, label)} applied."""
    rows, vals, missing = [], [], []
    for r in LAYER_ORDER:
        if r in subs:
            f1, lab, rnd = subs[r]
        else:
            f1, rnd = holdout_f1(r, root)
            lab = r
        rows.append([r, lab, "r%s" % ("%02d" % rnd if rnd is not None else "–"),
                     fnum(f1, 5) if f1 is not None else "**missing**"])
        if f1 is None:
            missing.append(r)
        vals.append(f1)
    holds = all(vals[i] is not None and vals[i + 1] is not None and vals[i] > vals[i + 1]
                for i in range(len(vals) - 1))
    return rows, holds, missing


def sec_verdict(o, n, root=CT):
    """Section 5 for one pair. Returns (lines, headline) where headline is UPHELD / WRONG / PENDING."""
    L = ["### 5. Verdict against the pre-registered criterion", ""]
    orig = o["run"] if o else "?"
    crit = CRITERION.get(orig)
    if crit is None:
        L += ["No pre-registered criterion exists for a pair whose original is `%s` "
              "(criteria are defined for %s). No verdict." % (orig, ", ".join(sorted(CRITERION))), ""]
        return L, "NO CRITERION (%s)" % orig
    kind, a, b = crit
    L += ["Pre-registered: the rerun's **final held-out chrIV MacIsaac pooled F1** must be %s, "
          "**and** the layer ordering %s (held-out MacIsaac pooled F1) must be unchanged when the "
          "rerun's number is substituted for `%s`." % (criterion_text(kind, a, b),
                                                       " > ".join(LAYER_ORDER), orig), ""]
    if not complete(n):
        rnd = n["last"] if (n and n["last"] is not None) else 0
        mx = (n["st"].get("max_rounds") if n else None) or MAX_ROUNDS_EXPECTED
        head = "PENDING — rerun at round %d of %d" % (rnd, mx)
        L += ["> # ⏳ %s" % head, ">",
              "> **NO VERDICT YET.** `%s` has not stopped, so its final held-out F1 does not exist. "
              "Status: %s" % (n["run"] if n else "?", where(n)), ">",
              "> Re-run `python bracketfix_compare.py` when it stops.", ""]
        f1_now, t_now = (holdout_f1(n["run"], root) if n else (None, None))
        if f1_now is not None:
            L += ["Latest held-out MacIsaac pooled F1 available for `%s` is %s at round r%02d — this is "
                  "**not** the final round and is not tested against the criterion."
                  % (n["run"], fnum(f1_now, 5), t_now), ""]
        return L, head

    f1, t = holdout_f1(n["run"], root)
    f1o, to = holdout_f1(orig, root)
    ok1 = f1 is not None and criterion_ok(kind, a, b, f1)
    subs = {orig: (f1, "%s (rerun %s)" % (orig, n["run"]), t)}
    rows, holds, missing = layer_table(subs, root)
    L += table(["campaign", "F1 source", "round", "held-out MacIsaac pooled F1"], rows)
    L += ["", "- rerun `%s` final held-out MacIsaac pooled F1 = **%s** (r%02d, `%s/validation.tsv`)"
          % (n["run"], fnum(f1, 5), t, n["run"]),
          "- original `%s` = %s (r%02d); difference %s"
          % (orig, fnum(f1o, 5) if f1o is not None else "–", to if to is not None else -1,
             "%+.5f" % (f1 - f1o) if (f1 is not None and f1o is not None) else "–"),
          "- criterion on F1 (%s): **%s**" % (criterion_text(kind, a, b), "PASS" if ok1 else "FAIL"),
          "- layer ordering %s with `%s` substituted: **%s**%s"
          % (" > ".join(LAYER_ORDER), n["run"], "unchanged" if holds else "CHANGED",
             (" (missing campaigns: %s)" % ", ".join(missing)) if missing else ""), ""]
    ok = ok1 and holds and not missing
    head = "UPHELD" if ok else "WRONG"
    L += ["> # %s" % head, ">"]
    if ok:
        L += ["> In plain English: rerunning `%s` with the bracket fix moved the held-out MacIsaac F1 "
              "to %s, inside the bound set before the result was seen, and no layer-ordering "
              "conclusion changed — the earlier claim that no campaign needed rerunning holds."
              % (orig, fnum(f1, 5)), ""]
    else:
        why = []
        if not ok1:
            why.append("the held-out F1 %s is outside the pre-registered %s" % (fnum(f1, 5),
                                                                                criterion_text(kind, a, b)))
        if not holds:
            why.append("the layer ordering changes once `%s` is substituted" % n["run"])
        if missing:
            why.append("campaigns %s have no held-out F1, so the ordering cannot be checked"
                       % ", ".join(missing))
        L += ["> In plain English: the earlier claim does **not** hold for `%s` — %s."
              % (orig, "; and ".join(why)), ""]
    return L, head


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", nargs="+", default=["bw01:bw01x", "fw01:fw01x"],
                    help="orig:rerun pairs (default bw01:bw01x fw01:fw01x)")
    ap.add_argument("--out", default=os.path.join(HERE, "bracketfix_rerun", "bracketfix_results.md"))
    ap.add_argument("--root", default=CT, help="campaign root (default analysis/conc_tuning)")
    ap.add_argument("--no-per-group", action="store_true", help="omit the per-group P/R/F1 tables")
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    L = ["# Bracket-fix reruns: %s" % ", ".join("`%s` vs `%s`" % tuple(p.split(":", 1)) for p in a.pairs), "",
         "_generated %s by `analysis/bracketfix_compare.py` from `%s` (read-only). Campaigns still "
         "running are marked %s — re-run to refresh._"
         % (time.strftime("%Y-%m-%d %H:%M"), os.path.relpath(root, HERE), INC), "",
         "### Rule 7 — what differs between an original and its rerun", "",
         "**Under test:** the bisection-bracket fix — the rerun's `state.json` carries "
         "`bracket_max_age: 3`, so a bracket end not re-confirmed for more than 3 updates is dropped; "
         "the originals keep every end forever (`tune_w.propose`).", "",
         "**Also different (stated confound, not corrected for):** the reruns use deadband 1.1× from "
         "round 0 and a 25-round cap set at init; the originals ran 1.25× for rounds 0–7 then 1.1×, "
         "with the cap staged 8 → 15 → 25 via two `continue` calls.", "",
         "**Identical:** driver, pkgvar tree, `robocop_train_fiberonly` source, the 58-group / 61-motif "
         "mr58 set with ARO80/RFX1/RPH1 frozen, untuned start, w_nuc 35, w_unknown 1e-3, no φ, no "
         "nucleosome hold, 1-decade step cap, β seed, ρ cap 0.70, 40 tasks, tune chrXIV+chrII, "
         "holdout chrIV.", "",
         "Definitions: within 1.1× = tuned group with |ln((E+1)/(T+1))| ≤ ln 1.1; bisect-set = final "
         "`report_NN.tsv` row whose `why` contains `bisect`; at ρ cap = `rho_next` ≥ %.2f; wasted "
         "rounds = trailing rounds sharing the final round's pooled tuning call count exactly; "
         "P / R / F1 = calls at posterior ≥ 0.10 matched one-to-one within 30 bp (`tw_validate.py`), "
         "`fitted == \"all\"`." % RHO_MAX, ""]

    heads = []
    banners = []
    for spec in a.pairs:
        if ":" not in spec:
            L += ["## %s — malformed pair (expected `orig:rerun`)" % spec, ""]
            heads.append((spec, "MALFORMED"))
            continue
        orig, rerun = spec.split(":", 1)
        o, n = load(orig, root), load(rerun, root)
        L += ["---", "", "## %s → %s" % (orig, rerun), ""]
        if o is None:
            L += ["`%s` has no `state.json` under `%s`; nothing to compare." % (orig, root), ""]
            heads.append((spec, "MISSING ORIGINAL"))
            continue
        if n is None:
            L += ["> ⏳ `%s` has not been initialised (no `%s/state.json`). The original is summarised "
                  "below; there is no rerun to compare yet." % (rerun, rerun), ""]
        if orig == rerun:
            L += ["> Self-comparison (`%s` against itself): every orig → rerun column must show the "
                  "same value on both sides. This is the machinery self-test." % orig, ""]
        bma = lambda c: (c["st"].get("bracket_max_age") if c else None)
        L += ["`bracket_max_age`: %s = %r, %s = %r  (the fix is on iff this is an integer)."
              % (orig, bma(o), rerun, bma(n)), ""]
        L += sec_stop(o, n)
        L += sec_counts(o, n)
        L += sec_pinned(o, n, PINNED.get(orig, sorted(report_flags(o, o["last"])[0])
                                         if o["last"] is not None and report_flags(o, o["last"]) else []))
        L += sec_accuracy(o, n, per_group=not a.no_per_group)
        vl, head = sec_verdict(o, n, root)
        L += vl
        heads.append((spec, head))
        if head.startswith("PENDING"):
            banners.append("`%s`: %s" % (spec, head))

    summary = ["## Summary", ""] + table(["pair", "verdict"], [[p, h] for p, h in heads]) + [""]
    if banners:
        summary = (["> # ⏳ PENDING — no verdict yet", ">"]
                   + ["> " + b for b in banners]
                   + [">", "> The rerun(s) above have not stopped; every number below them is an "
                      "interim reading, not a result.", ""]) + summary
    # summary goes right after the header block, before the first pair
    cut = L.index("---") if "---" in L else len(L)
    L = L[:cut] + summary + L[cut:]

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(out + ".tmp", out)
    print("wrote %s" % out)
    for p, h in heads:
        print("  %-14s %s" % (p, h))


if __name__ == "__main__":
    main()
