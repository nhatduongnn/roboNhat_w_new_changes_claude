"""Continuation 2026-09-15 report: the six tuner-v2 campaigns continued from their last round with the
deadband tightened 1.25x -> 1.1x and the round limit raised 8 -> 15 (`tune_w.py continue`).
Writes ../presentation/continue_results.md comparing, per campaign, the OLD final (1.25x stop) with the
NEW final (1.1x stop). Safe to run any time; a campaign still running shows its latest round, marked
**INCOMPLETE**. Re-run by Slurm (twSUM_<run>_NN) whenever a continued campaign stops.

Inputs (read-only)
  conc_tuning/<run>/{state.json, validation.tsv, STOPPED, STOPPED_<date>_round<NN>}
  overnight/chereji_chrXIV_tw_<run>_NN.json      (score_robocop.py on the final chrXIV decode)

Definitions
  within 1.1x / 1.25x : tuned groups (55) with |ln((E+1)/(T+1))| <= ln fold  (the tuner's deadband test)
  within 2x           : tuned groups with T >= 5 and |ln(E/T)| <= ln 2        (the tuner's within2x)
  P / R / F1          : 58 groups pooled, calls = posterior >= 0.10, 30 bp one-to-one (tw_validate.py)
  bracket-pinned      : outside the deadband, step set by the bisection bracket, bracket < 0.1 decade

Usage:  python continue_summary.py [--out ../presentation/continue_results.md]
"""
import argparse
import csv
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CT = os.path.join(HERE, "conc_tuning")
ON = os.path.join(HERE, "overnight")
INC = "**INCOMPLETE**"
RUNS = [("bt10", "both, fiber tempered φ=10"), ("bt05", "both, φ=5"), ("sw01", "sequence only"),
        ("bw01", "both layers (φ=1)"), ("bt02", "both, φ=2"), ("fw01", "fiber only")]
LN10 = math.log(10)


def f(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if v != v else "%.*f" % (nd, v)


def prf(r):
    return "%s / %s / %s" % (f(r["P"]), f(r["R"]), f(r["F1"])) if r else "–"


def table(hdr, body):
    return (["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
            + ["| " + " | ".join(str(x) for x in r) + " |" for r in body])


def within(st, h):
    tuned = st["tuned"]
    n11 = sum(1 for g in tuned if abs(h["gap"][g]) <= math.log(1.1) + 1e-12)
    n125 = sum(1 for g in tuned if abs(h["gap"][g]) <= math.log(1.25) + 1e-12)
    return n11, n125, len(tuned), "%d/%d" % (h["within2x"], h["n_T5"])


def pinned(st, h, deadband):
    """Tuned groups outside the deadband whose step at round h was set by the bisection bracket
    (history stores `why`, not the bracket itself; at the continuation round every such group was also
    bracket-pinned, bracket < 0.1 decade — see chain.log's CONTINUE line)."""
    out = []
    for g in st["tuned"]:
        if "bisect" in h["why"][g] and abs(h["gap"][g]) > deadband:
            out.append(g)
    return out


def chereji(run, t):
    p = os.path.join(ON, "chereji_chrXIV_tw_%s_%02d.json" % (run, t))
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["nucleosome"]
    except Exception:
        return None


def campaign(run, desc):
    sp = os.path.join(CT, run, "state.json")
    if not os.path.exists(sp):
        return None
    st = json.load(open(sp))
    sh = st.get("settings_history") or []
    V = list(csv.DictReader(open(os.path.join(CT, run, "validation.tsv")), delimiter="\t")) \
        if os.path.exists(os.path.join(CT, run, "validation.tsv")) else []
    idx = {(int(r["round"]), r["set"], r["ref"], r["fitted"]): r for r in V}
    hist = {h["iter"]: h for h in st["history"]}
    stp = os.path.join(CT, run, "STOPPED")
    stopped = open(stp).read().strip() if os.path.exists(stp) else None
    old = sh[0]["from_round"] if sh else max(hist)
    new = max(hist)
    return dict(run=run, desc=desc, st=st, sh=sh, idx=idx, hist=hist, stopped=stopped, old=old, new=new,
                continued=bool(sh), done=bool(stopped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "presentation", "continue_results.md"))
    ap.add_argument("--root", default=None, help="campaign root (testing; default conc_tuning)")
    a = ap.parse_args()
    global CT
    if a.root:
        CT = os.path.abspath(a.root)
    L = ["# Tuner-v2 continuation: deadband 1.25× → 1.1×, up to 15 rounds", "",
         "_generated %s by `analysis/continue_summary.py`; campaigns still running are marked %s — "
         "re-run `python continue_summary.py` to refresh._" % (time.strftime("%Y-%m-%d %H:%M"), INC), "",
         "### Rule 7 — what differs", "",
         "Each campaign differs from its original run **only in the stop criteria, from the continuation round "
         "onward: deadband 1.25× → 1.1× and max rounds 8 → 15 (rounds 0..14)**. Everything else is identical: "
         "step cap 10×/round, zero-count ×10 step, per-bp weight cap 0.70, bisection bracket (unchanged), β seed/secant, "
         "α damping, unknown fixed w = 1e-3, w_nuc 35 with no hold, 58 groups with the mr58 mask, MacIsaac targets, "
         "tune chrXIV+chrII, holdout chrIV, same driver/pkgvar per campaign. The continuation re-ran the last round's "
         "update from its pre-update snapshot under 1.1× (the original post-update state is kept as "
         "`state_after_update_NN_deadband1p25.json`), so the old final's decode is shared by both columns below.", "",
         "Definitions: within 1.1× / 1.25× = tuned groups (of 55) with |ln((E+1)/(T+1))| ≤ ln fold (the deadband "
         "test); within 2× = the tuner's metric (T ≥ 5, |ln E/T| ≤ ln 2). P / R / F1 pooled over all 58 groups "
         "(calls posterior ≥ 0.10, 30 bp one-to-one). Bracket-pinned = outside the deadband with the step set by the "
         "bisection bracket (bracket < 0.1 decade).", ""]
    C = [c for c in (campaign(r, d) for r, d in RUNS) if c]
    body = []
    for c in C:
        st, ho, hn = c["st"], c["hist"][c["old"]], c["hist"][c["new"]]
        o11, o125, n, o2 = within(st, ho)
        n11, n125, _, n2 = within(st, hn)
        state = ("stopped: " + c["stopped"].replace("stopped after ", "")) if c["done"] else \
            INC + " (running; next round to update %d)" % st["iter"]
        prev = c["sh"][0]["previous"]["stopped"].replace("stopped after ", "") if c["sh"] else "–"
        body.append([c["run"], c["desc"], "r%02d (%s)" % (c["old"], prev),
                     "r%02d%s" % (c["new"], "" if c["done"] else " so far"), state,
                     "%d → %d" % (o11, n11), "%d → %d" % (o125, n125), "%s → %s" % (o2, n2),
                     "%d → %d" % (len(pinned(st, ho, math.log(1.1))), len(pinned(st, hn, math.log(1.1))))])
    L += ["## Campaigns", ""]
    L += table(["run", "config", "old final (1.25× stop)", "new final", "state", "within 1.1× (of 55)",
                "within 1.25×", "within 2× (T≥5)", "bisect-set outside 1.1× (old → new)"], body)
    L.append("")

    for which, title in (("tune", "tuning chromosomes chrXIV+chrII"), ("holdout", "holdout chrIV")):
        L += ["## P / R / F1, %s — old final → new final" % title, ""]
        body = []
        for c in C:
            idx = c["idx"]
            for ref, rname in (("macisaac", "MacIsaac"), ("rossi_cx", "Rossi _CX")):
                row = [c["run"], rname]
                for cat in ("all", "fitted", "nonfitted"):
                    o = idx.get((c["old"], which, ref, cat))
                    nw = idx.get((c["new"], which, ref, cat))
                    if c["new"] == c["old"]:
                        nw_txt = "(same round)"
                    elif nw:
                        nw_txt = prf(nw)
                    else:
                        nw_txt = INC
                    row += [prf(o), nw_txt]
                calls_o = idx.get((c["old"], which, ref, "all"))
                calls_n = idx.get((c["new"], which, ref, "all"))
                row.append("%s → %s" % (calls_o["calls"] if calls_o else "–", calls_n["calls"] if calls_n else "–"))
                body.append(row)
        L += table(["run", "ref", "all old", "all new", "fitted old", "fitted new", "nonfitted old", "nonfitted new",
                    "calls"], body)
        L.append("")

    L += ["## Nucleosomes", ""]
    body = []
    for c in C:
        ho, hn = c["hist"][c["old"]], c["hist"][c["new"]]
        hold = lambda t: c["idx"].get((t, "holdout", "macisaac", "all"))
        cho, chn = chereji(c["run"], c["old"]), chereji(c["run"], c["new"])
        body.append([c["run"],
                     "%.0f (%+.2f%%)" % (ho["nucleosome_copies"], 100 * ho["nucleosome_rel_r0"]),
                     "%.0f (%+.2f%%)" % (hn["nucleosome_copies"], 100 * hn["nucleosome_rel_r0"]),
                     "%s" % f(hold(c["old"])["nuc_copies"], 0) if hold(c["old"]) else "–",
                     "%s" % f(hold(c["new"])["nuc_copies"], 0) if hold(c["new"]) else INC,
                     ("%s (n_ref %s)" % (f(cho["recall"]), cho["n_ref"])) if cho else INC,
                     ("%s (n_ref %s)" % (f(chn["recall"]), chn["n_ref"])) if chn else INC])
    L += table(["run", "tune copies old (vs r0)", "tune copies new (vs r0)", "holdout chrIV copies old",
                "holdout chrIV copies new", "Chereji +1/-1 recall chrXIV old", "Chereji recall new"], body)
    L.append("")

    for c in C:
        st, idx = c["st"], c["idx"]
        L += ["## %s (%s) — per round from the old final" % (c["run"], c["desc"]), ""]
        body = []
        for t in sorted(k for k in c["hist"] if k >= c["old"]):
            h = c["hist"][t]
            n11, n125, n, w2 = within(st, h)
            m, x = idx.get((t, "tune", "macisaac", "all")), idx.get((t, "tune", "rossi_cx", "all"))
            body.append(["r%02d" % t, n11, n125, w2, h["n_zero_step"], m["calls"] if m else "–", prf(m), prf(x),
                         "%.0f (%+.2f%%)" % (h["nucleosome_copies"], 100 * h["nucleosome_rel_r0"]),
                         ",".join(pinned(st, h, math.log(1.1))) or "–",
                         ",".join(h["capped_under"]) or "–"])
        L += table(["round", "within 1.1×", "within 1.25×", "within 2×", "zero steps", "calls", "MacIsaac P/R/F1",
                    "Rossi P/R/F1", "nucleosome copies", "bisect-set (outside 1.1×)", "capped_under"], body)
        L.append("")
        jobs = {k: v for k, v in st["jobs"].items() if int(k) >= c["old"]}
        L += ["jobs: `%s`" % json.dumps(jobs, sort_keys=True), ""]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print("wrote %s" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()
