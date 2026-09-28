"""Masked tuner-v2 campaigns (2026-09-16): report comparing each masked campaign with its unmasked counterpart.

Writes ../presentation/masked_results.md. Safe to run at any time: a campaign without STOPPED is marked
**INCOMPLETE** and shows its latest round; a campaign not yet initialised is listed as "not started".
Re-run by Slurm (twSUM_<run>_NN, tune_w.on_stop_refresh) whenever a masked campaign stops.

Campaigns (plan okay-so-i-want-humming-hearth.md §3)
  ba01 both/abf1only   fa01 fiber/abf1only   sa01 seq/abf1only
  bf09 both/fit9       ff09 fiber/fit9       sf09 seq/fit9
Counterparts (same layer config, 58 groups + unknown live): bw01 (both), fw01 (fiber only), sw01 (sequence only).

Inputs (read-only)
  <root>/<run>/{state.json, validation.tsv, validation_groups.tsv, STOPPED}      masked campaigns
  conc_tuning/<cpt>/{state.json, validation.tsv, validation_groups.tsv, STOPPED}   counterparts (always)
  overnight/chereji_chrXIV_tw_<run>_NN.json                                        score_robocop.py, final chrXIV decode

Definitions
  within 1.1x  : tuned group with |ln((E+1)/(T+1))| <= ln 1.1 at that round (the tuner's gap / deadband test)
  E            : summed occupancy (copies) of the group's LIVE motifs on chrXIV+chrII (state history)
  P / R / F1   : tw_validate.py rows: calls = posterior >= 0.10, 30 bp greedy one-to-one; the "set total" row
                 pools sites/calls/matched over the factor set's groups only (for the counterpart as well)
  final        : masked = last updated round; counterpart = its last updated round (tune) and its last holdout round

Usage:  python masked_summary.py [--out ../presentation/masked_results.md] [--root conc_tuning]
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
INC = "**INCOMPLETE**"
CAMPAIGNS = [("ba01", "both layers", "abf1only", "bw01"), ("bf09", "both layers", "fit9", "bw01"),
             ("sa01", "sequence only", "abf1only", "sw01"), ("sf09", "sequence only", "fit9", "sw01"),
             ("fa01", "fiber only", "abf1only", "fw01"), ("ff09", "fiber only", "fit9", "fw01")]
REFS = (("macisaac", "MacIsaac"), ("rossi_cx", "Rossi _CX"))
LN11 = math.log(1.1)


def fnum(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if v != v else "%.*f" % (nd, v)


def table(hdr, body):
    return (["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
            + ["| " + " | ".join(str(x) for x in r) + " |" for r in body])


def load(root, run):
    d = os.path.join(root, run)
    sp = os.path.join(d, "state.json")
    if not os.path.exists(sp):
        return None
    st = json.load(open(sp))

    def rows(name):
        p = os.path.join(d, name)
        return list(csv.DictReader(open(p), delimiter="\t")) if os.path.exists(p) else []
    grp = {(int(r["round"]), r["set"], r["ref"], r["group"]): r for r in rows("validation_groups.tsv")}
    agg = {(int(r["round"]), r["set"], r["ref"], r["fitted"]): r for r in rows("validation.tsv")}
    stp = os.path.join(d, "STOPPED")
    stopped = open(stp).read().strip() if os.path.exists(stp) else None
    hist = {h["iter"]: h for h in st["history"]}
    hold = sorted({k[0] for k in grp if k[1] == "holdout"})
    return dict(run=run, st=st, grp=grp, agg=agg, stopped=stopped, hist=hist,
                last=max(hist) if hist else None, hold=hold)


def chereji(run, t):
    p = os.path.join(ON, "chereji_chrXIV_tw_%s_%02d.json" % (run, t))
    if t is None or not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["nucleosome"]
    except Exception:
        return None


def latest_chereji(run):
    best = None
    for p in glob.glob(os.path.join(ON, "chereji_chrXIV_tw_%s_[0-9][0-9].json" % run)):
        t = int(re.search(r"_(\d\d)\.json$", p).group(1))
        if best is None or t > best:
            best = t
    return best, chereji(run, best) if best is not None else None


def pooled(c, t, which, ref, groups):
    """(sites, calls, matched) summed over `groups` from validation_groups rows, or None if any is missing."""
    s = n = m = 0
    for g in groups:
        r = c["grp"].get((t, which, ref, g)) if t is not None else None
        if r is None:
            return None
        s, n, m = s + int(r["sites"]), n + int(r["calls"]), m + int(r["matched"])
    return s, n, m


def prf_txt(x):
    if x is None:
        return "–"
    s, n, m = x
    P = m / n if n else float("nan")
    R = m / s if s else float("nan")
    F = 2 * P * R / (P + R) if (n and s and P + R > 0) else (0.0 if (n and s) else float("nan"))
    return "%s / %s / %s (%d calls)" % (fnum(P), fnum(R), fnum(F), n)


def within(st, h, groups):
    return sum(1 for g in groups if abs(h["gap"][g]) <= LN11 + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "presentation", "masked_results.md"))
    ap.add_argument("--root", default=CT, help="root of the MASKED campaigns (testing); counterparts always conc_tuning")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    L = ["# Masked tuner-v2 campaigns: ABF1 only and the 9 fitted TFs vs the unmasked runs", "",
         "_generated %s by `analysis/masked_summary.py` (masked campaigns from `%s`); campaigns still running are "
         "marked %s — re-run `python masked_summary.py` to refresh._"
         % (time.strftime("%Y-%m-%d %H:%M"), os.path.relpath(root, HERE), INC), "",
         "### Rule 7 — what differs", "",
         "Versus bw01 / fw01 / sw01 at the same layer configuration, **only the live factor set differs** "
         "(ABF1 only = `Abf1_murphy`; or the 9 fitted TFs `Abf1_murphy, Cin5_murphy, Fhl1_zhu, Fkh1_zhu, Mcm1_zhu, "
         "Rap1_telomeric, Reb1_badis, Sko1_murphy, Ume6_zhu`; every other motif **and `unknown`** hard-masked) "
         "**and the stop settings are 1.1× deadband / 25 rounds from round 0** (the counterparts started at 1.25× / 8 "
         "rounds and were continued; their stop-setting history is listed below). Everything else is identical: untuned start (round-0 "
         "weights identical to `robocop_train_tw_bw01_00`), no φ (fiber untempered), no nucleosome hold (w_nuc 35), "
         "step cap 10×/round, per-bp cap 0.70, bisection, β seed, MacIsaac targets, tune chrXIV+chrII, holdout chrIV.", "",
         "Definitions: within 1.1× = |ln((E+1)/(T+1))| ≤ ln 1.1 (the tuner's deadband test). E = summed occupancy of "
         "the group's live motifs. P / R / F1 = calls at posterior ≥ 0.10 matched one-to-one within 30 bp "
         "(tw_validate.py). **Counterpart rows are restricted to the same groups**; their RAP1 group pools the motifs "
         "listed below, the fit9 campaigns' RAP1 is `Rap1_telomeric` only.", ""]
    cpts = {r: load(CT, r) for r in ("bw01", "fw01", "sw01")}
    for r, c in cpts.items():
        if c:
            L.append("- counterpart %s: %s; stop settings %s; RAP1 = `%s`; last round %s; holdout rounds %s"
                     % (r, c["stopped"] or INC + " (running)",
                        " → ".join(["%gx/%d" % (sh["previous"]["deadband_fold"], sh["previous"]["max_rounds"])
                                    for sh in c["st"].get("settings_history", [])[:1]]
                                   + ["%gx/%d (from r%02d)" % (sh["deadband_fold"], sh["max_rounds"], sh["from_round"])
                                      for sh in c["st"].get("settings_history", [])]) or "defaults",
                        "+".join(c["st"]["groups"].get("RAP1", [])), c["last"], c["hold"]))
    L.append("")

    C = []
    body = []
    for run, layers, fset, cpt in CAMPAIGNS:
        c = load(root, run)
        k = cpts.get(cpt)
        if c is None or not c["hist"]:
            body.append([run, layers, fset, cpt, "not started" if c is None else INC + " (round 0 not yet updated)"]
                        + ["–"] * 8)
            if c is not None:
                C.append((c, k, layers, fset, cpt))
            continue
        C.append((c, k, layers, fset, cpt))
        st, h, groups = c["st"], c["hist"][c["last"]], sorted(c["st"]["groups"])
        state = c["stopped"].replace("stopped after ", "") if c["stopped"] else INC + " (next round to update %d)" % st["iter"]
        kh = k["hist"][k["last"]] if k and k["hist"] else None
        hN = lambda cc, t: cc["agg"].get((t, "holdout", "macisaac", "all")) if (cc and t is not None) else None
        h0, hf = hN(c, 0), hN(c, c["hold"][-1] if c["hold"] else None)
        ch = chereji(run, c["last"])
        kt, kch = latest_chereji(cpt)
        hold_txt = "r00 %s → " % (fnum(h0["nuc_copies"], 0) if h0 else "–")
        hold_txt += ("r%02d %s" % (c["hold"][-1], fnum(hf["nuc_copies"], 0))) if (hf and c["hold"][-1] != 0 and c["stopped"]) else INC
        body.append([run, layers, fset, cpt, state, "r00–r%02d" % c["last"],
                     "%d/%d" % (within(st, h, groups), len(groups)),
                     ("%d/%d (r%02d)" % (within(k["st"], kh, groups), len(groups), k["last"])) if kh else "–",
                     "%.0f (%+.2f%% vs r0)" % (h["nucleosome_copies"], 100 * h["nucleosome_rel_r0"]),
                     ("%.0f (%+.2f%%, r%02d)" % (kh["nucleosome_copies"], 100 * kh["nucleosome_rel_r0"], k["last"])) if kh else "–",
                     hold_txt,
                     ("%s (n_ref %s)" % (fnum(ch["recall"]), ch["n_ref"])) if ch else INC,
                     ("%s (r%02d)" % (fnum(kch["recall"]), kt)) if kch else "–"])
    L += ["## Campaigns", ""]
    L += table(["run", "layers", "factor set", "counterpart", "state", "rounds", "within 1.1× (final)",
                "counterpart within 1.1× (same groups)", "nucleosome copies chrXIV+chrII (final)",
                "counterpart nucleosome copies", "holdout chrIV nucleosome copies", "Chereji +1/-1 recall chrXIV (final)",
                "counterpart Chereji recall (latest)"], body)
    L.append("")

    for c, k, layers, fset, cpt in C:
        st, groups = c["st"], sorted(c["st"]["groups"])
        L += ["## %s — %s, %s (vs %s)" % (c["run"], layers, fset, cpt), ""]
        if not c["hist"]:
            L += [INC + ": no round updated yet.", ""]
            continue
        last, h0 = c["last"], c["hist"][0]
        hl = c["hist"][last]
        kl = k["last"] if k else None
        kh = k["hist"][kl] if k else None
        body = []
        for g in groups:
            T = st["target"][g]
            body.append([g, "+".join(st["groups"][g]), T,
                         "%.1f → %.1f" % (h0["E"][g], hl["E"][g]),
                         "%.3gx" % (math.exp(-hl["gap"][g])) if hl["gap"][g] == hl["gap"][g] else "–",
                         "yes" if abs(hl["gap"][g]) <= LN11 + 1e-12 else "no",
                         "%.4g" % math.exp(hl["delta_next"][g]),
                         hl["why"][g],
                         ("%.1f → %.1f" % (k["hist"][0]["E"][g], kh["E"][g])) if kh and g in kh["E"] else "–",
                         ("yes" if abs(kh["gap"][g]) <= LN11 + 1e-12 else "no") if kh and g in kh["gap"] else "–",
                         "+".join(k["st"]["groups"].get(g, [])) if k else "–"])
        L += ["### Counts (masked r00 → r%02d; counterpart %s r00 → r%02d)" % (last, cpt, kl if kl is not None else -1), ""]
        L += table(["group", "live motif(s)", "T", "E masked", "(E+1)/(T+1) final", "within 1.1×", "λ next",
                    "last step", "E %s" % cpt, "%s within 1.1×" % cpt, "%s motifs" % cpt], body)
        L.append("")
        mh = c["hold"][-1] if c["hold"] else None
        kho = k["hold"][-1] if (k and k["hold"]) else None
        for which, title, mt, kt in (("tune", "tuning chrXIV+chrII", last, kl), ("holdout", "holdout chrIV", mh, kho)):
            L += ["### P / R / F1 — %s (masked r00, masked r%s; %s r00, %s r%s)"
                  % (title, "%02d" % mt if mt is not None else "–", cpt, cpt, "%02d" % kt if kt is not None else "–"), ""]
            body = []
            for ref, rname in REFS:
                for g in groups + [None]:
                    gs = groups if g is None else [g]
                    label = "set total (%d groups)" % len(groups) if g is None else g
                    fin = prf_txt(pooled(c, mt, which, ref, gs))
                    if which == "holdout" and (mt in (None, 0) or not c["stopped"]):
                        fin = INC if (mt in (None, 0)) else fin + " " + INC
                    body.append([rname, label, prf_txt(pooled(c, 0, which, ref, gs)), fin,
                                 prf_txt(pooled(k, 0, which, ref, gs)) if k else "–",
                                 prf_txt(pooled(k, kt, which, ref, gs)) if k else "–"])
            L += table(["ref", "group", "masked r00", "masked final", "%s r00" % cpt, "%s final" % cpt], body)
            L.append("")
        body = []
        for t in sorted(c["hist"]):
            h = c["hist"][t]
            body.append(["r%02d" % t, "%d/%d" % (within(st, h, groups), len(groups)),
                         prf_txt(pooled(c, t, "tune", "macisaac", groups)),
                         prf_txt(pooled(c, t, "tune", "rossi_cx", groups)),
                         "%.0f (%+.2f%%)" % (h["nucleosome_copies"], 100 * h["nucleosome_rel_r0"]),
                         "%.3g" % h.get("unknown_occ", float("nan")),
                         ", ".join("%s %s" % (g, h["why"][g]) for g in groups if h["why"][g] != "deadband") or "all deadband"])
        L += ["### Per round (%s)" % c["run"], ""]
        L += table(["round", "within 1.1×", "MacIsaac P/R/F1 (set)", "Rossi P/R/F1 (set)", "nucleosome copies",
                    "unknown occ (masked: expect 0)", "non-deadband steps"], body)
        L += ["", "state: %s" % (c["stopped"] or INC), "", "jobs: `%s`" % json.dumps(st["jobs"], sort_keys=True), ""]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print("wrote %s" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()
