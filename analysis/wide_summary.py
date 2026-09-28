"""Widened-ABF1 tuner-v2 campaigns (2026-09-19): report comparing each width x layer config with the
unpadded ABF1-only campaign of the same layer config (ba01 / fa01 / sa01).

Writes ../presentation/wide_results.md. Adapted from masked_summary.py. Safe to run at any time: a
campaign without STOPPED is marked **INCOMPLETE** and shows its latest round; a campaign with no
state is "not started"; one with a state but no updated round is "round 0 not yet updated".
Re-run by Slurm (twSUM_<run>_NN, tune_w.on_stop_refresh via state on_stop.summary) whenever a
widened campaign stops.

Campaigns (plan okay-so-i-want-humming-hearth.md, 2026-09-19)
  width  both   fiber  seq    ABF1 block (bp)  trainDir
  7/2    bp72   fp72   sp72   23              robocop_train_widememe
  +/-20  bp20   fp20   sp20   54              robocop_train_abf1_pm20
  +/-100 bp100  fp100  sp100  214             robocop_train_abf1_pm100
Baselines (unpadded 14 bp ABF1, same factor set / tuner): ba01 (both), fa01 (fiber), sa01 (seq).

Inputs (read-only)
  <root>/<run>/{state.json, validation.tsv, validation_groups.tsv, STOPPED}   widened campaigns
  conc_tuning/<base>/{...}                                                  baselines (always)
  overnight/chereji_chrXIV_tw_<run>_NN.json                                 score_robocop.py, final chrXIV decode

Definitions
  E          : summed ABF1 occupancy (copies) on chrXIV+chrII at that round (state history); T = 58 (MacIsaac)
  within 1.1x: |ln((E+1)/(T+1))| <= ln 1.1 (the tuner's deadband test)
  P / R / F1 : tw_validate.py rows -- calls = centre of a posterior run >= 0.10, matched one-to-one within 30 bp
  final      : last updated round (tune); last holdout round (holdout, only once the campaign has stopped)

Usage:  python wide_summary.py [--out ../presentation/wide_results.md] [--root conc_tuning]
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
LAYERS = {"b": ("both layers", "ba01"), "f": ("fiber only", "fa01"), "s": ("sequence only", "sa01")}
WIDTHS = [("72", "7/2", 23, "robocop_train_widememe"), ("20", "±20", 54, "robocop_train_abf1_pm20"),
          ("100", "±100", 214, "robocop_train_abf1_pm100")]
CAMPAIGNS = [(p + "p" + w, LAYERS[p][0], wl, blen, td, LAYERS[p][1])
             for w, wl, blen, td in WIDTHS for p in ("b", "f", "s")]
BASELINES = [("ba01", "both layers"), ("fa01", "fiber only"), ("sa01", "sequence only")]
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
    stp = os.path.join(d, "STOPPED")
    stopped = open(stp).read().strip() if os.path.exists(stp) else None
    hist = {h["iter"]: h for h in st.get("history", [])}
    hold = sorted({k[0] for k in grp if k[1] == "holdout"})
    return dict(run=run, st=st, grp=grp, stopped=stopped, hist=hist,
                last=max(hist) if hist else None, hold=hold)


def chereji(run, t):
    p = os.path.join(ON, "chereji_chrXIV_tw_%s_%02d.json" % (run, t)) if t is not None else None
    if p is None or not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["nucleosome"]
    except Exception:
        return None


def pooled(c, t, which, ref, groups):
    """(sites, calls, matched) summed over `groups`, or None if any row is missing."""
    if c is None or t is None:
        return None
    s = n = m = 0
    for g in groups:
        r = c["grp"].get((t, which, ref, g))
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


def summary_row(c, run, layers, width, blen, base):
    """One row of the campaign table; tolerates a missing state or no updated round."""
    if c is None:
        return [run, layers, width, blen, base, "not started"] + ["–"] * 9
    if not c["hist"]:
        return [run, layers, width, blen, base, INC + " (round 0 not yet updated)"] + ["–"] * 9
    st, groups = c["st"], sorted(c["st"]["groups"])
    h = c["hist"][c["last"]]
    state = c["stopped"].replace("stopped after ", "") if c["stopped"] else INC + " (next round to update %d)" % st["iter"]
    E = sum(h["E"][g] for g in groups)
    T = sum(st["target"][g] for g in groups)
    lam = ", ".join("%s %.4g" % (g, math.exp(h["delta"][g])) for g in groups)
    hf = c["hold"][-1] if c["hold"] else None
    hold_fin = prf_txt(pooled(c, hf, "holdout", "macisaac", groups)) if (hf not in (None, 0) and c["stopped"]) else INC
    ch = chereji(run, c["last"])
    return [run, layers, width, blen, base, state, "r00–r%02d" % c["last"],
            "%.1f / %d = %.3g" % (E, T, E / T if T else float("nan")), lam,
            prf_txt(pooled(c, c["last"], "tune", "macisaac", groups)),
            prf_txt(pooled(c, c["last"], "tune", "rossi_cx", groups)),
            prf_txt(pooled(c, 0, "holdout", "macisaac", groups)), hold_fin,
            "%.0f (%+.2f%% vs r0)" % (h["nucleosome_copies"], 100 * h["nucleosome_rel_r0"]),
            ("%s (n_ref %s)" % (fnum(ch["recall"]), ch["n_ref"])) if ch else (INC if not c["stopped"] else "–")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "presentation", "wide_results.md"))
    ap.add_argument("--root", default=CT, help="root of the WIDENED campaigns (testing); baselines always conc_tuning")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    L = ["# Widened-ABF1 tuner-v2 campaigns: 7/2, ±20, ±100 × both / fiber / sequence", "",
         "_generated %s by `analysis/wide_summary.py` (widened campaigns from `%s`); campaigns still running are "
         "marked %s — re-run `python wide_summary.py` to refresh._"
         % (time.strftime("%Y-%m-%d %H:%M"), os.path.relpath(root, HERE), INC), "",
         "### Rule 7 — what differs from ba01 / fa01 / sa01", "",
         "**Under test:** ABF1's width. Each width brings its own trainDir (built from a widened MEME whose pad "
         "columns are the flank composition of the 341 Rossi ABF1 sites; the 14 bp core rows are the shipped ones "
         "verbatim) and a widened ABF1 Fiber-seq p-vector sliced from the Rossi refit (pm50 for 7/2 and ±20, "
         "pm200 for ±100).", "",
         "**Also different, all stated:** max_rounds 50 (baselines 25); bracket expiry on (bracket_max_age 3; the "
         "baselines never bisected); rpy2 imported lazily in the new pkgvar trees (no numeric effect); the per-bp "
         "cap is judged on the 14 bp **core** (cap_delta 9.571, numerically the baselines' cap, but not the widened "
         "motif's own ρ).", "",
         "**Identical:** untuned start (λ = 1 on the widened model's own Kd), no φ, `unknown` hard-masked (w 1e-3 "
         "gauge), w_nuc 35, no nucleosome hold, step cap 1 decade, deadband 1.1×, β seed, MacIsaac target 58, tune "
         "chrXIV+chrII, holdout chrIV.", "",
         "**Caveats.** Circularity: sequence and fiber pads are estimated at Rossi sites, so Rossi scoring is fully "
         "circular and MacIsaac partly (77.5% of MacIsaac ABF1 sites lie inside Rossi, per the 2026-09-19 plan). A call is the centre of a posterior "
         "run ≥ 0.10, so ±100 (214 bp) plateaus can merge nearby sites: compare calls with E.", ""]

    body = []
    base = {r: load(CT, r) for r, _ in BASELINES}
    for r, layers in BASELINES:
        body.append(summary_row(base[r], r, layers, "none (14 bp core)", 14, "—"))
    camp = []
    for run, layers, width, blen, td, b in CAMPAIGNS:
        c = load(root, run)
        camp.append((c, run, layers, width, blen, td, b))
        body.append(summary_row(c, run, layers, width, blen, b))
    L += ["## Campaigns (baselines first)", ""]
    L += table(["run", "layers", "ABF1 pad", "block bp", "baseline", "state", "rounds", "E / T (final)",
                "λ (final round)", "tune MacIsaac P/R/F1 (final)", "tune Rossi _CX P/R/F1 (final)",
                "holdout chrIV MacIsaac P/R/F1 r00", "holdout chrIV MacIsaac P/R/F1 final",
                "nucleosome copies chrXIV+chrII (final)", "Chereji +1/-1 recall chrXIV (final)"], body)
    L.append("")

    for c, run, layers, width, blen, td, b in camp:
        L += ["## %s — %s, ABF1 %s (%d bp; vs %s)" % (run, layers, width, blen, b), ""]
        if c is None:
            L += ["not started.", ""]
            continue
        st, groups = c["st"], sorted(c["st"]["groups"])
        L += ["src trainDir `%s`, driver `%s`, cap basis `%s`, max_rounds %s, deadband %.3gx"
              % (st.get("src_traindir"), st.get("driver"),
                 (st.get("cap_basis") or {}).get("traindir", "own motif"), st.get("max_rounds", "default"),
                 math.exp(st.get("deadband", math.log(1.25)))), ""]
        if not c["hist"]:
            L += [INC + ": no round updated yet.", "", "jobs: `%s`" % json.dumps(st.get("jobs", {}), sort_keys=True), ""]
            continue
        k = base.get(b)
        mh = c["hold"][-1] if c["hold"] else None
        kho = k["hold"][-1] if (k and k["hold"]) else None
        for which, title, mt, kt in (("tune", "tuning chrXIV+chrII", c["last"], k["last"] if k else None),
                                     ("holdout", "holdout chrIV", mh, kho)):
            L += ["### P / R / F1 — %s (%s r00 / final; %s r00 / final)" % (title, run, b), ""]
            rows = []
            for ref, rname in REFS:
                fin = prf_txt(pooled(c, mt, which, ref, groups))
                if which == "holdout" and (mt in (None, 0) or not c["stopped"]):
                    fin = INC if mt in (None, 0) else fin + " " + INC
                rows.append([rname, prf_txt(pooled(c, 0, which, ref, groups)), fin,
                             prf_txt(pooled(k, 0, which, ref, groups)), prf_txt(pooled(k, kt, which, ref, groups))])
            L += table(["ref", "%s r00" % run, "%s final" % run, "%s r00" % b, "%s final" % b], rows)
            L.append("")
        rows = []
        for t in sorted(c["hist"]):
            h = c["hist"][t]
            E = sum(h["E"][g] for g in groups)
            n_calls = pooled(c, t, "tune", "macisaac", groups)
            rows.append(["r%02d" % t, "%.1f" % E, "%s" % (n_calls[1] if n_calls else "–"),
                         ", ".join("%.4g → %.4g" % (math.exp(h["delta"][g]), math.exp(h["delta_next"][g])) for g in groups),
                         ", ".join(h["why"][g] for g in groups),
                         prf_txt(pooled(c, t, "tune", "macisaac", groups)),
                         prf_txt(pooled(c, t, "tune", "rossi_cx", groups)),
                         "%.0f (%+.2f%%)" % (h["nucleosome_copies"], 100 * h["nucleosome_rel_r0"]),
                         "%.3g" % h.get("unknown_occ", float("nan"))])
        L += ["### Per round (%s; T = %s)" % (run, ", ".join("%s %d" % (g, st["target"][g]) for g in groups)), ""]
        L += table(["round", "E", "calls (MacIsaac rows)", "λ → λ next", "step", "MacIsaac P/R/F1", "Rossi P/R/F1",
                    "nucleosome copies", "unknown occ (masked: expect 0)"], rows)
        L += ["", "state: %s" % (c["stopped"] or INC), "", "jobs: `%s`" % json.dumps(st.get("jobs", {}), sort_keys=True), ""]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print("wrote %s" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()
