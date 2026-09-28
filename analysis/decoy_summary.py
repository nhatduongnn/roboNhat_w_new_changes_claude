"""ABF1-vs-decoy tuner-v2 campaigns (2026-09-21): each decoy campaign next to its no-decoy comparator.

Writes ../presentation/decoy_results.md. Adapted from wide_summary.py. Safe to run at any time: a
campaign without STOPPED is marked **INCOMPLETE** and shows its latest round; a campaign with no state
is "not started"; one with a state but no updated round is "round 0 not yet updated". Re-run by Slurm
(twSUM_<run>_NN, tune_w.on_stop_refresh via state on_stop.summary) whenever a decoy campaign stops.

Campaigns (plan ~/.claude/plans/okay-so-i-want-humming-hearth.md)
  width        both   fiber  seq    ABF1/decoy block  trainDir                     comparators
  plain 14 bp  bd14   fd14   sd14   14                robocop_train_decoy14        ba01 fa01 sa01
  7/2   23 bp  bd72   fd72   sd72   23                robocop_train_wide_decoy23   bp72 fp72 sp72
The decoy `Zz_decoy_abf1` has a background-equal PWM (Kd 1.0) and a flat Fiber-seq footprint at ABF1's
own mean rate; its weight is tied to ABF1's every round (w_decoy = w_ABF1; tune_w.py --tie). It has
no target, is never tuned and is never scored as ABF1.

Definitions
  E            : summed ABF1 occupancy (copies) on chrXIV+chrII (state history); T = 58 (MacIsaac)
  decoy occ    : summed decoy occupancy (copies) on chrXIV+chrII (state history["tied"])
  calls        : centre of a posterior run >= 0.10 (count_calls n_pred_fixed / calls/<chrom>.tsv)
  P / R / F1   : tw_validate.py rows (ABF1 group only), matched one-to-one within 30 bp
  spurious     : a comparator ABF1 call with NO MacIsaac ABF1 site within 30 bp (nearest distance, not
                 one-to-one). "Moved to decoy" = that call has a decoy call within 30 bp in the decoy
                 campaign's decode of the same round; "kept as ABF1" = an ABF1 call within 30 bp.
                 Computed from counts_tw_<run>_NN/calls/<chrom>.tsv, so only where both call files exist.

Usage:  python decoy_summary.py [--out ../presentation/decoy_results.md] [--root conc_tuning]
"""
import argparse
import bisect
import collections
import csv
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CT = os.path.join(HERE, "conc_tuning")
INC = "**INCOMPLETE**"
DECOY = "Zz_decoy_abf1"
ABF1 = "Abf1_murphy"
TOL = 30
TUNE = ["chrXIV", "chrII"]
HOLD = ["chrIV"]
LAYERS = {"b": "both layers", "f": "fiber only", "s": "sequence only"}
WIDTHS = [("14", "plain 14 bp", 14, "robocop_train_decoy14", {"b": "ba01", "f": "fa01", "s": "sa01"}),
          ("72", "7/2 23 bp", 23, "robocop_train_wide_decoy23", {"b": "bp72", "f": "fp72", "s": "sp72"})]
CAMPAIGNS = [(p + "d" + w, LAYERS[p], wl, blen, td, comp[p]) for w, wl, blen, td, comp in WIDTHS for p in "bfs"]
REFS = (("macisaac", "MacIsaac"), ("rossi_cx", "Rossi _CX"))


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
    return dict(run=run, root=root, st=st, grp=grp, stopped=stopped, hist=hist,
                last=max(hist) if hist else None, hold=hold)


def pooled(c, t, which, ref, groups):
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


def tied_of(h):
    return (h or {}).get("tied", {}).get(DECOY)


def summary_row(c, run, layers, width, blen, comp, decoy):
    if c is None:
        return [run, layers, width, blen, comp, "not started"] + ["–"] * 9
    if not c["hist"]:
        return [run, layers, width, blen, comp, INC + " (round 0 not yet updated)"] + ["–"] * 9
    st, groups = c["st"], sorted(c["st"]["groups"])
    h = c["hist"][c["last"]]
    state = c["stopped"].replace("stopped after ", "") if c["stopped"] else INC + " (next round to update %d)" % st["iter"]
    E = sum(h["E"][g] for g in groups)
    T = sum(st["target"][g] for g in groups)
    lam = ", ".join("%s %.4g" % (g, math.exp(h["delta"][g])) for g in groups)
    tv = tied_of(h)
    dec = ("occ %.1f, calls %d" % (tv["occ"], tv["calls"])) if tv else ("–" if not decoy else "not recorded")
    hf = c["hold"][-1] if c["hold"] else None
    hold_fin = (prf_txt(pooled(c, hf, "holdout", "macisaac", groups)) if (hf not in (None, 0) and c["stopped"])
                else INC)
    return [run, layers, width, blen, comp, state, "r00–r%02d" % c["last"],
            "%.1f / %d = %.3g" % (E, T, E / T if T else float("nan")), lam, dec,
            prf_txt(pooled(c, c["last"], "tune", "macisaac", groups)),
            prf_txt(pooled(c, c["last"], "tune", "rossi_cx", groups)),
            prf_txt(pooled(c, 0, "holdout", "macisaac", groups)), hold_fin,
            "%.0f (%+.2f%% vs r0)" % (h["nucleosome_copies"], 100 * h["nucleosome_rel_r0"])]


# ---------------------------------------------------------------- spurious-call migration
_MAC = None


def macisaac_abf1():
    global _MAC
    if _MAC is None:
        import tw_validate as TV
        _MAC = {c: sorted(v) for c, v in TV.references({"ABF1": [ABF1]})["ABF1"]["macisaac"].items()}
    return _MAC


def read_calls(root, run, t, chroms):
    """{factor: {chrom: sorted centres}} or None when any chrom file is missing."""
    d = os.path.join(root, "counts_tw_%s_%02d" % (run, t), "calls")
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in chroms:
        p = os.path.join(d, c + ".tsv")
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            fh.readline()
            for line in fh:
                f, ch, centre = line.split("\t")[:3]
                out[f][ch].append(int(centre))
    for f in out:
        for ch in out[f]:
            out[f][ch].sort()
    return out


def near(sorted_xs, x, tol=TOL):
    i = bisect.bisect_left(sorted_xs, x - tol)
    return i < len(sorted_xs) and sorted_xs[i] <= x + tol


def migration(croot, comp, t_comp, droot, run, t_dec, chroms):
    """Comparator's spurious ABF1 calls at round t_comp, and where they went in the decoy run's round t_dec."""
    cc = read_calls(croot, comp, t_comp, chroms)
    dc = read_calls(droot, run, t_dec, chroms)
    if cc is None or dc is None:
        return None
    mac = macisaac_abf1()
    n_calls = n_sp = to_dec = kept = both = 0
    for ch in chroms:
        for x in cc.get(ABF1, {}).get(ch, []):
            n_calls += 1
            if near(mac.get(ch, []), x):
                continue
            n_sp += 1
            a = near(dc.get(ABF1, {}).get(ch, []), x)
            d = near(dc.get(DECOY, {}).get(ch, []), x)
            kept += a
            to_dec += d
            both += a and d
    dec_calls = sum(len(v) for v in dc.get(DECOY, {}).values())
    dec_on_mac = sum(1 for ch in chroms for x in dc.get(DECOY, {}).get(ch, []) if near(mac.get(ch, []), x))
    return dict(n_calls=n_calls, n_sp=n_sp, to_dec=to_dec, kept=kept, both=both,
                gone=n_sp - to_dec - kept + both, dec_calls=dec_calls, dec_on_mac=dec_on_mac)


def mig_txt(m):
    if m is None:
        return ["–"] * 6
    pct = lambda k: "%d (%.1f%%)" % (m[k], 100.0 * m[k] / m["n_sp"]) if m["n_sp"] else "%d" % m[k]
    return ["%d / %d" % (m["n_sp"], m["n_calls"]), pct("to_dec"), pct("kept"), pct("gone"),
            "%d" % m["dec_calls"], "%d" % m["dec_on_mac"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "presentation", "decoy_results.md"))
    ap.add_argument("--root", default=CT, help="root of the DECOY campaigns (testing); comparators always conc_tuning")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    L = ["# ABF1 vs a same-length decoy: tuner-v2 campaigns bd14/fd14/sd14 and bd72/fd72/sd72", "",
         "_generated %s by `analysis/decoy_summary.py` (decoy campaigns from `%s`); campaigns still running are "
         "marked %s — re-run `python decoy_summary.py` to refresh._"
         % (time.strftime("%Y-%m-%d %H:%M"), os.path.relpath(root, HERE), INC), "",
         "### Rule 7 — what differs from the comparators (ba01/fa01/sa01 and bp72/fp72/sp72)", "",
         "**Under test:** a decoy `Zz_decoy_abf1` of ABF1's length (14 or 23 bp) and equal prior weight "
         "(w_decoy = w_ABF1 every round), with a flat Fiber-seq footprint at ABF1's own mean rate (14 bp: W 0.0843 / "
         "C 0.0797; 23 bp: W 0.0892 / C 0.0960) and no sequence motif (PWM = background, Kd 1.0; the reverse-strand "
         "block is the reverse complement of background, see the build notes).", "",
         "**Also different, all stated:** each run's trainDir is retrained with one extra motif (the untuned "
         "shared-root tf_prob shifts, but every tuner round rebuilds the priors from weights); the plain-width runs "
         "have a 50-round cap (ba01/fa01/sa01 had 25); bracket expiry is on (bracket_max_age 3; the plain "
         "comparators never had it); rpy2 is imported lazily in the plain-width decoy trees (no numeric effect).", "",
         "**Identical:** targets (MacIsaac ABF1 58 on chrXIV+chrII), chromosomes, chrIV holdout, step cap, ρ cap "
         "(core basis for 7/2), w_nuc 35, w_unknown 1e-3 (masked), no φ, deadband 1.1×, untuned start.", "",
         "**Seq-only is a control:** there the decoy's sequence emission is (forward block) exactly background and "
         "the fiber layer is off, so it only reflects its prior.", ""]

    body = []
    comps = {}
    for w, wl, blen, td, comp in WIDTHS:
        for p in "bfs":
            r = comp[p]
            comps[r] = load(CT, r)
            body.append(summary_row(comps[r], r, LAYERS[p], wl + " (no decoy)", blen, "—", False))
    camp = []
    for run, layers, width, blen, td, comp in CAMPAIGNS:
        c = load(root, run)
        camp.append((c, run, layers, width, blen, td, comp))
        body.append(summary_row(c, run, layers, width, blen, comp, True))
    L += ["## Campaigns (comparators first)", ""]
    L += table(["run", "layers", "width", "block bp", "comparator", "state", "rounds", "ABF1 E / T (final)",
                "λ (final round)", "decoy occ / calls (final)", "tune MacIsaac P/R/F1 (final)",
                "tune Rossi _CX P/R/F1 (final)", "holdout chrIV MacIsaac P/R/F1 r00",
                "holdout chrIV MacIsaac P/R/F1 final", "nucleosome copies chrXIV+chrII (final)"], body)
    L.append("")

    L += ["## Spurious ABF1 calls: where do they go?", "",
          "Comparator ABF1 calls with no MacIsaac ABF1 site within %d bp, checked against the decoy campaign's calls "
          "in the same round (r00: both untuned, λ = 1 on the same ABF1 Kd; final: each campaign's own last round). "
          "'gone' = neither an ABF1 nor a decoy call within %d bp." % (TOL, TOL), ""]
    rows = []
    for c, run, layers, width, blen, td, comp in camp:
        k = comps.get(comp)
        pairs = [("tune r00", TUNE, 0, 0), ("holdout r00", HOLD, 0, 0)]
        if c and c["last"] is not None and k and k["last"] is not None:
            pairs.append(("tune final (r%02d vs r%02d)" % (k["last"], c["last"]), TUNE, k["last"], c["last"]))
        for name, chroms, tc, td_ in pairs:
            m = migration(CT, comp, tc, root, run, td_, chroms) if c else None
            rows.append([run, comp, name] + mig_txt(m))
    L += table(["run", "comparator", "set / rounds", "comparator spurious / all ABF1 calls", "moved to decoy",
                "kept as ABF1", "gone", "decoy calls (all)", "decoy calls on a MacIsaac ABF1 site"], rows)
    L.append("")

    for c, run, layers, width, blen, td, comp in camp:
        L += ["## %s — %s, %s (vs %s)" % (run, layers, width, comp), ""]
        if c is None:
            L += ["not started.", ""]
            continue
        st, groups = c["st"], sorted(c["st"]["groups"])
        L += ["src trainDir `%s`, driver `%s`, cap basis `%s`, tied `%s`, max_rounds %s, deadband %.3gx"
              % (st.get("src_traindir"), st.get("driver"), (st.get("cap_basis") or {}).get("traindir", "own motif"),
                 json.dumps({d: "%s x%g" % (v["group"], v["ratio"]) for d, v in st.get("tied", {}).items()}),
                 st.get("max_rounds", "default"), math.exp(st.get("deadband", math.log(1.25)))), ""]
        if not c["hist"]:
            L += [INC + ": no round updated yet.", "", "jobs: `%s`" % json.dumps(st.get("jobs", {}), sort_keys=True), ""]
            continue
        k = comps.get(comp)
        mh = c["hold"][-1] if c["hold"] else None
        kho = k["hold"][-1] if (k and k["hold"]) else None
        for which, title, mt, kt in (("tune", "tuning chrXIV+chrII", c["last"], k["last"] if k else None),
                                     ("holdout", "holdout chrIV", mh, kho)):
            L += ["### ABF1 P / R / F1 — %s (%s r00 / final; %s r00 / final)" % (title, run, comp), ""]
            rows = []
            for ref, rname in REFS:
                fin = prf_txt(pooled(c, mt, which, ref, groups))
                if which == "holdout" and (mt in (None, 0) or not c["stopped"]):
                    fin = INC if mt in (None, 0) else fin + " " + INC
                rows.append([rname, prf_txt(pooled(c, 0, which, ref, groups)), fin,
                             prf_txt(pooled(k, 0, which, ref, groups)), prf_txt(pooled(k, kt, which, ref, groups))])
            L += table(["ref", "%s r00" % run, "%s final" % run, "%s r00" % comp, "%s final" % comp], rows)
            L.append("")
        rows = []
        for t in sorted(c["hist"]):
            h = c["hist"][t]
            E = sum(h["E"][g] for g in groups)
            tv = tied_of(h)
            n_calls = pooled(c, t, "tune", "macisaac", groups)
            rows.append(["r%02d" % t, "%.1f" % E, "%s" % (n_calls[1] if n_calls else "–"),
                         "%.1f" % tv["occ"] if tv else "–", "%d" % tv["calls"] if tv else "–",
                         ("%.4g → %.4g" % (tv["w"], tv["w_next"])) if tv else "–",
                         ", ".join("%.4g → %.4g" % (math.exp(h["delta"][g]), math.exp(h["delta_next"][g])) for g in groups),
                         ", ".join(h["why"][g] for g in groups),
                         prf_txt(pooled(c, t, "tune", "macisaac", groups)),
                         prf_txt(pooled(c, t, "tune", "rossi_cx", groups)),
                         "%.0f (%+.2f%%)" % (h["nucleosome_copies"], 100 * h["nucleosome_rel_r0"])])
        L += ["### Per round (%s; T = %s)" % (run, ", ".join("%s %d" % (g, st["target"][g]) for g in groups)), ""]
        L += table(["round", "ABF1 E", "ABF1 calls", "decoy occ", "decoy calls", "w_decoy = w_ABF1 (→ next)",
                    "λ_ABF1 → next", "step", "ABF1 MacIsaac P/R/F1", "ABF1 Rossi P/R/F1", "nucleosome copies"], rows)
        L += ["", "state: %s" % (c["stopped"] or INC), "", "jobs: `%s`" % json.dumps(st.get("jobs", {}), sort_keys=True), ""]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print("wrote %s" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()
