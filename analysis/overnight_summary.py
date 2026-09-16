"""Overnight 2026-09-15 morning report: reads whatever has finished and writes
../presentation/overnight_results.md. Safe to run at any time (anything not finished is marked
**INCOMPLETE**). Re-run by Slurm after the A and C scoring jobs and whenever a B campaign stops.

Inputs
  A   overnight/A_validation.tsv, A_veto_groups.tsv            (overnight_score.py A)
  C   overnight/C_validation.tsv, C_chroms.tsv                  (overnight_score.py C)
  B   conc_tuning/{bt02,bt05,bt10,bw01,sw01}/{state.json,validation.tsv,STOPPED}
      overnight/chereji_chrXIV_tw_<run>_NN.json                 (score_robocop.py, chrXIV only)
  job ids: overnight/jobs.json

Usage:  python overnight_summary.py [--out ../presentation/overnight_results.md]
"""
import argparse
import csv
import glob
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "overnight")
CT = os.path.join(HERE, "conc_tuning")
INC = "**INCOMPLETE**"
B_RUNS = [("bt02", 2), ("bt05", 5), ("bt10", 10)]
B_REFS = [("bw01", "φ=1 (untempered both-layers reference)"), ("sw01", "seq-only reference")]


def rows(path):
    return list(csv.DictReader(open(path), delimiter="\t")) if os.path.exists(path) else None


def f(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if v != v else ("%.*f" % (nd, v))


def prf(r):
    return "%s / %s / %s" % (f(r.get("P")), f(r.get("R")), f(r.get("F1"))) if r else "–"


def mtime(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path))) if os.path.exists(path) else "–"


def table(hdr, body):
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in body]
    return out


# ================================================================ A
def section_a(jobs):
    L = ["## A — fixed-prior fiber veto test (sw01 round-05 weights)", "",
         "Rule 7: **only the layer configuration differs from sw01_05** (same trainDir "
         "`robocop_train_tw_sw01_05`, same weights). seq-only = sw01's own round-05 decode; "
         "+fiber = `run_split_revfix_seq_maskoff_mr58.py`; fiber-only = `run_split_revfix_fiber_maskoff_mr58.py`. "
         "58 groups, calls = posterior >= 0.10, 30 bp greedy one-to-one match (tw_validate code). "
         "tune = chrXIV+chrII, holdout = chrIV.", ""]
    V = rows(os.path.join(OUT, "A_validation.tsv"))
    if V is None:
        return L + ["%s — `overnight/A_validation.tsv` not written yet (scoring job %s)." %
                    (INC, jobs.get("score", {}).get("A", "?")), ""]
    L.append("_scored %s_" % mtime(os.path.join(OUT, "A_validation.tsv")))
    L.append("")
    missing = [r for r in V if r["status"] != "ok"]
    for r in missing:
        L.append("- %s: A %s / %s (`%s`): %s" % (INC, r["config"], r["set"], r["decode"], r["status"]))
    if missing:
        L.append("")
    idx = {(r["config"], r["set"], r["ref"], r["fitted"]): r for r in V if r["status"] == "ok"}
    names = [("seqonly", "seq-only (sw01_05)"), ("both", "seq + fiber"), ("fib", "fiber-only")]
    for which, title in (("tune", "tuning chromosomes chrXIV+chrII"), ("holdout", "holdout chrIV")):
        L.append("**%s** — P / R / F1" % title)
        L.append("")
        body = []
        for cfg, nm in names:
            if (cfg, which, "macisaac", "all") not in idx:
                body.append([nm, INC] + [""] * 6)
                continue
            m, x = idx[(cfg, which, "macisaac", "all")], idx[(cfg, which, "rossi_cx", "all")]
            body.append([nm, m["calls"], prf(m), prf(idx.get((cfg, which, "macisaac", "fitted"))),
                         prf(idx.get((cfg, which, "macisaac", "nonfitted"))), prf(x),
                         prf(idx.get((cfg, which, "rossi_cx", "fitted"))), prf(idx.get((cfg, which, "rossi_cx", "nonfitted")))])
        L += table(["config", "calls", "MacIsaac all", "MacIsaac fitted", "MacIsaac nonfitted", "Rossi all",
                    "Rossi fitted", "Rossi nonfitted"], body)
        L.append("")
    VG = rows(os.path.join(OUT, "A_veto_groups.tsv")) or []
    ok = [r for r in VG if r["status"] == "ok"]
    if not ok:
        return L + ["Fiber veto / added: %s" % INC, ""]
    L.append("**Fiber veto / fiber-added calls vs seq-only** (30 bp, not one-to-one; `supp` = a MacIsaac / Rossi site within 30 bp)")
    L.append("")
    body = []
    for comp in ("both_vs_seqonly", "fib_vs_seqonly"):
        for which in ("tune", "holdout"):
            rs = [r for r in ok if r["comparison"] == comp and r["set"] == which]
            if not rs:
                body.append([comp, which, INC] + [""] * 8)
                continue
            s = {k: sum(int(r[k]) for r in rs) for k in ("calls_seqonly", "calls_cmp", "veto", "veto_mac", "veto_rossi",
                                                         "added", "added_mac", "added_rossi", "kept_mac", "kept_rossi")}
            body.append([comp, which, s["calls_seqonly"], s["calls_cmp"],
                         "%d (%d / %d)" % (s["veto"], s["veto_mac"], s["veto_rossi"]),
                         "%d (%d / %d)" % (s["added"], s["added_mac"], s["added_rossi"]),
                         "%d / %d" % (s["kept_mac"], s["kept_rossi"]),
                         sum(1 for r in rs if int(r["veto"])), sum(1 for r in rs if int(r["added"]))])
    L += table(["comparison", "set", "seq-only calls", "compared calls", "veto (MacIsaac supp / Rossi supp)",
                "added (MacIsaac supp / Rossi supp)", "kept seq-only calls supp (Mac / Rossi)",
                "groups with veto", "groups with added"], body)
    L.append("")
    L.append("Per group, seq + fiber vs seq-only (groups with any veto or added call; full table "
             "`overnight/A_veto_groups.tsv`, call list `overnight/A_veto_calls.tsv`):")
    L.append("")
    body = []
    for which in ("tune", "holdout"):
        rs = [r for r in ok if r["comparison"] == "both_vs_seqonly" and r["set"] == which
              and (int(r["veto"]) or int(r["added"]))]
        for r in sorted(rs, key=lambda r: -(int(r["veto"]) + int(r["added"]))):
            body.append([which, r["group"], r["fitted"], r["calls_seqonly"], r["calls_cmp"],
                         "%s (%s/%s)" % (r["veto"], r["veto_mac"], r["veto_rossi"]),
                         "%s (%s/%s)" % (r["added"], r["added_mac"], r["added_rossi"])])
    L += table(["set", "group", "fitted", "seq-only calls", "+fiber calls", "veto (Mac/Rossi supp)",
                "added (Mac/Rossi supp)"], body) if body else ["(no group has a veto or added call)"]
    L.append("")
    return L


# ================================================================ C
def section_c(jobs):
    L = ["## C — genome-wide validation of the tuned models", "",
         "Rule 7: **only the decode scope (whole genome, 48-task split of `coord_genome_full.tsv`) differs "
         "from each campaign's final round**: sw01_05 with seqonly_mr58, fw01_07 with fiber_mr58, bw01_07 with "
         "seq_mr58. Legacy u001/m001 round 7 re-scored from the existing `calls_ct_*_07` (no new decode) on the same "
         "58 groups. n_excl = invalid-posterior positions dropped by count_calls (chrXII rDNA, chrVIII end, ...).", ""]
    V = rows(os.path.join(OUT, "C_validation.tsv"))
    if V is None:
        return L + ["%s — `overnight/C_validation.tsv` not written yet (scoring job %s)." %
                    (INC, jobs.get("score", {}).get("C", "?")), ""]
    L += ["_scored %s_" % mtime(os.path.join(OUT, "C_validation.tsv")), ""]
    idx = {(r["model"], r["scope"], r["ref"], r["fitted"]): r for r in V if r.get("ref")}
    status = {(r["model"], r["scope"]): r["status"] for r in V}
    excl = {(r["model"], r["scope"]): r.get("n_excluded", "") for r in V}
    models = ["tw_sw01_05", "tw_fw01_07", "tw_bw01_07", "u001_07", "m001_07"]
    for scope, title in (("genome", "genome (16 chromosomes)"), ("tuned_chrXIV+chrII", "tuned chromosomes chrXIV+chrII"),
                         ("untuned_14", "never-tuned chromosomes (other 14)")):
        L += ["**%s** — P / R / F1" % title, ""]
        body = []
        for m in models:
            st = status.get((m, scope), "missing")
            if (m, scope, "macisaac", "all") not in idx:
                body.append([m, INC + " (%s)" % st] + [""] * 7)
                continue
            g = lambda ref, cat: idx.get((m, scope, ref, cat))
            body.append([m + ("" if st == "ok" else " " + INC + " " + st), g("macisaac", "all")["calls"],
                         prf(g("macisaac", "all")), prf(g("macisaac", "fitted")), prf(g("macisaac", "nonfitted")),
                         prf(g("rossi_cx", "all")), prf(g("rossi_cx", "fitted")), prf(g("rossi_cx", "nonfitted")),
                         excl.get((m, scope), "")])
        L += table(["model", "calls", "MacIsaac all", "MacIsaac fitted", "MacIsaac nonfitted", "Rossi all",
                    "Rossi fitted", "Rossi nonfitted", "n_excl"], body)
        L.append("")
    CH = rows(os.path.join(OUT, "C_chroms.tsv")) or []
    if CH:
        chroms = []
        for r in CH:
            if r["scope"] not in chroms:
                chroms.append(r["scope"])
        cidx = {(r["model"], r["scope"], r.get("ref"), r.get("fitted")): r for r in CH}
        L += ["**Per chromosome** — F1 vs MacIsaac / F1 vs Rossi (all 58 groups; `*` = tuned chromosome; "
              "[n] = excluded positions)", ""]
        body = []
        for c in chroms:
            row = [c + ("*" if c in ("chrXIV", "chrII") else "")]
            for m in models:
                a, b = cidx.get((m, c, "macisaac", "all")), cidx.get((m, c, "rossi_cx", "all"))
                if not a:
                    row.append(INC)
                else:
                    ex = int(float(a.get("n_excluded") or 0))
                    row.append("%s / %s%s" % (f(a["F1"]), f(b["F1"]) if b else "–", (" [%d]" % ex) if ex else ""))
            body.append(row)
        L += table(["chrom"] + models, body)
        L.append("")
    return L


# ================================================================ B
def chereji(label):
    p = os.path.join(OUT, "chereji_chrXIV_%s.json" % label)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["nucleosome"]
    except Exception:
        return None


def section_b(jobs):
    L = ["## B — fiber-tempering tuning sweep (bt02 / bt05 / bt10)", "",
         "Rule 7: **differs from bw01 only in the fiber temper φ (2 / 5 / 10)**: Fiber-seq layers 5 and 6 raised to "
         "1/φ for every state, baked into `pkgvar/seq_maskoff_mr58_phi{2,5,10}` (after the 1e-30 floor, before the mr58 "
         "mask). Everything else identical to bw01: weight start, update rule, caps, targets, 58 groups, tune "
         "chrXIV+chrII, holdout chrIV (round 0 and final), MAX_ROUNDS 8.", ""]
    runs = [(r, "φ=%d" % p) for r, p in B_RUNS] + B_REFS
    summary = []
    for run, desc in runs:
        sp = os.path.join(CT, run, "state.json")
        if not os.path.exists(sp):
            summary.append([run, desc, INC + " (no state)"] + [""] * 7)
            continue
        st = json.load(open(sp))
        stp = os.path.join(CT, run, "STOPPED")
        stopped = open(stp).read().strip() if os.path.exists(stp) else None
        V = rows(os.path.join(CT, run, "validation.tsv")) or []
        H = st["history"]
        last = H[-1]["iter"] if H else None
        hold = {int(r["round"]): r for r in V if r["set"] == "holdout" and r["ref"] == "macisaac" and r["fitted"] == "all"}
        holdR = {int(r["round"]): r for r in V if r["set"] == "holdout" and r["ref"] == "rossi_cx" and r["fitted"] == "all"}
        final_hold = hold.get(last) if last else None
        state_txt = ("stopped: " + stopped.replace("stopped after ", "")) if stopped else INC + " (running, next iter %d)" % st["iter"]
        if stopped and last and last not in hold:
            state_txt += "; holdout of r%02d %s" % (last, INC)
        ch = chereji("tw_%s_%02d" % (run, last)) if last is not None else None
        summary.append([run, desc, state_txt,
                        "%d/%d" % (H[-1]["within2x"], H[-1]["n_T5"]) if H else "–",
                        "%.0f (%+.1f%%)" % (H[-1]["nucleosome_copies"], 100 * H[-1]["nucleosome_rel_r0"]) if H else "–",
                        prf(hold.get(0)), prf(final_hold) if final_hold else (INC if H else "–"),
                        prf(holdR.get(last)) if last in holdR else "–",
                        ("%s (n_ref %s)" % (f(ch["recall"]), ch["n_ref"])) if ch else
                        ("not scored" if run == "sw01" else INC)])
    L += ["**Campaign summary** (final round; holdout chrIV P / R / F1; Chereji +1/-1 recall on chrXIV at the final round)", ""]
    L += table(["run", "config", "state", "within-2× (T≥5)", "nucleosome copies vs r0", "holdout r0 MacIsaac",
                "holdout final MacIsaac", "holdout final Rossi", "Chereji recall chrXIV"], summary)
    L.append("")
    for run, desc in runs:
        V = rows(os.path.join(CT, run, "validation.tsv"))
        sp = os.path.join(CT, run, "state.json")
        if not os.path.exists(sp):
            continue
        st = json.load(open(sp))
        hist = {h["iter"]: h for h in st["history"]}
        L += ["**%s (%s)** — per round, tuning chromosomes chrXIV+chrII" % (run, desc), ""]
        if not V:
            L += [INC + " — no validation rows yet", ""]
            continue
        idx = {(int(r["round"]), r["set"], r["ref"], r["fitted"]): r for r in V}
        body = []
        for t in sorted({int(r["round"]) for r in V if r["set"] == "tune"}):
            h = hist.get(t, {})
            m = idx.get((t, "tune", "macisaac", "all"))
            body.append(["r%02d" % t, "%s/%s" % (h.get("within2x", "–"), h.get("n_T5", "–")),
                         m["calls"] if m else "–", prf(m), prf(idx.get((t, "tune", "macisaac", "nonfitted"))),
                         prf(idx.get((t, "tune", "rossi_cx", "all"))), prf(idx.get((t, "tune", "rossi_cx", "nonfitted"))),
                         "%.0f (%+.1f%%)" % (h["nucleosome_copies"], 100 * h["nucleosome_rel_r0"]) if h else "–",
                         ",".join(h.get("capped_under", [])) or "–"])
        L += table(["round", "within-2×", "calls", "MacIsaac all", "MacIsaac nonfitted", "Rossi all", "Rossi nonfitted",
                     "nucleosome copies (vs r0)", "capped_under"], body)
        L.append("")
        hb = []
        for t in sorted({int(r["round"]) for r in V if r["set"] == "holdout"}):
            hb.append(["r%02d" % t] + [prf(idx.get((t, "holdout", ref, cat))) for ref in ("macisaac", "rossi_cx")
                                        for cat in ("all", "fitted", "nonfitted")]
                      + ["%s (%s)" % (f(idx[(t, "holdout", "macisaac", "all")].get("nuc_copies"), 0),
                                      f(100 * float(idx[(t, "holdout", "macisaac", "all")].get("nuc_rel_r0") or "nan"), 1) + "%")])
        if hb:
            L += ["holdout chrIV:", ""]
            L += table(["round", "MacIsaac all", "MacIsaac fitted", "MacIsaac nonfitted", "Rossi all", "Rossi fitted",
                        "Rossi nonfitted", "nucleosome copies (vs r0)"], hb)
            L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "presentation", "overnight_results.md"))
    a = ap.parse_args()
    jp = os.path.join(OUT, "jobs.json")
    jobs = json.load(open(jp)) if os.path.exists(jp) else {}
    L = ["# Overnight results (2026-09-15 launch)", "",
         "_generated %s by `analysis/overnight_summary.py`; anything not finished is marked %s — re-run "
         "`python overnight_summary.py` to refresh._" % (time.strftime("%Y-%m-%d %H:%M"), INC), "",
         "### Rule 7 — what differs", "",
         "- **A**: only the layer configuration differs from sw01_05 (same weights).",
         "- **C**: only the decode scope (whole genome) differs from each campaign's final round.",
         "- **B**: differs from bw01 only in the fiber temper φ (2/5/10); everything else identical "
         "(weights start, rule, caps, targets, groups, chromosomes, rounds).", ""]
    for sec in (section_a, section_c, section_b):
        try:
            L += sec(jobs)
        except Exception as e:           # a broken section must not hide the others
            import traceback
            L += ["## %s" % sec.__name__, "", "%s — summary section failed: `%r`" % (INC, e), "",
                  "```", traceback.format_exc(), "```", ""]
    L += ["### Jobs", "", "```", json.dumps(jobs, indent=1, sort_keys=True), "```", ""]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print("wrote %s" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()
