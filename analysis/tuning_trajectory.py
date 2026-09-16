"""What each concentration step bought: count agreement vs site accuracy, round by round.

Why this exists
---------------
The tuning loop moves each TF's lambda until its decoded count matches MacIsaac. Matching the
COUNT says nothing about whether the calls are at the right SITES: a raise can add calls that
are all junk, and a lowering can throw away real sites along with false ones. So for every
group and every round this reads the loop's own reports (`conc_tuning/<run>/report_NN.tsv`) and
asks, of each lambda step, what it did to precision and recall.

The numbers
-----------
Per group per round, from calls at posterior >= 0.10 matched to MacIsaac within 30 bp:

    precision = matched / calls          recall = matched / MacIsaac          F1

Per step (round t vs t-1):

    d_log10_lambda      the step taken
    d_abs_log_gap       < 0 means the count moved TOWARD MacIsaac
    d_precision, d_recall, d_f1
    marginal_precision  d_matched / d_calls -- of the calls this step added (or removed), the
                        fraction that were at MacIsaac sites. For a raise, ~0 means the new
                        calls were junk; for a lowering, a high value means real sites were lost.

MacIsaac requires cross-species conservation, so real but unconserved sites count as misses:
precision is a lower bound, meaningful for comparing rounds and steps, not as an absolute.

Usage
-----
    python tuning_trajectory.py --run u001           # writes conc_tuning/u001/trajectory.tsv
"""
import argparse
import glob
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NAN = float("nan")


def _f1(p, r):
    return 2 * p * r / (p + r) if p == p and r == r and (p + r) > 0 else (0.0 if p == p and r == r else NAN)


def read_reports(run):
    """{round: {group: row}} from every report_NN.tsv of the run."""
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, "conc_tuning", run, "report_*.tsv"))):
        t = int(re.search(r"report_(\d+)\.tsv$", path).group(1))
        rows = {}
        with open(path) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                r = dict(zip(hdr, line.rstrip("\n").split("\t")))
                T, E = float(r["macisaac"]), float(r["decoded"])
                calls, matched = int(r["calls"]), int(r["matched_30bp"])
                p = matched / calls if calls else NAN
                rec = matched / T if T else NAN
                rows[r["group"]] = dict(
                    group=r["group"], round=t, lam=float(r["lambda"]), target=T, decoded=E,
                    calls=calls, matched=matched, precision=p, recall=rec, f1=_f1(p, rec),
                    abs_log_gap=abs(math.log(T / E)) if T > 0 and E > 0 else NAN,
                    status=r["status"])
        out[t] = rows
    return out


def trajectory(reports):
    """Long rows, one per group per round, with step deltas against the previous round."""
    rows = []
    rounds = sorted(reports)
    for i, t in enumerate(rounds):
        prev = reports[rounds[i - 1]] if i else {}
        for g, r in sorted(reports[t].items()):
            r = dict(r)
            q = prev.get(g)
            if q:
                dc, dm = r["calls"] - q["calls"], r["matched"] - q["matched"]
                r.update(
                    d_log10_lambda=math.log10(r["lam"] / q["lam"]),
                    d_abs_log_gap=r["abs_log_gap"] - q["abs_log_gap"],
                    d_calls=dc, d_matched=dm,
                    d_precision=r["precision"] - q["precision"],
                    d_recall=r["recall"] - q["recall"],
                    d_f1=r["f1"] - q["f1"],
                    marginal_precision=dm / dc if dc else NAN)
            rows.append(r)
    return rows


def pooled(reports):
    """Per round: pooled precision / recall / F1 over all groups."""
    out = []
    for t in sorted(reports):
        rs = reports[t].values()
        c = sum(r["calls"] for r in rs)
        m = sum(r["matched"] for r in rs)
        T = sum(r["target"] for r in rs)
        p, rec = (m / c if c else NAN), (m / T if T else NAN)
        within = sum(1 for r in rs if r["status"] == "within_2x")
        out.append(dict(round=t, calls=c, matched=m, macisaac=T, precision=p, recall=rec,
                        f1=_f1(p, rec), within_2x=within, groups=len(reports[t])))
    return out


COLS = ["group", "round", "lam", "target", "decoded", "calls", "matched", "precision", "recall",
        "f1", "abs_log_gap", "status", "d_log10_lambda", "d_abs_log_gap", "d_calls",
        "d_matched", "d_precision", "d_recall", "d_f1", "marginal_precision"]


def write(rows, path):
    with open(path, "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for r in rows:
            fh.write("\t".join(
                ("%.6g" % r[c]) if isinstance(r.get(c), float) else str(r.get(c, ""))
                for c in COLS) + "\n")


def _pct(x):
    return "   n/a" if x != x else "%5.1f%%" % (100 * x)


def summarize(run, top=8):
    reports = read_reports(run)
    if not reports:
        print("no reports under conc_tuning/%s" % run)
        return
    rows = trajectory(reports)
    out = os.path.join(HERE, "conc_tuning", run, "trajectory.tsv")
    write(rows, out)

    print("\n== site accuracy by round (%s; calls at post >= 0.10, matched within 30 bp) ==" % run)
    print("  round  within2x      calls  matched  precision  recall     F1   (change in F1)")
    last = None
    for p in pooled(reports):
        d = "" if last is None else "  (%+.4f)" % (p["f1"] - last)
        print("  %5d  %5d/%-3d %9d  %7d     %s  %s  %.4f%s"
              % (p["round"], p["within_2x"], p["groups"], p["calls"], p["matched"],
                 _pct(p["precision"]), _pct(p["recall"]), p["f1"], d))
        last = p["f1"]

    rounds = sorted(reports)
    if len(rounds) < 2:
        print("  (one round so far -- step effects appear after the next round)")
        print("  wrote %s" % out)
        return
    t = rounds[-1]
    steps = [r for r in rows if r["round"] == t and "d_f1" in r and r["d_f1"] == r["d_f1"]]
    moved = [r for r in steps if abs(r["d_log10_lambda"]) > 1e-9]
    hdr = ("    %-8s %8s  %13s  %13s  %7s %7s %8s  %8s  %s"
           % ("group", "dlog10λ", "calls", "matched", "dprec", "drec", "dF1", "marg.prec", "count→MacIsaac"))

    def line(r):
        q = reports[rounds[-2]][r["group"]]
        toward = ("closer" if r["d_abs_log_gap"] < 0 else "further") if r["d_abs_log_gap"] == r["d_abs_log_gap"] else "n/a"
        return ("    %-8s %+8.2f  %5d → %-5d  %5d → %-5d  %+6.1f%% %+6.1f%% %+8.4f  %8s  %s"
                % (r["group"], r["d_log10_lambda"], q["calls"], r["calls"], q["matched"],
                   r["matched"], 100 * r["d_precision"] if r["d_precision"] == r["d_precision"] else 0,
                   100 * r["d_recall"], r["d_f1"], _pct(r["marginal_precision"]).strip(), toward))

    print("\n== step %d -> %d: %d groups moved lambda ==" % (rounds[-2], t, len(moved)))
    print("  biggest F1 gains:")
    print(hdr)
    for r in sorted(moved, key=lambda r: -r["d_f1"])[:top]:
        print(line(r))
    print("  biggest F1 losses:")
    print(hdr)
    for r in sorted(moved, key=lambda r: r["d_f1"])[:top]:
        print(line(r))

    if len(rounds) > 2:
        print("\n== per group: the step that changed F1 most (top %d by |dF1|) ==" % top)
        best = {}
        for r in rows:
            if "d_f1" in r and r["d_f1"] == r["d_f1"]:
                if r["group"] not in best or abs(r["d_f1"]) > abs(best[r["group"]]["d_f1"]):
                    best[r["group"]] = r
        for r in sorted(best.values(), key=lambda r: -abs(r["d_f1"]))[:top]:
            print("    %-8s round %d  dlog10λ %+6.2f  dF1 %+.4f  marginal precision %s"
                  % (r["group"], r["round"], r["d_log10_lambda"], r["d_f1"],
                     _pct(r["marginal_precision"]).strip()))
    print("  wrote %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="u001")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    summarize(args.run, args.top)


if __name__ == "__main__":
    main()
