"""
Score the four full-chrI decodes (2x2: Fiber-only vs Fiber+seq  X  mask-off vs mask-on)
against BOTH nucleosome references (Chereji +1/-1 and Brogaard top-2000) plus ABF1, and
print a side-by-side comparison. Writes chrI_compare.json.
"""
import os, json
import score_robocop as S

RUNS = [
    ("robocop_chrI_maskoff",     "Fiber-only  mask-OFF"),
    ("robocop_chrI_maskon",      "Fiber-only  mask-ON"),
    ("robocop_chrI_seq_maskoff", "Fiber+seq   mask-OFF"),
    ("robocop_chrI_seq_maskon",  "Fiber+seq   mask-ON"),
]

results = {}
for outDir, label in RUNS:
    if not os.path.isdir(os.path.join(outDir, "tmpDir")):
        print("SKIP (no decode yet):", label); continue
    print("scoring:", label, flush=True)
    r = S.score(outDir, label=label)
    results[label] = r
    with open(outDir.rstrip("/") + "/score_report.json", "w") as fh:
        json.dump(r, fh, indent=2, default=lambda o: None)

def g(r, *keys, default=None):
    d = r
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d

def fmt(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x) if isinstance(x, float) else str(x))

rows = [(lab, results[lab]) for _, lab in RUNS if lab in results]

print("\n" + "=" * 100)
print("FULL chrI  --  2x2 comparison   (nucleosome tol=20bp, ABF1 tol=20bp)")
print("=" * 100)

def block(title, path_recall, path_med, path_w30, path_nref):
    print("\n%s" % title)
    print("  %-22s %8s %8s %10s %10s" % ("run", "n_ref", "recall", "med_err", "within30"))
    for lab, r in rows:
        print("  %-22s %8s %8s %10s %10s" % (
            lab,
            fmt(g(r, *path_nref)),
            fmt(g(r, *path_recall)),
            fmt(g(r, *path_med), 1),
            fmt(g(r, *path_w30)),
        ))

block("[NUC] vs Chereji +1/-1",
      ("nucleosome", "recall"), ("nucleosome", "median_dyad_err"),
      ("nucleosome", "pct_within_30bp"), ("nucleosome", "n_ref"))
block("[NUC] vs Brogaard top-2000",
      ("nucleosome_brogaard", "recall"), ("nucleosome_brogaard", "median_dyad_err"),
      ("nucleosome_brogaard", "pct_within_30bp"), ("nucleosome_brogaard", "n_ref"))

print("\n[ABF1] vs MacIsaac")
print("  %-22s %8s %8s %10s %12s" % ("run", "n_ref", "recall", "enrich", "auroc"))
for lab, r in rows:
    print("  %-22s %8s %8s %10s %12s" % (
        lab,
        fmt(g(r, "abf1", "n_ref")),
        fmt(g(r, "abf1", "recall")),
        fmt(g(r, "abf1", "enrichment"), 2),
        fmt(g(r, "abf1", "auroc")),
    ))

print("\n[PHASING] median period bp / [ACCESS] corr")
for lab, r in rows:
    print("  %-22s  phasing=%s   access=%s" % (
        lab, fmt(g(r, "phasing", "median_period_bp"), 1),
        fmt(g(r, "accessibility_consistency", "mean_corr_access_vs_methylation"))))

with open("chrI_compare.json", "w") as fh:
    json.dump(results, fh, indent=2, default=lambda o: None)
print("\nwritten: chrI_compare.json")
