"""Score all 5 full-chrI runs vs Chereji + Brogaard (nucleosomes) and MacIsaac (ABF1).
Dump recall + supporting numbers to chrI_5run_metrics.json for charting."""
import os, json
import score_robocop as S

RUNS = [
    ("robocop_chrI_maskon",      "Fiber\nAbf1 only"),
    ("robocop_chrI_maskoff",     "Fiber\nall TFs"),
    ("robocop_chrI_seq_maskon",  "Fiber+seq\nAbf1 only"),
    ("robocop_chrI_seq_maskoff", "Fiber+seq\nall TFs"),
]

def g(r, *ks, d=None):
    x = r
    for k in ks:
        if not isinstance(x, dict) or x.get(k) is None:
            return d
        x = x[k]
    return x

out = []
for outDir, label in RUNS:
    if not os.path.isdir(os.path.join(outDir, "tmpDir")):
        print("SKIP (no decode):", outDir); continue
    print("scoring:", outDir, flush=True)
    r = S.score(outDir, label=label)
    with open(outDir.rstrip("/") + "/score_report.json", "w") as fh:
        json.dump(r, fh, indent=2, default=lambda o: None)
    out.append(dict(
        outDir=outDir, label=label,
        chereji_recall   = g(r, "nucleosome", "recall"),
        chereji_within30 = g(r, "nucleosome", "pct_within_30bp"),
        chereji_nref     = g(r, "nucleosome", "n_ref"),
        chereji_mederr   = g(r, "nucleosome", "median_dyad_err"),
        brogaard_recall  = g(r, "nucleosome_brogaard", "recall"),
        brogaard_within30= g(r, "nucleosome_brogaard", "pct_within_30bp"),
        brogaard_nref    = g(r, "nucleosome_brogaard", "n_ref"),
        brogaard_mederr  = g(r, "nucleosome_brogaard", "median_dyad_err"),
        abf1_recall      = g(r, "abf1", "recall"),
        abf1_nref        = g(r, "abf1", "n_ref"),
        abf1_enrich      = g(r, "abf1", "enrichment"),
    ))

with open("chrI_5run_metrics.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwritten: chrI_5run_metrics.json")
for e in out:
    print("  %-28s chereji_recall=%s  brogaard_recall=%s  abf1_recall=%s" %
          (e["outDir"], e["chereji_recall"], e["brogaard_recall"], e["abf1_recall"]))
print("SCOREDONE")
