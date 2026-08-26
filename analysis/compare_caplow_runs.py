"""Where does posterior mass go? baseline vs capA vs capB, whole chrI.

The stated problem: the 141 TFs without an individual Fiber-seq fit share one flat
combined_low_count scalar, so they win at highly-methylated linkers/promoters on
methylation LEVEL alone rather than footprint shape. This measures exactly that -- how
much chrI posterior mass each class of state holds, and how methylated the positions are
where the fallback TFs win.

Reuses score_robocop.load_decode / region_optable (the repo's single source of truth for
reading a decode) rather than re-reading the h5 files.

    python compare_caplow_runs.py
"""
import json

import numpy as np
import pandas as pd

import score_robocop as S

RUNS = [("baseline", "robocop_chrI_seq_maskoff_revfix"),
        ("capA", "robocop_chrI_seq_maskoff_capA"),
        ("capB", "robocop_chrI_seq_maskoff_capB")]
FITTED = ["Abf1_murphy", "Cin5_murphy", "Fhl1_zhu", "Fkh1_zhu", "Mcm1_zhu", "Nhp6a_zhu",
          "Rap1_telomeric", "Reb1_badis", "Sko1_murphy", "Spt15_zhu", "Tbf1_zhu", "Ume6_zhu"]
CHROM, START, END = "chrI", 1, 230218
CALL = 0.30


def main():
    rows, detail = [], {}
    for label, outDir in RUNS:
        dec = S.load_decode(outDir)
        op, covered, fiber = S.region_optable(dec, CHROM, START, END)
        cols = list(op.columns)
        nuc = [c for c in cols if c.lower().startswith("nuc")]
        unk = [c for c in cols if c.lower() == "unknown"]
        fit = [c for c in cols if c in FITTED]
        fallback = [c for c in cols if c not in nuc + unk + fit]

        m = covered & ~np.isnan(fiber) if fiber is not None else covered
        tot = m.sum()
        mass = {k: float(op.loc[m, v].to_numpy().sum() / tot)
                for k, v in (("nucleosome", nuc), ("unknown", unk),
                             ("fitted_TFs", fit), ("fallback_TFs", fallback))}
        mass["background"] = max(0.0, 1.0 - sum(mass.values()))

        # where do the fallback TFs actually win?
        fb = op.loc[m, fallback].to_numpy().sum(1) if fallback else np.zeros(tot)
        ft = op.loc[m, fit].to_numpy().sum(1)
        fr = fiber[m]
        hi_fb = fb >= CALL
        hi_ft = ft >= CALL
        rows.append(dict(
            run=label, n_factors=len(cols), n_fallback_cols=len(fallback),
            bg=round(mass["background"], 4), nuc=round(mass["nucleosome"], 4),
            unknown=round(mass["unknown"], 4),
            fitted=round(mass["fitted_TFs"], 4), fallback=round(mass["fallback_TFs"], 4),
            fb_calls=int(hi_fb.sum()), fit_calls=int(hi_ft.sum()),
            fb_meth_med=round(float(np.median(fr[hi_fb])), 4) if hi_fb.any() else None,
            fb_meth_p90=round(float(np.percentile(fr[hi_fb], 90)), 4) if hi_fb.any() else None,
            fit_meth_med=round(float(np.median(fr[hi_ft])), 4) if hi_ft.any() else None,
            genome_meth_med=round(float(np.median(fr)), 4)))
        detail[label] = dict(mass=mass, fallback_cols=fallback)
        print("%-9s done" % label, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("caplow_comparison.tsv", sep="\t", index=False)
    json.dump(detail, open("caplow_comparison.json", "w"), indent=2)

    pd.set_option("display.width", 200)
    print("\nPOSTERIOR MASS on chrI (fraction of all covered positions)")
    print(df[["run", "bg", "nuc", "unknown", "fitted", "fallback"]].to_string(index=False))
    print("\nCALLS at posterior >= %.2f, and the METHYLATION where they land" % CALL)
    print(df[["run", "fb_calls", "fb_meth_med", "fb_meth_p90",
              "fit_calls", "fit_meth_med", "genome_meth_med"]].to_string(index=False))
    print("\nwrote caplow_comparison.tsv / .json")


if __name__ == "__main__":
    main()
