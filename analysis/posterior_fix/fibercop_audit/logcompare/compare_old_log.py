"""Old (original .so, clipped viewer tables) vs log-space (exact emission) runs, in the viewer windows.

    cd analysis && python posterior_fix/fibercop_audit/logcompare/compare_old_log.py <out.tsv>

Both sides are read the way the Run Browser reads them: factor_tables/part_*.npz (clipped to [0,1]),
overlapping segments averaged (viewer_site/extract_run.py load_parts + stitch). Each position is also tagged
`old_broken` when any OLD segment covering it has an emitting row sum > 1 + 1e-4 there (the C1 rows), so the
differences split into "the old row was broken" vs "the emission changed" (exact emission, no 1e-30 floor).
Read-only.
"""
import sys, os, glob, json, pickle
import numpy as np, pandas as pd, h5py

A = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(A, "viewer_site"))
from extract_run import load_parts, stitch

MODELS = {"bo09_48": "bo09", "bu01_11": "bu01", "bv75_49": "bv75"}
OLD = {"chrXIV": "robocop_chrXIV_chrII_tw_%s", "chrII": "robocop_chrXIV_chrII_tw_%s", "chrIV": "robocop_chrIV_tw_%s"}
NON_MOTIF = {"background", "unknown", "nuc_padding", "nucleosome", "nuc_center", "nuc_start", "nuc_end"}
win = pd.read_csv(os.path.join(A, "viewer_site", "windows.tsv"), sep="\t")
win = win[win.chrom.isin(OLD)]


def broken_mask(d, chrom, ws, we):
    """True where any segment of old decode d covering the position has emitting row sum > 1 + 1e-4."""
    ri = json.load(open(os.path.join(A, d, "RUN_INFO.json")))
    ssb = int(pickle.load(open(os.path.join(A, ri["traindir"], "HMMconfig.pkl"), "rb"))["silent_states_begin"])
    coords = pd.read_csv(os.path.join(A, d, "coords.tsv"), sep="\t")
    want = set(coords.index[(coords.chr == chrom) & (coords.end >= ws) & (coords.start <= we)])
    m = np.zeros(we - ws + 1, bool)
    for fn in glob.glob(os.path.join(A, d, "tmpDir", "info*.h5")):
        with h5py.File(fn, "r") as f:
            for k in f.keys():
                i = int(k.split("_")[1])
                if i not in want:
                    continue
                g = f[k + "/posterior"]; data = g["data"][:]; ind = g["indices"][:]; ptr = g["indptr"][:]
                rs = np.add.reduceat(np.concatenate([np.where(ind < ssb, data, 0.0), [0.0]]), ptr[:-1])
                s = int(coords.loc[i, "start"]); bad = np.flatnonzero(rs > 1 + 1e-4) + s
                bad = bad[(bad >= ws) & (bad <= we)]
                m[bad - ws] = True
    return m


rows = []
for key, lab in MODELS.items():
    new = "robocop_tw3_%s_logfull" % key
    for _, w in win.iterrows():
        c, ws, we = w.chrom, int(w.start), int(w.end)
        old = OLD[c] % key
        so, co, _ = load_parts(old, c); sn, cn, _ = load_parts(new, c)
        cols = [x for x in co if x in cn]
        vo, cov_o = stitch(so, len(co), ws, we); vn, cov_n = stitch(sn, len(cn), ws, we)
        vo = vo[:, [co.index(x) for x in cols]]; vn = vn[:, [cn.index(x) for x in cols]]
        cov = cov_o & cov_n
        br = broken_mask(old, c, ws, we)
        motif = [i for i, x in enumerate(cols) if x not in NON_MOTIF]
        d = np.abs(vo - vn).max(1)
        fake = ((vo[:, motif] >= 0.5) & (vn[:, motif] < 0.5)).any(1)
        lost = ((vn[:, motif] >= 0.5) & (vo[:, motif] < 0.5)).any(1)
        abf = cols.index("Abf1_murphy"); nuc = cols.index("nucleosome")
        for tag, sel in (("old row broken (C1)", cov & br), ("old row fine", cov & ~br)):
            rows.append(dict(model=lab, window="%s:%d-%d" % (c, ws, we), rows=tag, positions=int(sel.sum()),
                             diff_ge_0p1=int((d[sel] >= 0.1).sum()), diff_ge_0p5=int((d[sel] >= 0.5).sum()),
                             abf1_diff_ge_0p1=int((np.abs(vo[sel, abf] - vn[sel, abf]) >= 0.1).sum()),
                             nuc_diff_ge_0p1=int((np.abs(vo[sel, nuc] - vn[sel, nuc]) >= 0.1).sum()),
                             motif_call_only_old=int(fake[sel].sum()), motif_call_only_log=int(lost[sel].sum()),
                             max_diff=float(d[sel].max()) if sel.any() else 0.0))
        print(rows[-2]); print(rows[-1]); sys.stdout.flush()
df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], sep="\t", index=False)
print("wrote", sys.argv[1])
