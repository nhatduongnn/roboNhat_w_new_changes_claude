"""What each downstream reader showed at every position touched by an un-normalised posterior row.

    cd analysis && python posterior_fix/fibercop_audit/extent/reader_impact.py <decode dir> <out.tsv.gz>

Readers (all average overlapping segments equally, as the code does):
  correct  each segment's rows / emitting row sum, collapsed, averaged   (score_robocop.region_optable since 2026-10-05)
  viewer   each segment collapsed raw, clipped to [0,1], averaged        (write_factor_table.py -> viewer_site/extract_run.py)
  old      raw rows averaged, collapsed, no clip                          (region_optable before 2026-10-05: score_robocop,
           make_posterior_viewer, count_calls; count_calls then EXCLUDES a position if any column is outside [0, 1+1e-4])
Collapse = the run's own pkgvar tree's get_posterior_binding_probability_df (linear, so collapse/average commute).
Positions written: every position of a segment whose emitting row sum exceeds 1+1e-6, plus the overlapping
rows of its neighbours. Read-only.
"""
import sys, os, json, glob, pickle, contextlib
import numpy as np, pandas as pd, h5py, scipy.sparse as sp

d, out = sys.argv[1].rstrip("/"), sys.argv[2]
ri = json.load(open(os.path.join(d, "RUN_INFO.json")))
sys.path.insert(0, ri["pkg_tree"])
import robocop
hc = pickle.load(open(os.path.join(ri["traindir"], "HMMconfig.pkl"), "rb")); hc["info_file"] = None
ssb = int(hc["silent_states_begin"])
coords = pd.read_csv(os.path.join(d, "coords.tsv"), sep="\t")

seg_file = {}
for fn in glob.glob(os.path.join(d, "tmpDir", "info*.h5")):
    with h5py.File(fn, "r") as f:
        for k in f.keys():
            seg_file[int(k.split("_")[1])] = fn

def load(idx):
    with h5py.File(seg_file[idx], "r") as f:
        g = f["segment_%d/posterior" % idx]
        P = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g.attrs["shape"])).toarray()
    rs = P[:, :ssb].sum(1)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        raw = robocop.get_posterior_binding_probability_df(hc, P)
    cols = list(raw.columns); raw = raw.values
    norm = np.where(rs[:, None] > 0, raw / np.where(rs > 0, rs, 1)[:, None], 0.0)
    return cols, raw, norm, rs

# bad segments
bad = []
for idx in sorted(seg_file):
    with h5py.File(seg_file[idx], "r") as f:
        g = f["segment_%d/posterior" % idx]; data = g["data"][:]; ind = g["indices"][:]; ptr = g["indptr"][:]
    rs = np.add.reduceat(np.concatenate([np.where(ind < ssb, data, 0), [0]]), ptr[:-1]); rs[ptr[:-1] == ptr[1:]] = 0
    if (rs > 1 + 1e-6).any():
        bad.append(idx)
print(d, "segments with a row sum > 1+1e-6:", len(bad), flush=True)

frames = []
for idx in bad:
    chrom, s, e = coords.loc[idx, "chr"], int(coords.loc[idx, "start"]), int(coords.loc[idx, "end"])
    group = [j for j in coords.index if coords.loc[j, "chr"] == chrom and coords.loc[j, "start"] <= e and coords.loc[j, "end"] >= s]
    acc = {m: None for m in ("correct", "viewer", "old")}; cnt = np.zeros(e - s + 1)
    rsum_raw = np.zeros(e - s + 1)
    for j in group:
        cols, raw, norm, rs = load(j)
        js = int(coords.loc[j, "start"]); n = raw.shape[0]
        a, b = max(s, js), min(e, js + n - 1)
        sl_dst, sl_src = slice(a - s, b - s + 1), slice(a - js, b - js + 1)
        for m, v in (("correct", norm), ("viewer", np.clip(raw, 0, 1)), ("old", raw)):
            if acc[m] is None: acc[m] = np.zeros((e - s + 1, raw.shape[1]))
            acc[m][sl_dst] += v[sl_src]
        cnt[sl_dst] += 1
        if j == idx: rsum_raw[sl_dst] = rs[sl_src]
    cov = cnt > 0
    for m in acc: acc[m][cov] /= cnt[cov, None]
    ci = {c: i for i, c in enumerate(cols)}
    motif = [i for c, i in ci.items() if c not in ("background", "unknown", "nuc_padding", "nucleosome", "nuc_center", "nuc_start", "nuc_end")]
    C, V, O = acc["correct"], acc["viewer"], acc["old"]
    old_excl = ~((O >= -1e-4).all(1) & (O <= 1 + 1e-4).all(1) & np.isfinite(O).all(1))
    df = pd.DataFrame(dict(
        chrom=chrom, pos=np.arange(s, e + 1), segment=idx, n_segs=cnt.astype(int), rowsum=rsum_raw,
        viewer_maxerr=np.abs(V - C).max(1), old_excluded=old_excl,
        old_maxerr=np.where(old_excl, np.nan, np.abs(O - C).max(1)),
        abf1_correct=C[:, ci["Abf1_murphy"]], abf1_viewer=V[:, ci["Abf1_murphy"]],
        nuc_correct=C[:, ci["nucleosome"]], nuc_viewer=V[:, ci["nucleosome"]],
        bg_correct=C[:, ci["background"]], bg_viewer=V[:, ci["background"]],
        motif_maxerr_viewer=np.abs(V[:, motif] - C[:, motif]).max(1),
        worst_col_viewer=[cols[k] for k in np.abs(V - C).argmax(1)],
        n_motif_shown_ge05_true_lt05=((V[:, motif] >= 0.5) & (C[:, motif] < 0.5)).sum(1),
        n_motif_true_ge05_shown_lt05=((C[:, motif] >= 0.5) & (V[:, motif] < 0.5)).sum(1)))
    frames.append(df[cov])
res = pd.concat(frames).drop_duplicates(["chrom", "pos"], keep="first")
res.to_csv(out, sep="\t", index=False)
print("wrote", out, len(res), "positions")
