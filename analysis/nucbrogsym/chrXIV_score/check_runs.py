"""Step 4 checks (2026-10-30): posterior validity per run, and nucleosome-occupancy differences
old vs new on chrXIV. Run from analysis/:  python nucbrogsym/chrXIV_score/check_runs.py"""
import sys, os, glob, json, h5py, numpy as np
sys.path.insert(0, '.')
import score_robocop as sr

RUNS = ['robocop_chrXIV_logfull_oldnuc', 'robocop_chrXIV_logfull_nucbrogsym_oldm6a',
        'robocop_chrXIV_logfull_nucbrogsym_all7']
OUT = 'nucbrogsym/chrXIV_score/'
res = {}
tracks = {}
for r in RUNS:
    dec = sr.load_decode(r)
    sb = int(dec['dshared']['silent_states_begin'])
    nseg = 0; nnan = 0; rmin = np.inf; rmax = -np.inf; nbad = 0; silmax = 0.0; cellmax = 0.0
    for fn in dec['infofiles']:
        with h5py.File(fn, 'r') as f:
            for k in f.keys():
                if not k.startswith('segment_') or 'posterior' not in f[k]:
                    continue
                dp = sr._get_sparse_todense(f, k + '/posterior')
                nseg += 1
                nnan += int((~np.isfinite(dp)).sum())
                rs = dp[:, :sb].sum(1)
                rmin = min(rmin, rs.min()); rmax = max(rmax, rs.max())
                nbad += int((np.abs(rs - 1) > 1e-6).sum())
                cellmax = max(cellmax, float(dp[:, :sb].max()))
                if dp.shape[1] > sb:
                    silmax = max(silmax, float(np.abs(dp[:, sb:]).max()))
    res[r] = dict(n_segments=nseg, n_nonfinite=nnan, rowsum_min=float(rmin), rowsum_max=float(rmax),
                  rows_off_by_1e6=nbad, max_emitting_cell=cellmax, max_abs_silent_cell=silmax)
    print(r, res[r], flush=True)
    (chrm, s, e), = sr.merge_regions(dec['coords'])
    opt, cov, fr = sr.region_optable(dec, chrm, s, e)
    tracks[r] = dict(occ=opt['nucleosome'].values, dyad=opt['nuc_center'].values, cov=cov,
                     fr=fr, start=s, end=e)
np.savez_compressed(OUT + 'tracks.npz', **{r + '__' + k: v for r in RUNS for k, v in tracks[r].items()
                                           if k in ('occ', 'dyad', 'cov', 'fr')})
s0 = tracks[RUNS[0]]['start']
pos = np.arange(s0, tracks[RUNS[0]]['end'] + 1)
for a, b in [(RUNS[0], RUNS[2]), (RUNS[0], RUNS[1]), (RUNS[1], RUNS[2])]:
    d = tracks[b]['occ'] - tracks[a]['occ']
    m = tracks[a]['cov'] & tracks[b]['cov']
    # 1 kb windows, mean |diff|
    w = 1000; nb = len(d) // w
    wd = np.array([np.mean(np.abs(d[i*w:(i+1)*w])) for i in range(nb)])
    wsd = np.array([np.mean(d[i*w:(i+1)*w]) for i in range(nb)])
    top = np.argsort(-wd)[:8]
    key = '%s -> %s' % (a, b)
    res[key] = dict(mean_occ_a=float(tracks[a]['occ'][m].mean()), mean_occ_b=float(tracks[b]['occ'][m].mean()),
                    mean_abs_diff=float(np.abs(d[m]).mean()), frac_pos_absdiff_gt_0p5=float((np.abs(d[m]) > 0.5).mean()),
                    corr=float(np.corrcoef(tracks[a]['occ'][m], tracks[b]['occ'][m])[0, 1]),
                    top_1kb_windows=[dict(region='chrXIV:%d-%d' % (pos[i*w], pos[i*w] + w - 1),
                                          mean_abs_diff=float(wd[i]), mean_signed_diff=float(wsd[i]))
                                     for i in top])
    print(key, json.dumps(res[key], indent=1), flush=True)
    if tracks[a]['fr'] is not None:
        acc = 1 - tracks[a]['occ']; accb = 1 - tracks[b]['occ']; fr = tracks[a]['fr']
        mm = m & np.isfinite(fr)
        res[key]['corr_access_vs_mratio_a'] = float(np.corrcoef(acc[mm], fr[mm])[0, 1])
        res[key]['corr_access_vs_mratio_b'] = float(np.corrcoef(accb[mm], fr[mm])[0, 1])
json.dump(res, open(OUT + 'checks.json', 'w'), indent=1)
print('wrote', OUT + 'checks.json')
