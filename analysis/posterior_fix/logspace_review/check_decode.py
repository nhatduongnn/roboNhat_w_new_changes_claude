"""(d) Sanity-check an exact-mode decode's info h5: completes, no NaN, emitting row sums, silent
columns 0, loglik finite, masked (bo09) states 0.  python check_decode.py <outDir> <trainDir> [keep-set: fit9|all]"""
import sys, glob, pickle, json, h5py, numpy as np
from scipy import sparse
out, td, mode = sys.argv[1], sys.argv[2], sys.argv[3]
h = h5py.File(glob.glob(out + '/tmpDir/info_*.h5')[0], 'r')
d = pickle.load(open(td + '/HMMconfig.pkl', 'rb')); ssb = int(d['silent_states_begin'])
R = dict(outDir=out)
for seg in sorted(h.keys()):
    g = h[seg + '/posterior']
    P = np.asarray(sparse.csr_matrix((g['data'][:], g['indices'][:], g['indptr'][:]), shape=tuple(g.attrs['shape'])).todense())
    rs = P[:, :ssb].sum(1)
    r = dict(shape=list(P.shape), nan=int(np.isnan(P).sum()), max=float(P.max()), min=float(P.min()),
             rowsum_min=float(rs.min()), rowsum_max=float(rs.max()), rows_below_0p99=int((rs < 0.99).sum()),
             silent_max=float(np.abs(P[:, ssb:]).max()), loglik=float(h[seg].attrs['log_likelihood']))
    if mode == 'fit9':
        keep = {'Abf1_murphy', 'Cin5_murphy', 'Fhl1_zhu', 'Fkh1_zhu', 'Mcm1_zhu', 'Rap1_telomeric', 'Reb1_badis', 'Sko1_murphy', 'Ume6_zhu'}
        m = np.zeros(P.shape[1], bool)
        for i, nm in enumerate(d['tfs']):
            if nm not in keep:
                s = int(d['tf_starts'][i]); m[s:s + 2 * int(d['tf_lens'][i])] = True
        r['masked_states'] = int(m.sum()); r['masked_post_max'] = float(P[:, m].max())
    R[seg] = r
print(json.dumps(R))
