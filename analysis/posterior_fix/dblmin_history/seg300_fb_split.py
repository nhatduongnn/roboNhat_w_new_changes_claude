"""Which pass carries defect 3 at the 147 bad rows of seg 300? Swap tables between A (+DBL_MIN) and L (log).
    python seg300_fb_split.py <cap.npz> unused"""
import sys, os, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seg300_forward_replay.py')).read().split("with np.errstate(all='ignore'):")[0])
with np.errstate(all='ignore'):
    fA, sfA, bA, pA = linear(R + '/pkg/robocop/librobocop.so')
lf, lsf, lb, pL = logspace()
def norm_rows(x): return x / x[:, :ssb].sum(1, keepdims=True)
fLn = np.exp(lf); bLn = np.exp(lb)                      # each row already normalised over emitting cols
def post(f, b):                                          # b stored reversed
    p = f[:, :ssb] * b[::-1, :ssb]; return p / p.sum(1, keepdims=True)
P_LL = post(fLn, bLn); P_AA = post(fA, bA); P_AL = post(fA, bLn); P_LA = post(fLn, bA)
bad = np.arange(1682, 1829)
for name, P in [('f=A b=A (fix B)', P_AA), ('f=A b=L', P_AL), ('f=L b=A', P_LA)]:
    d = np.abs(P - P_LL).max(1)
    print('%-16s rows>0.01 vs log: whole seg %4d, inside 1682-1828 %3d, max %.3f' % (name, (d > .01).sum(), (d[bad] > .01).sum(), d.max()))
db = np.abs(bA[::-1, :ssb] - bLn[::-1, :ssb]).max(1); df = np.abs(fA[:, :ssb] - fLn[:, :ssb]).max(1)
print('normalised backward rows differing >0.01: %d (positions %s..%s)' % ((db > .01).sum(), np.flatnonzero(db > .01).min(), np.flatnonzero(db > .01).max()))
print('normalised forward  rows differing >0.01: %d (positions %s..%s)' % ((df > .01).sum(), np.flatnonzero(df > .01).min(), np.flatnonzero(df > .01).max()))
