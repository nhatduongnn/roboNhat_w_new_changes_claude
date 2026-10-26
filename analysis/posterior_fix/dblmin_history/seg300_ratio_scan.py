"""Where does +DBL_MIN first inflate a real (non-floor) forward/backward cell? seg 300.
    python seg300_ratio_scan.py <cap.npz> unused
For each row: max over emitting cells of log10(A/L), restricted to cells that L says are possible
(finite log) and that A holds well above its own padding floor (> 1e3 * DBL_MIN/S)."""
import sys, os, ctypes, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seg300_forward_replay.py')).read().split("with np.errstate(all='ignore'):")[0])
with np.errstate(all='ignore'):
    fA, sfA, bA, pA = linear(R + '/pkg/robocop/librobocop.so')
lf, lsf, lb, pL = logspace()
DB = 1e3 * DBL_MIN
def scan(A, Llog, s, label, rev):
    out = np.full(n, np.nan); arg = np.zeros(n, int)
    for i in range(n):
        a = A[i, :ssb]; l = Llog[i, :ssb]
        ok = np.isfinite(l) & (a > DB / s[i if not rev else n - 1 - i])
        if ok.any():
            r = np.log10(a[ok]) - l[ok] / np.log(10)
            k = np.argmax(r); out[i] = r[k]; arg[i] = np.flatnonzero(ok)[k]
    return out, arg
rf, af = scan(fA, lf, sfA, 'f', False)
# backward tables are stored reversed (row r <-> position n-1-r); compare in storage order, report position
sbA = np.zeros(n)  # recompute backward scale per storage row: sum of emitting cells before normalize is not kept; use floor-free test
rb = np.full(n, np.nan); ab = np.zeros(n, int)
for r in range(n):
    a = bA[r, :ssb]; l = lb[r, :ssb]; ok = np.isfinite(l) & (a > 1e-200)
    if ok.any():
        x = np.log10(a[ok]) - l[ok] / np.log(10); k = np.argmax(x); rb[n - 1 - r] = x[k]; ab[n - 1 - r] = np.flatnonzero(ok)[k]
for name, rr, aa in [('forward', rf, af), ('backward(by position)', rb, ab)]:
    big = np.flatnonzero(rr > 1)
    print('%s: rows with some real cell inflated >10x: %d' % (name, big.size))
    if big.size:
        i0 = big.min() if name == 'forward' else big.max()
        print('  first (in pass direction) at position %d, state %s, log10 inflation %.2f' % (i0, state_name(int(aa[i0])), rr[i0]))
        rng = range(max(i0 - 4, 0), min(i0 + 6, n))
        for i in rng:
            print('   pos %d  max log10(A/L) %.2f  at %s  log10 S_f %.2f' % (i, rr[i], state_name(int(aa[i])), np.log10(sfA[i])))
    print('  max inflation anywhere %.2f at pos %d (%s)' % (np.nanmax(rr), int(np.nanargmax(rr)), state_name(int(aa[np.nanargmax(rr)]))))
