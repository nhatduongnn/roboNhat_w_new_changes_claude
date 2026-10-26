"""Brute-force check of librobocop_logfull.so (log path) and librobocop.so (fix B) on tiny HMMs
built in the RoboCOP shape, against exact path enumeration.

    cd analysis && python posterior_fix/logspace_review/bruteforce.py [--seeds N] > out.txt

Truth (textbook HMM with silent states). A path is an emitting sequence s_1..s_T plus, between
consecutive emitting states, a route i -> (silent chain) -> j or a direct edge i -> j.
  P(path, x) = init[s_1] e_1(s_1) * prod_t R(s_t -> s_{t+1}) e_{t+1}(s_{t+1}) * end[s_T]
  gamma_t(j)       = P(s_t = j | x)                        (emitting posterior; sums to 1 over j)
  silent post g    = P(route between t and t+1 visits sigma | x)
  loglik           = log sum_paths P(path, x)
Enumeration is done in log space (logsumexp over every emitting sequence; routes are summed per
pair by DFS over the silent DAG), so emissions of 1e-600 are representable.

Bindings copy pkgvar/seq_maskoff_tune75_logfull/robocop/robocop.py (log) and
pkgvar/seq_maskoff_tune75/robocop/robocop.py (linear fix B) argument lists exactly.
"""
import sys, argparse, itertools, ctypes, json
import numpy as np
from ctypes import c_int, c_double
from numpy.ctypeslib import ndpointer

FORK = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude"
LOGLIB = FORK + "/pkg/robocop/librobocop_logfull.so"
FIXB = FORK + "/pkg/robocop/librobocop.so"
ABI = 20261005
D = ndpointer(np.double, flags="C_CONTIGUOUS"); L = ndpointer(np.longlong, flags="C_CONTIGUOUS")
LDp = ndpointer(np.longdouble, flags="C_CONTIGUOUS")


def lse(a):
    a = np.asarray(a, float)
    m = np.max(a) if a.size else -np.inf
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.sum(np.exp(a - m)))


# ---------------------------------------------------------------- HMM generator (RoboCOP shape)
def make_hmm(rng, kind):
    """States: emitting [0, E), silent [E, N). RoboCOP shape: a central silent C = E (lowest silent
    index); C -> bg(0) and C -> begin-silent of each 'motif'; motif = begin-silent -> chain of
    emitting -> end-silent -> C' ... In RoboCOP every silent->silent edge goes to a higher index
    (checked on HMMconfig.pkl), so here: C(E) -> B_m (E+1+2m) -> ... and end-silent X_m (E+2+2m)
    -> a 'sink' silent K (N-1) -> emitting. Plus random extra emit<->emit edges."""
    if kind == "robocop":
        n_mot = 2
        mot_len = [2, 1]
        E = 1 + sum(mot_len)                  # bg + motif states
        # silent: C=E, B0=E+1, X0=E+2, B1=E+3, X1=E+4, K=E+5
        N = E + 2 + 2 * n_mot
        C, K = E, N - 1
        # every silent->silent edge goes low -> high (as in RoboCOP): C -> B_m, X_m -> K
        T = np.zeros((N, N))
        # bg: self loop or go into central C
        T[0, 0] = rng.uniform(0.3, 0.8); T[0, C] = 1 - T[0, 0]
        # C -> B0, C -> B1, C -> bg directly (silent -> emitting)
        w = rng.dirichlet([1, 1, 1]); T[C, E + 1] = w[0]; T[C, E + 3] = w[1]; T[C, 0] = w[2]
        s = 1
        for m in range(n_mot):
            B, X = E + 1 + 2 * m, E + 2 + 2 * m
            T[B, s] = 1.0                          # begin silent -> first motif state
            for q in range(mot_len[m] - 1):
                T[s + q, s + q + 1] = 1.0
            T[s + mot_len[m] - 1, X] = 1.0         # last motif state -> end silent
            T[X, K] = 1.0                          # end silent -> sink K (higher index)
            s += mot_len[m]
        # K -> bg, K -> motif-1 first state (silent -> emitting)
        T[K, 0] = 0.6; T[K, 1 + mot_len[0]] = 0.4
        ssb = E
    else:  # random DAG
        E = int(rng.integers(2, 5)); S = int(rng.integers(1, 4)); N = E + S; ssb = E
        T = np.zeros((N, N))
        for i in range(N):
            cand = [j for j in range(N) if (i < E) or (j < E) or (j > i)]   # silent->silent only upward
            k = int(rng.integers(1, min(len(cand), 4) + 1))
            js = rng.choice(cand, size=k, replace=False)
            w = rng.dirichlet(np.ones(k))
            T[i, js] = w
        ssb = E
    return T, ssb


def silent_routes(T, ssb):
    """R[i, j] = sum of probabilities of every route from emitting i to emitting j (direct or via a
    silent chain); Rs[sig][i, j] = the part of R[i, j] whose route visits silent sig. Linear (routes
    are products of transition weights only, never underflow here)."""
    N = T.shape[0]; E = ssb
    R = np.zeros((E, E)); Rs = np.zeros((N - E, E, E))

    def dfs(i0, node, prob, visited):
        for j in np.nonzero(T[node])[0]:
            p = prob * T[node, j]
            if j < E:
                R[i0, j] += p
                for v in visited:
                    Rs[v - E, i0, j] += p
            else:
                assert j not in visited, "silent cycle"
                dfs(i0, j, p, visited + [j])

    for i in range(E):
        dfs(i, i, 1.0, [])
    return R, Rs


def brute(T, ssb, init, endp, logE):
    """Exact enumeration. logE: (n_obs, E) log emissions of emitting states."""
    n = logE.shape[0]; E = ssb; N = T.shape[0]
    R, Rs = silent_routes(T, ssb)
    with np.errstate(divide='ignore'):
        lR = np.log(R); lRs = np.log(Rs); li = np.log(init[:E]); le = np.log(endp[:E])
    seqs = np.array(list(itertools.product(range(E), repeat=n)), dtype=int)   # (E^n, n)
    lp = li[seqs[:, 0]] + logE[0, seqs[:, 0]]
    for t in range(1, n):
        lp = lp + lR[seqs[:, t - 1], seqs[:, t]] + logE[t, seqs[:, t]]
    lp = lp + le[seqs[:, n - 1]]
    Z = lse(lp)
    gam = np.zeros((n, E))
    for t in range(n):
        for j in range(E):
            gam[t, j] = np.exp(lse(lp[seqs[:, t] == j]) - Z) if np.isfinite(Z) else np.nan
    # silent posterior for the gap (t, t+1), t = 0..n-2: fraction of route mass through sigma
    sil = np.zeros((max(n - 1, 0), N - E))
    for t in range(n - 1):
        a, b = seqs[:, t], seqs[:, t + 1]
        for v in range(N - E):
            with np.errstate(divide='ignore', invalid='ignore'):
                frac = lRs[v][a, b] - lR[a, b]
            frac[~np.isfinite(lR[a, b])] = -np.inf
            sil[t, v] = np.exp(lse(lp + frac) - Z)
    return Z, gam, sil


# ---------------------------------------------------------------- C bindings (as robocop.py)
def graph(lib, T, ssb):
    ns = T.shape[0]
    par = np.zeros((ns, ns), np.longlong); chi = np.zeros((ns, ns), np.longlong)
    npar = np.zeros(ns, np.longlong); nchi = np.zeros(ns, np.longlong)
    lib.find_parents_and_children.argtypes = [L, L, L, L, c_int, c_int, D]
    lib.find_parents_and_children(par, chi, npar, nchi, ns, ssb, T)
    return par, chi, npar, nchi


def run_log(T, ssb, init, endp, logE_full):
    lib = ctypes.CDLL(LOGLIB)
    ns = T.shape[0]; n = logE_full.shape[0]
    par, chi, npar, nchi = graph(lib, T, ssb)
    f = np.full((n, ns), -np.inf); b = np.full((n, ns), -np.inf); p = np.zeros((n, ns))
    fs = np.zeros(n); bs = np.zeros(n)
    lib.fward_log.argtypes = [D, D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.fward_log.restype = c_int
    lib.bward_log.argtypes = [D, D, D, c_int, c_int, c_int, L, L, D, D, c_int]; lib.bward_log.restype = c_int
    lib.posterior_decoding_log.argtypes = [D, D, c_int, c_int, c_int, D, c_int]; lib.posterior_decoding_log.restype = c_int
    with np.errstate(divide='ignore'):
        rf = lib.fward_log(init, T, logE_full, endp, ns, ssb, n, par, npar, f, fs, ABI)
        rb = lib.bward_log(T, logE_full, endp, ns, ssb, n, chi, nchi, b, bs, ABI)
    rp = lib.posterior_decoding_log(f, b, ns, n, ssb, p, ABI)
    ll = 0.0
    for i in range(n - 1, 0, -1):
        ll += fs[i]
    ll += fs[0]
    return dict(rc=(rf, rb, rp), post=p, ll=ll, f=f, b=b)


def run_fixb(T, ssb, init, endp, logE_full):
    lib = ctypes.CDLL(FIXB)
    ns = T.shape[0]; n = logE_full.shape[0]
    par, chi, npar, nchi = graph(lib, T, ssb)
    Elin = np.ascontiguousarray(np.exp(logE_full)[None, :, :])        # n_vars = 1
    f = np.zeros((n, ns)); b = np.zeros((n, ns)); p = np.zeros((n, ns))
    fs = np.zeros(n); bs = np.zeros(n); sr = np.zeros(n, dtype=np.longdouble)
    ms = np.zeros(1, np.longlong); ml = np.zeros(1, np.longlong)
    lib.fward.argtypes = [D, D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.bward.argtypes = [D, D, D, c_int, c_int, c_int, c_int, L, L, L, L, c_int, c_int, c_int, c_int, D, D]
    lib.calc_sr.argtypes = [D, D, c_int, LDp]; lib.calc_sr.restype = c_double
    lib.posterior_decoding_v2.argtypes = [D, D, D, LDp, c_int, c_int, c_int, D]
    lib.fward(init, T, Elin, endp, ns, ssb, n, 1, par, npar, ms, ml, 0, 0, 0, 0, f, fs)
    lib.bward(T, Elin, endp, ns, ssb, n, 1, chi, nchi, ms, ml, 0, 0, 0, 0, b, bs)
    ll = lib.calc_sr(fs, bs, n, sr)
    lib.posterior_decoding_v2(f, b, bs, sr, ns, n, ssb, p)
    return dict(post=p, ll=ll)


def case(rng, kind, n, emis, endmode, maskfrac):
    T, ssb = make_hmm(rng, kind)
    N = T.shape[0]; E = ssb
    init = np.zeros(N); init[:E] = rng.dirichlet(np.ones(E))
    endp = np.ones(N)
    if endmode == "random":
        endp[:E] = rng.uniform(0.05, 1.0, E)
    logE = np.log(rng.uniform(0.05, 1.0, (n, E)))
    if emis == "tiny":            # every emitting state ~1e-300 .. 1e-310 (linear: near-denormal)
        logE = logE + np.log(1e-300)
    elif emis == "ultra":         # ~1e-600: unrepresentable in linear double
        logE = logE - 600 * np.log(10)
    elif emis == "mixed":         # one state per position ~1e-600, others O(1)
        for t in range(n):
            logE[t, rng.integers(E)] -= 600 * np.log(10)
    if maskfrac > 0:              # -inf masks, never a whole row
        M = rng.random((n, E)) < maskfrac
        for t in range(n):
            if M[t].all():
                M[t, rng.integers(E)] = False
        logE[M] = -np.inf
    full = np.zeros((n, N)); full[:, :E] = logE        # silent columns 0.0 = log 1
    Z, gam, sil = brute(T, ssb, init, endp, logE)
    return T, ssb, init, endp, full, Z, gam, sil


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--json'); a = ap.parse_args()
    rows = []
    for kind in ("robocop", "random"):
        for n in (1, 2, 3, 6):
            for emis in ("normal", "tiny", "ultra", "mixed"):
                for endmode in ("one", "random"):
                    for maskfrac in (0.0, 0.25):
                        worst = dict(lp=0.0, lpi=0.0, llz=0.0, lb=0.0, lbi=0.0, lbll=0.0, ncase=0, nz=0,
                                     sil_t=0.0, sil_tm1=0.0, rc_bad=0, fb_nan=0, logsil_nonzero=0)
                        for s in range(a.seeds):
                            rng = np.random.default_rng(10000 * n + 100 * s + (kind == "random") * 7)
                            T, ssb, init, endp, full, Z, gam, sil = case(rng, kind, n, emis, endmode, maskfrac)
                            if not np.isfinite(Z):
                                worst['nz'] += 1; continue   # no valid path: skip (C returns -2)
                            worst['ncase'] += 1
                            r = run_log(T, ssb, init, endp, full)
                            if any(x != ABI for x in r['rc']):
                                worst['rc_bad'] += 1
                            P = r['post']
                            worst['logsil_nonzero'] += int(np.any(P[:, ssb:] != 0))
                            Pe = P[:, :ssb]
                            if endmode == "one" or n == 1:
                                worst['lp'] = max(worst['lp'], float(np.abs(Pe - gam).max()))
                            else:   # last position double-counts end_probs (C5); score rows < n-1 here
                                worst['lp'] = max(worst['lp'], float(np.abs(Pe[:-1] - gam[:-1]).max()))
                                # last row: compare against the end^2 hypothesis
                                g2 = gam[-1] * endp[:ssb]; g2 = g2 / g2.sum()
                                worst['lpi'] = max(worst['lpi'], float(np.abs(Pe[-1] - g2).max()))
                                worst['lb'] = max(worst['lb'], float(np.abs(Pe[-1] - gam[-1]).max()))
                            worst['llz'] = max(worst['llz'], abs(r['ll'] - Z))
                            if n >= 2 and emis != "ultra":
                                fb = run_fixb(T, ssb, init, endp, full)
                                Pb = fb['post'][:, :ssb]
                                if np.isnan(Pb).any():
                                    worst['fb_nan'] += 1
                                else:
                                    rr = slice(None) if endmode == "one" else slice(0, n - 1)
                                    worst['lbi'] = max(worst['lbi'], float(np.abs(Pb[rr] - gam[rr]).max()))
                                    worst['lbll'] = max(worst['lbll'], abs(fb['ll'] - Z))
                                # C4: fix B's silent column at row t vs textbook gap (t,t+1) and (t-1,t)
                                if emis == "normal" and n >= 3:
                                    Sb = fb['post'][:, ssb:]
                                    worst['sil_t'] = max(worst['sil_t'], float(np.abs(Sb[:n - 1] - sil).max()))
                                    worst['sil_tm1'] = max(worst['sil_tm1'], float(np.abs(Sb[1:] - sil).max()))
                        row = dict(kind=kind, n=n, emis=emis, end=endmode, mask=maskfrac, **worst)
                        rows.append(row)
                        print("%-7s n=%d %-6s end=%-6s mask=%.2f cases=%2d | LOG emit max|err| %.1e  LOG lastrow vs "
                              "end^2-hyp %.1e (vs truth %.1e)  LOG |ll-Z| %.1e  rc_bad %d  logsil!=0 %d | FIXB emit %.1e "
                              "|ll-Z| %.1e nan %d | FIXB silent vs gap(t,t+1) %.2e vs gap(t-1,t) %.2e"
                              % (kind, n, emis, endmode, maskfrac, worst['ncase'], worst['lp'], worst['lpi'],
                                 worst['lb'], worst['llz'], worst['rc_bad'], worst['logsil_nonzero'],
                                 worst['lbi'], worst['lbll'], worst['fb_nan'], worst['sil_t'], worst['sil_tm1']),
                              flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == "__main__":
    main()
