"""(c2) Does the logfull tree in ROBOCOP_LOG_EMISSION=floored hand the recursion log(E) of what the
linear tree hands it?  python compare_caps.py <cap_linear.npz> <cap_logfloored.npz> [out.json]
Per layer: log(E_lin) vs E_log (cells where E_lin == 0 must be -inf in E_log, and vice versa).
Collapsed: sum_l E_log vs log(prod_l E_lin). Also checks every non-emission field is identical."""
import sys, json, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
R = dict(lin=sys.argv[1], log=sys.argv[2], lin_tree=str(a['tree']), log_tree=str(b['tree']),
         log_env=str(b['log_emission_env']), lin_lib=str(a['robocopC']), log_lib=str(b['robocopC']))
for k in ['n_states', 'silent_states_begin', 'n_vars', 'n_tfs', 'nuc_present', 'nuc_start', 'nuc_len', 'timepoints', 'n_obs',
          'initial_probs', 'end_probs', 'transition_matrix', 'tf_starts', 'tf_lens']:
    R['same_' + k] = bool(np.array_equal(a[k], b[k]))
El, Eg = a['emission'], b['emission']
ssb = int(a['silent_states_begin'])
R['shape'] = list(El.shape)
R['log_nan'] = int(np.isnan(Eg).sum()); R['log_posinf'] = int(np.isposinf(Eg).sum()); R['log_max'] = float(np.max(Eg))
layers = []
with np.errstate(divide='ignore'):
    for l in range(El.shape[0]):
        L = np.log(El[l]); G = Eg[l]
        zl, zg = np.isneginf(L), np.isneginf(G)
        fin = ~zl & ~zg
        layers.append(dict(layer=l, lin_zero=int(zl.sum()), log_neginf=int(zg.sum()), zero_mismatch=int((zl != zg).sum()),
                           max_abs_diff_finite=float(np.abs(L[fin] - G[fin]).max()) if fin.any() else 0.0,
                           lin_floor_cells=int((El[l] == 1e-30).sum())))
    R['layers'] = layers
    Llin = np.zeros(El.shape[1:])
    for l in range(El.shape[0]):
        Llin += np.log(El[l])                 # = log(prod), exact sum of per-layer logs
    plin = np.prod(El, axis=0)                # what the linear recursion multiplies (may underflow)
    Lprod = np.log(plin)
Lg = Eg.sum(0)
zl, zg = np.isneginf(Llin), np.isneginf(Lg); fin = ~zl & ~zg
R['collapsed'] = dict(neginf_lin=int(zl.sum()), neginf_log=int(zg.sum()), mismatch=int((zl != zg).sum()),
                      max_abs_diff=float(np.abs(Llin[fin] - Lg[fin]).max()),
                      max_rel_diff=float((np.abs(Llin[fin] - Lg[fin]) / np.maximum(1, np.abs(Lg[fin]))).max()),
                      linear_product_underflow_cells=int((np.isneginf(Lprod) & ~zl).sum()),
                      silent_cols_log_all_zero=bool(np.all(Lg[:, ssb:] == 0.0)),
                      emitting_neginf_cols=int(np.all(zg[:, :ssb], axis=0).sum()))
print(json.dumps(R, indent=1))
if len(sys.argv) > 3:
    json.dump(R, open(sys.argv[3], 'w'), indent=1)
