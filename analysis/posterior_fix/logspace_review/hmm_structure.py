"""Structural checks on the three trainDirs' HMMconfig.pkl: silent->silent edge order,
initial/end probs, n_states. Read-only."""
import pickle, sys, numpy as np
for td in ["robocop_train_tw_bo09_48", "robocop_train_tw_bu01_11", "robocop_train_tw_bv75_49"]:
    d = pickle.load(open(td + "/HMMconfig.pkl", "rb"))
    if td.endswith("bo09_48"): print("keys:", sorted(d.keys()))
    T = np.asarray(d["transition_matrix"], float); ssb = int(d["silent_states_begin"]); n = int(d["n_states"])
    ip = np.asarray(d["initial_probs"], float); ep = np.asarray(d["end_probs"], float)
    r, c = np.nonzero(T)
    ss = (r >= ssb) & (c >= ssb)
    print("\n==", td, "n_states", n, "ssb", ssb, "n_silent", n - ssb, "edges", len(r))
    print(" silent->silent edges", ss.sum(), " backward(to lower/equal idx):", (ss & (c <= r)).sum())
    print(" emit->emit edges", ((r < ssb) & (c < ssb)).sum(), " emit->silent", ((r < ssb) & (c >= ssb)).sum(), " silent->emit", ((r >= ssb) & (c < ssb)).sum())
    print(" self-loops", (r == c).sum(), " min nonzero T", T[T != 0].min(), " row sums range", T.sum(1).min(), T.sum(1).max())
    print(" initial_probs: sum", ip.sum(), " mass on silent", ip[ssb:].sum(), " nnz", (ip > 0).sum(), " shape", ip.shape)
    print(" end_probs: unique", np.unique(ep)[:10], " #!=1 among emitting", (ep[:ssb] != 1).sum(), " shape", ep.shape)
    print(" nuc_start", d.get("nuc_start"), "nuc_len", d.get("nuc_len"), "timepoints", d.get("timepoints"), "n_vars", d.get("n_vars"))
    # silent states reachable at position 0 path start? silent states with emitting parents
    print(" n_states^2 =", n * n, "< 2^31:", n * n < 2**31, "; n_obs*n_states at 5000:", 5000 * n)
