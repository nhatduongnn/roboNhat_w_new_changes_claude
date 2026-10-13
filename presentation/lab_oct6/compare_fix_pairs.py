"""Compare fix B (Bprod) vs log-space (logfull) decodes over their whole windows, read through
score_robocop.region_optable (normalised per position over emitting states, never clipped).
Prints per-factor max |diff|, the count of positions where any factor differs by >= 0.05,
and the 2,001 bp windows with the most such positions. Saves the tables to npz for reuse."""
import os, sys, numpy as np
A = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis"
sys.path.insert(0, A); os.chdir(A)
import score_robocop as S
OUT = os.path.dirname(os.path.abspath(__file__))
PAIRS = [("chrXIV", 396001, 433000, "robocop_w30k_bv75_49_Bprod_both", "robocop_w30k_bv75_49_logfull_both"),
         ("chrXII", 450001, 471000, "robocop_rdna_chrXII_Bprod_both", "robocop_rdna_chrXII_logfull_both")]
for chrom, s, e, b, l in PAIRS:
    ops = {}
    for d in (b, l):
        op, cov, _ = S.region_optable(S.load_decode(d), chrom, s, e)
        ops[d] = op
        print(d, op.shape, "covered", int(cov.sum()), "max", float(op.values.max()), "min", float(op.values.min()))
    B, L = ops[b], ops[l]
    assert list(B.columns) == list(L.columns)
    D = (B.values - L.values)
    ad = np.abs(D)
    print("== %s:%d-%d  overall max |diff| %.3g" % (chrom, s, e, ad.max()))
    pf = ad.max(0); order = np.argsort(-pf)
    for j in order[:12]:
        print("   %-22s max|diff| %.4f  positions>=0.05 %d" % (B.columns[j], pf[j], int((ad[:, j] >= 0.05).sum())))
    anyd = (ad >= 0.05).any(1)
    pos = np.arange(s, e + 1)
    print("   positions with any factor >= 0.05: %d" % anyd.sum(),
          ("span %d-%d" % (pos[anyd].min(), pos[anyd].max())) if anyd.any() else "")
    W = 2001
    c = np.convolve(anyd.astype(int), np.ones(W, int), "valid")
    best = np.argsort(-c)[:1]
    for k in best:
        print("   best window %d-%d: %d positions differ" % (pos[k], pos[k] + W - 1, c[k]))
    np.savez_compressed(os.path.join(OUT, "pair_%s.npz" % chrom), cols=np.array(B.columns), B=B.values.astype(np.float32),
                        L=L.values.astype(np.float32), s=s, e=e)
