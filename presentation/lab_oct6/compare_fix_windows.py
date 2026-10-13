"""Fix B vs log space on the two 500 bp RPC19 windows (bv75 round 49 model, fib+seq):
max |diff| over every state column of region_optable (normalised per position, never clipped),
and ABF1 at the MacIsaac c1 site 412,615-412,627 (max within +-20 bp) and at bo09's
two downstream calls 413,418-413,431 and 413,461-413,474."""
import os, sys, numpy as np
A = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis"
sys.path.insert(0, A); os.chdir(A)
import score_robocop as S
B, L = "robocop_w30k_bv75_49_Bprod_both", "robocop_w30k_bv75_49_logfull_both"
dB, dL = S.load_decode(B), S.load_decode(L)
for s, e in [(412410, 412909), (413212, 413711)]:
    ob, cb, _ = S.region_optable(dB, "chrXIV", s, e)
    ol, cl, _ = S.region_optable(dL, "chrXIV", s, e)
    assert list(ob.columns) == list(ol.columns) and cb.all() and cl.all()
    d = np.abs(ob.values - ol.values)
    a, b = ob["Abf1_murphy"].values, ol["Abf1_murphy"].values
    msg = "chrXIV:%d-%d  max|diff| all %d columns %.3g | ABF1 max B %.6f log %.6f | nuc max |diff| %.3g" % (
        s, e, d.shape[1], d.max(), a.max(), b.max(), np.abs(ob["nucleosome"].values - ol["nucleosome"].values).max())
    if s <= 412615 <= e:
        lo, hi = 412615 - 20 - s, 412627 + 20 - s + 1
        msg += " | ABF1 at site B %.6f log %.6f" % (a[lo:hi].max(), b[lo:hi].max())
    for x, y in [(413418, 413431), (413461, 413474)]:
        if s <= x <= e:
            msg += " | ABF1 %d-%d B %.6f log %.6f" % (x, y, a[x - s:y - s + 1].max(), b[x - s:y - s + 1].max())
    print(msg)
