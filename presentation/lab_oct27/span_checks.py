"""Mean / max of a few tracks in named spans, BEFORE (original .so, factor tables) vs AFTER (log space), bo09.
Same Run Browser reader as build_payloads.py.   cd analysis && python ../presentation/lab_oct27/span_checks.py"""
import sys, os
A = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis"))
sys.path.insert(0, os.path.join(A, "viewer_site")); os.chdir(A)
from extract_run import load_parts, stitch
def rd(d, c, s, e):
    segs, cols, _ = load_parts(d, c); v, cov = stitch(segs, len(cols), s, e); assert cov.all(); return v, cols
for (lab, c, s, e, od, nd) in [
        ("loc4 after broken rows", "chrIV", 1201698, 1201829, "robocop_chrIV_tw_bo09_48", "robocop_tw3_bo09_48_logfull"),
        ("loc4 broken rows", "chrIV", 1200001, 1201697, "robocop_chrIV_tw_bo09_48", "robocop_tw3_bo09_48_logfull"),
        ("loc3 broken rows", "chrXIV", 724001, 726207, "robocop_chrXIV_chrII_tw_bo09_48", "robocop_tw3_bo09_48_logfull"),
        ("loc2 seg314 broken", "chrXIV", 440001, 442954, "robocop_chrXIV_chrII_tw_bo09_48", "robocop_tw3_bo09_48_logfull"),
        ("loc2 444001-444389", "chrXIV", 444001, 444389, "robocop_chrXIV_chrII_tw_bo09_48", "robocop_tw3_bo09_48_logfull")]:
    o, co = rd(od, c, s, e); n, cn = rd(nd, c, s, e)
    for f in ("background", "nucleosome", "Abf1_murphy", "Fhl1_zhu"):
        a = o[:, co.index(f)]; b = n[:, cn.index(f)]
        print("%-22s %-12s before mean %.3f max %.3f | after mean %.3f max %.3f | n|d|>=.1 %d" % (lab, f, a.mean(), a.max(), b.mean(), b.max(), (abs(a - b) >= .1).sum()))
