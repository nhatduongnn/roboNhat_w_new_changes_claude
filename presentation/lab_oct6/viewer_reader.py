"""Old Run Browser reader for the locus slides: what the viewer drew BEFORE the posterior fix.

Reads <decode dir>/factor_tables/part_*.npz (write_factor_table.py without --normalise:
per-segment posteriors clipped to [0,1], stored as uint16 / scale) and averages overlapping
segments equally per position -- the same load_parts/stitch logic as
analysis/viewer_site/extract_run.py:45-84 (re-implemented here, nothing shared is imported or edited).

    python viewer_reader.py rpc19_up.json rpc19_up_viewer.json bo09_48=robocop_chrXIV_chrII_tw_bo09_48 ...

Takes a payload built by build_locus_slide.py (corrected, region_optable reader), replaces each
run's ABF1 (payload "factor") and nucleosome tracks with the viewer-reader values, and prints
every position where the two readers differ by >= 0.05 (per run, per plotted track and over
any column).
"""
import glob, json, os, sys
import numpy as np
A = "/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis"


def load_parts(d, chrom):
    segs, cols = {}, None
    for p in sorted(glob.glob(os.path.join(A, d, "factor_tables", "part_*.npz"))):
        z = np.load(p, allow_pickle=False)
        c = [str(x) for x in z["cols"]]
        if cols is None: cols = c
        elif c != cols: raise RuntimeError("%s: column set differs between part files" % d)
        scale = json.loads(str(z["meta"])).get("scale", 65535)
        ro, q = z["row_off"], None
        for k in range(len(z["seg_idx"])):
            if str(z["seg_chr"][k]) != chrom: continue
            if q is None: q = z["q"]
            segs[int(z["seg_idx"][k])] = (int(z["seg_start"][k]), int(z["seg_end"][k]),
                                          q[ro[k]:ro[k + 1]].astype(np.float32) / scale, int(z["n_invalid"][k]))
    return [segs[k] for k in sorted(segs)], cols


def stitch(segs, ncol, ws, we):
    n = we - ws + 1
    acc = np.zeros((n, ncol)); cnt = np.zeros(n, dtype=np.int32)
    for s, e, v, _ in segs:
        if e < ws or s > we: continue
        a, b = max(s, ws), min(e, we, s + v.shape[0] - 1)
        acc[a - ws:b - ws + 1] += v[a - s:b - s + 1]; cnt[a - ws:b - ws + 1] += 1
    cov = cnt > 0
    acc[cov] /= cnt[cov, None]
    return acc, cov


def spans(pos):
    out = []
    for p in pos:
        if out and p == out[-1][1] + 1: out[-1][1] = p
        else: out.append([p, p])
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    dirs = dict(a.split("=", 1) for a in sys.argv[3:])
    D = json.load(open(src)); s, e, chrom = D["s"], D["e"], D["chrom"]
    sys.path.insert(0, A); os.chdir(A)
    import score_robocop as S
    for label in D["runOrder"]:
        segs, cols = load_parts(dirs[label], chrom)
        V, cov = stitch(segs, len(cols), s, e)
        assert cov.all(), "%s: viewer tables do not cover %s:%d-%d" % (label, chrom, s, e)
        op, ccov, _ = S.region_optable(S.load_decode(dirs[label]), chrom, s, e)
        common = [c for c in cols if c in op.columns]
        C = op[common].values; Vc = V[:, [cols.index(c) for c in common]]
        segs_here = [(a, b, ni) for a, b, _, ni in segs if not (b < s or a > e)]
        print("== %s %s  segments %s" % (label, dirs[label], segs_here))
        for name in (D["factor"], "nucleosome"):
            v, c = V[:, cols.index(name)], op[name].values
            d = np.abs(v - c); pos = [s + i for i in np.flatnonzero(d >= 0.05)]
            print("   %-12s positions |viewer-corrected|>=0.05: %d  spans %s  max diff %.4f" % (
                name, len(pos), [(a, b, round(float(v[a - s:b - s + 1].max()), 4), round(float(c[a - s:b - s + 1].max()), 6)) for a, b in spans(pos)], d.max()))
        anyd = (np.abs(Vc - C) >= 0.05).any(1)
        print("   any column: %d positions differ >= 0.05" % int(anyd.sum()))
        for p in (412655, 412625, 413425, 413467):
            if s <= p <= e:
                i = p - s
                print("   at %d: %s viewer %.4f corrected %.6f" % (p, D["factor"], V[i, cols.index(D["factor"])], op[D["factor"]].values[i]))
        D["runs"][label]["tf"] = [int(round(x * 1000)) for x in V[:, cols.index(D["factor"])]]
        D["runs"][label]["nuc"] = [int(round(x * 1000)) for x in V[:, cols.index("nucleosome")]]
    D["reader"] = "viewer (factor_tables, clipped per segment, overlapping segments averaged)"
    os.chdir(os.path.dirname(os.path.abspath(__file__)) if False else os.getcwd())
    json.dump(D, open(dst, "w"), separators=(",", ":"))
    print("wrote", dst)


if __name__ == "__main__":
    main()
