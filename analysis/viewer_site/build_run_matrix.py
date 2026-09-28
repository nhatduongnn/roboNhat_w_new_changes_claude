#!/usr/bin/env python
"""Run registry for the viewer site: every decode dir -> runs.json / runs.tsv / decode_dirs.json.

    python viewer_site/build_run_matrix.py [--out /usr/project/xtmp/nd141/viewer_site]
        [--backfill-run-info] [--tasks FILE] [--dirs-only]

WHAT A RUN IS
  * every non-campaign decode dir (analysis/robocop_*, not robocop_train_*) is its own run;
  * each tuner-v2 campaign (conc_tuning/<r>/state.json with "groups") gives an UNTUNED run
    (round 00) and a FINAL run (the round named in STOPPED, or, while still running, the latest
    round whose tuning decode is complete, flagged in_progress). The tuning decode
    robocop_chrXIV_chrII_tw_<r>_NN supplies chrXIV+chrII and the holdout decode
    robocop_chrIV_tw_<r>_NN (if one exists for that round) supplies chrIV;
  * each tuner-v1 campaign (state.json without "groups": u001, m001, fm00x, sm00x) likewise, from
    robocop_genome_ct_<r>_NN.

PROVENANCE (driver, pkg tree, real trainDir) per decode dir, first hit wins:
  1. RUN_INFO.json written at decode time (sbatch_genome_decode.sh hook);
  2. campaign state.json: built[NN].traindir, driver in force at round NN (driver_history);
  3. the decode logs (logs/*.out: DRIVER=/TRAINDIR=/OUTDIR= or `pkg variant:` + `coordFile: |
     trainDir: | outDir:`), matched on the dir's name;
  4. run_overrides.tsv `original_outdir` (renamed dirs) -> the logs under the ORIGINAL name.
  config.ini is NOT used: it is copied from whichever trainDir was cloned (367 dirs claim
  robocop_train_fiberonly). A provenance is cross-checked: the h5 posterior's state count must
  equal the trainDir HMMconfig n_states (field nstates_ok).

DERIVED ATTRIBUTES come only from (tree, trainDir): layers + mask from the tree's
robocopExtras.py (active lines; KEEP_*/FITTED_* set literals via ast), phi (FIBER_TEMPER_PHI),
emission-parameter pkls (the tree's robocop.py `open('inputs/*.pkl')`), EM iterations
(likelihood.txt lines - 1, following conc_patch/w_patch "src"), lambda (conc_patch.json),
ABF1 state width (HMMconfig tf_lens), pads (pads table for the trainDir's pwmFile, from
run_overrides.tsv), decoy (Zz_decoy_abf1 in tfs), nucleosome-model md5. Anything not derivable
is "?" -- never a guess.

COMPLETENESS is segment-level: every coords.tsv row must have segment_<i>/posterior in its
info_<i mod N>_<N>.h5 (an h5 being written by a running decode is unreadable -> incomplete).
Cached by (mtime, size) in /usr/project/xtmp/nd141/scratch_viewer/h5cache.json.
"""
import argparse
import ast
import collections
import configparser
import glob
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.dirname(HERE)                       # analysis/
sys.path.insert(0, A)
from write_run_info import driver_tree  # noqa: E402

SCRATCH = "/usr/project/xtmp/nd141/scratch_viewer"
OUT_DEFAULT = "/usr/project/xtmp/nd141/viewer_site"
H5CACHE = os.path.join(SCRATCH, "h5cache.json")
TDCACHE = os.path.join(SCRATCH, "traindir_cache.json")
WINDOWS = os.path.join(HERE, "windows.tsv")
Q = "?"
CAMPAIGN_RE = re.compile(r"^robocop_(chrXIV_chrII|chrIV|genome)_(tw|ct)_([a-z0-9]+)_(\d\d)$")


# ------------------------------------------------------------------ small helpers
def jload(p, default):
    try:
        return json.load(open(p))
    except Exception:
        return default


def jsave(p, obj):
    tmp = p + ".tmp%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, p)


def md5_file(p):
    try:
        return hashlib.md5(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def norm_td(td):
    if td is None:
        return None
    td = td.strip()
    if td.startswith(A + "/"):
        td = td[len(A) + 1:]
    return os.path.normpath(td.lstrip("./") if td.startswith("./") else td)


def norm_tree(t):
    if not t:
        return None
    t = t.strip()
    if t.startswith(A + "/"):
        t = t[len(A) + 1:]
    return os.path.normpath(t)


# ------------------------------------------------------------------ overrides
def load_overrides():
    """run_overrides.tsv: key <tab> field <tab> value <tab> note. key = decode dir, run id,
    'traindir:<name>' or 'pwmfile:<path>'."""
    ov = collections.defaultdict(dict)
    p = os.path.join(HERE, "run_overrides.tsv")
    if not os.path.exists(p):
        return ov
    for line in open(p):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        parts += [""] * (4 - len(parts))
        k, f, v, note = parts[:4]
        ov[k.strip()][f.strip()] = (v.strip(), note.strip())
    return ov


# ------------------------------------------------------------------ logs
def build_log_index():
    """outdir basename -> Counter{(driver, tree, traindir, coords)} from logs/*.out headers."""
    cache = os.path.join(SCRATCH, "logindex.txt")
    logs = os.path.join(A, "logs")
    fresh = os.path.exists(cache) and os.path.getmtime(cache) > time.time() - 6 * 3600
    if not fresh:
        cmd = ("cd %s && ls | grep '\\.out$' | xargs -n 2000 grep -H -m 4 -E "
               "'^DRIVER=|^pkg variant:|^coordFile:|^=== [A-Za-z0-9_.]+ ===$' > %s.tmp; mv %s.tmp %s"
               % (logs, cache, cache, cache))
        subprocess.run(cmd, shell=True, check=False)
    recs = {}
    for line in open(cache):
        f, _, rest = line.rstrip("\n").partition(":")
        d = recs.setdefault(f, {})
        if rest.startswith("DRIVER="):
            m = re.match(r"DRIVER=(\S+)\s+TRAINDIR=(\S+)\s+OUTDIR=(\S+)(?:\s+COORDS=(\S+))?", rest)
            if m:
                d.update(driver=m.group(1), traindir=m.group(2), outdir=m.group(3), coords=m.group(4))
        elif rest.startswith("pkg variant:"):
            d["tree"] = rest.split(":", 1)[1].strip()
        elif rest.startswith("coordFile:"):
            m = re.match(r"coordFile:\s*(\S+)\s*\|\s*trainDir:\s*(\S+)\s*\|\s*outDir:\s*(\S+)", rest)
            if m:
                d.update(coords2=m.group(1), traindir2=m.group(2), outdir2=m.group(3))
        elif rest.startswith("==="):
            d.setdefault("banner", rest.strip("= ").strip())
    idx = collections.defaultdict(collections.Counter)
    for f, d in recs.items():
        od = d.get("outdir") or d.get("outdir2")
        if not od:
            continue
        od = os.path.basename(os.path.normpath(od))
        drv = d.get("driver")
        if not drv and d.get("banner"):
            drv = d["banner"] if d["banner"].endswith(".py") else d["banner"] + ".py"
        key = (drv, norm_tree(d.get("tree")), norm_td(d.get("traindir") or d.get("traindir2")),
               d.get("coords") or d.get("coords2"))
        idx[od][key] += 1
    return idx


# ------------------------------------------------------------------ completeness
def check_dir(d, cache):
    """-> dict(complete, n_expected, n_present, chroms, infofiles, unreadable)."""
    od = os.path.join(A, d)
    res = dict(complete=False, n_expected=0, n_present=0, chroms=[], n_info=0, unreadable=[])
    try:
        coords = pd.read_csv(os.path.join(od, "coords.tsv"), sep="\t")
    except Exception:
        res["why"] = "no coords.tsv"
        return res
    res["n_expected"] = len(coords)
    res["chroms"] = list(dict.fromkeys(coords["chr"].tolist()))
    infos = sorted(glob.glob(os.path.join(od, "tmpDir", "info_*_*.h5")))
    res["n_info"] = len(infos)
    if not infos:
        res["why"] = "no tmpDir/info_*.h5"
        return res
    present = set()
    for p in infos:
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = os.path.relpath(p, A)
        c = cache.get(key)
        if c and c[0] == st.st_mtime and c[1] == st.st_size:
            segs = c[2]
        else:
            try:
                with h5py.File(p, "r") as f:
                    segs = sorted(int(k.split("_")[1]) for k in f.keys()
                                  if k.startswith("segment_") and "posterior" in f[k])
                    nst = None
                    if segs:
                        nst = int(f["segment_%d/posterior" % segs[0]].attrs["shape"][1])
                cache[key] = [st.st_mtime, st.st_size, segs, nst]
            except Exception as e:
                res["unreadable"].append(os.path.basename(p))
                continue
        present.update(segs)
    res["n_present"] = len(present & set(range(len(coords))))
    res["complete"] = res["n_present"] == len(coords) and not res["unreadable"]
    # state count from the first readable file (for the provenance cross-check)
    for p in infos:
        c = cache.get(os.path.relpath(p, A))
        if c and len(c) > 3 and c[3]:
            res["n_states_h5"] = c[3]
            break
    if not res["complete"]:
        res["why"] = "%d/%d segments%s" % (res["n_present"], len(coords),
                                            (", unreadable: %s" % ",".join(res["unreadable"][:3])) if res["unreadable"] else "")
    return res


# ------------------------------------------------------------------ campaigns
def campaigns():
    out = {}
    for p in sorted(glob.glob(os.path.join(A, "conc_tuning", "*", "state.json"))):
        s = jload(p, None)
        if not s or not s.get("run"):
            continue
        r = s["run"]
        stop = None
        sp = os.path.join(os.path.dirname(p), "STOPPED")
        if os.path.exists(sp):
            stop = open(sp).read().strip()
        out[r] = dict(state=s, stopped=stop, version="v2" if "groups" in s else "v1")
    return out


def driver_at_round(s, t):
    drv = s.get("driver")
    hist = sorted(s.get("driver_history") or [], key=lambda h: h.get("from_round", 0))
    if hist:
        drv = hist[0].get("old", drv)
        for h in hist:
            if t >= h.get("from_round", 10 ** 9):
                drv = h.get("new", drv)
    return drv


# ------------------------------------------------------------------ provenance
def provenance(d, camps, logidx, ov):
    """-> dict(driver, tree, traindir, coords, campaign, round, role, source, original_outdir)."""
    od = os.path.join(A, d)
    ri = jload(os.path.join(od, "RUN_INFO.json"), None)
    if ri and ri.get("source") == "decode-time":
        return dict(driver=ri.get("driver"), tree=norm_tree(ri.get("pkg_tree")), traindir=norm_td(ri.get("traindir")),
                    coords=ri.get("coords"), campaign=ri.get("campaign"), round=ri.get("round"),
                    role=ri.get("role"), source="RUN_INFO decode-time",
                    original_outdir=ri.get("original_outdir", d))
    m = CAMPAIGN_RE.match(d)
    if m and m.group(3) in camps and not (m.group(1) == "genome" and m.group(2) == "tw"):
        kind, fam, r, nn = m.groups()
        t = int(nn)
        c = camps[r]
        s = c["state"]
        role = "holdout" if kind == "chrIV" else "tune"
        if c["version"] == "v2":
            td = (s.get("built", {}).get(str(t)) or {}).get("traindir") or "robocop_train_tw_%s_%02d" % (r, t)
            drv = driver_at_round(s, t)
            coords = s.get("coords_holdout") if role == "holdout" else s.get("coords_tune")
        else:
            td = "robocop_train_ct_%s_%02d" % (r, t)
            drv = s.get("driver")
            coords = None
        tree, how = driver_tree(drv) if drv else (None, "no driver in state")
        src = "state.json"
        # cross-check / fill from the logs
        hits = logidx.get(d)
        if hits:
            (ldrv, ltree, ltd, lco), _ = hits.most_common(1)[0]
            if not drv and ldrv:
                drv, src = ldrv, "state.json+log"
            if ltree and (tree is None):
                tree = ltree
            if ltd and ltd != norm_td(td):
                src += " (log trainDir %s differs!)" % ltd
            coords = coords or lco
        if tree is None and drv:
            tree, how = driver_tree(drv)
        return dict(driver=drv, tree=norm_tree(tree), traindir=norm_td(td), coords=coords, campaign=r, round=t,
                    role=role, source=src, original_outdir=d)
    orig = ov.get(d, {}).get("original_outdir", (None,))[0]
    for name, src in ((d, "log"), (orig, "hand+log")):
        if not name:
            continue
        hits = logidx.get(name)
        if hits:
            keys = list(hits)
            (drv, tree, td, co), _ = hits.most_common(1)[0]
            if tree is None and drv:
                tree, _ = driver_tree(drv)
            note = "" if len(keys) == 1 else " (%d distinct log headers; most common used)" % len(keys)
            p = dict(driver=drv, tree=norm_tree(tree), traindir=td, coords=co, campaign=None, round=None, role=None,
                     source="backfill:" + src + note, original_outdir=name)
            # hand corrections for renamed pkgvar trees / trainDirs (e.g. widefp -> wide12)
            for f in ("tree", "traindir", "driver"):
                if f in ov.get(d, {}):
                    p[f] = ov[d][f][0]
                    p["source"] += " +hand %s" % f
            return p
    p = dict(driver=None, tree=None, traindir=None, coords=None, campaign=None, round=None, role=None,
             source=Q, original_outdir=orig or d)
    for f in ("tree", "traindir", "driver"):
        if f in ov.get(d, {}):
            p[f] = ov[d][f][0]
            p["source"] = "backfill:hand"
    return p


# ------------------------------------------------------------------ derived attributes
def active_src(path):
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return None
    return "\n".join(l.split("#", 1)[0] if not l.lstrip().startswith("#") else "" for l in lines)


def tree_attrs(tree):
    """layers, mask and phi from the tree's robocopExtras.py; emission pkls from robocop.py."""
    out = dict(seq_layer=Q, fiber_layer=Q, mask=Q, keep=None, phi=Q, emission_pkls=Q)
    if not tree or tree == "../pkg":
        out["tree_note"] = "../pkg is edited in place (hand toggles); its state at decode time is unknown"
        return out
    root = os.path.join(A, tree)
    ex = os.path.join(root, "robocop", "utils", "robocopExtras.py")
    src = active_src(ex)
    if src is None:
        out["tree_note"] = "tree not on disk"
        return out
    out["seq_layer"] = "off" if re.search(r"data_emission_matrix\[0\]\[:\]\s*=\s*1\b", src) else "on"
    fib_off = re.search(r"data_emission_matrix\[5\]\[:\]\s*=\s*1\b", src) and \
        re.search(r"data_emission_matrix\[6\]\[:\]\s*=\s*1\b", src)
    out["fiber_layer"] = "off" if fib_off else "on"
    keep = None
    try:
        for node in ast.walk(ast.parse(open(ex).read())):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and re.match(r"^(KEEP_\w+|FITTED_\w+)$", node.targets[0].id) and isinstance(node.value, ast.Set):
                vals = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
                if keep is not None:
                    out["tree_note"] = "more than one KEEP/FITTED set; first used"
                    continue
                keep = (node.targets[0].id, sorted(vals))
    except SyntaxError:
        pass
    if keep:
        out["mask"] = keep[0].replace("KEEP_", "").lower()
        out["keep"] = keep[1]
    elif re.search(r"data_emission_matrix\[[056]\]\[:,\s*29:dshared\['nuc_start'\]\]\s*=\s*0", src):
        out["mask"] = "abf1 (29:nuc_start)"
        out["keep"] = ["Abf1_murphy"]
    else:
        out["mask"] = "none"
    m = re.search(r"^\s*FIBER_TEMPER_PHI\s*=\s*([0-9.eE+-]+)", src, re.M)
    out["phi"] = float(m.group(1)) if m else 1.0
    try:
        rp = open(os.path.join(root, "robocop", "robocop.py")).read()
        base = open(os.path.join(A, "..", "pkg", "robocop", "robocop.py")).read()
        pk = re.findall(r"open\('inputs/([^']+\.pkl)'", rp)
        pb = re.findall(r"open\('inputs/([^']+\.pkl)'", base)
        diff = [x for x in pk if x not in pb]
        out["emission_pkls"] = "default" if not diff else ",".join(diff)
    except OSError:
        pass
    return out


def traindir_attrs(td, cache, ov):
    if not td:
        return {}
    p = os.path.join(A, td, "HMMconfig.pkl")
    try:
        st = os.stat(p)
    except OSError:
        return dict(traindir_note="no HMMconfig.pkl")
    key = td
    c = cache.get(key)
    if c and c.get("_mtime") == st.st_mtime and c.get("_v") == 4:
        return c
    dsh = pickle.load(open(p, "rb"))
    tfs = list(dsh["tfs"])
    out = dict(_mtime=st.st_mtime, _v=4, n_states=int(dsh["n_states"]), n_tfs=len(tfs), tfs=tfs)
    if "Abf1_murphy" in tfs:
        out["abf1_width"] = int(dsh["tf_lens"][tfs.index("Abf1_murphy")])
    out["decoy"] = "Zz_decoy_abf1" in tfs
    out["nuc_md5"] = (md5_file(os.path.join(A, td, "nuc_emission.npy")) or "")[:10]
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(A, td, "config.ini"))
    out["pwm_file"] = cfg.get("main", "pwmFile", fallback=None)
    # EM iterations: likelihood.txt lines - 1, following the patch chain
    em, cur, seen, last = None, td, set(), td
    lam = None
    while cur and cur not in seen:
        seen.add(cur)
        last = cur
        lk = os.path.join(A, cur, "likelihood.txt")
        if os.path.exists(lk):
            em = max(0, sum(1 for l in open(lk) if l.strip()) - 1)
            break
        nxt = None
        for pj in ("conc_patch.json", "w_patch.json"):
            j = jload(os.path.join(A, cur, pj), None)
            if j:
                if cur == td and pj == "conc_patch.json":
                    if "lams" in j:
                        lam = j["lams"]
                    elif "tf" in j:
                        lam = {j["tf"]: j.get("lam")}
                if cur == td and pj == "w_patch.json":
                    lam = "w_patch (tuner v2 weights)"
                nxt = norm_td(j.get("src"))
                break
        cur = nxt
    cur = cur or last
    if em is None:
        o = ov.get("traindir:" + cur, {}).get("em_iters")
        if o:
            em = o[0] + " (hand)"
    out["em_iters"] = em if em is not None else Q
    out["em_root"] = cur
    out["lam"] = lam
    cache[key] = out
    return out


def pads_for(pwm_file, ov, width=None):
    o = ov.get("pwmfile:" + (pwm_file or ""), {}).get("pads_table")
    if not o:
        # the shipped Abf1_murphy motif is 14 columns (robocop_train_fiberonly/pwm.p), so a 14-wide
        # ABF1 state block carries no pad
        return "0/0" if width == 14 else None
    tbl = os.path.join(A, o[0])
    try:
        for line in open(tbl):
            if line.startswith("Abf1_murphy"):
                f = line.split()
                return "%s/%s" % (f[1], f[2])
    except OSError:
        return None
    return None


def lam_str(lam):
    if lam is None:
        return "1 (untouched)"
    if isinstance(lam, str):
        return lam
    if len(lam) <= 3:
        return ", ".join("%s=%.3g" % (k, float(v)) for k, v in lam.items())
    return "%d factors patched" % len(lam)


# ------------------------------------------------------------------ labels
def label_tables():
    lab = {}
    pats = ["viewer_runs_*.tsv", "layer_runs_*.tsv", "chrXIV_runs.tsv", "conc_*_runs.tsv"]
    for pat in pats:
        for p in glob.glob(os.path.join(A, pat)):
            for line in open(p):
                if line.startswith("#") or not line.strip():
                    continue
                f = re.split(r"\t+|\s{2,}", line.strip())
                if len(f) >= 2 and f[1].startswith("robocop_"):
                    lab.setdefault(f[1], f[0])
    return lab


def family_of(d, prov):
    n = d[len("robocop_"):]
    if prov.get("campaign"):
        return "tuner"
    rules = [(r"^genome_tw_", "genome decode of tuned trainDir"), (r"twS05", "layer transfer (sw01 r05 weights)"),
             (r"conctune", "tuner-v1 seed"), (r"sweep_unk", "unknown-state sweep"), (r"conclo_|_conc\d", "ABF1 lambda sweep"),
             (r"JASPAR", "JASPAR PWM"), (r"wide", "widened footprint"), (r"em10", "EM"), (r"cap[AB]", "fitted conc (cap)"),
             (r"lam0p01", "ABF1 lambda"), (r"macisaac", "MacIsaac mask"), (r"12tfs|bgtss|lowabf1", "emission-param variant"),
             (r"^(chrI|chrXIV)_(fib|seq|fib_seq)$", "layers"), (r"revfix", "layers (revfix)"),
             (r"^(chrI_maskon|chrI_maskoff|chrI_seq_maskon|chrI_seq_maskoff|erv46_|seqlayer_|all)", "early (pre-revfix)")]
    for pat, fam in rules:
        if re.search(pat, n):
            return fam
    return "other"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--backfill-run-info", action="store_true",
                    help="write RUN_INFO.json (source backfill:*) into in-scope decode dirs lacking one")
    ap.add_argument("--tasks", default=None,
                    help="write the factor-table backfill task list (dir, idx, total, driver, traindir, tree)")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    ov = load_overrides()
    camps = campaigns()
    logidx = build_log_index()
    h5c = jload(H5CACHE, {})
    tdc = jload(TDCACHE, {})
    labels = label_tables()

    dirs = sorted(os.path.basename(p.rstrip("/")) for p in glob.glob(os.path.join(A, "robocop_*/"))
                  if not os.path.basename(p.rstrip("/")).startswith("robocop_train_"))
    info = {}
    for d in dirs:
        c = check_dir(d, h5c)
        p = provenance(d, camps, logidx, ov)
        info[d] = dict(check=c, prov=p)
    jsave(H5CACHE, h5c)

    # ---- group into runs
    runs = []
    used = set()
    camp_dirs = collections.defaultdict(dict)       # run -> round -> {"tune": d, "holdout": d}
    for d in dirs:
        m = CAMPAIGN_RE.match(d)
        if m and m.group(3) in camps and not (m.group(1) == "genome" and m.group(2) == "tw"):
            kind, fam, r, nn = m.groups()
            camp_dirs[r].setdefault(int(nn), {})["holdout" if kind == "chrIV" else "tune"] = d
    for r, rounds in sorted(camp_dirs.items()):
        c = camps[r]
        stop = c["stopped"]
        final, in_prog = None, False
        mm = re.search(r"stopped after iter (\d+)", stop or "")
        if mm and int(mm.group(1)) in rounds:
            final = int(mm.group(1))
        if final is None:
            done = [t for t, x in rounds.items() if x.get("tune") and info[x["tune"]]["check"]["complete"]]
            if done:
                final = max(done)
            in_prog = not stop
        picks = [(0, "untuned")]
        if final is not None and final != 0:
            picks.append((final, "final"))
        for t, role in picks:
            if t not in rounds:
                continue
            x = rounds[t]
            chroms = {}
            for part in ("tune", "holdout"):
                dd = x.get(part)
                if not dd:
                    continue
                used.add(dd)
                for ch in info[dd]["check"]["chroms"]:
                    chroms[ch] = dd
            prim = x.get("tune") or x.get("holdout")
            runs.append(dict(run_id="%s_%s" % (r, "r00" if role == "untuned" else "final"),
                             label="%s %s" % (r, "r00" if role == "untuned" else "final r%02d" % t),
                             family="tuner-%s" % c["version"], campaign=r, role=role, round=t,
                             stop_reason=(stop or ("in progress" if in_prog else Q)) if role == "final" else "",
                             converged=("converged" in (stop or "")) if role == "final" else "",
                             in_progress=bool(in_prog and role == "final"),
                             dirs=chroms, primary=prim, tie=c["state"].get("tied")))
    for d in dirs:
        if d in used or CAMPAIGN_RE.match(d) and not d.startswith("robocop_genome_tw_"):
            continue
        chroms = {ch: d for ch in info[d]["check"]["chroms"]}
        runs.append(dict(run_id=d[len("robocop_"):], label=labels.get(d, d[len("robocop_"):]),
                         family=family_of(d, info[d]["prov"]), campaign="", role="", round="",
                         stop_reason="", converged="", in_progress=False, dirs=chroms, primary=d, tie=None))

    # ---- attributes
    for r in runs:
        p = info[r["primary"]]["prov"]
        chk = info[r["primary"]]["check"]
        tr = tree_attrs(p.get("tree"))
        td = traindir_attrs(p.get("traindir"), tdc, ov) if p.get("traindir") else {}
        keep = tr.get("keep")
        tfs = td.get("tfs") or []
        motifs = [x for x in tfs if x != "unknown"]
        if tr["mask"] == Q:
            live, unk = Q, Q
        elif keep is None:
            live, unk = (len(motifs) if tfs else Q), ("yes" if "unknown" in tfs else (Q if not tfs else "no"))
        else:
            live = len([k for k in keep if k != "unknown"])
            unk = "yes" if "unknown" in keep else "no"
        tie = r.pop("tie")
        tie_s = ""
        if tie:
            tie_s = ", ".join("%s=%s x %s" % (k, v.get("ratio", Q), v.get("group", Q)) for k, v in tie.items())
        nst = chk.get("n_states_h5")
        nok = (nst == td.get("n_states")) if (nst and td.get("n_states")) else Q
        all_ok = [info[dd]["check"]["complete"] for dd in set(r["dirs"].values())]
        ov_r = ov.get(r["run_id"], {})
        ov_d = ov.get(r["primary"], {})
        r.update(
            seq_layer=tr["seq_layer"], fiber_layer=tr["fiber_layer"], mask=tr["mask"], live_motifs=live,
            unknown_live=unk, phi=tr["phi"], em_iters=td.get("em_iters", Q),
            lam=(lam_str(td.get("lam")) if td and not r["campaign"] else ("tuned" if r["campaign"] and r["round"] else
                                                                        (lam_str(td.get("lam")) if td else Q))),
            abf1_width=td.get("abf1_width", Q), abf1_pads=pads_for(td.get("pwm_file"), ov, td.get("abf1_width")) or Q,
            decoy=("yes" if td.get("decoy") else "no") if td else Q, tie=tie_s,
            emission_pkls=tr["emission_pkls"], nuc_md5=td.get("nuc_md5", Q), pwm_file=td.get("pwm_file") or Q,
            traindir=p.get("traindir") or Q, pkgvar=p.get("tree") or Q, driver=p.get("driver") or Q,
            provenance=p.get("source"), nstates_ok=nok,
            chroms=sorted(r["dirs"], key=lambda c: (len(c), c)),
            complete=all(all_ok), incomplete_dirs=[dd for dd in set(r["dirs"].values()) if not info[dd]["check"]["complete"]],
            legacy=(ov_r.get("legacy") or ov_d.get("legacy") or ("",))[0] or ("yes" if p.get("tree") == "../pkg" else ""),
            notes="; ".join(x for x in [tr.get("tree_note"), td.get("traindir_note")] if x),
        )
        for f in ("label", "family"):
            o = ov_r.get(f) or ov_d.get(f)
            if o:
                r[f] = o[0]
        r["attr_source"] = "derived from (pkgvar tree, trainDir)"
    jsave(TDCACHE, tdc)

    # ---- outputs
    cols = ["run_id", "label", "family", "campaign", "role", "round", "stop_reason", "converged", "in_progress",
            "seq_layer", "fiber_layer", "mask", "live_motifs", "unknown_live", "phi", "em_iters", "lam",
            "abf1_width", "abf1_pads", "decoy", "tie", "emission_pkls", "nuc_md5", "pwm_file", "traindir", "pkgvar",
            "driver", "provenance", "nstates_ok", "chroms", "complete", "legacy", "notes"]
    meta = dict(built=time.strftime("%Y-%m-%d %H:%M:%S"), n_dirs=len(dirs), n_runs=len(runs),
                n_campaign_dirs_not_shown=sum(1 for d in dirs if CAMPAIGN_RE.match(d) and d not in used
                                              and not d.startswith("robocop_genome_tw_")))
    jsave(os.path.join(a.out, "runs.json"), dict(meta=meta, columns=cols, runs=runs))
    with open(os.path.join(a.out, "runs.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in runs:
            fh.write("\t".join(",".join(r[c]) if isinstance(r[c], list) else str(r[c]) for c in cols) + "\n")
    jsave(os.path.join(SCRATCH, "decode_dirs.json"), info)

    # ---- summary
    fam = collections.Counter(r["family"] for r in runs)
    nq = collections.Counter(c for r in runs for c in cols if r.get(c) == Q)
    print("dirs %d; runs %d (%s); campaign rounds not shown %d; %.0fs"
          % (len(dirs), len(runs), ", ".join("%s %d" % kv for kv in fam.most_common()),
             meta["n_campaign_dirs_not_shown"], time.time() - t0))
    print("runs with '?' per field: %s" % (dict(nq) or "none"))
    print("runs incomplete: %d; nstates mismatch: %d"
          % (sum(1 for r in runs if not r["complete"]), sum(1 for r in runs if r["nstates_ok"] is False)))

    in_scope = sorted({dd for r in runs for dd in r["dirs"].values()})
    if a.backfill_run_info:
        nw = 0
        for d in in_scope:
            p = info[d]["prov"]
            if os.path.exists(os.path.join(A, d, "RUN_INFO.json")) or not (p.get("driver") and p.get("traindir")):
                continue
            src = p["source"] if p["source"].startswith("backfill:") else "backfill:" + p["source"].split(" ")[0].replace(".json", "")
            args = ["--outdir", d, "--driver", p["driver"], "--traindir", p["traindir"], "--source", src,
                    "--original-outdir", p.get("original_outdir") or d]
            if p.get("coords"):
                args += ["--coords", p["coords"]]
            if p.get("campaign"):
                args += ["--campaign", p["campaign"], "--round", str(p["round"]), "--role", p["role"]]
            extra = {}
            if p.get("tree") and driver_tree(p["driver"])[0] != p["tree"]:
                extra = dict(pkg_tree=p["tree"], pkg_tree_how="from %s (driver default differs)" % src)
            if extra:
                args += ["--extra", json.dumps(extra)]
            r = subprocess.run([sys.executable, os.path.join(A, "write_run_info.py")] + args,
                               capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                                    if not k.startswith(("TW_", "SLURM"))})
            nw += r.returncode == 0
            if r.returncode:
                print("RUN_INFO failed for %s: %s" % (d, r.stderr.strip()[-200:]))
        print("RUN_INFO backfill: wrote %d" % nw)
    if a.tasks:
        wins = pd.read_csv(WINDOWS, sep="\t")
        rows = []
        for d in in_scope:
            p = info[d]["prov"]
            if not info[d]["check"]["complete"] or not (p.get("tree") and p.get("traindir")):
                continue
            coords = pd.read_csv(os.path.join(A, d, "coords.tsv"), sep="\t")
            want = set()
            for _, w in wins.iterrows():
                cc = coords[(coords["chr"] == w["chrom"]) & (coords["start"] <= w["end"]) & (coords["end"] >= w["start"])]
                want.update(int(i) for i in cc.index)
            if not want:
                continue
            infos = glob.glob(os.path.join(A, d, "tmpDir", "info_*_*.h5"))
            for f in sorted(infos):
                i, n = map(int, re.search(r"info_(\d+)_(\d+)\.h5$", f).groups())
                segs = set(h5c.get(os.path.relpath(f, A), [0, 0, []])[2])
                need = sorted(segs & want)
                if not need:
                    continue
                part = os.path.join(A, d, "factor_tables", "part_%d_%d.npz" % (i, n))
                if os.path.exists(part):
                    try:
                        have = set(int(x) for x in np.load(part)["seg_idx"])
                        if not (set(need) - have):
                            continue
                    except Exception:
                        pass
                rows.append((d, i, n, p["driver"] or "-", p["traindir"], p["tree"], ",".join(map(str, need))))
        with open(a.tasks, "w") as fh:
            for row in rows:
                fh.write("\t".join(map(str, row)) + "\n")
        print("backfill tasks: %d (dir x info file) over %d dirs -> %s"
              % (len(rows), len({r[0] for r in rows}), a.tasks))


if __name__ == "__main__":
    main()
