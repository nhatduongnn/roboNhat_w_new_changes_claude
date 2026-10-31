#!/usr/bin/env python
"""Strand-symmetric Brogaard nucleosome sequence model -> new trainDir (2026-10-28).

  python nucbrogsym/build_trainDir.py [--offset 0] [--out robocop_train_fiberonly_nucbrogsym]

1. 147-mers at the 2,000 Brogaard top dyads (all of them, no random draw), centre offset
   --offset (0 = 1-based base BED start; chosen by centring_seq.py).
2. Two symmetric fits:
     pooled : counts(Watson) + counts(reverse complement), normalised once (the ML fit of the
              symmetric training set)                                              <- USED
     average: fit Watson, fit reverse complement, average the normalised parameters
   plus Watson-only, the shipped model, and Chereji all-9,271 + rc for reference.
   Compared by: max |param difference|, training log-likelihood, asymmetry on ERV46 147-mers.
3. New trainDir = copy of --src (sidecars copied byte for byte) with the new
   nuc_dinucleotide_model.txt / nuc_emission.npy, and HMMconfig.pkl rebuilt by upstream's own
   _build_transition_matrix -> set_transition -> set_initial_probs -> set_end_probs (the order
   build_transition_matrix uses, robocop.py) with every prior read from the source HMMconfig,
   and pwm_emission's nucleosome rows replaced. The same rebuild with the SOURCE nucleosome
   files must reproduce the source HMMconfig bit for bit (asserted first).
Writes OUT/build_trainDir.json.
"""
import argparse
import copy
import filecmp
import json
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nucmodel as NM  # noqa: E402

os.chdir(NM.ANALYSIS)
sys.path.insert(0, "../pkg/")
from robocop import robocop as R  # noqa: E402

OUTDIR = "/usr/project/xtmp/nd141/permol_proto_logfull/nucbrog"
CSHARED = "../pkg/robocop/librobocop.so"
SIDECARS = ("config.ini", "pwm.p", "negParamsMNase.pkl")


def avg_models(m1, m2):
    return {k: 0.5 * (m1[k] + m2[k]) for k in m1}


def rebuild(cfg0, dinuc_file, nuc_em):
    cfg = copy.deepcopy(cfg0)
    cfg["robocopC"] = CSHARED
    R._build_transition_matrix(cfg, dinuc_file)
    R.set_transition(cfg, [], cfg["background_prob"], cfg["nucleosome_prob"])
    R.set_initial_probs(cfg)
    R.set_end_probs(cfg)
    ns, ssb = int(cfg["nuc_start"]), int(cfg["silent_states_begin"])
    pe = np.array(cfg["pwm_emission"], copy=True)
    pe[ns:ssb] = nuc_em
    cfg["pwm_emission"] = np.ascontiguousarray(pe)
    cfg["robocopC"] = cfg0["robocopC"]
    return cfg


def diff_cfg(a, b, ns, ssb):
    out = {}
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            va, vb = np.asarray(va), np.asarray(vb)
            if va.shape != vb.shape:
                out[k] = "shape %s vs %s" % (va.shape, vb.shape)
                continue
            if va.dtype.kind in "fc":
                d = va != vb
                n = int(d.sum())
                if n:
                    idx = np.argwhere(d)
                    inside = np.all((idx >= ns) & (idx < ssb), axis=1) if va.ndim == 2 and k == "transition_matrix" \
                        else (idx[:, 0] >= ns) & (idx[:, 0] < ssb)
                    out[k] = dict(n_diff=n, all_inside_nuc_block=bool(inside.all()),
                                  max_abs=float(np.abs(va - vb).max()))
            elif not np.array_equal(va, vb):
                out[k] = "differs"
        elif va != vb:
            out[k] = "%r vs %r" % (va, vb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--src", default="robocop_train_fiberonly")
    ap.add_argument("--out", default="robocop_train_fiberonly_nucbrogsym")
    a = ap.parse_args()
    res = dict(offset=a.offset, src=a.src, out=a.out, training_bed=NM.BROGAARD)

    g = NM.load_genome()
    rows = NM.read_bed(NM.BROGAARD)
    W, dropped = NM.extract(rows, g, a.offset)
    C = [NM.revcomp(s) for s in W]
    res.update(n_rows=len(rows), n_seqs=len(W), n_dropped=dropped)
    print("Brogaard rows %d, 147-mers %d, dropped %d" % (len(rows), len(W), dropped))

    cW, cC = NM.fit_counts(W), NM.fit_counts(C)
    for nm, cc in (("watson", cW), ("revcomp", cC)):
        res["min_dinuc_row_count_" + nm] = float(cc["dinuc"].sum(2).min())
    mW, mC = NM.normalise(cW), NM.normalise(cC)
    NM.check_against_upstream(W, mW)
    pooled = NM.normalise({k: cW[k] + cC[k] for k in cW})
    NM.check_against_upstream(W + C, pooled)
    average = avg_models(mW, mC)

    # --- pooled vs average ----------------------------------------------------------
    cmp = {}
    for k in pooled:
        d = np.abs(pooled[k] - average[k])
        cmp[k] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()))
    # marginal sums: pooled vs average differ only where n_W(a, pos) != n_C(a, pos)
    nW, nC = cW["dinuc"].sum(2), cC["dinuc"].sum(2)
    cmp["max_rel_count_imbalance_nW_vs_nC"] = float((np.abs(nW - nC) / (nW + nC)).max())
    train = W + C
    llp = np.array([NM.logp(s, pooled) for s in train])
    lla = np.array([NM.logp(s, average) for s in train])
    cmp["train_mean_loglik_nats_pooled"] = float(llp.mean())
    cmp["train_mean_loglik_nats_average"] = float(lla.mean())
    cmp["train_loglik_gain_pooled_minus_average_nats_per_seq"] = float((llp - lla).mean())
    kmers = NM.window_kmers(g)
    lp = np.array([NM.logp(s, pooled) for s in kmers])
    la = np.array([NM.logp(s, average) for s in kmers])
    cmp["erv46_mean_abs_log10P_pooled_minus_average"] = float(np.abs(lp - la).mean() / np.log(10))
    cmp["erv46_max_abs_log10P_pooled_minus_average"] = float(np.abs(lp - la).max() / np.log(10))
    res["pooled_vs_average"] = cmp
    print("pooled vs average:", json.dumps(cmp, indent=1))

    # --- asymmetry ---------------------------------------------------------------------
    src_cfg = pickle.load(open(os.path.join(a.src, "HMMconfig.pkl"), "rb"), encoding="latin1")
    shipped = NM.model_from_trainfiles(os.path.join(a.src, "nuc_dinucleotide_model.txt"),
                                       os.path.join(a.src, "nuc_emission.npy"))
    ch_rows = NM.read_bed(NM.CHEREJI)
    chW, _ = NM.extract(ch_rows, g, 0)
    cc1, cc2 = NM.fit_counts(chW), NM.fit_counts([NM.revcomp(s) for s in chW])
    chereji_sym = NM.normalise({k: cc1[k] + cc2[k] for k in cc1})
    asym = {}
    asym["shipped_from_HMMconfig"], d_hmm = NM.asymmetry(kmers, cfg=src_cfg)
    asym["shipped_from_trainfiles"], d_tf = NM.asymmetry(kmers, model=shipped)
    res["shipped_hmm_vs_trainfiles_max_abs_log10"] = float(np.abs(d_hmm - d_tf).max())
    for nm, m in (("chereji_all_plus_rc", chereji_sym), ("brogaard_watson_only", mW),
                  ("brogaard_average", average), ("brogaard_pooled", pooled)):
        asym[nm], _ = NM.asymmetry(kmers, model=m)
    res["asymmetry_erv46"] = asym
    res["n_erv46_kmers"] = len(kmers)
    print("asymmetry:", json.dumps(asym, indent=1))

    # --- trainDir ----------------------------------------------------------------------
    out = a.out
    if os.path.exists(out):
        sys.exit("refusing to overwrite existing trainDir %s" % out)
    os.makedirs(out)
    for fn in SIDECARS:
        shutil.copy2(os.path.join(a.src, fn), os.path.join(out, fn))
        assert filecmp.cmp(os.path.join(a.src, fn), os.path.join(out, fn), shallow=False)
    # config.ini: only nucleosomeFile and trainDir change (they describe this trainDir)
    lines = open(os.path.join(out, "config.ini")).read().splitlines(True)
    for i, l in enumerate(lines):
        if l.startswith("nucleosomeFile"):
            lines[i] = "nucleosomeFile = %s\n" % NM.BROGAARD
        elif l.startswith("trainDir"):
            lines[i] = "trainDir = ./%s/\n" % out.strip("./")
    open(os.path.join(out, "config.ini"), "w").write("".join(lines))
    dinuc_file = os.path.join(out, "nuc_dinucleotide_model.txt")
    NM.CD.printDiNuc(pooled["strt"], pooled["dinuc"], dinuc_file)
    nuc_em = NM.nuc_emission_from(pooled)
    np.save(os.path.join(out, "nuc_emission"), nuc_em)
    back = NM.model_from_trainfiles(dinuc_file, os.path.join(out, "nuc_emission.npy"))
    res["written_model_roundtrip_max_abs"] = float(max(np.abs(back[k] - pooled[k]).max() for k in back))

    ns, ssb = int(src_cfg["nuc_start"]), int(src_cfg["silent_states_begin"])
    # validation: the rebuild with the SOURCE nucleosome files reproduces the source exactly
    src_em = np.load(os.path.join(a.src, "nuc_emission.npy"))
    same = rebuild(src_cfg, os.path.join(a.src, "nuc_dinucleotide_model.txt"), src_em)
    res["rebuild_with_source_files_diff"] = diff_cfg(src_cfg, same, ns, ssb)
    print("rebuild with source files, diff:", res["rebuild_with_source_files_diff"])
    assert res["rebuild_with_source_files_diff"] == {}, "rebuild does not reproduce the source"

    new = rebuild(src_cfg, dinuc_file, nuc_em)
    res["new_vs_source_diff"] = diff_cfg(src_cfg, new, ns, ssb)
    print("new vs source, diff:", json.dumps(res["new_vs_source_diff"], indent=1))
    assert set(res["new_vs_source_diff"]) <= {"transition_matrix", "pwm_emission", "initial_probs"}
    for k in ("tf_prob", "background_prob", "nucleosome_prob"):
        assert np.array_equal(np.asarray(src_cfg[k]), np.asarray(new[k])), k
    res["nucleosome_entry_prob"] = dict(src=float(src_cfg["transition_matrix"][ssb, ns]),
                                        new=float(new["transition_matrix"][ssb, ns]))
    assert res["nucleosome_entry_prob"]["src"] == res["nucleosome_entry_prob"]["new"]
    T = new["transition_matrix"]
    rs = T[:ssb].sum(1)
    res["new_emitting_row_sum_range"] = [float(rs.min()), float(rs.max())]
    rs0 = src_cfg["transition_matrix"][:ssb].sum(1)
    res["src_emitting_row_sum_range"] = [float(rs0.min()), float(rs0.max())]
    asym["new_from_HMMconfig"], d_new = NM.asymmetry(kmers, cfg=new)
    _, d_pool = NM.asymmetry(kmers, model=pooled)
    res["new_hmm_vs_pooled_model_max_abs_log10"] = float(np.abs(d_new - d_pool).max())
    with open(os.path.join(out, "HMMconfig.pkl"), "wb") as f:
        pickle.dump(new, f)
    with open(os.path.join(out, "NUCBROGSYM_README.txt"), "w") as f:
        f.write("Copy of %s with ONLY the nucleosome sequence model replaced (2026-10-28).\n"
                "Model: Brogaard 2012 top-2000 dyads (%s), centre = 1-based BED start + %d,\n"
                "Watson 147-mers + reverse complements, counts pooled, upstream layout (531 states).\n"
                "Built by analysis/nucbrogsym/build_trainDir.py; details in\n"
                "%s/build_trainDir.json\n" % (a.src, NM.BROGAARD, a.offset, OUTDIR))
    json.dump(res, open(os.path.join(OUTDIR, "build_trainDir.json"), "w"), indent=1)
    print("asymmetry new HMMconfig %.4f" % asym["new_from_HMMconfig"])
    print("wrote", out)


if __name__ == "__main__":
    main()
