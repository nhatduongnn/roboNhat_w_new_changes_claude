#!/usr/bin/env python
"""Single-fiber RoboCOP: decode every molecule over one window, one molecule at a time.

Same 3,485-state HMM, same transition matrix, same PWM sequence layer and same 531-state
nucleosome dinucleotide model as an aggregate decode. The ONLY change is the Fiber-seq
emission: instead of a binomial over the pooled (k, n) at each position, each molecule
contributes its own 0/1 m6A call and the emission is a Bernoulli lookup

    calls[i] ==  1 -> ps[j]        calls[i] == 0 -> 1 - ps[j]        no call -> 1.0

(`robocop.update_data_emission_matrix_using_bernoulli_fiber_seq` in the frozen tree
`pkgvar/permol_seq_maskoff`). The pileup is never read: calls come straight from the BAM's
MM/ML tags via `read_calls.py`, which reproduces modkit's (k, n) exactly on every position
the emission consumes (see `validate_reader.py`).

Usage (from analysis/):

    python singlefiber/run_permol_window.py --arm A --eff 0 \\
        --out /usr/project/xtmp/nd141/permol_proto/armA_effoff

Outputs, under --out:
    config.ini, coords.tsv, RUN_INFO.json      so score_robocop.py / write_factor_table.py
    tmpDir/info_0_1.h5                         molecule-AVERAGE posterior, as segment_0
    permol.npz                                 per-molecule collapsed factor columns
    molecules.tsv                              per-molecule spans, call counts, eff scale
    run_stats.json                             timing, peak RSS, arm parameters
"""
import argparse
import datetime
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
os.chdir(ANALYSIS)                      # the package reads 'inputs/...' relatively
sys.path.insert(0, 'pkgvar/permol_seq_maskoff/')   # literal, so write_run_info.driver_tree resolves it
sys.path.insert(0, HERE)

import h5py                                                       # noqa: E402
import robocop                                                    # noqa: E402
from robocop.utils.robocopExtras import updatePerMoleculeEMMat     # noqa: E402
from robocop.utils.getNucleotides import getNucleotideSequence     # noqa: E402

import collapse as C                                              # noqa: E402
import permol_params as PP                                        # noqa: E402
import read_calls as RC                                           # noqa: E402

BAM = ("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/"
       "combine_sequencing_runs/merged_Mar20_barcode01_Jun25_barcode21-24_"
       "May07_barcode03-04.bam")
FASTA = os.path.join(ANALYSIS, "inputs", "SacCer3.fa")
KEEP_COLS = ["background", "nucleosome", "nuc_center", "nuc_padding",
             "Abf1_murphy", "Reb1_badis", "unknown"]

# ---------------------------------------------------------------------------
# Keep the full per-molecule posterior instead of the 1e-4-truncated one.
# robocop.save_sparse_posterior (robocop.py:31-39) zeroes every entry below 1e-4 before
# writing. That is fine for one aggregate decode, but applying it to each of 439
# molecules before averaging would throw away up to 3,485 x 1e-4 of mass per position
# per molecule. These two shims hand the dense p_table straight back instead of writing
# it, and the molecule AVERAGE is written with the real (truncated) function at the end,
# so the output file still looks exactly like an ordinary decode.
# ---------------------------------------------------------------------------
_GRAB = {}


def _grab_posterior(f, k, v):
    _GRAB["p"] = np.asarray(v)
    return None


def install_posterior_grab():
    # posterior_forward_backward_loop resolves save_sparse_posterior in the globals of the
    # MODULE robocop.robocop, not of the package `robocop`. Patch that module's globals
    # through a function object we hold, NOT through sys.modules['robocop.robocop'] --
    # score_robocop.use_pkg() deletes and re-imports every robocop module at import time,
    # so the sys.modules entry can be a different object from the one we will call.
    gl = robocop.posterior_forward_backward.__globals__
    gl["save_sparse_posterior"] = _grab_posterior
    gl["update_sparse_posterior"] = _grab_posterior
    robocop.save_sparse_posterior = _grab_posterior
    robocop.update_sparse_posterior = _grab_posterior


def save_sparse_dense(f, k, v):
    """save_sparse (getReads.py:8-15): plain CSR, no thresholding."""
    v = np.array(v)
    vs = sparse.csr_matrix(v)
    g = f.create_group(k)
    g.create_dataset("data", data=vs.data)
    g.create_dataset("indices", data=vs.indices)
    g.create_dataset("indptr", data=vs.indptr)
    g.attrs["shape"] = vs.shape


def save_posterior_truncated(f, k, v):
    """The real save_sparse_posterior, inlined so the shim above cannot affect it."""
    v = np.array(v)
    v[v < 1e-4] = 0
    vs = sparse.csr_matrix(v)
    g = f.create_group(k)
    g.create_dataset("data", data=vs.data)
    g.create_dataset("indices", data=vs.indices)
    g.create_dataset("indptr", data=vs.indptr)
    g.attrs["shape"] = vs.shape


def decode_one(dshared, chrom, seg_start1, seg_end1, codes_seg,
               ps_w, ps_c, calls_w, calls_c, M, keep_idx):
    """One molecule. Returns (posterior (n_obs, n_states), optable columns, loglik)."""
    n_obs = seg_end1 - seg_start1 + 1
    mem = h5py.File("permol_mem_%d" % os.getpid(), mode="w",
                    driver="core", backing_store=False)
    try:
        g = mem.create_group("segment_0")
        g.attrs["n_obs"] = n_obs
        g.attrs["chr"] = chrom
        g.attrs["start"] = seg_start1
        g.attrs["end"] = seg_end1
        g.attrs["segment"] = 0
        save_sparse_dense(mem, "segment_0/nucleotides", np.asarray(codes_seg))
        dshared["info_file"] = mem
        d = robocop.createDictionary(0, dshared, chrom, seg_start1, seg_end1)
        updatePerMoleculeEMMat(d, 0, dshared, ps_w, ps_c, calls_w, calls_c)

        # Pre-multiply the 7 emission layers into 1 and tell the C code n_vars = 1.
        # algo.c:57-58/:84-85/:110-111 multiply the n_vars layers together, so this is
        # algebraically identical and cuts the emission array (and the inner loop) 7x.
        emat = d["emission"]
        acc = np.array(emat[0], dtype=np.double, order="C")
        for layer in range(1, emat.shape[0]):
            acc *= emat[layer]
        d["emission"] = acc[np.newaxis, :, :]
        del emat, acc
        dshared["n_vars"] = 1

        robocop.posterior_forward_backward(d, 0, dshared)
        p = _GRAB.pop("p")
        ll = float(mem["segment_0"].attrs["log_likelihood"])
        op = C.collapse(p, M)
        return p, op[:, keep_idx], ll
    finally:
        mem.close()
        dshared["info_file"] = None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--pi", type=float, default=None, help="arm B mixing fraction")
    ap.add_argument("--eff", type=int, default=0, choices=[0, 1],
                    help="1 = per-read efficiency correction on")
    ap.add_argument("--window", default="chrI:60001-65000")
    ap.add_argument("--traindir", default="robocop_train_fiberonly")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bam", default=BAM)
    ap.add_argument("--flank", type=int, default=400)
    ap.add_argument("--max-molecules", type=int, default=0, help="0 = all; >0 for a smoke test")
    ap.add_argument("--min-span", type=int, default=1,
                    help="skip molecules whose in-window span is shorter than this")
    a = ap.parse_args(argv)
    t_all = time.time()
    install_posterior_grab()

    chrom, rng = a.window.split(":")
    win_start1, win_end1 = (int(x) for x in rng.split("-"))
    n_win = win_end1 - win_start1 + 1
    out = a.out.rstrip("/")
    os.makedirs(os.path.join(out, "tmpDir"), exist_ok=True)

    import pickle
    dshared = pickle.load(open(os.path.join(a.traindir, "HMMconfig.pkl"), "rb"))
    dshared["robocopC"] = os.path.join(ANALYSIS, "pkgvar", "permol_seq_maskoff",
                                       "robocop", "librobocop.so")
    dshared["tmpDir"] = os.path.join(out, "tmpDir") + "/"
    dshared["nucleotides"] = 1
    dshared["cache_parents_children"] = True   # see robocop.py PERMOL note
    dshared["info_file"] = None
    n_states = dshared["n_states"]

    # --- parameters ---------------------------------------------------------------
    PP.assert_verbatim(dshared, "watson_signal")
    PP.assert_verbatim(dshared, "crick_signal")
    ps_w, src_w, info_w = PP.arm_ps(dshared, "watson_signal", a.arm, a.pi)
    ps_c, src_c, info_c = PP.arm_ps(dshared, "crick_signal", a.arm, a.pi)
    print("arm params watson:", json.dumps(info_w))
    print("arm params crick :", json.dumps(info_c))

    # --- molecules ----------------------------------------------------------------
    ext_start0 = max(0, win_start1 - 1 - a.flank)
    ext_end0 = win_end1 + a.flank
    codes_ext = getNucleotideSequence(FASTA, chrom, ext_start0 + 1, ext_end0)
    t0 = time.time()
    mols, es, ee, rstats = RC.read_molecules(a.bam, chrom, win_start1, win_end1,
                                             codes_ext, flank=a.flank)
    print("read %d molecules in %.1fs; %s" % (len(mols), time.time() - t0,
                                              json.dumps(rstats, sort_keys=True)))
    if a.max_molecules:
        mols = mols[:a.max_molecules]
    pooled_f = PP.pooled_rate(mols, "flank")
    pooled_s = PP.pooled_rate(mols, "span")
    print("pooled flank m6A rate %.5f  span %.5f" % (pooled_f, pooled_s))

    # --- collapse machinery -------------------------------------------------------
    M = C.collapse_matrix(dshared)
    cols = C.column_names(dshared)
    keep = [c for c in KEEP_COLS if c in cols]
    keep_idx = np.array([cols.index(c) for c in keep])
    print("kept optable columns:", keep)

    win_off = (win_start1 - 1) - es          # window start inside the ext arrays
    sum_p = np.zeros((n_win, n_states))
    cov = np.zeros(n_win, dtype=np.int64)
    per_mol = np.zeros((len(mols), n_win, len(keep)), dtype=np.float32)
    per_mol_cov = np.zeros((len(mols), n_win), dtype=bool)
    rows = []
    checked_collapse = False
    t_dec = time.time()

    for mi, m in enumerate(mols):
        s0 = max(m.ref_start0, win_start1 - 1)
        e0 = min(m.ref_end0, win_end1)
        if e0 - s0 < a.min_span:
            rows.append(dict(i=mi, name=m.name, strand=m.strand, skipped="short",
                             span=e0 - s0))
            continue
        seg_start1, seg_end1 = s0 + 1, e0
        lo, hi = s0 - es, e0 - es
        codes_seg = codes_ext[lo:hi]
        calls = m.calls[lo:hi]
        no = np.full(hi - lo, RC.NO_OBS, dtype=np.int8)
        if m.strand == "watson":
            calls_w, calls_c = calls, no
        else:
            calls_w, calls_c = no, calls

        scale, basis = (1.0, "off")
        pw, pc = ps_w, ps_c
        if a.eff:
            scale, basis = PP.read_scale(m, pooled_f, pooled_s)
            pw, pc = PP.scaled_ps(ps_w, scale), PP.scaled_ps(ps_c, scale)

        t1 = time.time()
        p, op, ll = decode_one(dshared, chrom, seg_start1, seg_end1, codes_seg,
                               pw, pc, calls_w, calls_c, M, keep_idx)
        if not checked_collapse:
            d = C.assert_matches(dshared, p[:min(200, p.shape[0])])
            print("vectorised collapse == robocop.sum_for_dbf_probs (max|d| = %g)" % d)
            checked_collapse = True

        ws, we = s0 - (win_start1 - 1), e0 - (win_start1 - 1)
        sum_p[ws:we] += p
        cov[ws:we] += 1
        per_mol[mi, ws:we, :] = op.astype(np.float32)
        per_mol_cov[mi, ws:we] = True
        bad = int(((p < -1e-9) | (p > 1 + 1e-6)).sum())
        rows.append(dict(i=mi, name=m.name, strand=m.strand, skipped="",
                         span=e0 - s0, seg_start1=seg_start1, seg_end1=seg_end1,
                         n_informative=m.n_informative, n_meth=m.n_meth,
                         n_ambig=m.n_ambig, flank_n=m.flank_n, flank_meth=m.flank_meth,
                         eff_scale=scale, eff_basis=basis, loglik=ll,
                         n_invalid_posterior=bad, secs=time.time() - t1))
        if (mi + 1) % 25 == 0:
            print("  %d/%d  %.1fs elapsed  peak RSS %.2f GiB" % (
                mi + 1, len(mols), time.time() - t_dec,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0))
            sys.stdout.flush()
    secs_decode = time.time() - t_dec
    done = sum(1 for r in rows if not r.get("skipped"))
    print("decoded %d molecules in %.1fs (%.2fs each)" % (
        done, secs_decode, secs_decode / max(done, 1)))

    # --- molecule average, written as an ordinary decode --------------------------
    avg = np.zeros_like(sum_p)
    ok = cov > 0
    avg[ok] = sum_p[ok] / cov[ok, None]
    pu = RC.pileup_from_molecules(mols, es, ee)
    n_store = n_win + 1
    info_path = os.path.join(out, "tmpDir", "info_0_1.h5")
    if os.path.exists(info_path):
        os.remove(info_path)
    with h5py.File(info_path, "w") as f:
        g = f.create_group("segment_0")
        g.attrs["n_obs"] = n_win
        g.attrs["chr"] = chrom
        g.attrs["start"] = win_start1
        g.attrs["end"] = win_end1
        g.attrs["segment"] = 0
        g.attrs["log_likelihood"] = float(np.nansum([r.get("loglik", np.nan) for r in rows]))
        g.attrs["n_molecules"] = done
        save_sparse_dense(f, "segment_0/nucleotides",
                          np.asarray(codes_ext[win_off:win_off + n_win]))
        save_posterior_truncated(f, "segment_0/posterior", avg)
        for nm, key in (("meth_watson", "k_watson"), ("meth_crick", "k_crick"),
                        ("A_watson", "n_watson"), ("A_crick", "n_crick")):
            save_sparse_dense(f, "segment_0/Fiber_count_%s" % nm,
                              pu[key][win_off:win_off + n_store])
        save_sparse_dense(f, "segment_0/molecule_depth", cov)
    print("wrote", info_path)

    pd.DataFrame([dict(chr=chrom, start=win_start1, end=win_end1)]).to_csv(
        os.path.join(out, "coords.tsv"), sep="\t", index=False)
    with open(os.path.join(out, "config.ini"), "w") as f:
        f.write("[main]\n")
        f.write("nucFile = inputs/SacCer3.fa\n")
        f.write("nucleosomeFile = inputs/Chereji_2018_+1_-1_nucs.bed\n")
        f.write("tfFile = inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1.bed\n")
        f.write("cshared = pkgvar/permol_seq_maskoff/robocop/librobocop.so\n")
        f.write("bamFile = %s\n" % a.bam)
        f.write("nucleotide = A\ntech = MNase\ntech2 = Fiber\n")
        f.write("fragRangeLong = (127, 187)\nfragRangeShort = (0, 80)\n")
        f.write("trainDir = %s\n" % a.traindir)
    label = "permol_arm%s%s_eff%s" % (a.arm,
                                      ("" if a.pi is None else "_pi%s" % str(a.pi).replace(".", "p")),
                                      "on" if a.eff else "off")
    with open(os.path.join(out, "RUN_INFO.json"), "w") as f:
        json.dump(dict(schema=1, outdir=out, label=label,
                       driver="singlefiber/run_permol_window.py",
                       pkg_tree="pkgvar/permol_seq_maskoff",
                       pkg_tree_how="literal sys.path.insert",
                       traindir=a.traindir, coords="singlefiber (inline)",
                       emission="per-molecule Bernoulli",
                       arm=a.arm, pi=a.pi, efficiency=bool(a.eff),
                       window=a.window, n_molecules=done,
                       bam=a.bam, written=datetime.datetime.now().isoformat(timespec="seconds"),
                       slurm_job_id=os.environ.get("SLURM_JOB_ID")), f, indent=1)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "molecules.tsv"), sep="\t", index=False)
    np.savez_compressed(os.path.join(out, "permol.npz"),
                        per_mol=per_mol, per_mol_cov=per_mol_cov,
                        cols=np.array(keep), cov=cov,
                        names=np.array([m.name for m in mols]),
                        strands=np.array([m.strand for m in mols]),
                        win=np.array([win_start1, win_end1]),
                        avg_optable=C.collapse(avg, M)[:, keep_idx].astype(np.float32))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    stats = dict(label=label, arm=a.arm, pi=a.pi, eff=bool(a.eff), window=a.window,
                 traindir=a.traindir, n_molecules_fetched=len(mols),
                 n_molecules_decoded=done, reader=rstats,
                 pooled_flank_rate=float(pooled_f), pooled_span_rate=float(pooled_s),
                 arm_info_watson=info_w, arm_info_crick=info_c,
                 secs_decode=secs_decode, secs_total=time.time() - t_all,
                 secs_per_molecule=secs_decode / max(done, 1),
                 peak_rss_gib=peak,
                 n_positions_uncovered=int((cov == 0).sum()),
                 median_molecule_depth=float(np.median(cov)),
                 n_invalid_posterior=int(df.get("n_invalid_posterior", pd.Series([0])).sum()))
    with open(os.path.join(out, "run_stats.json"), "w") as f:
        json.dump(stats, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in stats.items() if k != "reader"}, indent=1, default=float))
    print("peak RSS %.2f GiB; total %.1fs" % (peak, time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
