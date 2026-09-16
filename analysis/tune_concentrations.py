"""Calibrate all 153 TF concentrations against an external per-TF site-count target.

The idea in five lines
----------------------
1. Every TF has a TARGET number of sites (make_conc_targets.py; MacIsaac p005_c1 by default).
2. One genome-wide decode reports the PREDICTED count for all 153 TFs at once.
3. Called too many -> lower that TF's lambda; too few -> raise it.
4. Rebuild the trainDir with the new lambda vector (~2 s) and decode again.
5. Stop when predicted ~= target, typically 4-8 rounds.

Step 2 is why this is cheap and why Gibbs sampling would not be: a single decode already
returns every coordinate, so a scheme that visits one TF at a time would spend 153 decodes
to learn what one decode gives.

Why the direction is never ambiguous
------------------------------------
The response is MONOTONE. The 17 decodes of the existing Abf1 lambda sweep take n_pred from
5 to 306 as lambda goes 1e-5 -> 1e3, never decreasing. More prior always means more calls, so
"too many -> go down" cannot backfire.

The step size, and the bug it fixes
-----------------------------------
The response is also strongly SUBLINEAR. Fitting those same 17 points gives

    n_pred ~ lambda ** 0.230

and the two full decodes (robocop_chrI_fib_seq vs ..._lam0p01) independently give 0.238 from
posterior mass. So the obvious update `lambda *= (T/E)**alpha` closes only ~16% of the log-gap
per iteration -- about 25 rounds. The step must be divided by that elasticity:

    log lambda_i  +=  (alpha / beta_i) * (log T_i - log E_i)

`beta_i` is seeded at BETA0 and then re-estimated per TF by secant from the last two
iterations, because motif length and site density will not share one exponent.

Coupling
--------
Changing one TF does move others -- measured, by comparing the two decodes above: of the
other 147 TF states the median ratio was 1.0000 and the 95th percentile 1.0029, but Rsc3
moved +24.7%, Nhp6a +9.3%, Spt15 +5.5%. Real but SPARSE. Damping (alpha < 1) absorbs it, and
`cross_talk` in the per-iteration log measures it: observed change divided by the change this
TF's own lambda predicts. ~1.0 means uncoupled.

Groups
------
Rap1 has four motifs against one target. They are alternative descriptions of one factor, so
the group's predicted counts are SUMMED and compared to the single target, and one lambda is
applied to all four. `make_conc_targets.py` emits the `group` column that drives this.

Fixed settings and the report
-----------------------------
`unknown` is pinned at FIXED_LAM on every build and the nucleosome prior is held at its
source value (`make_conc_trainDir --hold-nucleosome`). Only groups with a MacIsaac target
move; a group measured at the lambda cap and still >2x short goes on the `cant_fix` list.
Each `update` writes `conc_tuning/<run>/report_NN.tsv`: MacIsaac count, decoded occupancy,
calls, calls within 30 bp of a MacIsaac site, status, and the lambda step.

Usage
-----
    python tune_concentrations.py status
    python tune_concentrations.py build  --iter 0      # write the trainDir
    python tune_concentrations.py submit --iter 0      # decode + count arrays, chained
    python tune_concentrations.py update --iter 0      # counts -> lambda for iter 1
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from count_calls import match_path   # noqa: E402

# Each campaign lives in its own namespace (state, trainDirs, decodes, counts), so a restart
# never collides with an earlier one. `u001` is the restart at lambda_unknown = 0.01; the
# first campaign (lambda_unknown = 1) keeps its original un-namespaced files.
RUN = "u001"


def run_dir():
    return os.path.join(HERE, "conc_tuning", RUN)


def state_path():
    return os.path.join(run_dir(), "state.json")

SRC_TRAINDIR = "robocop_train_fiberonly"     # PRISTINE; never patch a patch (see below)
DRIVER = "run_split_revfix_seq_maskoff.py"   # fib+seq, the baseline layer configuration
NTASK = 48

ALPHA = 0.7          # damping; absorbs the sparse cross-TF coupling
BETA0 = 0.23         # seed elasticity d log E / d log lambda, from the existing sweep
BETA_LO, BETA_HI = 0.10, 1.00
LAM_LO, LAM_HI = 1e-6, 1e3   # above ~1e4 the shared root collapses and p^147 kills nucleosomes
NUC_TOL = 0.05       # nucleosome_prob must stay within 5% of baseline

# Held fixed for the whole campaign, never updated: `unknown` has no target (user decision
# 2026-09-10: 0.01). The nucleosome is not tuned either -- every build solves the lambda that
# keeps its prior at the source value (make_conc_trainDir --hold-nucleosome), because moving
# `unknown` and the TFs shifts the shared root, which the 147 bp nucleosome amplifies ~21x.
FIXED_LAM = {"unknown": 0.01}
NUC_BASELINE = 66521.3   # genome nucleosome copies, untuned model (counts_conctune_00)

NON_TF = {"background", "nucleosome", "unknown", "nuc_center", "nuc_start", "nuc_end"}


# ---------------------------------------------------------------------------
def load_targets(source):
    path = os.path.join(HERE, "inputs", "conc_targets_%s.tsv" % source)
    if not os.path.exists(path):
        sys.exit("missing %s -- run make_conc_targets.py --source %s" % (path, source))
    rows = []
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    for r in rows:
        r["n_total"] = int(r["n_total"])
        r["has_target"] = int(r["has_target"])
    return rows


def load_state():
    if not os.path.exists(state_path()):
        return None
    with open(state_path()) as fh:
        return json.load(fh)


def save_state(st):
    os.makedirs(run_dir(), exist_ok=True)
    with open(state_path(), "w") as fh:
        json.dump(st, fh, indent=2, sort_keys=True)


def init_state(source, driver=None, no_nuc_stop=False, no_nuc_hold=False):
    tg = load_targets(source)
    # The driver is frozen into state.json here and used by every round's 48 decode tasks, so
    # a typo would fail all of them. Legacy campaigns never pass through this function again.
    if not os.path.exists(os.path.join(HERE, driver or DRIVER)):
        sys.exit("driver %s does not exist in %s" % (driver or DRIVER, HERE))
    st = dict(source=source, run=RUN, iter=0, driver=driver or DRIVER,
              # Recorded once, at creation. States without the key (u001, m001) keep the rule on.
              nuc_stop_rule=not no_nuc_stop,
              # Recorded once, at creation. States without the key (u001, m001, fm001, sm001)
              # keep --hold-nucleosome and its build-time prior-drift abort.
              nuc_hold=not no_nuc_hold,
              motifs=[r["motif"] for r in tg],
              group={r["motif"]: r["group"] for r in tg},
              has_target={r["motif"]: r["has_target"] for r in tg},
              target={r["group"]: r["n_total"] for r in tg if r["has_target"]},
              lam={r["motif"]: 1.0 for r in tg},
              beta={r["group"]: BETA0 for r in tg if r["has_target"]},
              alpha={r["group"]: ALPHA for r in tg if r["has_target"]},
              fixed_lam=dict(FIXED_LAM),
              nuc_lam={},        # iter -> solved lambda_nucleosome
              cant_fix={},       # group -> why concentration alone cannot reach its target
              history=[])
    save_state(st)
    return st


def traindir(t):
    return "robocop_train_ct_%s_%02d" % (RUN, t)


def outdir(t):
    return "robocop_genome_ct_%s_%02d" % (RUN, t)


def tag(t):
    return "ct_%s_%02d" % (RUN, t)


# ---------------------------------------------------------------------------
def cmd_build(st, t):
    """Write the trainDir for iteration t from the CUMULATIVE lambda vector.

    Always against the pristine source. make_conc_trainDir's FIDELITY gate recomputes every
    concentration from calculateKD and demands a bit-exact match to its source, so a patched
    trainDir cannot itself be a source -- lambda=0.5 then lambda=0.5 does NOT give 0.25.
    The cumulative vector lives in state.json and the full value is re-applied each time.
    """
    out = traindir(t)
    if os.path.isdir(out):
        print("%s exists; not rebuilding" % out)
        return out
    lam = st["lam"]
    moved = {m: v for m, v in lam.items() if abs(v - 1.0) > 1e-12}
    fixed = st.get("fixed_lam", {})
    hold = st.get("nuc_hold", True)
    args = [sys.executable, os.path.join(HERE, "make_conc_trainDir.py"),
            "--src", SRC_TRAINDIR, "--out", out]
    if hold:
        args.append("--hold-nucleosome")
    # --set needs at least one factor; with nothing moved and nothing fixed, pass a no-op
    # lambda=1 for one motif. The FIDELITY gate then also proves the pipeline is a faithful
    # identity before any real patch is applied.
    sets = dict(moved, **fixed) or {st["motifs"][0]: 1.0}
    for m, v in sets.items():
        args += ["--set", "%s=%.10g" % (m, v)]
    print("building %s  (%d factors moved from 1.0, fixed %s, nucleosome %s)"
          % (out, len(moved), fixed or "none", "held" if hold else "NOT held"))
    r = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("make_conc_trainDir failed:\n%s\n%s" % (r.stdout[-3000:], r.stderr[-3000:]))

    patch = json.load(open(os.path.join(HERE, out, "conc_patch.json")))
    nb, na = patch["nucleosome_prob_before"], patch["nucleosome_prob_after"]
    drift = abs(na / nb - 1.0)
    if not hold:
        # No hold: lambda_nucleosome stays 1 and the prior floats with the TF lambda vector.
        # The drift abort below guards only a failed hold solve, so it does not apply here.
        st.setdefault("nuc_lam", {})[str(t)] = None
        st.setdefault("nuc_prob", {})[str(t)] = na
        save_state(st)
        print("  lambda_nucleosome 1 (not held)")
        print("  nucleosome_prob %.6g -> %.6g  (x%.4g of source) [hold off, no drift abort]"
              % (nb, na, na / nb))
        print("  max rel change to untouched TFs: %.3g" % patch["max_rel_change_other_tfs"])
        return out
    st.setdefault("nuc_lam", {})[str(t)] = patch["nucleosome_lambda_solved"]
    save_state(st)
    print("  lambda_nucleosome solved = %.6g" % patch["nucleosome_lambda_solved"])
    print("  nucleosome_prob %.6g -> %.6g  (%.3f%% drift)" % (nb, na, 100 * drift))
    print("  max rel change to untouched TFs: %.3g" % patch["max_rel_change_other_tfs"])
    if drift > NUC_TOL:
        sys.exit("ABORT: nucleosome_prob moved %.1f%% (> %.0f%%) even though it is held. The "
                 "--hold-nucleosome solve failed; do not decode this trainDir."
                 % (100 * drift, 100 * NUC_TOL))
    return out


def cmd_submit(st, t, chain=False):
    td, od, tg = traindir(t), outdir(t), tag(t)
    if not os.path.isdir(os.path.join(HERE, td)):
        sys.exit("%s missing -- run `build --iter %d` first" % (td, t))
    if os.path.isdir(os.path.join(HERE, od)):
        sys.exit("%s already exists; move it aside or skip to `update`" % od)

    driver = st.get("driver", DRIVER)
    env = dict(os.environ, DRIVER=driver, TRAINDIR=td, OUTDIR=od, NTASK=str(NTASK))
    j1 = subprocess.run(
        ["sbatch", "--parsable", "--job-name=ctDec_%s_%02d" % (RUN, t),
         "--array=0-%d" % (NTASK - 1), "sbatch_genome_decode.sh"],
        cwd=HERE, env=env, capture_output=True, text=True)
    if j1.returncode != 0:
        sys.exit("decode submit failed: %s" % j1.stderr)
    jid1 = j1.stdout.strip()

    env2 = dict(os.environ, OUTDIR=od, TAG=tg)
    # --kill-on-invalid-dep: if a decode task fails for good, Slurm cancels this count instead
    # of leaving it pending forever (DependencyNeverSatisfied), so the chained `next` job still
    # runs, finds no counts, and writes STOPPED rather than the chain stalling silently.
    j2 = subprocess.run(
        ["sbatch", "--parsable", "--job-name=ctCnt_%s_%02d" % (RUN, t),
         "--dependency=afterok:%s" % jid1, "--kill-on-invalid-dep=yes",
         "sbatch_count_calls.sh"],
        cwd=HERE, env=env2, capture_output=True, text=True)
    if j2.returncode != 0:
        sys.exit("count submit failed: %s" % j2.stderr)
    jid2 = j2.stdout.strip()

    print("iter %d  driver %s  decode %s (%d tasks)  ->  count %s (16 tasks)"
          % (t, driver, jid1, NTASK, jid2))
    if chain:
        # `next` runs update -> stop rules -> build + submit the following round. afterany, so it
        # runs even when the decode or count failed; `update` then finds fewer than 16 count
        # tables, exits, and cmd_next records STOPPED -- a failure stops the chain visibly
        # instead of tuning on bad counts or pending forever.
        env3 = dict(os.environ, RUN=RUN, ITER=str(t))
        j3 = subprocess.run(
            ["sbatch", "--parsable", "--job-name=ctNext_%s_%02d" % (RUN, t),
             "--dependency=afterany:%s" % jid2, "sbatch_tune_next.sh"],
            cwd=HERE, env=env3, capture_output=True, text=True)
        if j3.returncode != 0:
            sys.exit("next-round submit failed: %s" % j3.stderr)
        print("  chained: next-round job %s runs `next --iter %d` after the count" % (j3.stdout.strip(), t))
        chain_log(st, t, "submitted decode %s count %s next %s" % (jid1, jid2, j3.stdout.strip()))
    else:
        print("watch:  squeue -u $USER")
        print("then:   python tune_concentrations.py update --iter %d" % t)
    return jid1, jid2


MAX_ROUNDS = 8       # rounds 0..7


def chain_log(st, t, msg):
    import time
    os.makedirs(run_dir(), exist_ok=True)
    with open(os.path.join(run_dir(), "chain.log"), "a") as fh:
        fh.write("%s  iter %02d  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), t, msg))


def stop(st, t, reason):
    with open(os.path.join(run_dir(), "STOPPED"), "w") as fh:
        fh.write("stopped after iter %d: %s\n" % (t, reason))
    chain_log(st, t, "STOPPED: " + reason)
    print("CHAIN STOPPED after iter %d: %s" % (t, reason))


def stop_reason(st, t):
    """The auto-chain's stop rules, evaluated after `update --iter t`. None = keep going."""
    h = st["history"]
    nuc = h[-1].get("nucleosome_copies", float("nan"))
    # A campaign created with --no-nuc-stop reports nucleosome copies but never stops on them.
    if st.get("nuc_stop_rule", True) and not abs(nuc / NUC_BASELINE - 1.0) <= NUC_TOL:
        return ("nucleosome copies %.0f are %+.1f%% from the untuned %.0f (limit %.0f%%) -- "
                "review before tuning further" % (nuc, 100 * (nuc / NUC_BASELINE - 1),
                                                   NUC_BASELINE, 100 * NUC_TOL))
    w = [x.get("status", {}).get("within_2x", 0) for x in h]
    if len(w) >= 3 and max(w[-2:]) <= w[-3]:
        return "within-2x count has not risen for 2 rounds (%s)" % " -> ".join(map(str, w[-3:]))
    if t + 1 >= MAX_ROUNDS:
        return "reached the %d-round limit" % MAX_ROUNDS
    return None


def cmd_next(st, t):
    """update --iter t, then either stop or build + submit iter t+1 with the chain attached."""
    try:
        cmd_update(st, t)
        why = stop_reason(st, t)
        if why:
            return stop(st, t, why)
        cmd_build(st, t + 1)
        cmd_submit(st, t + 1, chain=True)
    except BaseException as e:       # sys.exit() inside update/build/submit lands here too
        stop(st, t, "error: %s" % (e,))
        raise


def read_counts(t):
    d = os.path.join(HERE, "conc_tuning", "counts_%s" % tag(t))
    parts = sorted(glob.glob(os.path.join(d, "*.tsv")))
    if len(parts) != 16:
        sys.exit("expected 16 per-chromosome count tables in %s, found %d" % (d, len(parts)))
    merged = os.path.join(HERE, "conc_tuning", "counts_%s.tsv" % tag(t))
    r = subprocess.run([sys.executable, os.path.join(HERE, "count_calls.py"),
                        "--merge"] + parts + ["--out", merged],
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("merge failed:\n%s" % r.stderr[-3000:])
    out = {}
    with open(merged) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(hdr, line.rstrip("\n").split("\t")))
            out[row["factor"]] = float(row["occ"])
    match = {}
    mpath = match_path(merged)
    if not os.path.exists(mpath):
        sys.exit("missing MacIsaac match table %s -- were the counts made by the current "
                 "count_calls.py?" % mpath)
    with open(mpath) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(hdr, line.rstrip("\n").split("\t")))
            match[row["group"]] = {k: int(row[k]) for k in ("n_macisaac", "n_calls", "n_matched")}
    return out, match


def cmd_update(st, t):
    """Counts for iteration t -> lambda for iteration t+1."""
    occ, match = read_counts(t)
    groups = {}
    for m in st["motifs"]:
        g = st["group"][m]
        if g in st["target"]:
            groups.setdefault(g, []).append(m)

    prev = st["history"][-1] if st["history"] else None
    rec, rows = {}, []
    for g, motifs in sorted(groups.items()):
        T = float(st["target"][g])
        E = sum(occ.get(m, 0.0) for m in motifs)     # group counts are SUMMED
        lam = st["lam"][motifs[0]]
        beta, alpha = st["beta"][g], st["alpha"][g]

        if T <= 0:
            continue
        if E <= 0:
            # The state carries no mass at all. A ratio is undefined; nudge up by a fixed
            # factor rather than dividing by zero, and let the next iteration measure it.
            new_lam, gap, beta_new, cross = min(lam * 10.0, LAM_HI), float("nan"), beta, float("nan")
        else:
            gap = math.log(T) - math.log(E)
            # secant estimate of d log E / d log lambda from the previous iteration
            beta_new = beta
            cross = float("nan")
            if prev and g in prev["E"] and prev["E"][g] > 0 and prev["lam"][g] > 0:
                dl = math.log(lam) - math.log(prev["lam"][g])
                de = math.log(E) - math.log(prev["E"][g])
                if abs(dl) > 1e-9:
                    beta_new = min(BETA_HI, max(BETA_LO, de / dl))
                    cross = de / (dl * beta) if beta else float("nan")
                elif abs(de) > 1e-9:
                    # lambda did not move but the count did -> pure cross-talk
                    cross = float("inf")
            # oscillation: the gap changed sign, so we overshot -> damp this factor
            if prev and g in prev["gap"] and prev["gap"][g] == prev["gap"][g]:
                if gap * prev["gap"][g] < 0:
                    alpha = max(0.1, alpha * 0.5)
            new_lam = lam * math.exp((alpha / beta_new) * gap)
            new_lam = min(LAM_HI, max(LAM_LO, new_lam))

        # Can't-fix: measured AT the cap and still more than 2x short. Concentration has no
        # headroom left, so the shortfall is the motif/emission, not the prior. It stays at
        # the cap (user decision 2026-09-10) and is listed; re-judged every round because
        # coupling can still move it.
        if lam >= LAM_HI * (1 - 1e-9) and E < T / 2:
            try:
                need = lam * (T / E) ** (1.0 / beta_new) if E > 0 else float("inf")
            except OverflowError:
                need = float("inf")
            st["cant_fix"][g] = dict(iter=t, target=T, decoded=E,
                                     fold_short=(T / E) if E > 0 else float("inf"),
                                     lambda_needed=need)
        else:
            st["cant_fix"].pop(g, None)

        for m in motifs:
            st["lam"][m] = new_lam
        st["beta"][g], st["alpha"][g] = beta_new, alpha
        rec[g] = dict(T=T, E=E, lam=lam, new_lam=new_lam, gap=gap,
                      beta=beta_new, alpha=alpha, cross=cross)
        rows.append((g, T, E, lam, new_lam, gap, beta_new, alpha, cross))

    def status(g, gap):
        if g in st["cant_fix"]:
            return "cant_fix"
        if gap != gap:
            return "no_mass"
        return "within_2x" if abs(gap) <= math.log(2) else ("under" if gap > 0 else "over")

    log = os.path.join(run_dir(), "report_%02d.tsv" % t)
    with open(log, "w") as fh:
        fh.write("group\tmacisaac\tdecoded\tcalls\tmatched_30bp\tpct_macisaac_hit\t"
                 "pct_calls_hit\tfold_off\tlambda\tlambda_next\tstatus\tlog_gap\tbeta\talpha\t"
                 "cross_talk\n")
        for r in sorted(rows, key=lambda x: -abs(x[5]) if x[5] == x[5] else 0):
            g, T, E, lam, nl, gap = r[:6]
            mt = match.get(g, dict(n_macisaac=0, n_calls=0, n_matched=0))
            pm = 100.0 * mt["n_matched"] / mt["n_macisaac"] if mt["n_macisaac"] else float("nan")
            pc = 100.0 * mt["n_matched"] / mt["n_calls"] if mt["n_calls"] else float("nan")
            fold = (E / T) if T else float("nan")
            fh.write("%s\t%.6g\t%.6g\t%d\t%d\t%.1f\t%.1f\t%.4g\t%.6g\t%.6g\t%s\t%.4f\t%.4f\t"
                     "%.3f\t%.4g\n" % (g, T, E, mt["n_calls"], mt["n_matched"], pm, pc, fold,
                                       lam, nl, status(g, gap), gap, r[6], r[7], r[8]))

    finite = [abs(r[5]) for r in rows if r[5] == r[5]]
    tally = {}
    for r in rows:
        s = status(r[0], r[5])
        tally[s] = tally.get(s, 0) + 1
    nuc = occ.get("nucleosome", float("nan"))
    st["history"].append(dict(iter=t,
                              E={g: v["E"] for g, v in rec.items()},
                              lam={g: v["lam"] for g, v in rec.items()},
                              gap={g: v["gap"] for g, v in rec.items()},
                              status=tally, nucleosome_copies=nuc,
                              max_abs_gap=max(finite) if finite else float("nan"),
                              median_abs_gap=sorted(finite)[len(finite) // 2] if finite else float("nan")))
    st["iter"] = t + 1
    save_state(st)

    def as_factor(g):
        """exp() of a log-gap, reported as a fold-change. Guarded: a state with almost no
        posterior gives a huge gap, and math.exp would raise OverflowError on the report
        rather than on anything that matters."""
        try:
            return "%.1fx" % math.exp(g)
        except OverflowError:
            return ">1e300x"

    print("iteration %d  (%d groups with a target)" % (t, len(rows)))
    if not finite:
        print("  WARNING: no group produced a finite gap. Every predicted count was 0 or the")
        print("  targets are empty -- check conc_tuning/counts_%s.tsv before iterating." % tag(t))
        return
    med = sorted(finite)[len(finite) // 2]
    print("  within 2x of MacIsaac %d | under %d | over %d | can't fix %d | no mass %d"
          % tuple(tally.get(k, 0) for k in ("within_2x", "under", "over", "cant_fix", "no_mass")))
    print("  typical factor is %s off its MacIsaac count (median |log gap| %.3f); worst %s"
          % (as_factor(med), med, as_factor(max(finite))))
    ncalls = sum(match.get(r[0], {}).get("n_calls", 0) for r in rows)
    nmat = sum(match.get(r[0], {}).get("n_matched", 0) for r in rows)
    nmac = sum(match.get(r[0], {}).get("n_macisaac", 0) for r in rows)
    print("  calls (post >= 0.10) %d, of which %d within 30 bp of a MacIsaac site "
          "(%.1f%% of calls, %.1f%% of %d MacIsaac sites)"
          % (ncalls, nmat, 100.0 * nmat / max(ncalls, 1), 100.0 * nmat / max(nmac, 1), nmac))
    if st.get("nuc_stop_rule", True):
        dn = nuc / NUC_BASELINE - 1.0
        print("  nucleosome copies %.0f vs %.0f untuned (%+.2f%%); lambda_nucleosome %s"
              % (nuc, NUC_BASELINE, 100 * dn, st.get("nuc_lam", {}).get(str(t), "n/a")))
        if abs(dn) > NUC_TOL:
            print("  ** nucleosome copies moved more than %.0f%% -- pause and review before the "
                  "next round **" % (100 * NUC_TOL))
    elif len(st["history"]) < 2:
        # Stop rule off: the layer configuration differs from the untuned fib+seq model behind
        # NUC_BASELINE, so the only meaningful reference is this campaign's own round 0.
        print("  nucleosome copies %.0f (round 0, the reference for later rounds); "
              "lambda_nucleosome %s" % (nuc, st.get("nuc_lam", {}).get(str(t), "n/a")))
    else:
        nuc0 = st["history"][0]["nucleosome_copies"]
        print("  nucleosome copies %.0f vs %.0f at round 0 (%+.2f%%); lambda_nucleosome %s "
              "[stop rule off]" % (nuc, nuc0, 100 * (nuc / nuc0 - 1.0),
                                   st.get("nuc_lam", {}).get(str(t), "n/a")))
    if not st.get("nuc_hold", True):
        print("  nucleosome prior [hold off]: "
              + "  ".join("r%s %.4g" % (k, v) for k, v in
                          sorted(st.get("nuc_prob", {}).items(), key=lambda kv: int(kv[0]))))
    if st["cant_fix"]:
        print("  can't fix with concentration (at lambda %g, still >2x short):" % LAM_HI)
        for g, c in sorted(st["cant_fix"].items(), key=lambda kv: -kv[1]["fold_short"]):
            print("    %-10s target %6.0f  decoded %8.1f  (%.1fx short, would need lambda ~%.3g)"
                  % (g, c["target"], c["decoded"], c["fold_short"], c["lambda_needed"]))
    print("  worst offenders:")
    for r in sorted(rows, key=lambda x: -abs(x[5]) if x[5] == x[5] else 0)[:8]:
        mt = match.get(r[0], dict(n_macisaac=0, n_calls=0, n_matched=0))
        print("    %-10s MacIsaac %6.0f  decoded %9.1f  calls %6d  matched %5d  lambda %9.3g -> %-9.3g"
              % (r[0], r[1], r[2], mt["n_calls"], mt["n_matched"], r[3], r[4]))
    print("  wrote %s" % log)
    import tuning_trajectory
    tuning_trajectory.summarize(RUN)
    print("  next: python tune_concentrations.py build --iter %d" % (t + 1))


def cmd_status(st):
    if st is None:
        print("no state; run `build --iter 0` to start")
        return
    print("run         %s  (%s)" % (RUN, run_dir()))
    print("driver      %s" % st.get("driver", DRIVER))
    if "nuc_stop_rule" in st:
        print("nuc stop rule  %s" % ("on" if st["nuc_stop_rule"] else "off"))
    if "nuc_hold" in st:
        print("nuc hold  %s" % ("on" if st["nuc_hold"] else "off"))
    sp = os.path.join(run_dir(), "STOPPED")
    if os.path.exists(sp):
        print("chain       %s" % open(sp).read().strip())
    print("source      %s" % st["source"])
    print("iteration   %d" % st["iter"])
    print("targets     %d groups" % len(st["target"]))
    print("fixed       %s, nucleosome %s"
          % (st.get("fixed_lam", {}), "held" if st.get("nuc_hold", True) else "NOT held"))
    moved = sum(1 for v in st["lam"].values() if abs(v - 1.0) > 1e-12)
    print("lambda      %d of %d motifs moved from 1.0" % (moved, len(st["lam"])))
    print("can't fix   %s" % (", ".join(sorted(st.get("cant_fix", {}))) or "none"))
    nprob = st.get("nuc_prob", {}) if "nuc_hold" in st else {}
    for h in st["history"]:
        s = h.get("status", {})
        print("  iter %d   within2x %d  under %d  over %d  cant_fix %d   median|gap| %.3f   "
              "nucleosome %.0f" % (h["iter"], s.get("within_2x", 0), s.get("under", 0),
                                   s.get("over", 0), s.get("cant_fix", 0), h["median_abs_gap"],
                                   h.get("nucleosome_copies", float("nan")))
              + ("   nuc prior %.4g" % nprob[str(h["iter"])] if str(h["iter"]) in nprob else ""))
    # Rounds built but not yet counted (no history entry): still show their prior.
    done = {str(h["iter"]) for h in st["history"]}
    for k in sorted((k for k in nprob if k not in done), key=int):
        print("  iter %s   (built, not yet counted)   nuc prior %.4g" % (k, nprob[k]))


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "build", "submit", "update", "next"])
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--source", default="macisaac_c1")
    ap.add_argument("--run", default=RUN, help="campaign namespace (default %(default)s)")
    ap.add_argument("--driver", default=None,
                    help="decode driver, recorded when a campaign is first created "
                         "(default %s)" % DRIVER)
    ap.add_argument("--no-nuc-stop", action="store_true",
                    help="turn off the nucleosome-copies stop rule; recorded when a campaign is "
                         "first created (copies are then reported vs the campaign's round 0)")
    ap.add_argument("--no-nuc-hold", action="store_true",
                    help="do not pass --hold-nucleosome: lambda_nucleosome stays 1 and the "
                         "nucleosome prior floats with the TF lambdas (no build-time drift "
                         "abort); recorded when a campaign is first created")
    ap.add_argument("--chain", action="store_true",
                    help="with submit: attach the auto-chain (update -> next round)")
    args = ap.parse_args()
    RUN = args.run
    st = load_state()
    if args.cmd == "status":
        return cmd_status(st)
    if st is None:
        st = init_state(args.source, args.driver, args.no_nuc_stop, args.no_nuc_hold)
    else:
        if args.driver and args.driver != st.get("driver", DRIVER):
            sys.exit("campaign %s already uses driver %s" % (RUN, st.get("driver", DRIVER)))
        if args.no_nuc_stop and st.get("nuc_stop_rule", True):
            sys.exit("campaign %s already has the nucleosome stop rule on" % RUN)
        if args.no_nuc_hold and st.get("nuc_hold", True):
            sys.exit("campaign %s already holds the nucleosome prior" % RUN)
    t = args.iter if args.iter is not None else st["iter"]

    if args.cmd == "build":
        cmd_build(st, t)
    elif args.cmd == "submit":
        cmd_submit(st, t, chain=args.chain)
    elif args.cmd == "update":
        cmd_update(st, t)
    elif args.cmd == "next":
        cmd_next(st, t)


if __name__ == "__main__":
    main()
