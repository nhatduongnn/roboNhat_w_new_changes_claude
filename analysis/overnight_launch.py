"""Overnight 2026-09-15 launcher for jobs A (fixed-prior fiber veto test) and C (genome-wide
validation of the three tuned tuner-v2 models). B (fiber-tempered campaigns bt02/bt05/bt10) is
launched with tune_w.py; `watch-b` only arranges the report refresh when a campaign stops.

Rule 7
  A: only the layer configuration differs from sw01 round 05 (same trainDir, same weights).
  C: only the decode scope (whole genome, 48-task split of coord_genome_full.tsv) differs from
     each campaign's final round (sw01_05 seq-only, fw01_07 fiber, bw01_07 both).
Nothing here edits a trainDir, a pkgvar tree or an existing decode; every decode dir is new and
the submit refuses to write over an existing one.

Usage
  python overnight_launch.py decode          # A + C decode arrays and their count arrays
  python overnight_launch.py score           # A/C scoring jobs + the summary job (afterany)
  python overnight_launch.py watch-b --run bt10   # (run as a Slurm job) re-summarise when bt10 stops
  python overnight_launch.py chereji --label tw_bw01_07 --decode robocop_chrXIV_chrII_tw_bw01_07
Job ids go to overnight/jobs.json.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "overnight")
JOBS = os.path.join(OUT, "jobs.json")
SLURM_STRIP = ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_STEP_ID", "SLURM_NODELIST",
               "SLURM_JOB_NODELIST", "SLURM_NTASKS", "SLURM_NPROCS", "SLURM_TASKS_PER_NODE",
               "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "SLURM_MEM_PER_NODE")
EXCLUDE = "linux[31-40]"
WATCH_DEADLINE = "2026-09-15 20:00"   # B watchers stop resubmitting after this
GENOME_CHROMS = ["chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI", "chrVII", "chrVIII", "chrIX",
                 "chrX", "chrXI", "chrXII", "chrXIII", "chrXIV", "chrXV", "chrXVI"]

# A: (decode dir == label, driver, coords, chroms, ntask)
A_DECODES = [
    ("robocop_chrXIV_chrII_twS05_both", "run_split_revfix_seq_maskoff_mr58.py", "coord_tw_chrXIV_chrII.tsv", ["chrXIV", "chrII"]),
    ("robocop_chrXIV_chrII_twS05_fib", "run_split_revfix_fiber_maskoff_mr58.py", "coord_tw_chrXIV_chrII.tsv", ["chrXIV", "chrII"]),
    ("robocop_chrIV_twS05_both", "run_split_revfix_seq_maskoff_mr58.py", "coord_tw_chrIV.tsv", ["chrIV"]),
    ("robocop_chrIV_twS05_fib", "run_split_revfix_fiber_maskoff_mr58.py", "coord_tw_chrIV.tsv", ["chrIV"]),
]
A_TRAINDIR = "robocop_train_tw_sw01_05"
# C: (decode dir, driver, trainDir)
C_DECODES = [
    ("robocop_genome_tw_sw01_05", "run_split_revfix_seqonly_maskoff_mr58.py", "robocop_train_tw_sw01_05"),
    ("robocop_genome_tw_fw01_07", "run_split_revfix_fiber_maskoff_mr58.py", "robocop_train_tw_fw01_07"),
    ("robocop_genome_tw_bw01_07", "run_split_revfix_seq_maskoff_mr58.py", "robocop_train_tw_bw01_07"),
]


def tag(decode_dir):
    return decode_dir[len("robocop_"):]


def load_jobs():
    return json.load(open(JOBS)) if os.path.exists(JOBS) else {}


def save_jobs(j):
    os.makedirs(OUT, exist_ok=True)
    with open(JOBS + ".tmp", "w") as fh:
        json.dump(j, fh, indent=1, sort_keys=True)
    os.replace(JOBS + ".tmp", JOBS)


def sbatch(args, env_extra):
    env = {k: v for k, v in os.environ.items() if k not in SLURM_STRIP}
    env.update(env_extra)
    r = subprocess.run(["sbatch", "--parsable"] + args, cwd=HERE, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("sbatch failed: %s %s" % (args, r.stderr))
    jid = r.stdout.strip().split(";")[0]
    print("  %s  %s  %s" % (jid, " ".join(args), env_extra))
    return jid


def cmd_decode():
    jobs = load_jobs()
    for od, drv, td in [(a[0], a[1], A_TRAINDIR) for a in A_DECODES] + C_DECODES:
        if os.path.exists(os.path.join(HERE, od)):
            sys.exit("%s exists; refusing to decode over it" % od)
        if not os.path.isdir(os.path.join(HERE, td)):
            sys.exit("missing trainDir %s" % td)
    os.makedirs(OUT, exist_ok=True)
    # priority 1: A (submitted first)
    for od, drv, coords, chroms in A_DECODES:
        d = sbatch(["--job-name=onA_D_%s" % tag(od), "--array=0-39", "--mem=24G", "--exclude=" + EXCLUDE,
                    "sbatch_genome_decode.sh"],
                   dict(DRIVER=drv, TRAINDIR=A_TRAINDIR, OUTDIR=od, NTASK="40", COORDS=coords))
        c = sbatch(["--job-name=onA_C_%s" % tag(od), "--array=0-%d" % (len(chroms) - 1),
                    "--dependency=afterok:%s" % d, "--kill-on-invalid-dep=yes", "--exclude=" + EXCLUDE,
                    "sbatch_tw_count.sh"],
                   dict(OUTDIR=od, TAG=tag(od), CHROMS=" ".join(chroms), COUNTS_ROOT="overnight"))
        jobs.setdefault("A", {})[od] = dict(decode=d, count=c, driver=drv, traindir=A_TRAINDIR, coords=coords)
        save_jobs(jobs)
    # priority 2: C
    for od, drv, td in C_DECODES:
        d = sbatch(["--job-name=onC_D_%s" % tag(od), "--array=0-47", "--mem=48G", "--exclude=" + EXCLUDE,
                    "sbatch_genome_decode.sh"],
                   dict(DRIVER=drv, TRAINDIR=td, OUTDIR=od, NTASK="48", COORDS="coord_genome_full.tsv"))
        c = sbatch(["--job-name=onC_C_%s" % tag(od), "--array=0-15", "--mem=64G",
                    "--dependency=afterok:%s" % d, "--kill-on-invalid-dep=yes", "--exclude=" + EXCLUDE,
                    "sbatch_tw_count.sh"],
                   dict(OUTDIR=od, TAG=tag(od), CHROMS=" ".join(GENOME_CHROMS), COUNTS_ROOT="overnight"))
        jobs.setdefault("C", {})[od] = dict(decode=d, count=c, driver=drv, traindir=td,
                                            coords="coord_genome_full.tsv")
        save_jobs(jobs)


def submit_chereji(label, decode, dep=None):
    """score_robocop.py on chrXIV only (Chereji +1/-1 nucleosomes), as layerXIV_scores did (200G)."""
    args = ["--job-name=onNUC_%s" % label, "--mem=200G", "--cpus-per-task=2", "--time=3:00:00"] \
        + (["--dependency=afterany:%s" % dep] if dep else []) + ["sbatch_overnight_py.sh"]
    j = sbatch(args, dict(PYARGS="score_robocop.py %s --regions overnight/regions_chrXIV.tsv --label %s "
                               "--out overnight/chereji_chrXIV_%s.json" % (decode, label, label)))
    jobs = load_jobs()
    jobs.setdefault("chereji", {})[label] = j
    save_jobs(jobs)
    return j


def cmd_score():
    jobs = load_jobs()
    a_counts = [v["count"] for v in jobs["A"].values()]
    c_counts = [v["count"] for v in jobs["C"].values()]
    sa = sbatch(["--job-name=onA_S", "--dependency=afterany:" + ":".join(a_counts), "--mem=16G",
                 "--time=1:00:00", "sbatch_overnight_py.sh"], dict(PYARGS="overnight_score.py A"))
    sc = sbatch(["--job-name=onC_S", "--dependency=afterany:" + ":".join(c_counts), "--mem=16G",
                 "--time=2:00:00", "sbatch_overnight_py.sh"], dict(PYARGS="overnight_score.py C"))
    ss = sbatch(["--job-name=onSUM", "--dependency=afterany:%s:%s" % (sa, sc), "--mem=8G",
                 "--time=0:30:00", "sbatch_overnight_py.sh"], dict(PYARGS="overnight_summary.py"))
    jobs["score"] = dict(A=sa, C=sc, summary=ss)
    save_jobs(jobs)


def cmd_watch_b(run, every_min):
    """Runs inside a Slurm job. If <run> is STOPPED and its holdout validate job is known, submit
    the summary with afterany on it and exit; otherwise resubmit this watcher to begin later."""
    st_p = os.path.join(HERE, "conc_tuning", run, "state.json")
    stop_p = os.path.join(HERE, "conc_tuning", run, "STOPPED")
    jobs = load_jobs()
    if os.path.exists(stop_p) and os.path.exists(st_p):
        st = json.load(open(st_p))
        why = open(stop_p).read().strip()
        last = max([h["iter"] for h in st["history"]], default=0)
        hv = st["jobs"].get(str(last), {}).get("holdout_validate") if last > 0 else None
        # STOPPED is written just before the final holdout is submitted: wait for its job id,
        # unless the campaign stopped on an error (then no holdout is coming).
        if hv or "error" in why or last == 0:
            deps = [hv] if hv else []
            ch = None
            dec = "robocop_chrXIV_chrII_tw_%s_%02d" % (run, last)
            if os.path.isdir(os.path.join(HERE, dec)) and os.path.exists(os.path.join(HERE, dec, "config.ini")):
                ch = submit_chereji("tw_%s_%02d" % (run, last), dec)
                deps.append(ch)
            args = ["--job-name=onSUM_%s" % run, "--mem=8G", "--time=0:30:00"] \
                + (["--dependency=afterany:%s" % ":".join(deps)] if deps else []) + ["sbatch_overnight_py.sh"]
            j = sbatch(args, dict(PYARGS="overnight_summary.py"))
            jobs = load_jobs()
            jobs.setdefault("B_refresh", {})[run] = dict(summary=j, after_holdout_validate=hv, chereji=ch, stopped=why)
            save_jobs(jobs)
            return
    import time
    if time.time() > time.mktime(time.strptime(WATCH_DEADLINE, "%Y-%m-%d %H:%M")):
        print("watcher for %s past its deadline %s; not resubmitting" % (run, WATCH_DEADLINE))
        return
    j = sbatch(["--job-name=onW_%s" % run, "--begin=now+%dminutes" % every_min, "--mem=1G",
                "--time=0:10:00", "sbatch_overnight_py.sh"],
               dict(PYARGS="overnight_launch.py watch-b --run %s --every %d" % (run, every_min)))
    jobs = load_jobs()
    jobs.setdefault("B_watch", {})[run] = j
    save_jobs(jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["decode", "score", "watch-b", "chereji"])
    ap.add_argument("--label")
    ap.add_argument("--decode")
    ap.add_argument("--run")
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "decode":
        cmd_decode()
    elif a.cmd == "score":
        cmd_score()
    elif a.cmd == "chereji":
        submit_chereji(a.label, a.decode)
    else:
        cmd_watch_b(a.run, a.every)


if __name__ == "__main__":
    main()
