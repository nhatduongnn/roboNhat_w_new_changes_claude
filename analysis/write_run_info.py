#!/usr/bin/env python
"""Write a slim provenance stamp, <outdir>/RUN_INFO.json, into a decode directory.

Only PROVENANCE facts are recorded (what produced this decode): original outdir name, driver,
the pkgvar tree the driver imports, the real trainDir, coords file, campaign/round/role, git
commit + dirty flag, date, SLURM job id and where the facts came from. Derived model attributes
(layers, mask, phi, lambda, pads, decoy, ...) are NOT stored here: viewer_site/build_run_matrix.py
derives them from (tree, trainDir), so there is one derivation path.

Why: a decode's config.ini is copied from whichever trainDir was cloned, so it is not a record of
the model (367 decodes claim robocop_train_fiberonly), and the log link to a decode breaks when the
directory is renamed.

    python write_run_info.py --outdir robocop_chrI_x --driver run_split_revfix_seq_maskoff.py \
        --traindir robocop_train_fiberonly [--coords coord_erv46.tsv] \
        [--campaign bd14 --round 3 --role tune] [--source decode-time] [--force]

Campaign fields default to $TW_RUN / $TW_ITER / $TW_ROLE (exported by tune_w.py to decode jobs).
Without --force an existing RUN_INFO.json is left alone (exit 0).
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def driver_tree(driver, argv6=None):
    """The package tree a driver puts first on sys.path, as a path relative to analysis/.

    Handles literal inserts ('pkgvar/seq_maskoff/', '../pkg/') and the two variant drivers
    that insert 'pkgvar/%s/' % variant, where variant = argv[6] or os.environ.get(VAR, default).
    Returns (tree, how) or (None, reason).
    """
    p = driver if os.path.isabs(driver) else os.path.join(HERE, driver)
    try:
        src = open(p).read()
    except OSError as e:
        return None, "driver unreadable: %s" % e
    m = re.search(r"""^sys\.path\.insert\(0,\s*['"]([^'"%]+)['"]\s*\)""", src, re.M)
    if m:
        return os.path.normpath(m.group(1)), "literal sys.path.insert"
    m = re.search(r"""^sys\.path\.insert\(0,\s*['"]pkgvar/%s/?['"]\s*%\s*variant\s*\)""", src, re.M)
    if m:
        if argv6:
            return os.path.normpath("pkgvar/%s" % argv6), "variant from argv[6]"
        e = re.search(r"""os\.environ\.get\(\s*['"](\w+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)""", src)
        if e:
            var, default = e.group(1), e.group(2)
            v = os.environ.get(var)
            if v:
                return os.path.normpath("pkgvar/%s" % v), "variant from $%s" % var
            return os.path.normpath("pkgvar/%s" % default), "variant default (no $%s)" % var
    return None, "no recognisable sys.path.insert in driver"


def git_info():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=60).stdout.strip() or None
        st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                            capture_output=True, text=True, timeout=120)
        dirty = bool(st.stdout.strip()) if st.returncode == 0 else None
        return head, dirty
    except Exception:
        return None, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--traindir", required=True)
    ap.add_argument("--coords", default=None)
    ap.add_argument("--variant", default=None, help="argv[6] of a variant driver, if one was passed")
    ap.add_argument("--campaign", default=os.environ.get("TW_RUN") or None)
    ap.add_argument("--round", type=int, default=int(os.environ["TW_ITER"]) if os.environ.get("TW_ITER") else None)
    ap.add_argument("--role", default=os.environ.get("TW_ROLE") or None, choices=[None, "tune", "holdout"])
    ap.add_argument("--source", default="decode-time",
                    help="decode-time | backfill:state | backfill:log | backfill:hand")
    ap.add_argument("--original-outdir", default=None,
                    help="name the decode was written under, if it has since been renamed")
    ap.add_argument("--extra", default=None, help="JSON object merged in (backfill notes)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    od = a.outdir if os.path.isabs(a.outdir) else os.path.join(HERE, a.outdir)
    od = od.rstrip("/")
    out = os.path.join(od, "RUN_INFO.json")
    if not os.path.isdir(od):
        print("write_run_info: no such outdir %s" % od, file=sys.stderr)
        return 2
    if os.path.exists(out) and not a.force:
        print("write_run_info: %s exists; leaving it (use --force)" % out)
        return 0
    tree, how = driver_tree(a.driver, a.variant)
    head, dirty = git_info()
    td = a.traindir.rstrip("/")
    info = dict(
        schema=1,
        outdir=os.path.basename(od),
        original_outdir=a.original_outdir or os.path.basename(od),
        driver=os.path.basename(a.driver),
        pkg_tree=tree,
        pkg_tree_how=how,
        traindir=os.path.relpath(os.path.join(HERE, td), HERE) if not os.path.isabs(td) else td,
        traindir_exists=os.path.isfile(os.path.join(HERE, td, "HMMconfig.pkl")),
        coords=a.coords,
        campaign=a.campaign,
        round=a.round,
        role=a.role,
        git_head=head,
        git_dirty=dirty,
        written=datetime.datetime.now().isoformat(timespec="seconds"),
        slurm_job_id=(os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"))
        if a.source == "decode-time" else None,
        source=a.source,
    )
    if a.extra:
        info.update(json.loads(a.extra))
    tmp = out + ".tmp%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump(info, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, out)
    print("write_run_info: wrote %s (tree %s via %s)" % (out, tree, how))
    return 0


if __name__ == "__main__":
    sys.exit(main())
