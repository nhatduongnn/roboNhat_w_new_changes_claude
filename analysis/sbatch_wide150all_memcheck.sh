#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=400G
#SBATCH --time=0:40:00
#SBATCH --output=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.out
#SBATCH --error=/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/logs/%x_%j.err

# Standalone go/no-go for wide150all (n_states 95285) BEFORE burning an hour of training.
#
# Allocates and touches the three dense square matrices robocop.py builds -- t_mat
# (float64), parents_mat and children_mat (int64) -- at the real n_states, and then
# reports what the (7, n_obs, n_states) per-segment emission tensors would additionally
# cost. It also re-runs the linear-index check against the shipped librobocop.so, which is
# the thing that actually decides feasibility.
#
# Nothing here writes to any run directory.

source /home/users/nd141/miniconda3/etc/profile.d/conda.sh
conda activate robocop-2024
set -eo pipefail
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis
export MPLBACKEND=Agg
export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R

echo "Host: $(hostname)  Start: $(date)"
free -g | head -2

python - <<'EOF'
import ctypes, os, resource, sys, time
import numpy as np

N = 95285          # wide150all n_states, read off the built meme file
N_OBS, N_SEG = 5000, 20    # 5 kb windows; 20 chrII training windows

GB = 1024.0 ** 3
def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0 / GB

print("n_states = %d" % N)
print("predicted: t_mat float64 %.1f GB, parents_mat int64 %.1f GB, "
      "children_mat int64 %.1f GB" % (N * N * 8 / GB, N * N * 8 / GB, N * N * 8 / GB))
sys.stdout.flush()

mats = []
for name, dtype in (("t_mat", np.float64), ("parents_mat", np.int64),
                    ("children_mat", np.int64)):
    t0 = time.time()
    a = np.empty((N, N), dtype=dtype)
    # np.zeros is lazy -- calloc hands back untouched pages and RSS stays at 0, which
    # proves nothing. fill() writes every page, so the RSS below is the real footprint
    # and an over-commit would OOM here rather than an hour into training.
    a.fill(0)
    mats.append(a)
    print("  allocated+touched %-13s %s %s   peak RSS %.1f GB  (%.0fs)"
          % (name, a.shape, a.dtype, rss(), time.time() - t0))
    sys.stdout.flush()

print("\nTHREE SQUARE MATRICES OK -- peak RSS %.1f GB" % rss())

emis = 7 * N_OBS * N * 8 / GB
tables = 3 * N_OBS * N * 8 / GB
print("\nstill to come, on top of the above:")
print("  emission tensor (7, %d, %d) float64      %8.1f GB  per segment" % (N_OBS, N, emis))
print("  robocop_em holds all %d training segments live: %8.1f GB" % (N_SEG, N_SEG * emis))
print("  ftable+btable+p_table (%d, %d) x3        %8.1f GB  transient" % (N_OBS, N, tables))
print("  -> TRAIN peak  ~%.0f GB ; DECODE peak ~%.0f GB (1 segment at a time)"
      % (3 * N * N * 8 / GB + N_SEG * emis + tables,
         3 * N * N * 8 / GB + emis + tables))
print("  -> HMMconfig.pkl on disk ~%.0f GB (it pickles transition_matrix)" % (N * N * 8 / GB))

print("\n--- linear-index arithmetic in the shipped librobocop.so ---")
lib = ctypes.CDLL("../pkg/robocop/librobocop.so")
lib.I.argtypes = [ctypes.c_int] * 3; lib.I.restype = ctypes.c_int
lib.I3.argtypes = [ctypes.c_int] * 5; lib.I3.restype = ctypes.c_int
bad = 0
for n in (9605, 10685, N):
    got, want = lib.I(n - 1, n - 1, n), (n - 1) * n + (n - 1)
    ok = got == want
    bad += not ok
    print("  I(n-1,n-1,n)  n_states=%-6d got %-14d want %-14d %s"
          % (n, got, want, "ok" if ok else "*** OVERFLOW ***"))
    got = lib.I3(6, N_OBS - 1, N_OBS, n - 1, n)
    want = 6 * N_OBS * n + (N_OBS - 1) * n + (n - 1)
    ok = got == want
    bad += not ok
    print("  I3(6,n_obs-1,..,n-1,n)      =%-6d got %-14d want %-14d %s"
          % (n, got, want, "ok" if ok else "*** OVERFLOW ***"))
print("\nVERDICT: %s" % ("index arithmetic is safe at every n_states tested"
                         if not bad else
                         "%d index computation(s) OVERFLOW -- librobocop.so uses 32-bit "
                         "int for linearized indices, so any n_states >= 46341 writes out "
                         "of bounds. wide150all CANNOT run correctly against this .so." % bad))
EOF
echo "End: $(date)"
