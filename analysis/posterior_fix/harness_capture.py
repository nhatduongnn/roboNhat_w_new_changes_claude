"""Phase A harness, step 1: run the REAL decode pipeline of one pkgvar tree on one segment and
capture the exact inputs handed to the forward/backward recursion (POSTERIOR_FIX_PLAN.md, gates A1/A2).

    cd analysis && python posterior_fix/harness_capture.py <tree> <coords> <outDir> <out.npz> [--run-original]

<tree> is a pkgvar path (e.g. pkgvar/seq_maskoff_tune75). The tree's robocop.posterior_forward_backward
is wrapped: d['emission'] (7, n_obs, n_states) and every dshared field the recursion reads are saved to
<out.npz> BEFORE the recursion runs. For a stock (linear) tree the values are linear; for a *_logfull
tree they are logs. --run-original then lets the tree's own recursion run and write its info h5 too.
Must be run from analysis/ (the trees read inputs/*.pkl relative to cwd). Read-only on all inputs.
"""
import sys, os, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('tree'); ap.add_argument('coords'); ap.add_argument('outdir'); ap.add_argument('npz')
ap.add_argument('--traindir', default='robocop_train_tw_bv75_49')
ap.add_argument('--run-original', action='store_true')
a = ap.parse_args()

sys.path.insert(0, a.tree)
import robocop
import run_robocop
assert os.path.abspath(os.path.dirname(robocop.__file__)).startswith(os.path.abspath(a.tree)), robocop.__file__

_orig = robocop.posterior_forward_backward
KEYS_INT = ['n_states', 'silent_states_begin', 'n_vars', 'n_tfs', 'nuc_present', 'nuc_start', 'nuc_len', 'timepoints']
KEYS_ARR = ['initial_probs', 'end_probs', 'transition_matrix', 'tf_starts', 'tf_lens']
captured = []

def capture(d, segment, dshared):
    k = 'segment_' + str(segment) + '/'
    n_obs = int(dshared['info_file'][k].attrs['n_obs'])
    out = dict(emission=np.array(d['emission'], dtype=np.double), n_obs=n_obs, segment=segment,
               robocopC=str(dshared['robocopC']), tree=os.path.abspath(a.tree),
               log_emission_env=str(os.environ.get('ROBOCOP_LOG_EMISSION')))
    for key in KEYS_INT: out[key] = int(dshared[key])
    for key in KEYS_ARR: out[key] = np.array(dshared[key])
    np.savez(a.npz, **out)
    print("captured segment %s -> %s  emission %s  robocopC=%s" % (segment, a.npz, out['emission'].shape, out['robocopC']))
    sys.stdout.flush()
    captured.append(segment)
    if a.run_original:
        _orig(d, segment, dshared)
    else:
        # the stored log_likelihood attr must exist for downstream readers of this throwaway h5
        dshared['info_file']['segment_' + str(segment)].attrs['log_likelihood'] = np.nan

robocop.posterior_forward_backward = capture
run_robocop.run_robocop_without_em(a.coords, a.traindir, a.outdir, idx=0, total=1)
assert captured == [0], captured
print("done")
