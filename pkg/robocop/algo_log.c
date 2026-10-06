/* algo_log.c -- log-space reference decoder (POSTERIOR_FIX_PLAN.md, Phase A), 2026-10-06.

   A strict SUPERSET of algo.c: nothing in algo.c or bc.c is changed, so librobocop.so stays
   reproducible from source and librobocop_logfull.so (bc.c + algo.c + algo_log.c) can run the
   linear and the log path in one process.

   Conventions (all mirror algo.c literally, only the arithmetic moves to log space):
     - log_emis is 2-D (n_obs, n_states), PRE-COLLAPSED over the 7 emission channels by the caller
       (sum of per-channel logs). Silent columns [silent_states_begin, n_states) are 0.0 = log 1.
       -INFINITY means "forbidden" (pwm_emission zero, tune75/ABF1 hard mask).
     - transition_mat, initial_probs, end_probs stay LINEAR; their logs are taken here.
       Only edges listed in parents/children (transition != 0.0) are visited, so log() is finite.
     - log_ftable rows are in position order; log_btable rows are stored REVERSED
       (row r <-> position n_obs-1-r), exactly as bward does (algo.c:131-207); read with Ir().
     - silent-state order is preserved literally: forward ASCENDING, backward DESCENDING, with
       in-place accumulation, so a silent->silent edge sees exactly the partial value the linear
       code would have seen (154 silent->silent edges make this a correctness requirement).
     - log_normalize reproduces normalize()'s asymmetry: offset = logsumexp over [0, idx) only,
       subtracted from the FULL row [0, ncol). Only the + DBL_MIN is dropped. The offset is
       returned and stored where scaling_factors lived, so sum(offsets) == calc_sr's return.
     - every exported entry point takes a trailing `int abi` that must equal ROBOCOP_LOG_ABI and
       is checked before ANY pointer is dereferenced. Each returns ROBOCOP_LOG_ABI on success and
       a negative code on failure; the Python side sets restype = c_int and asserts the value.
     - C initialises log_ftable / log_btable to -INFINITY itself (0.0 is log 1 -- the worst wrong
       identity); Python also passes np.full(-inf), redundantly.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bc.h"

#define ROBOCOP_LOG_ABI 20261005

static int abi_bad(const char *fn, int abi)
{
    if (abi != ROBOCOP_LOG_ABI) {
        fprintf(stderr, "%s: abi=%d, expected %d -- ctypes argument list does not match "
                        "algo_log.c; refusing to touch any pointer\n", fn, abi, ROBOCOP_LOG_ABI);
        return 1;
    }
    return 0;
}

/* log(exp(a) + exp(b)), exact at -INFINITY. */
double log_add(double a, double b)
{
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    if (a >= b) return a + log1p(exp(b - a));
    return b + log1p(exp(a - b));
}

/* Log-space normalize(): offset = logsumexp(row[0:idx)) (max-subtracted, two pass), then the
   whole row [0:ncol) has the offset subtracted. Returns the offset (log of normalize()'s sum,
   minus the DBL_MIN terms). Returns -INFINITY, leaving the row untouched, if every entry of
   [0:idx) is -INFINITY. */
double log_normalize(double *table, int row, int ncol, int idx)
{
    int i;
    double m = -INFINITY, s = 0.0, off;

    for (i = 0; i < idx; i++)
        if (table[I(row, i, ncol)] > m) m = table[I(row, i, ncol)];
    if (m == -INFINITY) return -INFINITY;

    for (i = 0; i < idx; i++)
        s += exp(table[I(row, i, ncol)] - m);
    off = m + log(s);

    for (i = 0; i < ncol; i++)
        table[I(row, i, ncol)] -= off;      /* -INFINITY - finite stays -INFINITY */

    return off;
}

/* log of the transition weight of every listed edge, laid out compactly per state:
   lw[off[j] + k] = log(transition_mat[from, to]) for the k-th neighbour of state j.
   parents: edge (nbr[j][k] -> j); children: edge (j -> nbr[j][k]). */
static double *edge_log_weights(const double *transition_mat, const long *nbr_mat, const long *n_nbr,
                                int n_states, int is_parent, long **off_out)
{
    int j, k;
    long tot = 0, *off;
    double *lw;

    off = (long *) ALLOC(sizeof(long) * (size_t) n_states);
    for (j = 0; j < n_states; j++) { off[j] = tot; tot += n_nbr[j]; }
    lw = (double *) ALLOC(sizeof(double) * (size_t) (tot > 0 ? tot : 1));
    for (j = 0; j < n_states; j++) {
        for (k = 0; k < n_nbr[j]; k++) {
            int o = (int) nbr_mat[I(j, k, n_states)];
            double t = is_parent ? transition_mat[I(o, j, n_states)] : transition_mat[I(j, o, n_states)];
            lw[off[j] + k] = log(t);
        }
    }
    *off_out = off;
    return lw;
}

int fward_log(
    // input (transition/initial/end LINEAR; log_emis 2-D, log, pre-collapsed)
    double *initial_probs,
    double *transition_mat, double *log_emis,
    double *end_probs,
    int n_states, int silent_states_begin, int n_obs,
    long *parents_mat, long *n_parents,
    // output
    double *log_ftable, double *log_scaling_factors,
    int abi
    )
{
    int i, j, k, rc = ROBOCOP_LOG_ABI;
    long *poff;
    double *plw;
    double off;

    if (abi_bad("fward_log", abi)) return -1;
    if (silent_states_begin <= 0 || silent_states_begin > n_states) {
        fprintf(stderr, "fward_log: silent_states_begin=%d out of range (n_states=%d)\n",
                silent_states_begin, n_states);
        return -1;
    }

    for (i = 0; i < n_obs * n_states; i++) log_ftable[i] = -INFINITY;
    plw = edge_log_weights(transition_mat, parents_mat, n_parents, n_states, 1, &poff);

    /* position 0 (algo.c:56-74) */
    for (j = 0; j < silent_states_begin; j++)
        log_ftable[I(0, j, n_states)] = log(initial_probs[j]) + log_emis[I(0, j, n_states)];
    for (j = silent_states_begin; j < n_states; j++)
        for (k = 0; k < n_parents[j]; k++) {
            int parent = (int) parents_mat[I(j, k, n_states)];
            log_ftable[I(0, j, n_states)] = log_add(log_ftable[I(0, j, n_states)],
                log_ftable[I(0, parent, n_states)] + plw[poff[j] + k]);
        }
    off = log_normalize(log_ftable, 0, n_states, silent_states_begin);
    log_scaling_factors[0] = off;
    if (off == -INFINITY) { fprintf(stderr, "fward_log: no emitting mass at position 0\n"); rc = -2; }

    /* positions 1 .. n_obs-1 (algo.c:79-125); the last one also takes end_probs */
    for (i = 1; i < n_obs; i++) {
        for (j = 0; j < silent_states_begin; j++) {
            for (k = 0; k < n_parents[j]; k++) {
                int parent = (int) parents_mat[I(j, k, n_states)];
                log_ftable[I(i, j, n_states)] = log_add(log_ftable[I(i, j, n_states)],
                    log_ftable[I(i - 1, parent, n_states)] + plw[poff[j] + k]);
            }
            if (i == n_obs - 1)
                log_ftable[I(i, j, n_states)] += log(end_probs[j]);
            log_ftable[I(i, j, n_states)] += log_emis[I(i, j, n_states)];
        }
        /* silent states, ASCENDING (algo.c:92, :117) */
        for (j = silent_states_begin; j < n_states; j++)
            for (k = 0; k < n_parents[j]; k++) {
                int parent = (int) parents_mat[I(j, k, n_states)];
                log_ftable[I(i, j, n_states)] = log_add(log_ftable[I(i, j, n_states)],
                    log_ftable[I(i, parent, n_states)] + plw[poff[j] + k]);
            }
        off = log_normalize(log_ftable, i, n_states, silent_states_begin);
        log_scaling_factors[i] = off;
        if (off == -INFINITY && rc > 0) {
            fprintf(stderr, "fward_log: no emitting mass at position %d\n", i);
            rc = -2;
        }
    }

    free(plw); free(poff);
    return rc;
}

int bward_log(
    double *transition_mat, double *log_emis,
    double *end_probs,
    int n_states, int silent_states_begin, int n_obs,
    long *children_mat, long *n_children,
    // output, rows REVERSED like bward
    double *log_btable, double *log_scaling_factors,
    int abi
    )
{
    int i, j, k, rc = ROBOCOP_LOG_ABI;
    long *coff;
    double *clw;
    double off;

    if (abi_bad("bward_log", abi)) return -1;
    if (silent_states_begin <= 0 || silent_states_begin > n_states) {
        fprintf(stderr, "bward_log: silent_states_begin=%d out of range (n_states=%d)\n",
                silent_states_begin, n_states);
        return -1;
    }

    for (i = 0; i < n_obs * n_states; i++) log_btable[i] = -INFINITY;
    clw = edge_log_weights(transition_mat, children_mat, n_children, n_states, 0, &coff);

    /* reversed row 0 = position n_obs-1 (algo.c:151-168) */
    for (j = 0; j < silent_states_begin; j++)
        log_btable[I(0, j, n_states)] = log(end_probs[j]);
    /* silent states, DESCENDING (algo.c:156) */
    for (j = n_states - 1; j > silent_states_begin - 1; j--)
        for (k = 0; k < n_children[j]; k++) {
            int child = (int) children_mat[I(j, k, n_states)];
            log_btable[I(0, j, n_states)] = log_add(log_btable[I(0, j, n_states)],
                log_btable[I(0, child, n_states)] + clw[coff[j] + k]
                + log_emis[I(n_obs - 1, child, n_states)]);
        }
    off = log_normalize(log_btable, 0, n_states, silent_states_begin);
    log_scaling_factors[n_obs - 1] = off;
    if (off == -INFINITY) { fprintf(stderr, "bward_log: no emitting mass at position %d\n", n_obs - 1); rc = -2; }

    /* reversed rows 1 .. n_obs-1 (algo.c:171-204) */
    for (i = 1; i < n_obs; i++) {
        for (j = 0; j < silent_states_begin; j++)
            for (k = 0; k < n_children[j]; k++) {
                int child = (int) children_mat[I(j, k, n_states)];
                log_btable[I(i, j, n_states)] = log_add(log_btable[I(i, j, n_states)],
                    log_btable[I(i - 1, child, n_states)] + clw[coff[j] + k]
                    + log_emis[I(n_obs - i, child, n_states)]);
            }
        /* silent states, DESCENDING (algo.c:187) */
        for (j = n_states - 1; j > silent_states_begin - 1; j--)
            for (k = 0; k < n_children[j]; k++) {
                int child = (int) children_mat[I(j, k, n_states)];
                log_btable[I(i, j, n_states)] = log_add(log_btable[I(i, j, n_states)],
                    log_btable[I(i, child, n_states)] + clw[coff[j] + k]
                    + log_emis[I(n_obs - i - 1, child, n_states)]);
            }
        off = log_normalize(log_btable, i, n_states, silent_states_begin);
        log_scaling_factors[n_obs - 1 - i] = off;
        if (off == -INFINITY && rc > 0) {
            fprintf(stderr, "bward_log: no emitting mass at position %d\n", n_obs - 1 - i);
            rc = -2;
        }
    }

    free(clw); free(coff);
    return rc;
}

/* Posterior from log tables. Exponentiates internally and writes LINEAR values, normalised per
   position over the EMITTING states [0, silent_states_begin); silent columns are written 0.0.
   So the h5 contract (linear, row-normalised, float64) is unchanged. A row whose emitting
   log-posterior is all -INFINITY is written as zeros and makes the call return -2. */
int posterior_decoding_log(
    double *log_ftable, double *log_btable,
    int n_states, int n_obs, int silent_states_begin,
    double *posterior_table,
    int abi
    )
{
    int i, j, nbad = 0;
    double m, s, lse, v;

    if (abi_bad("posterior_decoding_log", abi)) return -1;
    if (silent_states_begin <= 0 || silent_states_begin > n_states) {
        fprintf(stderr, "posterior_decoding_log: silent_states_begin=%d out of range (n_states=%d)\n",
                silent_states_begin, n_states);
        return -1;
    }

    for (i = 0; i < n_obs; i++) {
        m = -INFINITY;
        for (j = 0; j < silent_states_begin; j++) {
            v = log_ftable[I(i, j, n_states)] + log_btable[Ir(i, j, n_obs, n_states)];
            posterior_table[I(i, j, n_states)] = v;      /* scratch: log value */
            if (v > m) m = v;
        }
        for (j = silent_states_begin; j < n_states; j++)
            posterior_table[I(i, j, n_states)] = 0.0;
        if (m == -INFINITY) {
            for (j = 0; j < silent_states_begin; j++) posterior_table[I(i, j, n_states)] = 0.0;
            nbad++;
            continue;
        }
        s = 0.0;
        for (j = 0; j < silent_states_begin; j++)
            s += exp(posterior_table[I(i, j, n_states)] - m);
        lse = m + log(s);
        for (j = 0; j < silent_states_begin; j++)
            posterior_table[I(i, j, n_states)] = exp(posterior_table[I(i, j, n_states)] - lse);
    }

    if (nbad) {
        fprintf(stderr, "posterior_decoding_log: %d of %d positions have no emitting mass\n", nbad, n_obs);
        return -2;
    }
    return ROBOCOP_LOG_ABI;
}

/* Linear twin of posterior_decoding_log, same output convention (emitting columns normalised
   over [0, silent_states_begin), silent columns 0.0). Takes the LINEAR f/b tables from
   fward/bward (algo.c). Exists so the linear and the log path can be compared column for column
   in one process (gate A2); on emitting columns it equals algo.c's posterior_decoding (fix B). */
int posterior_decoding_normlin(
    double *f_table, double *b_table,
    int n_states, int n_obs, int silent_states_begin,
    double *posterior_table,
    int abi
    )
{
    int i, j, nbad = 0;
    double rowsum;

    if (abi_bad("posterior_decoding_normlin", abi)) return -1;
    if (silent_states_begin <= 0 || silent_states_begin > n_states) {
        fprintf(stderr, "posterior_decoding_normlin: silent_states_begin=%d out of range (n_states=%d)\n",
                silent_states_begin, n_states);
        return -1;
    }

    for (i = 0; i < n_obs; i++) {
        rowsum = 0.0;
        for (j = 0; j < silent_states_begin; j++) {
            posterior_table[I(i, j, n_states)] = b_table[Ir(i, j, n_obs, n_states)] * f_table[I(i, j, n_states)];
            rowsum += posterior_table[I(i, j, n_states)];
        }
        for (j = silent_states_begin; j < n_states; j++)
            posterior_table[I(i, j, n_states)] = 0.0;
        if (rowsum > 0.0) {
            for (j = 0; j < silent_states_begin; j++) posterior_table[I(i, j, n_states)] /= rowsum;
        } else {
            for (j = 0; j < silent_states_begin; j++) posterior_table[I(i, j, n_states)] = 0.0;
            nbad++;
        }
    }
    if (nbad) {
        fprintf(stderr, "posterior_decoding_normlin: %d of %d positions have no emitting mass\n", nbad, n_obs);
        return -2;
    }
    return ROBOCOP_LOG_ABI;
}

/* Log twin of bc.c's build_emission_mat_from_pwm (bc.c:110-131), which is left untouched for
   librobocop.so. Emitting columns get log(pwm) (log(0) = -INFINITY = forbidden); silent columns
   get 0.0 = log 1 (bc.c:126 writes 1.0). */
int build_log_emission_mat_from_pwm(
    long *sequence, double *pwm,
    int n_obs, int n_states, int silent_states_begin, int alphabet_length,
    double *log_emission_mat,
    int abi
    )
{
    int i, j;

    if (abi_bad("build_log_emission_mat_from_pwm", abi)) return -1;

    for (i = 0; i < n_obs; i++) {
        for (j = 0; j < silent_states_begin; j++)
            log_emission_mat[I(i, j, n_states)] = log(pwm[j * alphabet_length + sequence[i]]);
        for (j = silent_states_begin; j < n_states; j++)
            log_emission_mat[I(i, j, n_states)] = 0.0;
    }
    return ROBOCOP_LOG_ABI;
}
