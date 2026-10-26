#ifndef _bc_h
#define _bc_h

int I3(int, int, int, int, int);
int I(int row, int col, int ncol);
int Ir(int, int, int, int);
void *ALLOC(size_t);
void construct_transition_matrix(int, int, int, long*, long*, int, int, double*, char*);
int build_emission_mat_from_pwm(long *, double*, int, int, int, int, double*);
void set_initial_probs( long*, long*, int, int, int, int, int, int, double*, double*);
void find_parents_and_children(long*, long*, long*, long*, int, int, double*);
void print2dmatrix(double*, int, int);

/* algo_log.c -- log-space reference decoder (POSTERIOR_FIX_PLAN.md Phase A). Built only into
   librobocop_logfull.so (gccCompile_logfull); librobocop.so (gccCompile) does not compile it.
   Every int-returning entry point takes a trailing abi == 20261005 and returns it on success. */
double log_add(double, double);
double log_normalize(double*, int, int, int);
int fward_log(double*, double*, double*, double*, int, int, int, long*, long*, double*, double*, int);
int bward_log(double*, double*, double*, int, int, int, long*, long*, double*, double*, int);
int posterior_decoding_log(double*, double*, int, int, int, double*, int);
int posterior_decoding_normlin(double*, double*, int, int, int, double*, int);
int build_log_emission_mat_from_pwm(long*, double*, int, int, int, int, double*, int);

#endif
