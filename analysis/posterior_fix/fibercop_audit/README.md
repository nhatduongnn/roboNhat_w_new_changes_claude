# fiberCOP C-audit: data for the PI / line-by-line / empirical tabs

Reproduces the C audit's C1 tabs (`robocop_c_audit/walkthrough`, bug C1 = `bward` computes silent states
from un-rescaled children) on the Fiber-seq model: same 12-state toy, sequence `AATTAA`, `n_vars = 7`,
layers 5/6 = Watson/Crick binomial m6A with the 1e-30 floor, MNase layers off. The audit tree is only read
and imported, never modified.

## Scripts

| script | writes | what it does |
|---|---|---|
| `fiber_model.py` | — | shared model: `make_model`, `SCENARIOS`, `high`, `run_lin`, `run_logfull`, `brute`, library paths |
| `gen_f_lbl.py` | `build/f_lbl.json`, `build/f_trace.tsv` | "Now" backward steps from the traced upstream algo.c (asserted bit-identical to `librobocop_prenorm.so`, and its bward to production fix B); "Fix" steps by Python replay over 7 layers (asserted bit-identical to the audit's `librobocop_fixed.so` btable); fork `algo.c` listings (fork = upstream + 2, asserted); C1 cells |
| `gen_f_d.py` | `build/f_d.json` | model and data object for the c1 scenario: labels, 7-layer emission product, transitions, and the Fiber-seq counts / per-layer factors / floored cells |
| `gen_f_emp.py` | `build/f_emp.json` | 4 scenarios × 7 methods: posteriors, log-likelihood, error vs floored and vs exact truth, downstream dbf / EM sums |
| `gen_f_dep.py` | `build/f_dep.json` | depth sweep: coverage n at positions 1-4 (31 values, 2-10000), methylated fraction 1.0 / 0.0 / 0.6, error of 5 methods vs both truths |

`build/SCHEMA.md` describes every key and what differs from the audit's JSON.

## Rebuild

```bash
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
cd /usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis/posterior_fix/fibercop_audit
export MPLBACKEND=Agg R_HOME=$CONDA_PREFIX/lib/R
python gen_f_lbl.py && python gen_f_d.py && python gen_f_emp.py && python gen_f_dep.py
```

Each takes a few seconds. Output is deterministic (two runs gave identical md5s on 2026-10-13).

## Verified results (from `build/f_emp.json`)

Max |posterior − truth| over emitting states. Left = vs the floored model (`brute` on log of the
floored emissions: what production builds); right = vs the exact model (`brute_exact`: the real data).

| scenario | current (orig) | fixB | fixed (C1 fix) | logD0 | logD1 (exact) |
|---|---|---|---|---|---|
| ordinary (25 reads) | 7.7e-15 / 1.0e-15 | 7.5e-15 / 1.0e-15 | 7.7e-15 / 1.0e-15 | 7.1e-15 / 8.9e-16 | 7.4e-15 / 8.9e-16 |
| c1 (220 reads, k=n, pos 1-4) | **1.5e5 / 1.5e5** | **0.99 / 1** | 2.8e-14 / **1** | 2.8e-14 / **1** | **1** / 9.8e-14 |
| c1x (250 reads, all 6) | **1 / 1** | **1 / 1** | **1 / 1** | 5.5e-14 / **1** | **1** / 5.4e-113 |
| rdna (4379 reads, k=0, pos 1-4) | 1.4e-14 / **1** | 1.4e-14 / **1** | 1.4e-14 / **1** | 1.4e-14 / **1** | **1** / 1.6e-98 |

Floored binomial cells: 0 / 16 / 36 / 4. Brute force enumerates 136 paths in each scenario.

C1 cells in the c1 line-by-line (`f_lbl.json` `c1_rows`): silent values below DBL_MIN before
normalize at btable row 2 (t=3) column 11 (TF-entry), and row 3 (t=2) columns 10 (CENTRAL) and 11.
Rows 4-5 then differ by propagation. (The audit's MNase example had C1 at row 3, column 10.)

Depth sweep (`f_dep.json`, frac 1.0): against the floored truth, `current` fails only at n = 210 (2.2e-5) and
220 (1.5e5) and `fixB` only at 220 (0.99); from n = 230 on, the floor makes the linear codes "right" for the
floored model again. Against the exact truth, every floored method is wrong from n = 210 on (error 1, then 0.38
from n = 749); `logD1` stays at or below 3e-12 everywhere. For frac 0.0 the exact-truth failure starts at
n = 3292; for frac 0.6 at n = 517.
