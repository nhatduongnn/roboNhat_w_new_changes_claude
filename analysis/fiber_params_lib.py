"""Safe access to the functions in abf1_reb1_dms_parameter_Fiber-seq_w_binom.py.

That file cannot simply be imported for two reasons:

  1. Its module name contains hyphens, so `import` will not resolve it.
  2. More seriously, its bottom ~20 lines are TOP-LEVEL pipeline statements --
     `a = compute_Fiber_seq_TFPhisMus(...)` followed by a pickle.dump that
     OVERWRITES inputs/all_TFs_1000pealVal_params_pseudo.pkl. Importing it would
     re-run the whole parameter fit and clobber the current inputs.

So this loader parses the file and execs ONLY its imports and function
definitions, dropping every top-level Expr/Assign/With. The functions are
therefore byte-identical to the originals -- no copy-paste drift -- while none of
the pipeline runs.

Needs the robocop-2024 env (rpy2) with R_HOME set:
    conda activate robocop-2024
    export R_HOME=/home/users/nd141/miniconda3/envs/robocop-2024/lib/R
"""
import ast
import os
import sys

GENERATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "abf1_reb1_dms_parameter_Fiber-seq_w_binom.py")

# functions the pipeline statements would have called -- listed so a future edit
# to the generator that removes one fails loudly here instead of silently later
REQUIRED = ("compute_Fiber_seq_background", "computeLinkers", "fit_binomial_parameters",
            "combine_motif_counts_binom", "add_pseudocounts_binomial",
            "compute_individual_Fiber_seq_TF_binom", "create_default_params_binomial")


def load(verbose=True):
    """exec the generator's imports + defs into a fresh namespace and return it."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
    src = open(GENERATOR).read()
    tree = ast.parse(src)
    keep, dropped = [], []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            keep.append(node)
        else:
            dropped.append(ast.unparse(node).split("\n")[0][:76])
    ns = {"__name__": "fiber_params_generator", "__file__": GENERATOR}
    exec(compile(ast.Module(body=keep, type_ignores=[]), GENERATOR, "exec"), ns)
    missing = [f for f in REQUIRED if f not in ns]
    if missing:
        raise ImportError("generator no longer defines: %s" % missing)
    if verbose:
        print("fiber_params_lib: loaded %d defs from %s" % (
            sum(isinstance(n, ast.FunctionDef) for n in keep), os.path.basename(GENERATOR)))
        print("  NOT executed (%d top-level statements): %s" % (len(dropped), "; ".join(dropped)))
    return ns


if __name__ == "__main__":
    ns = load()
    print("\navailable:", ", ".join(sorted(k for k, v in ns.items() if callable(v))))
