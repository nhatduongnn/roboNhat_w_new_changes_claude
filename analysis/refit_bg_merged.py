"""Re-fit the Fiber-seq background rate on the MERGED pileup.

The shipped inputs/bg_params.pkl was fitted on 03202025_barcode01 alone (the commented
pipeline block at the bottom of abf1_reb1_dms_parameter_Fiber-seq_w_binom.py), while every
TF parameter in all_TFs_1000pealVal_params_pseudo.pkl was fitted on the merged 7-barcode
pileup (make_params_pm50.PILEUP). Background and foreground therefore come from different
amounts of data. This re-runs the IDENTICAL fit -- same computeLinkers definition, same
compute_Fiber_seq_background call, same binomial estimator -- on the merged pileup.

Reuses the generator's own functions via fiber_params_lib (importing the generator
directly would re-run its pipeline and clobber the params pkl).

Writes inputs/bg_params_merged.pkl. Does NOT touch inputs/bg_params.pkl.
"""
import pickle

import pandas as pd

import fiber_params_lib
from make_params_pm50 import PILEUP

NUCFILE = "inputs/Chereji_2018_+1_-1_nucs.bed"
FASTA = "inputs/sacCer3_genome.fa"
OUT = "inputs/bg_params_merged.pkl"

ns = fiber_params_lib.load(verbose=False)

# NOTE: pass computeLinkers' output straight through. It yields dicts keyed
# chrm/start/stop, which is exactly what compute_Fiber_seq_background renames to
# Chromosome/Start/End. The commented pipeline block in the generator renames them to
# chr/start/end first -- that block is stale and would raise inside pr.PyRanges.
segments = ns["computeLinkers"](NUCFILE)
print("linker segments: %d" % len(segments), flush=True)

print("loading merged pileup ...", flush=True)
df = pd.read_csv(PILEUP, sep="\t", header=None)
split = df[9].str.split(" ", expand=True)
split.columns = [i for i in range(9, 9 + split.shape[1])]
df = pd.concat([df.drop(columns=[9]), split], axis=1)
df[11] = df[11].astype(int)
df[9] = df[9].astype(int)
print("pileup rows: %d" % len(df), flush=True)

bg = ns["compute_Fiber_seq_background"](df, segments, FASTA,
                                        dist="binomial", successes_col=11, trials_col=9)
print("\nREFIT background (merged pileup):")
print("  watson A = %.6f" % bg["p"]["watson_signal"]["A"][0])
print("  crick  A = %.6f" % bg["p"]["crick_signal"]["A"][0])

old = pickle.load(open("inputs/bg_params.pkl", "rb"))
print("SHIPPED (single-barcode):")
print("  watson A = %.6f" % old["p"]["watson_signal"]["A"][0])
print("  crick  A = %.6f" % old["p"]["crick_signal"]["A"][0])

with open(OUT, "wb") as f:
    pickle.dump(bg, f)
print("\nwrote %s" % OUT)
