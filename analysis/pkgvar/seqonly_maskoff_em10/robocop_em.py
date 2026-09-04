import os
import sys
import robocop
from robocop.utils.readWriteOps import *
from robocop.utils.robocopExtras import *
import robocop.utils.parameterize as parameterize
import numpy as np
import pickle
import re
from configparser import ConfigParser
import robocop.utils.getReads as getReads
import robocop.utils.readData as readData
import pandas
import gc
import h5py
import random
random.seed(9)

# create posterior table for each segment
def createInstances(tf_prob, dbf_conc, coords, pwm, cshared, tmpDir, info_file, fasta_file, nucleosome_file, nucleotide_sequence, mnaseParams, tech):
    segments = len(coords)
    dshared = {} 
    d_segments = {}
    dshared["robocopC"] = cshared 
    robocop.createSharedDictionary(dshared, fasta_file, nucleosome_file, tf_prob, dbf_conc['background'], dbf_conc['nucleosome'], pwm, tmpDir, info_file, nucleotide_sequence)
    for t in range(segments):
        d_segments[t] = createInstance((t, dshared, coords.iloc[t]['chr'], coords.iloc[t]['start'], coords.iloc[t]['end']))
        # d = createInstance((t, dshared, coords.iloc[t]['chr'], coords.iloc[t]['start'], coords.iloc[t]['end']))
    if mnaseParams != None:
        for s in range(segments):
            updateMNaseEMMatNB((d_segments[s], s, dshared, mnaseParams, tech))
            # updateMNaseEMMatNB((d, s, dshared, mnaseParams, tech))
    for t in range(segments):
        posterior_forward_backward_wrapper((d_segments[t], t, dshared))
        # posterior_forward_backward_wrapper((d, t, dshared))
    gc.collect()
    return dshared, d_segments
    
def runROBOCOP_EM(coordFile, config, outDir, tmpDir, info_file_name, mnaseFile, dnaseFiles = ""):

    info_file = h5py.File(info_file_name, mode = 'w') 
    fragRangeLong = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeLong"))])
    fragRangeShort = tuple([int(x) for x in re.findall(r'\d+', config.get("main", "fragRangeShort"))])
    fragRange = (fragRangeLong, fragRangeShort)
    nucFile = config.get("main", "nucFile")
    mnaseFiles = mnaseFile 
    cshared = config.get("main", "cshared")
    tech = config.get("main", "tech")
    tech2 = config.get("main", "tech2")
    if tech2 == 'Fiber':
        modkitFile = config.get("main", "pileupFile")
        nucleotide = config.get("main", "nucleotide")
        # Load Modkit file only once outside loop
        modified_bases_df = pandas.read_csv(modkitFile, sep='\t', header=None)
        # Split the 9th column into multiple columns (if following previous code pattern)
        split_columns = modified_bases_df[9].str.split(' ', expand=True)
        split_columns.columns = [i for i in range(9,9+split_columns.shape[1])]
        modified_bases_df = pandas.concat([modified_bases_df.drop(columns=[9]), split_columns], axis=1)

    # chromosome segments in pandas data frame
    coords = pandas.read_csv(coordFile, sep = "\t")

    # select suubset for training 
    if len(coords) > 500: 
        idx = list(range(len(coords)))
        random.shuffle(idx)
        idx = idx[:500]
        coords = coords.iloc[idx]
        coords = coords.reset_index()
        
    # dbf weights initlaized using KD values
    dbf_conc, pwm = parameterize.getDBFconc(nucFile, config.get("main", "pwmFile"), outDir)

    # read nucleotide sequence and return 1 if successful
    nucleotide_sequence = getReads.getNucSequence(nucFile, tmpDir, info_file, coords)
    # read MNase-seq midpoint counts for long and short fragments

    # readData.get2DValues(mnaseFiles, config.get('main', 'chrSizesFile'), (0, 200), tmpDir)
    # mnase_data_long, mnase_data_short = getReads.getMNaseSmoothed(tmpDir, coords, fragRange, tech = tech)

    mnase_data_long, mnase_data_short = getReads.getMNase(mnaseFiles, tmpDir, info_file, coords, fragRange, tech = tech)
    fiber_seq_data_count_meth_watson, fiber_seq_data_count_meth_crick = getReads.getFiber_seq(modified_bases_df, tmpDir, info_file, coords, nucleotide, tech = tech2)


    # make t copies of each tf_prob for every timepoint -- only 1 timepoint
    timepoints = len(coords)
    segments = len(coords)
    tf_prob = dict()
    threshold = 0
    thresholds = []
    tName = []
    # determine max prob for TFs to perform constrained EM optimization
    for i in list(dbf_conc.keys()):
        if i != 'nucleosome' and i != 'background':
            tf_prob[i] = dbf_conc[i] 
            if threshold < tf_prob[i] and i != 'unknown':
                threshold = tf_prob[i]
            if i != 'unknown': thresholds.append(tf_prob[i])
            tName.append((i, tf_prob[i]))
    # Constrained-EM ceiling on any single TF prior (except 'unknown', which adjustEM
    # skips). Upstream is mean + 2*sd of the INITIAL priors, computed once and never
    # recomputed. Over a distribution this skewed that ceiling is low -- in the first 10
    # iteration run, 28 of 154 states ended up pinned to it, Abf1 and Nhp6a among them, so
    # EM never expressed a relative preference between them. ROBOCOP_EM_CAP_SD changes the
    # multiplier; ROBOCOP_EM_CAP=off lifts it entirely (threshold 1.0 can never bind, since
    # every prior is a probability). Unset => byte-for-byte the upstream formula.
    _cap = os.environ.get('ROBOCOP_EM_CAP', '').strip().lower()
    if _cap in ('off', 'none', 'inf'):
        threshold = 1.0
        print('constrained-EM cap: OFF (threshold=1.0, never binds)', flush=True)
    else:
        _sd = float(os.environ.get('ROBOCOP_EM_CAP_SD', 2))
        threshold = np.mean(thresholds) + _sd*np.std(thresholds)
        print('constrained-EM cap: mean + %g*sd = %.6g' % (_sd, threshold), flush=True)
    # get MNase-seq count parameters
    if mnaseFiles: 
        # with open(outDir + "/negParamsMNase.pkl", 'rb') as readFile:
        #     mnaseParams = pickle.load(readFile, encoding = 'latin1')
        mnaseParams = parameterize.getParamsMNase(mnaseFiles, config.get("main", "nucleosomeFile"), config.get("main", "tfFile"), fragRange, tmpDir, tech)
    else:
        mnaseParams = None
    
    # create shared dictionary for all segments and build HMM transition matrix
    dshared, d_segments = createInstances(tf_prob, dbf_conc, coords, pwm, cshared, tmpDir, info_file, config.get("main", "nucFile"), config.get("main", "nucleosomeFile"), nucleotide_sequence, mnaseParams, tech)

    fLike = open(outDir + '/likelihood.txt', 'w')
    likelihood = getLogLikelihood(segments, dshared)
    fLike.write(str(likelihood) + '\n')
    fLike.close()
    # EM is ON in this variant. Published RoboCOP runs 10 iterations; this fork set it to 0
    # in commit ad6c2be (2025-12-13), which turned training into a pure config build --
    # tf_prob stayed exactly calculateKD's output and was never fitted.
    iterations = int(os.environ.get("ROBOCOP_EM_ITERS", 10))
    print("EM iterations:", iterations, flush=True)
    countMNase = 0

    print("Writing MNase params")
    if mnaseFiles != "":
        with open(outDir + "/negParamsMNase.pkl", 'wb') as writeFile:
            pickle.dump(mnaseParams, writeFile, pickle.HIGHEST_PROTOCOL)

    # Per-iteration trajectory of the transition prior. Upstream dumps a full ~97 MB
    # HMMconfig{i}.pkl here; only the prior actually changes between iterations, so record
    # that (~5 KB) instead. The final HMMconfig.pkl is still written in full below.
    traceDir = outDir + "/em_trace/"
    os.makedirs(traceDir, exist_ok = True)

    def writeTrace(i, ll):
        np.savez(traceDir + "iter" + str(i) + ".npz",
                 tfs = np.asarray(dshared['tfs'], dtype = 'U'),
                 tf_prob = np.asarray(dshared['tf_prob'], dtype = np.float64),
                 background_prob = np.float64(dshared['background_prob']),
                 nucleosome_prob = np.float64(dshared['nucleosome_prob']),
                 iteration = np.int64(i),
                 log_likelihood = np.float64(ll))

    # iter0 is the state ENTERING the first update, i.e. the unfitted calculateKD prior.
    writeTrace(0, likelihood)

    for i in range(iterations):

        # Baum-Welch on transition probabilities
        background_prob, _tf_prob, nucleosome_prob = update_transition_probs(dshared, segments, tmpDir, threshold)
        tf_prob = np.array([_tf_prob[_] for _ in np.array(sorted(_tf_prob.keys()), order = 'c')])
        robocop.set_transition(dshared, tf_prob, background_prob, nucleosome_prob)
        robocop.set_initial_probs(dshared)

        # posterior decoding with updated transition probabilities
        for t in range(segments):
            setValuesPosterior((d_segments[t], t, dshared, tf_prob, background_prob, nucleosome_prob, tmpDir))
            # setValuesPosterior((t, dshared, tf_prob, background_prob, nucleosome_prob, tmpDir))

        likelihood = getLogLikelihood(segments, dshared)
        fLike = open(outDir + '/likelihood.txt', 'a')
        fLike.write(str(likelihood) + '\n')
        fLike.close()
        # set_transition (robocop.py:827-829) already wrote tf_prob / background_prob /
        # nucleosome_prob back into dshared, so the trace reads the post-update state.
        writeTrace(i + 1, likelihood)
        print("EM iter %d/%d  loglik %.6f" % (i + 1, iterations, likelihood), flush = True)
    
    #create new dshared
    dsharedNew = {}
    for k in list(dshared.keys()):
        if k == 'info_file': dsharedNew['info_file_name'] = info_file_name
        else: dsharedNew[k] = dshared[k]
    print("Writing to HMMconfig")
    with open(outDir + "/HMMconfig.pkl", 'wb') as writeFile:
        pickle.dump(dsharedNew, writeFile, pickle.HIGHEST_PROTOCOL)

    # remove tmpDir
    if len(coords) <= 500: os.system ("rm -rf " + outDir + "tmpDir")

    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python robocop_em.py <coordinate file> <config file> <output directory>")
        exit(1)
    coordFile = sys.argv[1]
    configFile = sys.argv[2]
    outDir = sys.argv[3]

    run_robocop_with_em(coordFile, configFile, outDir)
