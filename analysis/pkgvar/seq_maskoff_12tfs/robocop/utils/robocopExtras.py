import robocop
from .readWriteOps import *

# save posterior distribution of DBFs in csv format
def printPosterior(segments, dshared, tmpDir, outDir):
    for s in range(segments):
        robocop.print_posterior_binding_probability(x, dshared, file_name = outDir + '/em_test.out')

# adjust TF weights so that it does not exceed a threshold
def adjustEM(dbf_posterior_start_probs_same_update, dshared, threshold):
    indices = list(range(len(dbf_posterior_start_probs_same_update)))
    thresholdSum = 0
    for i in range(1, dshared['n_tfs']):
        # set to threshold when exceeds threshold
        if dbf_posterior_start_probs_same_update[i] > threshold:
            dbf_posterior_start_probs_same_update[i] = threshold
            thresholdSum += threshold
            indices.remove(i)
    # normalize the rest
    dbf_posterior_start_probs_same_update[indices] = dbf_posterior_start_probs_same_update[indices] / np.sum(dbf_posterior_start_probs_same_update[indices])
    dbf_posterior_start_probs_same_update[indices] *= 1 - thresholdSum
    return dbf_posterior_start_probs_same_update

# Baum Welch update of transition probabilities
def update_transition_probs(dshared, segments, tmpDir, threshold):
    nucleosome_prob = 0 
    background_prob = 0 
    tf_starts = dshared['tf_starts'] 
    tf_lens = dshared['tf_lens'] 
    tfs = dshared['tfs']
    info_file = dshared['info_file']
    dbf_posterior_start_probs_same_update = np.zeros(dshared['n_tfs'] + 1 + 1) # assuming nucs always present
    
    for t in range(segments):
        k = 'segment_' + str(t) + '/'
        
        # ignore segments giving numerical issues
        p_table = robocop.get_sparse_todense(info_file, k+'posterior')
        # p_table = info_file[k + 'posterior'][:] # x['posterior_table']
        if np.isinf(np.sum(p_table)): continue
        if np.sum(p_table) > 1e10: continue
        # print("Ptable sum:", np.sum(p_table), t)
        dbf_posterior_start_probs_same_update[0] += p_table[:,0].sum()
        # tfs
        for i in range(dshared['n_tfs']):
            dbf_posterior_start_probs_same_update[i + 1] += np.sum(p_table[:,tf_starts[i]])
            dbf_posterior_start_probs_same_update[i + 1] += np.sum(p_table[:,tf_starts[i] + tf_lens[i]])
        dbf_posterior_start_probs_same_update[dshared['n_tfs'] + 1] += np.sum(p_table[:, dshared['nuc_start']])
    # re normalize
    dbf_posterior_start_probs_same_update_em = dbf_posterior_start_probs_same_update / np.sum(dbf_posterior_start_probs_same_update)
    
    # active constraint based EM to limit the max probability for any TF excepting unknown
    # unknown is the last TF in the list
    if np.any(dbf_posterior_start_probs_same_update_em[1 : dshared['n_tfs']] > threshold):
        dbf_posterior_start_probs_same_update_em = adjustEM(dbf_posterior_start_probs_same_update_em, dshared, threshold)
    background_prob = dbf_posterior_start_probs_same_update_em[0] 
    tf_prob = dict()
    for i in range(dshared['n_tfs']):
        tf_prob[tfs[i]] = dbf_posterior_start_probs_same_update_em[i+1] 
    nucleosome_prob = dbf_posterior_start_probs_same_update_em[dshared['n_tfs'] + 1]
    return background_prob, tf_prob, nucleosome_prob


def updateMNaseEMMatNorm(args):
    (t, dshared, countParams, tech) = args
    x = loadIdx(dshared['tmpDir'], t)
    robocop.update_data_emission_matrix_using_mnase_midpoint_counts_norm(x, dshared, nuc_mean = countParams['nucShort']['mean'], nuc_sd = countParams['nucShort']['sd'], tf_mean = countParams['tfShort']['mean'], tf_sd = countParams['tfShort']['sd'], other_mean = countParams['otherShort']['mean'], other_sd = countParams['otherShort']['sd'], mnaseType = 'tf')
    robocop.update_data_emission_matrix_using_mnase_midpoint_counts_norm(x, dshared, nuc_mean = countParams['nucLong']['mean'], nuc_sd = countParams['nucLong']['sd'], tf_mean = countParams['tfLong']['mean'], tf_sd = countParams['tfLong']['sd'], other_mean = countParams['otherLong']['mean'], other_sd = countParams['otherLong']['sd'], mnaseType = 'nuc')

def updateMNaseEMMatGamma(args):
    (t, dshared, countParams, tech) = args
    x = loadIdx(dshared['tmpDir'], t)
    
    robocop.update_data_emission_matrix_using_mnase_midpoint_counts_gamma(x, dshared, nuc_shape = countParams['nucShort']['shape'], nuc_rate = countParams['nucShort']['rate'], tf_shape = countParams['tfShort']['shape'], tf_rate = countParams['tfShort']['rate'], other_shape = countParams['otherShort']['shape'], other_rate = countParams['otherShort']['rate'], mnaseType = 'tf')
    robocop.update_data_emission_matrix_using_mnase_midpoint_counts_gamma(x, dshared, nuc_shape = countParams['nucLong']['shape'], nuc_rate = countParams['nucLong']['rate'], tf_shape = countParams['tfLong']['shape'], tf_rate = countParams['tfLong']['rate'], other_shape = countParams['otherLong']['shape'], other_rate = countParams['otherLong']['rate'], mnaseType = 'nuc')

# update data emission matrix using negative binomial distribution parameters
def updateMNaseEMMatNB(args):
    (d, t, dshared, countParams, tech) = args

    # robocop.update_data_emission_matrix_using_mnase_midpoint_counts_onePhi(d, t, dshared, nuc_phi = countParams['nucLong']['phi'], nuc_mus = countParams['nucLong']['mu']*countParams['nucLong']['scale'], tf_phi = countParams['tfLong']['phi'], tf_mu = countParams['tfLong']['mu'], other_phi = countParams['otherLong']['phi'], other_mu = countParams['otherLong']['mu'], mnaseType = 'long', tech = tech)
    # robocop.update_data_emission_matrix_using_mnase_midpoint_counts_onePhi(d, t, dshared, nuc_phi = countParams['nucShort']['phi'], nuc_mus = countParams['nucShort']['mu']*countParams['nucShort']['scale'], tf_phi = countParams['tfShort']['phi'], tf_mu = countParams['tfShort']['mu'], other_phi = countParams['otherShort']['phi'], other_mu = countParams['otherShort']['mu'], mnaseType = 'short', tech = tech)
    # # robocop.update_data_emission_matrix_using_mnase_midpoint_counts_onePhi(t, dshared, nuc_phi = countParams['nucLong']['phi'], nuc_mus = countParams['nucLong']['mu']*countParams['nucLong']['scale'], tf_phi = countParams['tfLong']['phi'], tf_mu = countParams['tfLong']['mu'], other_phi = countParams['otherLong']['phi'], other_mu = countParams['otherLong']['mu'], mnaseType = 'long', tech = tech)
    # robocop.update_data_emission_matrix_using_mnase_midpoint_counts_onePhi(t, dshared, nuc_phi = countParams['nucShort']['phi'], nuc_mus = countParams['nucShort']['mu']*countParams['nucShort']['scale'], tf_phi = countParams['tfShort']['phi'], tf_mu = countParams['tfShort']['mu'], other_phi = countParams['otherShort']['phi'], other_mu = countParams['otherShort']['mu'], mnaseType = 'short', tech = tech)
#     robocop.update_data_emission_matrix_using_fiber_seq_counts_onePhi(d, t, dshared, nuc_phi = countParams['nucShort']['phi'], nuc_mus = countParams['nucShort']['mu']*countParams['nucShort']['scale'], tf_phi = countParams['tfShort']['phi'], tf_mu = countParams['tfShort']['mu'], other_phi = countParams['otherShort']['phi'], other_mu = countParams['otherShort']['mu'], FiberType = 'watson', tech = tech)
#     robocop.update_data_emission_matrix_using_fiber_seq_counts_onePhi(d, t, dshared, nuc_phi = countParams['nucShort']['phi'], nuc_mus = countParams['nucShort']['mu']*countParams['nucShort']['scale'], tf_phi = countParams['tfShort']['phi'], tf_mu = countParams['tfShort']['mu'], other_phi = countParams['otherShort']['phi'], other_mu = countParams['otherShort']['mu'], FiberType = 'crick', tech = tech)
    robocop.update_data_emission_matrix_using_fiber_seq_counts_Bionomial(d, t, dshared, nuc_phi = countParams['nucShort']['phi'], nuc_mus = countParams['nucShort']['mu']*countParams['nucShort']['scale'], tf_phi = countParams['tfShort']['phi'], tf_mu = countParams['tfShort']['mu'], other_phi = countParams['otherShort']['phi'], other_mu = countParams['otherShort']['mu'], FiberType = 'watson', tech = 'Fiber')
    robocop.update_data_emission_matrix_using_fiber_seq_counts_Bionomial(d, t, dshared, nuc_phi = countParams['nucShort']['phi'], nuc_mus = countParams['nucShort']['mu']*countParams['nucShort']['scale'], tf_phi = countParams['tfShort']['phi'], tf_mu = countParams['tfShort']['mu'], other_phi = countParams['otherShort']['phi'], other_mu = countParams['otherShort']['mu'], FiberType = 'crick', tech = 'Fiber')

  ## Sets the sequence emission to uniform 1 value so that it doesn't interfere with the results 
    # info_file = dshared['info_file']
    # data_emission_matrix = info_file['segment_' + str(t) + '/emission'][:]
    # data_emission_matrix[0][:] = 1
    # emat = info_file['segment_' + str(t) + '/emission']
    # emat[...] = data_emission_matrix

    print('bob')
    data_emission_matrix = d['emission']

    ## Exclude sequence from robocop
    # data_emission_matrix[0][:] = 1   # SEQUENCE LAYER ON for this variant

    ## Turn all values of 0 into a small value
    epsilon = 1e-30
    data_emission_matrix[5][data_emission_matrix[5] == 0] = epsilon
    data_emission_matrix[6][data_emission_matrix[6] == 0] = epsilon

    ## ============ VARIANT 12_tfs ============
    ## Keep ONLY the 12 TFs that have a real fitted Fiber-seq footprint. Every other TF
    ## state -- 141 factors plus `unknown` -- currently falls back to combined_low_count,
    ## whose p (0.2489/0.2647) is ABOVE the background p (0.1383). Higher p means "more
    ## methylation expected", so those states outscore background exactly where DNA is
    ## most accessible, which is why TFs get decoded into linkers. This variant removes
    ## them from the model entirely rather than reparameterising them.
    ##
    ## `unknown` is masked too (user decision): it is the generic DBF and also rides on
    ## combined_low_count, so leaving it live would let the artifact through unchanged.
    ##
    ## Derived from dshared['tfs'] rather than hardcoded state indices, so a change to
    ## the motif file or state layout cannot silently mask the wrong block.
    ## Runs AFTER the 1e-30 floor so forbidden states stay EXACTLY 0 (the floor would
    ## otherwise lift them to 1e-30, which still leaks posterior).
    FITTED_12 = {'Abf1_murphy', 'Cin5_murphy', 'Fhl1_zhu', 'Fkh1_zhu', 'Mcm1_zhu',
                 'Nhp6a_zhu', 'Rap1_telomeric', 'Reb1_badis', 'Sko1_murphy',
                 'Spt15_zhu', 'Tbf1_zhu', 'Ume6_zhu'}
    _tfs = list(dshared['tfs'])
    _missing = FITTED_12 - set(_tfs)
    assert not _missing, "12_tfs mask: names absent from the model: %s" % _missing
    _kept, _masked = [], []
    for _i, _name in enumerate(_tfs):
        _s = int(dshared['tf_starts'][_i])
        _e = _s + 2 * int(dshared['tf_lens'][_i])
        if _name in FITTED_12:
            _kept.append(_name)
        else:
            data_emission_matrix[5][:, _s:_e] = 0
            data_emission_matrix[6][:, _s:_e] = 0
            _masked.append(_name)
    print("12_tfs mask: kept %d TFs %s | masked %d (incl 'unknown': %s)"
          % (len(_kept), sorted(_kept), len(_masked), 'unknown' in _masked))

    ## original ABF1-only mask, left commented for reference:
    ## ABF1 HARD-forbid mask (comment IN for a TRUE ABF1-only run). Runs AFTER the 1e-30 floor
    ## so non-ABF1 TF states (29..nuc_start, incl. 'unknown') stay EXACTLY 0. Emission is a
    ## product over channels, so a 0 in the Fiber channels => total emission 0 => posterior 0
    ## for every TF except ABF1 (states 1..28). Background (0) and nucleosomes (nuc_start..)
    ## keep the 1e-30 floor so no position becomes all-zero (no NaN). Supersedes robocop.py:798.
    # data_emission_matrix[5][:, 29:dshared['nuc_start']] = 0
    # data_emission_matrix[6][:, 29:dshared['nuc_start']] = 0

    d['emission'] = data_emission_matrix

    # # data_emission_matrix[0][:,2799-10:2799] = 0.
    # # data_emission_matrix[0][:,2799:3330] = 0.3

# Posterior decoding
def setValuesPosterior(args):
    (d, t, dshared, tf_prob, background_prob, nucleosome_prob, tmpDir) = args
    robocop.posterior_forward_backward(d, t, dshared)

# Compute log likelihood
def getLogLikelihood(segments, dshared):
    logLikelihood = 0
    for s in range(segments):
        logLikelihood += robocop.get_log_likelihood(dshared, s)
    return logLikelihood

def build_data_emission_matrix_wrapper(t, dshared):
    robocop.build_data_emission_matrix(dshared, t)
    dumpIdx(x, tmpDir)

# wrapper function to perform posterior decoding
def posterior_forward_backward_wrapper(args):
    (d, t, dshared) = args
    robocop.posterior_forward_backward(d, t, dshared)

# create dictionary for each segment
def createInstance(args):
    (t, dshared, chrm, start, end) = args
    x = robocop.createDictionary(t, dshared, chrm, start, end)
    dumpIdx(x, dshared['info_file'])
    return x
    