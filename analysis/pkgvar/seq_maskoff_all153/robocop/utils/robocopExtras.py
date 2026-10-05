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

    ## ============ VARIANT all153 (campaign bu01, 2026-09-30) ============
    ## A keep-set that keeps EVERYTHING: all 153 motifs plus `unknown`, so the loop below masks
    ## nothing and this tree decodes exactly like pkgvar/seq_maskoff. It exists only because
    ## tune_w.driver_keep (tune_w.py:341-350) requires exactly one KEEP_* set and raises
    ## "has 0 KEEP_* sets" on an unmasked tree; cmd_init then compares it to the campaign's
    ## motifs. Names come from dshared['tfs'], never hardcoded state indices.
    ##
    ## The emission parameters, not the mask, are what differ here (robocop.py:598 and :606):
    ##   combined_low_count 0.24894/0.26468 -> 0.08/0.08   (..._clc08.pkl; `unknown` inherits it)
    ##   background         0.13827/0.13841 -> 0.24894/0.26468 (bg_params_open.pkl)
    KEEP_ALL154 = {'Abf1_murphy', 'Abf2_badis', 'Ace2_badis', 'Adr1_badis', 'Aft1_zhu', 'Aft2_badis',
                 'Aro80_zhu', 'Asg1_zhu', 'Azf1_badis', 'Bas1_zhu', 'Cad1_murphy',
                 'Cat8_badis', 'Cbf1_zhu', 'Cep3_badis', 'Cha4_zhu', 'Cin5_murphy',
                 'Crz1_badis', 'Cst6_murphy', 'Cup9_badis', 'Dal80_badis', 'Dal82_badis',
                 'Ecm22_murphy', 'Ecm23_badis', 'Fhl1_zhu', 'Fkh1_zhu', 'Fkh2_zhu',
                 'Fzf1_badis', 'Gal4_zhu', 'Gat1_zhu', 'Gat3_zhu', 'Gat4_zhu', 'Gcn4_zhu',
                 'Gcr1_murphy', 'Gis1_badis', 'Gln3_badis', 'Gsm1_zhu', 'Gzf3_zhu',
                 'Hac1_badis', 'Hal9_zhu', 'Hap1_murphy', 'Hcm1_badis', 'Hmlalpha2_murphy',
                 'Hmra2_badis', 'Hsf1_badis', 'Leu3_zhu', 'Lys14_zhu', 'Matalpha2_zhu',
                 'Mbp1_zhu', 'Mcm1_zhu', 'Met31_badis', 'Met32_badis', 'Mga1_zhu', 'Mig1_zhu',
                 'Mig2_zhu', 'Mig3_zhu', 'Mot3_murphy', 'Msn1_murphy', 'Msn2_badis',
                 'Msn4_badis', 'Ndt80_zhu', 'Nhp10_badis', 'Nhp6a_zhu', 'Nhp6b_zhu',
                 'Nrg1_zhu', 'Nrg2_murphy', 'Oaf1_badis', 'Pbf1_zhu', 'Pbf2_zhu',
                 'Pdr1_badis', 'Pdr3_murphy', 'Pdr8_badis', 'Phd1_zhu', 'Pho2_badis',
                 'Pho4_zhu', 'Put3_badis', 'Rap1_motif1', 'Rap1_motif2', 'Rap1_telomeric',
                 'Rap1_zhu', 'Rdr1_zhu', 'Rds1_zhu', 'Rds2_badis', 'Reb1_badis', 'Rei1_badis',
                 'Rfx1_badis', 'Rgm1_badis', 'Rgt1_badis', 'Rim101_badis', 'Rox1_badis',
                 'Rph1_badis', 'Rpn4_badis', 'Rsc30_zhu', 'Rsc3_zhu', 'Rtg3_zhu', 'Sfl1_zhu',
                 'Sfp1_zhu', 'Sig1_badis', 'Sip4_badis', 'Skn7_badis', 'Sko1_murphy',
                 'Smp1_zhu', 'Sok2_badis', 'Spt15_zhu', 'Srd1_badis', 'Stb3_zhu',
                 'Stb4_murphy', 'Stb5_murphy', 'Ste12_murphy', 'Stp1_murphy', 'Stp2_zhu',
                 'Stp3_badis', 'Stp4_zhu', 'Sum1_zhu', 'Sut1_murphy', 'Sut2_zhu',
                 'Swi4_badis', 'Swi5_badis', 'Tbf1_zhu', 'Tbs1_zhu', 'Tea1_zhu', 'Tec1_badis',
                 'Tos8_badis', 'Tye7_zhu', 'Uga3_badis', 'Ume6_zhu', 'Upc2_murphy',
                 'Usv1_zhu', 'Vhr1_murphy', 'Xbp1_badis', 'Yap1_zhu', 'Yap3_murphy',
                 'Yap6_zhu', 'Ybr033w_murphy', 'Ybr239c_zhu', 'Ydr520c_badis',
                 'Yer064c_murphy', 'Yer130c_badis', 'Yer184c_murphy', 'Ygr067c_badis',
                 'Ykl222c_zhu', 'Yll054c_zhu', 'Ylr278c_murphy', 'Yml081w_zhu',
                 'Ynr063w_badis', 'Yox1_badis', 'Ypr013c_zhu', 'Ypr015c_zhu', 'Ypr022c_badis',
                 'Ypr196w_badis', 'Yrm1_badis', 'Yrr1_zhu', 'Zap1_murphy', 'Zms1_badis',
                 'unknown'}
    _tfs = list(dshared['tfs'])
    _missing = KEEP_ALL154 - set(_tfs)
    assert not _missing, "all153 mask: names absent from the model: %s" % _missing
    _kept, _masked = [], []
    for _i, _name in enumerate(_tfs):
        _s = int(dshared['tf_starts'][_i])
        _e = _s + 2 * int(dshared['tf_lens'][_i])
        if _name in KEEP_ALL154:
            _kept.append(_name)
        else:
            data_emission_matrix[5][:, _s:_e] = 0
            data_emission_matrix[6][:, _s:_e] = 0
            _masked.append(_name)
    assert not _masked, "all153 mask must mask NOTHING, but masked %d: %s" % (len(_masked), _masked)
    if not globals().get('_ALL153_MASK_REPORTED'):
        print("all153 mask: kept %d (incl 'unknown': %s) | masked %d"
              % (len(_kept), 'unknown' in _kept, len(_masked)))
        globals()['_ALL153_MASK_REPORTED'] = True

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
    