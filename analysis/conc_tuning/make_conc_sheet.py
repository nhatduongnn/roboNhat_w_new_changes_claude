"""Build the concentration sheet's data payload and inject it into conc_sheet.html.

Per factor: the starting concentration `calculateKD` derives from the motif alone, the lambda
the tuning loop settled on, the product of the two, the resulting HMM prior, the copies decoded,
and the group's MacIsaac counts / precision / recall for that round.

Per campaign it also emits the outcome, read from the campaign's own files rather than typed in:
the round-by-round convergence and pooled site accuracy from `report_NN.tsv`, the nucleosome copy
count from `counts_ct_<run>_NN.tsv` against the untuned decode, the stop reason from `STOPPED`,
and the census of factors pinned at either end of the lambda clamp, plus where the decoded copies
went (tuned motifs, untuned motifs, unknown, background) round by round.

It also carries the finished experiments that moved a concentration outside the campaigns: the
ABF1 lambda sweep (chrI, plus the chrXIV check), the lambda_unknown sweep, the four EM trainings,
and the wide150 prior crush. Every run is named explicitly (in this file or in a committed run
table); nothing globs campaign directories or counts files, and a campaign without a STOPPED
marker is refused, so an in-progress campaign cannot reach the page.

Usage:  python conc_tuning/make_conc_sheet.py u001=7 m001=7
"""
import collections, csv, json, math, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pkg'))
from robocop.utils.parameterize import calculateKD

HERE = os.path.dirname(os.path.abspath(__file__))          # analysis/conc_tuning
AN = os.path.dirname(HERE)                                  # analysis
RUNS = dict(kv.split('=') for kv in sys.argv[1:]) or {"u001": "7", "m001": "7"}

pwm = pickle.load(open(os.path.join(AN, 'robocop_train_fiberonly/pwm.p'), 'rb'))
KEEP = {r['motif'] for r in csv.DictReader(open(os.path.join(AN, 'inputs/conc_targets_macisaac_c1.tsv')), delimiter='\t') if r['has_target'] == '1'}
KEEPCOLS = ("name group kind length masked conc_start lam conc_after prior occ target calls "
            "matched precision recall status").split()
out = {"runs": {}}

LAM_FLOOR, LAM_CAP = 1e-6, 1e3          # the clamp tune_concentrations.py applies to every step
TARGETS = list(csv.DictReader(open(os.path.join(AN, 'inputs/conc_targets_macisaac_c1.tsv')), delimiter='\t'))
TARGET_FLAG = {r['motif']: r['has_target'] == '1' for r in TARGETS}
CHROMS = [c for c in TARGETS[0].keys() if c.startswith('chr')]
assert len(CHROMS) == 16, CHROMS
CHROM_LEN = {l.split('\t')[0]: int(l.split('\t')[1]) for l in open(os.path.join(AN, 'inputs/SacCer3.fa.fai'))}
NUC_SUBSTATES = {'nuc_start', 'nuc_center', 'nuc_end', 'nuc_padding'}


def _tsv(path):
    return list(csv.DictReader(open(path), delimiter='\t'))


def _occ(run, it):
    return {r['factor']: float(r['occ']) for r in _tsv(os.path.join(AN, 'conc_tuning/counts_ct_%s_%02d.tsv' % (run, it)))}


# The untuned genome decode every nucleosome count is judged against (tune_concentrations.py's
# NUC_BASELINE), read from its own counts file so the two can never drift apart.
NUC_BASELINE = [r for r in _tsv(os.path.join(AN, 'conc_tuning/counts_conctune_00.tsv'))
                if r['factor'] == 'nucleosome'][0]
NUC_BASELINE = float(NUC_BASELINE['occ'])


def outcome(run, last):
    """Round-by-round convergence, pooled accuracy, and how the campaign ended."""
    rounds = []
    for it in range(last + 1):
        rep = _tsv(os.path.join(AN, 'conc_tuning/%s/report_%02d.tsv' % (run, it)))
        calls = sum(int(r['calls']) for r in rep)
        matched = sum(int(r['matched_30bp']) for r in rep)
        sites = sum(float(r['macisaac']) for r in rep)
        p, rc = matched / calls, matched / sites
        folds = sorted(max(f, 1 / f) for f in (float(r['fold_off']) for r in rep) if f > 0)
        rounds.append(dict(
            round=it, groups=len(rep),
            within2x=sum(1 for r in rep if 0.5 <= float(r['fold_off']) <= 2.0),
            fold_off=folds[len(folds) // 2], calls=calls, matched=matched, sites=sites,
            precision=p, recall=rc, f1=2 * p * rc / (p + rc),
            nuc=_occ(run, it).get('nucleosome', 0.0)))

    rep = {r['group']: r for r in _tsv(os.path.join(AN, 'conc_tuning/%s/report_%02d.tsv' % (run, last)))}
    clamped = []
    for g, r in rep.items():
        lam, fold = float(r['lambda']), float(r['fold_off'])
        end = 'floor' if lam <= LAM_FLOOR else ('cap' if lam >= LAM_CAP else None)
        if end:
            clamped.append(dict(group=g, end=end, target=float(r['macisaac']),
                                decoded=float(r['decoded']), calls=int(r['calls']),
                                matched=int(r['matched_30bp']), fold_off=fold,
                                within2x=bool(0.5 <= fold <= 2.0)))
    clamped.sort(key=lambda c: (c['end'], -c['fold_off']))

    # Where the decoded copies went, round by round. Rows are classified with the MacIsaac
    # target table; the nucleosome's sub-states are not factors and are left out, background
    # is a 1 bp state so its copies are base pairs.
    mass = []
    for it in range(last + 1):
        m = dict(round=it, tuned=0.0, untuned=0.0, unknown=0.0, background=0.0)
        for f, v in _occ(run, it).items():
            if f in NUC_SUBSTATES or f == 'nucleosome':
                continue
            if f in ('unknown', 'background'):
                m[f] += v
            elif f in TARGET_FLAG:
                m['tuned' if TARGET_FLAG[f] else 'untuned'] += v
            else:
                raise SystemExit('unclassified state %r in counts_ct_%s_%02d.tsv' % (f, run, it))
        mass.append(m)

    # Marginal precision after the first step: of the calls rounds 1 -> last removed, how many
    # sat at a MacIsaac site, against the precision of the calls round 1 already had.
    r1, rz = rounds[1], rounds[-1]
    marginal = dict(d_calls=rz['calls'] - r1['calls'], d_matched=rz['matched'] - r1['matched'],
                    precision=(rz['matched'] - r1['matched']) / (rz['calls'] - r1['calls']),
                    base_precision=r1['precision'])

    # The nucleosome prior each round's trainDir actually carries, and how far it strayed.
    nucp = [float(pickle.load(open(os.path.join(AN, 'robocop_train_ct_%s_%02d/HMMconfig.pkl' % (run, it)), 'rb'),
                              encoding='latin1')['nucleosome_prob']) for it in range(last + 1)]

    # Positions count_calls.py dropped because the decode emitted a posterior outside [0, 1],
    # per chromosome, at their widest over the campaign (16 chromosome files per round, by name).
    excluded = {}
    for it in range(last + 1):
        for c in CHROMS:
            r = _tsv(os.path.join(AN, 'conc_tuning/counts_ct_%s_%02d/%s.tsv' % (run, it, c)))[0]
            if int(r['n_excluded']):
                excluded[c] = max(excluded.get(c, 0), int(r['n_excluded']))

    stop = os.path.join(AN, 'conc_tuning/%s/STOPPED' % run)
    return dict(excluded={c: dict(n=n, length=CHROM_LEN[c]) for c, n in excluded.items()}, rounds=rounds, clamped=clamped, nuc_baseline=NUC_BASELINE, mass=mass,
                marginal=marginal, nuc_prior=nucp[0],
                nuc_prior_rel_spread=(max(nucp) - min(nucp)) / nucp[0],
                lam_floor=LAM_FLOOR, lam_cap=LAM_CAP,
                stopped=open(stop).read().strip() if os.path.exists(stop) else None)

for run, t in RUNS.items():
    t = int(t)
    # Only finished campaigns are published. Nothing below globs campaign directories or
    # counts files: every path is built from the run names on the command line, and a run
    # without a STOPPED marker (a campaign still in progress) is refused outright.
    if not os.path.exists(os.path.join(AN, 'conc_tuning/%s/STOPPED' % run)):
        sys.exit('%s has no conc_tuning/%s/STOPPED: still running, refusing to publish it' % (run, run))
    st = json.load(open(os.path.join(AN, 'conc_tuning/%s/state.json' % run)))
    cfg = pickle.load(open(os.path.join(AN, 'robocop_train_ct_%s_%02d/HMMconfig.pkl' % (run, t)), 'rb'), encoding='latin1')
    tfs, tf_prob = list(cfg['tfs']), np.asarray(cfg['tf_prob'], float)
    occ = {r['factor']: float(r['occ']) for r in csv.DictReader(open(os.path.join(AN, 'conc_tuning/counts_ct_%s_%02d.tsv' % (run, t))), delimiter='\t')}
    rep = {r['group']: r for r in csv.DictReader(open(os.path.join(AN, 'conc_tuning/%s/report_%02d.tsv' % (run, t))), delimiter='\t')}
    rows = []
    for i, m in enumerate(tfs):
        unk = (m == 'unknown')
        R = None if unk else rep.get(st['group'][m])
        cal = int(R['calls']) if R else None
        mat = int(R['matched_30bp']) if R else None
        start = 0.1 if unk else float(calculateKD(pwm, m))
        lam = st['fixed_lam']['unknown'] if unk else st['lam'][m]
        rows.append(dict(name=m, group='unknown' if unk else st['group'][m],
                         kind='unknown' if unk else 'tf',
                         length=10 if unk else int(pwm[m].shape[1]),
                         masked=bool(run == 'm001' and not unk and m not in KEEP),
                         conc_start=start, lam=lam, conc_after=start * lam,
                         prior=float(tf_prob[i]), occ=occ.get(m, 0.0),
                         target=float(R['macisaac']) if R else None,
                         calls=cal, matched=mat,
                         precision=(mat / cal) if cal else None,
                         recall=(mat / float(R['macisaac'])) if R and float(R['macisaac']) else None,
                         status=R['status'] if R else ('fixed at 0.01, no MacIsaac target' if unk else 'no MacIsaac target')))
    nl = st['nuc_lam'][str(t)]
    rows.append(dict(name='nucleosome', group='nucleosome', kind='nucleosome', length=147, masked=False,
                     conc_start=35.0, lam=nl, conc_after=35.0 * nl, prior=float(cfg['nucleosome_prob']),
                     occ=occ.get('nucleosome', 0.0), target=None, calls=None, matched=None,
                     precision=None, recall=None, status='prior held at the untuned value'))
    rows.append(dict(name='background', group='background', kind='background', length=1, masked=False,
                     conc_start=1.0, lam=1.0, conc_after=1.0, prior=float(cfg['background_prob']),
                     occ=occ.get('background', 0.0), target=None, calls=None, matched=None,
                     precision=None, recall=None, status='pinned at 1.0, defines the scale'))
    trim = lambda r: {k: (float('%.6g' % r[k]) if isinstance(r[k], float) else r[k]) for k in KEEPCOLS}
    out['runs'][run] = dict(round=t, masked_run=(run == 'm001'), nuc_lambda=float('%.6g' % nl),
                            nuc_lambdas=[st['nuc_lam'][str(i)] for i in range(t + 1)],
                            rows=[trim(r) for r in rows], outcome=outcome(run, t))
    oc = out['runs'][run]['outcome']
    print('%s round %d: %d rows; %d rounds, %d/%d within 2x, F1 %.4f, %d clamped (%s), stop: %s'
          % (run, t, len(rows), len(oc['rounds']), oc['rounds'][-1]['within2x'],
             oc['rounds'][-1]['groups'], oc['rounds'][-1]['f1'], len(oc['clamped']),
             ' '.join('%s@%s' % (c['group'], c['end']) for c in oc['clamped']), oc['stopped']))


# ---------------------------------------------------------------------------------------------
# Finished experiments that moved a concentration outside the two campaigns. Every run is named
# explicitly below or in a committed run table -- nothing is globbed.
# ---------------------------------------------------------------------------------------------
def _cfg(d):
    return pickle.load(open(os.path.join(AN, d, 'HMMconfig.pkl'), 'rb'), encoding='latin1')


def _runs_tsv(name):
    return [l.rstrip('\n').split('\t') for l in open(os.path.join(AN, name))
            if l.strip() and not l.startswith('#')]


BASE = _cfg('robocop_train_fiberonly')
BASE_TFS, BASE_P = list(BASE['tfs']), np.asarray(BASE['tf_prob'], float)

# The worked example: starting weights straight from calculateKD.
kd = {m: float(calculateKD(pwm, m)) for m in ('Abf1_murphy', 'Nhp6a_zhu')}
out['worked'] = dict(abf1=kd['Abf1_murphy'], nhp6a=kd['Nhp6a_zhu'], abf1_len=int(pwm['Abf1_murphy'].shape[1]),
                     nhp6a_len=int(pwm['Nhp6a_zhu'].shape[1]), unknown=0.1, nucleosome=35.0)

# 1. The ABF1 lambda sweep on chrI (18 points) plus the chrXIV check at the chosen lambda.
lamtab = {}
for f in ('conc_sweep_lo_runs.tsv', 'conc_refine_runs.tsv'):
    for label, outdir, lam in _runs_tsv(f):
        lamtab[label] = (float(lam), outdir)


def _abf1(path):
    d = json.load(open(os.path.join(AN, path)))
    a, n = d['abf1'], d['nucleosome']
    fin = lambda v: v if math.isfinite(v) else None      # no calls -> NaN precision, inf enrichment
    return dict(n_ref=a['n_ref'], n_pred=a['n_pred'], tp=a['tp'], fp=a['fp'], f1=fin(a['f1']),
                precision=fin(a['precision']), recall=fin(a['recall']), auroc=fin(a['auroc']),
                enrichment=fin(a['enrichment']), nuc_recall=n['recall'])


sweep = []
for label, (lam, outdir) in lamtab.items():
    if label == 'em10':
        continue
    sweep.append(dict(label=label, lam=lam, outdir=outdir, **_abf1('conc_scores/report_%s.json' % label)))
sweep.sort(key=lambda r: r['lam'])
em_pt = dict(label='em10', lam=float(dict((r[0], r[2]) for r in _runs_tsv('conc_sweep_runs.tsv'))['em10']),
             outdir=lamtab['em10'][1], **_abf1('conc_scores/report_em10.json'))
xiv = [dict(lam=1.0, outdir='robocop_chrXIV_fib_seq', **_abf1('layerXIV_scores/report_fib+seq.json')),
       dict(lam=0.01, outdir='robocop_chrXIV_fib_seq_lam0p01', **_abf1('layerXIV_scores/report_fib+seq+lam0.01.json'))]
out['abf1_sweep'] = dict(chrI=sweep, em=em_pt, chrXIV=xiv)

# 2. The lambda_unknown sweep: chrIV+VII+XV, nucleosome prior held, control cut from the
#    genome-wide lambda_unknown = 1 decode (same logic as compare_unk_sweep.py).
UNK_CHROMS = ('chrIV', 'chrVII', 'chrXV')
UNK = [(1.0, 'counts_conctune_00', 'robocop_train_conctune_00'),
       (0.3, 'counts_unk_0p3', 'robocop_train_unk_0p3'), (0.1, 'counts_unk_0p1', 'robocop_train_unk_0p1'),
       (0.03, 'counts_unk_0p03', 'robocop_train_unk_0p03'), (0.01, 'counts_unk_0p01', 'robocop_train_unk_0p01')]
UNK_FOCUS = ('FKH2', 'MBP1', 'STB5', 'HAP1', 'ABF1', 'REB1')
ui = BASE_TFS.index('unknown')
groups, gtarget = {}, {}
for r in TARGETS:
    if r['has_target'] == '1':
        groups.setdefault(r['group'], []).append(r['motif'])
        gtarget[r['group']] = sum(int(r[c]) for c in UNK_CHROMS)
unk = []
for lam, cdir, tdir in UNK:
    occ = {}
    for c in UNK_CHROMS:
        for r in _tsv(os.path.join(AN, 'conc_tuning', cdir, c + '.tsv')):
            occ[r['factor']] = occ.get(r['factor'], 0.0) + float(r['occ'])
    real = sum(v for f, v in occ.items() if f in TARGET_FLAG)
    gaps = []
    for g, ms in groups.items():
        T, E = gtarget[g], sum(occ.get(m, 0.0) for m in ms)
        if T > 0 and E > 0:
            gaps.append(abs(math.log(T) - math.log(E)))
    gaps.sort()
    q = np.asarray(_cfg(tdir)['tf_prob'], float)
    unk.append(dict(lam=lam, unknown=occ['unknown'], real=real, background=occ['background'],
                    nucleosome=occ['nucleosome'], groups=len(gaps), median_fold=math.exp(gaps[len(gaps) // 2]),
                    within2x=sum(1 for x in gaps if x < math.log(2)),
                    prior_ratio=float(q[ui] / (q.sum() - q[ui])),
                    focus={g: sum(occ.get(m, 0.0) for m in groups[g]) for g in UNK_FOCUS}))
out['unk_sweep'] = dict(rows=unk, targets={g: gtarget[g] for g in UNK_FOCUS}, chroms=UNK_CHROMS,
                        bp=sum(CHROM_LEN[c] for c in UNK_CHROMS), genome_bp=sum(CHROM_LEN[c] for c in CHROMS),
                        hap1_genome=[dict(max=float(r['global_max']), occ=float(r['occ']))
                                     for r in _tsv(os.path.join(AN, 'conc_tuning/counts_conctune_00.tsv'))
                                     if r['factor'] == 'Hap1_murphy'][0])

# 3. EM. Four 10-iteration Baum-Welch trainings on chrII against the untrained priors, and what
#    their ABF1 did on chrXIV (19 MacIsaac sites) next to the matching untrained decode.
EM_CAP = float(BASE_P[:-1].mean() + 2 * BASE_P[:-1].std())      # robocop_em.py: initial priors, unknown excluded
EM = [('robocop_train_em10_chrII', 'Fiber-seq + sequence', 'chrXIV_scores/report_seq.json', 'chrXIV_scores/report_seqem10.json'),
      ('robocop_train_em10_chrII_nocap', 'Fiber-seq + sequence, cap off', 'chrXIV_scores/report_seq.json', 'chrXIV_scores/report_em10nocap.json'),
      ('robocop_train_em10_chrII_fiber', 'Fiber-seq only', 'chrXIV_scores/report_fib.json', 'chrXIV_scores/report_fibem10.json'),
      ('robocop_train_em10_chrII_seqonly', 'sequence only', 'layerXIV_scores/report_seq.json', 'layerXIV_scores/report_seq+em10.json')]
em = []
for tdir, layers, before, after in EM:
    c = _cfg(tdir)
    assert list(c['tfs']) == BASE_TFS, tdir
    p = np.asarray(c['tf_prob'], float)
    ai = BASE_TFS.index('Abf1_murphy')
    em.append(dict(trainDir=tdir, layers=layers, abf1_fold=float(p[ai] / BASE_P[ai]),
                   at_cap=int(np.sum(np.isclose(p, EM_CAP, rtol=1e-6))), zero=int(np.sum(p == 0)),
                   n=len(p), iters=sum(1 for _ in open(os.path.join(AN, tdir, 'likelihood.txt'))) - 1,   # first line is the start
                   before=_abf1(before), after=_abf1(after)))
out['em'] = dict(rows=em, cap=EM_CAP, abf1_prior=float(BASE_P[BASE_TFS.index('Abf1_murphy')]))

# 4. wide150: padding 12 footprints by +/-150 bp, split into its K_d and alpha_0^L parts.
W = _cfg('robocop_train_wide150')
wpwm = pickle.load(open(os.path.join(AN, 'robocop_train_wide150/pwm.p'), 'rb'))
wp = np.asarray(W['tf_prob'], float)
wide = []
for i, m in enumerate(BASE_TFS):
    if m == 'unknown' or wpwm[m].shape[1] == pwm[m].shape[1]:
        continue
    L0, L1 = int(pwm[m].shape[1]), int(wpwm[m].shape[1])
    wide.append(dict(name=m, L0=L0, L1=L1, prior_ratio=float(wp[i] / BASE_P[i]),
                     kd_ratio=float(calculateKD(wpwm, m) / calculateKD(pwm, m)),
                     alpha_ratio=float(W['background_prob'] ** L1 / BASE['background_prob'] ** L0)))
rest = np.array([wp[i] / BASE_P[i] for i, m in enumerate(BASE_TFS) if m not in {w['name'] for w in wide}])
wabf1 = [json.load(open(os.path.join(AN, f)))['abf1'] for f in
         ('layer_scores/report_fib+wide150.json', 'layer_scores/report_fib+seq+wide150.json',
          'layerXIV_scores/report_fib+wide150.json', 'layerXIV_scores/report_fib+seq+wide150.json')]
wide.sort(key=lambda w: -w['prior_ratio'])
out['wide150'] = dict(rows=wide, rest_lo=float(rest.min()), rest_hi=float(rest.max()),
                      abf1_post_at_sites_max=max(a['mean_post_at_sites'] for a in wabf1), decodes=len(wabf1))

print('abf1 sweep %d chrI points; unk sweep %d; em %d; wide150 %d factors'
      % (len(sweep), len(unk), len(em), len(wide)))


# ---------------------------------------------------------------------------------------------
# 5. The same rounds checked against Rossi ChIP-exo. Everything comes from rossi_validate.py's
#    tables (conc_tuning/rossi_validation/), which re-derive each round's MacIsaac match from the
#    call positions and refuse to proceed unless it equals the tuning loop's own sidecar. Only
#    the campaigns named on the command line are read, and each already passed the STOPPED guard.
# ---------------------------------------------------------------------------------------------
RV = os.path.join(HERE, 'rossi_validation')
REFMETA = _tsv(os.path.join(RV, 'references.tsv'))
assert len(REFMETA) == 81, len(REFMETA)
ALLG = [m['group'] for m in REFMETA]
COMMON = [m['group'] for m in REFMETA if m['cx_tf']]
FITG = {m['group'] for m in REFMETA if m['fitted'] == '1'}
META = {m['group']: m for m in REFMETA}


def _prf(calls, matched, sites):
    p = matched / calls if calls else 0.0
    r = matched / sites if sites else 0.0
    return dict(calls=calls, matched=matched, sites=sites, precision=p, recall=r,
                f1=(2 * p * r / (p + r)) if p + r > 0 else 0.0)


# Reference columns on the page. 'macisaac_all' is the sheet's existing number (all 81 groups);
# the other two share the groups with a Rossi ChExMix file, so their calls are the same calls.
# Rossi's motif-anchored subset is not a validation column (user decision: _CX is the vetted set).
SCOPES = [('macisaac_all', 'macisaac', ALLG), ('macisaac', 'macisaac', COMMON),
          ('rossi_cx', 'rossi_cx', COMMON)]
rossi = dict(runs={}, n_groups=len(ALLG), common=len(COMMON),
             alias=[dict(group=m['group'], rossi=m['cx_tf']) for m in REFMETA if m['cx_join'] == 'alias'],
             unjoined=[m['group'] for m in REFMETA if not m['cx_tf']],
             fitted=sorted(FITG & set(COMMON)),
             fitted_motifs=sorted({mm for g in FITG for mm in META[g]['motifs'].split(',')} &
                                  {'Abf1_murphy', 'Cin5_murphy', 'Fhl1_zhu', 'Fkh1_zhu', 'Mcm1_zhu', 'Nhp6a_zhu',
                                   'Rap1_telomeric', 'Reb1_badis', 'Sko1_murphy', 'Spt15_zhu', 'Tbf1_zhu', 'Ume6_zhu'}),
             sites={k: sum(int(META[g]['n_macisaac' if s == 'macisaac' else 'n_cx']) for g in gs)
                    for k, s, gs in SCOPES},
             sites_nofit={k: sum(int(META[g]['n_macisaac' if s == 'macisaac' else 'n_cx']) for g in gs if g not in FITG)
                          for k, s, gs in SCOPES},
             cx_fusable=sum(int(META[g]['cx_fusable_20bp']) for g in COMMON),
             mac_in_cx=sum(int(META[g]['macisaac_in_rossi_cx']) for g in COMMON),
             mac_in_cx_nofit=sum(int(META[g]['macisaac_in_rossi_cx']) for g in COMMON if g not in FITG),
             group_meta=[dict(group=g, fitted=g in FITG, motifs=META[g]['motifs'], rossi=META[g]['cx_tf'],
                              join=META[g]['cx_join'], n_macisaac=int(META[g]['n_macisaac']), n_cx=int(META[g]['n_cx']))
                         for g in ALLG])

for run, t in RUNS.items():
    t = int(t)
    M = collections.defaultdict(dict)                  # (round, ref) -> group -> (n_ref, n_calls, n_matched)
    for r in _tsv(os.path.join(RV, 'match_%s.tsv' % run)):
        assert r['run'] == run
        M[(int(r['round']), r['ref'])][r['group']] = (int(r['n_ref']), int(r['n_calls']), int(r['n_matched']))
    G = collections.defaultdict(dict)
    for r in _tsv(os.path.join(RV, 'genic_%s.tsv' % run)):
        G[int(r['round'])][r['group']] = (int(r['n_calls']), int(r['n_genic']))
    assert sorted(G) == list(range(t + 1)), (run, sorted(G))
    rep0 = {r['group']: r for r in _tsv(os.path.join(AN, 'conc_tuning/%s/report_%02d.tsv' % (run, 0)))}

    def pooled(it, ref, groups):
        rows = [M[(it, ref)][g] for g in groups]
        return _prf(sum(x[1] for x in rows), sum(x[2] for x in rows), sum(x[0] for x in rows))

    rounds = []
    for it in range(t + 1):
        d = dict(round=it)
        for fit in ('all', 'nofit'):
            d[fit] = {}
            for key, ref, gs in SCOPES:
                gs2 = gs if fit == 'all' else [g for g in gs if g not in FITG]
                d[fit][key] = pooled(it, ref, gs2)
        # the sheet's own MacIsaac round numbers must come out of these tables unchanged
        rep = _tsv(os.path.join(AN, 'conc_tuning/%s/report_%02d.tsv' % (run, it)))
        assert d['all']['macisaac_all']['calls'] == sum(int(r['calls']) for r in rep), (run, it)
        assert d['all']['macisaac_all']['matched'] == sum(int(r['matched_30bp']) for r in rep), (run, it)
        rounds.append(d)

    def marginal(a, z, fit):
        out = {}
        for key, _, _ in SCOPES:
            A, Z = rounds[a][fit][key], rounds[z][fit][key]
            dc, dm = Z['calls'] - A['calls'], Z['matched'] - A['matched']
            out[key] = dict(d_calls=dc, d_matched=dm, precision=dm / dc if dc else None,
                            base_precision=A['precision'])
        return out
    marg = {fit: dict(first=marginal(0, 1, fit), rest=marginal(1, t, fit),
                      steps=[marginal(i - 1, i, fit) for i in range(1, t + 1)]) for fit in ('all', 'nofit')}

    # Genic fraction of the calls, over groups with a Rossi _cx genic% (peak-weighted pooled Rossi
    # figure alongside), and per factor with >= 100 Rossi _cx peaks.
    GEN = [g for g in ALLG if META[g]['genic_pct_cx'] not in ('NA', '')]
    rossi_cx_n = {g: int(float(META[g]['n_cx_genic_tsv'])) for g in GEN}
    rossi_cx_pct = {g: float(META[g]['genic_pct_cx']) for g in GEN}
    genic_rounds = []
    for it in range(t + 1):
        c = sum(G[it][g][0] for g in GEN)
        n = sum(G[it][g][1] for g in GEN)
        genic_rounds.append(dict(round=it, calls=c, genic=n, pct=100.0 * n / c if c else None))
    rossi_pooled_cx = sum(rossi_cx_pct[g] * rossi_cx_n[g] for g in GEN) / sum(rossi_cx_n.values())
    big = [g for g in GEN if rossi_cx_n[g] >= 100]
    per_genic = []
    for g in big:
        c0, n0 = G[0][g]
        cz, nz = G[t][g]
        per_genic.append(dict(group=g, fitted=g in FITG, rossi_n=rossi_cx_n[g], rossi_cx=rossi_cx_pct[g],
                              calls0=c0, pct0=100.0 * n0 / c0 if c0 else None,
                              callsz=cz, pctz=100.0 * nz / cz if cz else None))
    # gap to Rossi _cx, first vs last round, for factors with calls in both
    both = [p for p in per_genic if p['pct0'] is not None and p['pctz'] is not None]
    closer = sum(1 for p in both if abs(p['pctz'] - p['rossi_cx']) < abs(p['pct0'] - p['rossi_cx']))
    gap0 = sorted(abs(p['pct0'] - p['rossi_cx']) for p in both)
    gapz = sorted(abs(p['pctz'] - p['rossi_cx']) for p in both)
    med = lambda v: (v[len(v) // 2] + v[(len(v) - 1) // 2]) / 2 if v else None
    genic = dict(rounds=genic_rounds, groups=len(GEN), rossi_pooled_cx=rossi_pooled_cx, per=per_genic,
                 n_big=len(big), n_both=len(both), closer=closer, med_gap0=med(gap0), med_gapz=med(gapz),
                 med_call0=med(sorted(p['pct0'] for p in both)), med_callz=med(sorted(p['pctz'] for p in both)),
                 med_rossi=med(sorted(p['rossi_cx'] for p in both)))

    # Per factor, round 0 vs the last round, on the common groups.
    per = []
    for g in COMMON:
        row = dict(group=g, fitted=g in FITG, lam0=float(rep0[g]['lambda']))
        for ref in ('macisaac', 'rossi_cx'):
            a, z = M[(0, ref)][g], M[(t, ref)][g]
            row[ref] = dict(sites=a[0], calls0=a[1], m0=a[2], callsz=z[1], mz=z[2],
                            f10=_prf(a[1], a[2], a[0])['f1'], f1z=_prf(z[1], z[2], z[0])['f1'])
        per.append(row)
    rossi['runs'][run] = dict(round=t, rounds=rounds, marginal=marg, genic=genic, per=per)
    z = rounds[-1]['all']
    print('%s rossi: F1 round 0 -> %d  MacIsaac(common) %.4f -> %.4f  cx %.4f -> %.4f; genic calls %.1f%% -> %.1f%% vs Rossi %.1f%%'
          % (run, t, rounds[0]['all']['macisaac']['f1'], z['macisaac']['f1'], rounds[0]['all']['rossi_cx']['f1'],
             z['rossi_cx']['f1'],
             genic_rounds[0]['pct'], genic_rounds[-1]['pct'], rossi_pooled_cx))

GENIC_NULL = float(_tsv(os.path.join(AN, 'rossi_genic/rossi_genic_all_TFs.tsv'))[0]['null_genic_pct'])
rossi['genic_null'] = GENIC_NULL
out['rossi'] = rossi

blob = json.dumps(out, separators=(',', ':'), allow_nan=False)
assert '</script' not in blob
tpl = open(os.path.join(HERE, 'conc_sheet_template.html')).read()
open(os.path.join(HERE, 'conc_sheet.html'), 'w').write(tpl.replace('__DATA__', blob))
print('payload %.0f KB -> conc_tuning/conc_sheet.html' % (len(blob) / 1024))
