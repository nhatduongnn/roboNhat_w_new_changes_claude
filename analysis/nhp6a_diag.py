"""Why do linkers prefer Nhp6a?  Empirical: where do Nhp6a calls land, and at what m6A?"""
import sys, numpy as np, pandas as pd, pickle
sys.path.insert(0,'../pkg/'); sys.path.insert(0,'.')
import score_robocop as S

z = np.load('abf1_slide_cov.npz')
ntot=z['ntot']; kw=z['kw']; kc=z['kc']
meth = np.where(ntot>0, (kw+kc)/np.maximum(ntot,1), np.nan)
print("chrI m6A ratio: median %.4f mean %.4f p75 %.4f p90 %.4f"%(
    np.nanmedian(meth), np.nanmean(meth), np.nanpercentile(meth,75), np.nanpercentile(meth,90)), flush=True)

DEFAULT_RUNS=["robocop_chrI_seq_maskoff_revfix","robocop_chrI_seq_maskoff_capA","robocop_chrI_seq_maskoff_capB"]
# extra decode dirs may be named on the command line, e.g.
#   python nhp6a_diag.py robocop_chrI_seq_maskoff_em10
RUNS=DEFAULT_RUNS+[a for a in sys.argv[1:] if not a.startswith("-")]
rows=[]
for run in RUNS:
    dec=S.load_decode(run)
    op,cov,fr=S.region_optable(dec,"chrI",1,230218)
    mass=op.sum(0).sort_values(ascending=False)
    print("\n===",run,"=== top 15 states by posterior mass", flush=True)
    print(mass.head(15).round(1).to_string(), flush=True)
    n=len(op)
    mm=meth[:n]
    for tf in ["Nhp6a_zhu","Abf1_murphy","Reb1_badis","Nhp6b_zhu","Sig1_badis","Rox1_badis","Sum1_zhu","unknown","background","nucleosome"]:
        if tf not in op.columns: continue
        v=op[tf].values
        pos=np.where(v>=0.30)[0]
        m=mm[pos]
        rows.append(dict(run=run.split('_')[-1], state=tf, mass=float(v.sum()),
            calls=int(len(pos)),
            meth_med=float(np.nanmedian(m)) if len(m) else np.nan,
            meth_p90=float(np.nanpercentile(m,90)) if len(m) else np.nan))
df=pd.DataFrame(rows)
print("\n\nCALLS at posterior>=0.30 and the m6A where they land")
print(df.to_string(index=False, float_format=lambda x:"%.4f"%x))
df.to_csv("nhp6a_diag.tsv",sep="\t",index=False)
