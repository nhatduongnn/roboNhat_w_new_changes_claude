import sys, os, numpy as np
A='/usr/project/xtmp/nd141/programs/roboNhat_w_new_changes_claude/analysis'
os.chdir(A); sys.path.insert(0,'pkgvar/permol_seq_maskoff/'); sys.path.insert(0,'singlefiber')
import read_calls as RC
from robocop.utils.getNucleotides import getNucleotideSequence
BAM=("/usr/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/combine_sequencing_runs/"
     "merged_Mar20_barcode01_Jun25_barcode21-24_May07_barcode03-04.bam")
chrom,ws,we,fl='chrI',60001,65000,400
codes=getNucleotideSequence('inputs/SacCer3.fa',chrom,ws-fl,we+fl)
mols,es,ee,st=RC.read_molecules(BAM,chrom,ws,we,np.asarray(codes),flank=fl)
# NDR runs from checks_*.json: (61102,61242) and (62625,62748); motifs 61163-61177, 62657-62671
for name,(s,e),(ns,ne) in [('site1',(61163,61177),(61102,61242)),('site2',(62657,62671),(62625,62748))]:
    ninf=[];nme=[];ndr_rate_n=0;ndr_rate_k=0;zero14=[]
    for m in mols:
        c=m.calls[s-1-es:e-es]
        if (c!=RC.NO_OBS).sum()==0 and not (m.ref_start0<=s-1 and m.ref_end0>=e): continue
        if not (m.ref_start0<=s-1 and m.ref_end0>=e): continue
        ninf.append(int((c!=RC.NO_OBS).sum())); nme.append(int((c==RC.METH).sum()))
        nd=m.calls[ns-1-es:ne-es]; ndr_rate_n+=(nd!=RC.NO_OBS).sum(); ndr_rate_k+=(nd==RC.METH).sum()
        # sliding 14bp windows across NDR: fraction with informative>=1 and zero meth
        for o in range(0,len(nd)-14):
            w=nd[o:o+14]
            if (w!=RC.NO_OBS).sum()>0: zero14.append(int((w==RC.METH).sum()==0))
    ninf=np.array(ninf);nme=np.array(nme)
    print(name,'molecules',len(ninf),'informative calls in 14bp: median',np.median(ninf),'mean',ninf.mean().round(2),
          'frac with 0 meth in motif',(nme==0).mean().round(3),
          'NDR per-call meth rate',round(ndr_rate_k/ndr_rate_n,4),'(n=%d)'%ndr_rate_n,
          'P(no m6A in a 14bp NDR window | >=1 call)',round(np.mean(zero14),3))
tot_n=sum(m.n_informative for m in mols); print('all informative per bp of span', round(tot_n/sum(m.ref_end0-m.ref_start0 for m in mols),3))
