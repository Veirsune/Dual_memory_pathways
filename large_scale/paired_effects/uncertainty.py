"""Within-run cluster intervals on the common-clean and all-example contrasts."""
import argparse
import csv
from pathlib import Path
import h5py
import numpy as np

def bootstrap(labels,clusters,clean_a,clean_b,a,b):
    _, idx=np.unique(clusters,return_inverse=True)
    n=int(idx.max()+1)
    delta=(b==labels).astype(int)-(a==labels)
    pool=(clean_a==labels)&(clean_b==labels)
    stats=np.stack([np.bincount(idx,weights=w,minlength=n) for w in
                    (delta,np.ones(len(idx)),delta*pool,pool)],axis=1)
    rng=np.random.default_rng(907)
    sampled=[]
    for _ in range(40):
        draws=rng.integers(n,size=(100,n))
        x=stats[draws].sum(1)
        sampled.extend(np.stack([x[:,0]/x[:,1],x[:,2]/x[:,3]],axis=1))
    return n,(100*np.quantile(sampled,[.025,.975],axis=0)).T.flatten()

def main(args):
    zg=np.load(args.experiments/'final-study/results/official-evaluation/predictions.npz')
    with h5py.File(args.rawssc) as raw:
        speakers=raw['extra/speaker'][:] if 'extra/speaker' in raw else raw['speaker'][:]
        y=raw['labels'][:]
    archives=[('gesture',args.pred,'',zg['subjects'],zg['labels'],'overlay-1.81265'),
              ('ssc-original',args.pred,'-original',speakers,y,'overlay-0.125'),
              ('ssc-repair',args.repair,'-paired',speakers,y,'overlay-0.125')]
    rows=[]
    for ds,root,suffix,clusters,truth,mid in archives:
        if ds=='gesture':
            a,b=[np.load(root/('gesture-'+f+'.npz')) for f in ('dmp','commit')]
            pref=''
        else:
            a,b=[np.load(root/(f+suffix+'.npz')) for f in ('dmp','gdsm')]
            pref='full_'
        assert np.array_equal(truth,a['labels']) and np.array_equal(truth,b['labels'])
        for cond in ('clean',mid,'tail80'):
            n,ci=bootstrap(truth,clusters,a[pref+'clean'],b[pref+'clean'],a[pref+cond],b[pref+cond])
            rows.append([ds,cond,n,*ci])
    with args.output.open('w') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['dataset','condition','clusters','all_lo','all_hi','shared_lo','shared_hi'])
        w.writerows(rows)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('experiments','rawssc','pred','repair','output'):
        p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
