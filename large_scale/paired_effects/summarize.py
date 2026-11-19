"""Generate aligned, sample-level contrasts. Each seed is a separate row."""
import argparse
import csv
from pathlib import Path
import numpy as np

def contrast(dataset, comparison, seed, condition, y, clean_a, clean_b, a, b):
    assert all(v.shape==y.shape for v in (clean_a,clean_b,a,b))
    pool = (clean_a==y) & (clean_b==y)
    def pct(mask):return 100*mask.mean()
    ok_a,ok_b = a==y,b==y
    aa,bb=ok_a[pool],ok_b[pool]
    return dict(dataset=dataset,comparison=comparison,seed=seed,condition=condition,n=len(y),
        clean_a=pct(clean_a==y),clean_b=pct(clean_b==y),acc_a=pct(ok_a),acc_b=pct(ok_b),
        delta_pp=pct(ok_b)-pct(ok_a),n_both_clean=int(pool.sum()),a_correct_shared=int(aa.sum()),
        b_correct_shared=int(bb.sum()),only_a_shared=int((aa&~bb).sum()),only_b_shared=int((bb&~aa).sum()),
        shared_delta_pp=pct(bb)-pct(aa))

def pairfiles(a,b):
    for name in ('ids','labels'):assert np.array_equal(a[name],b[name])
    assert set(a.files)==set(b.files)
    return [n for n in a.files if n not in ('labels','ids')]

def main(args):
    rows=[]
    for task,seeds,keys,label in [('shd',[301,302,303],('dmp','commit_dmp'),'DMP / scalar candidate-stable'),
                                ('shd-vector',[501,502,503],('stable-off','full'),'GDSM stable read off / full'),
                                ('gesture',[701],('dmp','commit'),'DMP / GDSM')]:
        for seed in seeds:
            prefix=(f'{task}-{seed}-' if task!='gesture' else 'gesture-')
            a,b=[np.load(args.pred/(prefix+k+'.npz')) for k in keys]
            for name in pairfiles(a,b):
                rows.append(contrast(task,label,seed,name,a['labels'],a['clean'],b['clean'],a[name],b[name]))
    if args.repair:
        a,b=[np.load(args.pred/n) for n in ('shd-301-dmp.npz','shd-matched-301-gdsm.npz')]
        for name in pairfiles(a,b):
            rows.append(contrast('shd-matched','DMP / GDSM',301,name,a['labels'],a['clean'],b['clean'],a[name],b[name]))
    zs=np.load(args.ssc/'ssc-speech/predictions.npz')
    zt=np.load(args.ssc/'ssc-position/predictions.npz')
    y=zs['labels'];assert np.array_equal(y,zt['labels'])
    for key,src,name in [('0',zs,'clean'),('1',zs,'overlay-0.0625'),('2',zs,'overlay-0.125'),('3',zs,'overlay-0.25'),('0',zt,'tail80')]:
        rows.append(contrast('ssc-archived','DMP / GDSM',801,name,y,zs['dmp_0'],zs['commit_0'],src['dmp_'+key],src['commit_'+key]))
    a,b=[np.load(args.pred/(k+'-original.npz')) for k in ('dmp','gdsm')]
    assert np.array_equal(a['labels'],b['labels']) and np.array_equal(a['ids'],b['ids'])
    for name in ['clean','overlay-0.0625','overlay-0.125','overlay-0.25','tail80']:
        rows.append(contrast('ssc-256-original','DMP / GDSM',42,name,a['labels'],a['full_clean'],b['full_clean'],a['full_'+name],b['full_'+name]))
    if args.repair:
        a,b=[np.load(args.repair/(k+'-paired.npz')) for k in ('dmp','gdsm')]
        assert np.array_equal(a['labels'],b['labels']) and np.array_equal(a['ids'],b['ids'])
        for mode,label,arm1,arm2,p1,p2 in [('full','DMP / GDSM',a,b,'full','full'),('stable','GDSM stable read off / full',b,b,'stable-off','full'),('dmp-memory','DMP all memory off / full',a,a,'all-off','full'),('gdsm-memory','GDSM all memory off / full',b,b,'all-off','full')]:
            for name in ['clean','overlay-0.0625','overlay-0.125','overlay-0.25','tail80']:
                rows.append(contrast('ssc-repair',label,42,name,arm1['labels'],arm1[p1+'_clean'],arm2[p2+'_clean'],arm1[p1+'_'+name],arm2[p2+'_'+name]))
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    for ds in dict.fromkeys(r['dataset'] for r in rows):
        group=[r for r in rows if r['dataset']==ds]
        for comparison in dict.fromkeys(r['comparison'] for r in group):
            sub=[r for r in group if r['comparison']==comparison]
            for condition in dict.fromkeys(r['condition'] for r in sub):
                selected=[r for r in sub if r['condition']==condition]
                print(ds,comparison,condition,*[round(float(np.mean([r[k] for r in selected])),2) for k in ('acc_a','acc_b','delta_pp','shared_delta_pp')])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pred',type=Path,required=True)
    p.add_argument('--ssc',type=Path,required=True)
    p.add_argument('--repair',type=Path)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
