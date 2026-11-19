"""Evaluate archived official-test checkpoints with aligned additive event noise."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch

def save(out, key, labels, inputs, predict, rates, tail_rate, batch):
    pred = {'labels': labels, 'ids': np.arange(len(labels))}
    with torch.no_grad():
        for name, rate in [('clean',0), *[(f'overlay-{r:g}',r) for r in rates], ('tail80',tail_rate)]:
            values = []
            for start in range(0,len(labels),batch):
                samples = []
                for i in range(start,min(start+batch,len(labels))):
                    x = np.asarray(inputs[i], dtype=np.float32)
                    if rate:
                        rng = np.random.default_rng(1101 + i*1009)
                        shape = (80,*x.shape[1:]) if name=='tail80' else x.shape
                        noise = rng.poisson(rate, shape).astype(np.float32)
                        x = np.concatenate((x, noise)) if name=='tail80' else x+noise
                    samples.append(x)
                scores = predict(torch.from_numpy(np.stack(samples)),name=='tail80')
                assert torch.isfinite(scores).all()
                values.extend(scores.argmax(-1).tolist())
            pred[name]=np.asarray(values)
            print(key,name,round(np.mean(pred[name]==labels),4),flush=True)
    np.savez_compressed(out/(key+'.npz'),**pred)

def shd(args):
    exp=args.experiments
    base=exp/('shd-supervision-ablation' if args.task=='shd-vector' else 'shd-streaming')
    sys.path.insert(0,str(base))
    from model import StreamModel
    from data import arrays
    x,y,_=arrays('test')
    for seed in ([501,502,503] if args.task=='shd-vector' else [301,302,303]):
        vector=args.task=='shd-vector'
        for name in (['full','stable-off'] if vector else ['dmp','commit_dmp']):
            key=f'{args.task}-{seed}-{name}'
            if (args.out/(key+'.npz')).exists():continue
            if vector:
                path=base/'runs'/f'zero_rule-d30-s{seed}'/'final.pt'
                model=StreamModel('commit_dmp',30,'zero',True,.02,'vector',True,True)
            else:
                run=json.loads((base/'evaluation'/f'{name}-d30-s{seed}'/'results.json').read_text())
                path=Path(run['checkpoint'])
                assert hashlib.sha256(path.read_bytes()).hexdigest()==run['checkpoint_sha256']
                model=StreamModel(name,30)
            model.load_state_dict(torch.load(path,map_location='cpu')['model_state_dict'])
            model.eval()
            if name=='stable-off':
                for cell in model.cells():cell.read_m.weight.data.zero_()
            def predict(batch,tail):
                model.reset_state()
                z=model(batch)
                return z[:,-20:].mean(1) if tail else z.mean(1)
            save(args.out,key,y,x,predict,[.0625,.125,.25],.25,48)

def gesture(args):
    base=args.experiments/'final-study'
    sys.path.insert(0,str(base/'code'))
    from vision_tasks.final_study import make_model,horizon
    data=base/'gesture-test-t100'
    samples=json.loads((data/'manifest.json').read_text())['samples']
    labels=np.array([r['label'] for r in samples])
    xs=[]
    for r in samples:
        with np.load(data/r['file']) as f:xs.append(f['frames'])
    trained=base/'results'/'finetuned'
    rate=json.loads((trained/'protocol.json').read_text())['noise_rate']
    for name in ('dmp','commit'):
        key='gesture-'+name
        if (args.out/(key+'.npz')).exists():continue
        model=make_model(torch.load(trained/name/'last.pt',map_location='cpu'),'cpu')
        def predict(batch,tail):
            horizon(model,batch.shape[1])
            z=model(batch)
            return model.lif4.v_seq[-20:].mean(0) if tail else z
        save(args.out,key,labels,xs,predict,[rate*.5,rate,rate*3],rate,4)
        arch=np.load(base/'results/official-evaluation/predictions.npz')
        fresh=np.load(args.out/(key+'.npz'))
        assert np.array_equal(labels,arch['labels'])
        print('archived clean agreement',name,np.mean(fresh['clean']==arch[f'finetuned__{name}__clean__mean']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['shd','shd-vector','gesture'])
    p.add_argument('--experiments',type=Path,default=Path('/workspace/experiments'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    (gesture if a.task=='gesture' else shd)(a)
