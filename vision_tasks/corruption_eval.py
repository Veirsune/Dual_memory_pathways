"""Paired, label-blind corruption evaluation of frozen gesture checkpoints."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from .models import build
from .cpu_lif import enable_cpu_scan

def corrupt(x, position, length, kind, rate, seed):
    if length == 0: return x
    rng = np.random.default_rng(seed)
    # Generate maximum length first: L20/40/80 are nested realizations.
    block = np.zeros((80, *x.shape[1:]), dtype=np.float32) if kind == 'silence' else rng.poisson(rate, (80, *x.shape[1:])).astype(np.float32)
    return np.concatenate((x[:position], block[:length], x[position:]), axis=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=True);start=time.time()
    checkpoints={f:torch.load(a.weights/f/'best.pt',map_location='cpu') for f in ('fsnn','dmp','commit')}
    cp=checkpoints['dmp'];split=cp['split']
    assert all(v['split']==split for v in checkpoints.values())
    manifest=json.loads((a.data/'manifest.json').read_text());byfile={r['file']:r for r in manifest['samples']}
    assert hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest()==split['manifest_sha256']
    # Noise intensity derives solely from TRAIN, with no per-test-example adaptation.
    total=0.;count=0
    for name in split['train_files']:
        with np.load(a.data/name) as z: x=z['frames'];total+=float(x.sum(dtype=np.float64));count+=x.size
    rate=total/count
    files=split['validation_files'][:4] if a.smoke else split['validation_files']
    xs=[]
    for name in files:
        with np.load(a.data/name) as z:xs.append(z['frames'].astype(np.float32))
    y=np.array([byfile[f]['label'] for f in files]);subjects=np.array([byfile[f]['subject'] for f in files])
    cases=[dict(name='clean',position=100,length=0,kind='silence',multiplier=0.,seed=0)]
    for pos in (50,100):
        for kind,mult in (('silence',0.),('poisson',1.),('poisson',3.)):
            for length in (20,40,80):
                for seed in ((0,) if kind=='silence' else (101,202,303)):
                    cases.append(dict(name=f'p{pos}_{kind}{mult:g}_L{length}_s{seed}',position=pos,length=length,kind=kind,multiplier=mult,seed=seed))
    if a.smoke:cases=[cases[0],cases[-1]]
    protocol=dict(training='frozen clean-trained CPU checkpoints, 50 epochs, seed42; best clean VALID selection',train_noise_rate_per_pooled_pixel=rate,cases=cases,validation_files=files,official_test_evaluated=False,checkpoint_sha256={f:hashlib.sha256((a.weights/f/'best.pt').read_bytes()).hexdigest() for f in checkpoints},device=a.device,torch=torch.__version__,metrics=['mean (training readout)','last10 (diagnostic)','last (diagnostic)'],limitations=['One training seed; noise seeds are not training replications.','VALID previously used for checkpoint selection.','Commit has more parameters and states than DMP.','Insertion extends sequence, does not replace gesture frames; theta stays fixed.'])
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];predictions={'labels':y,'subjects':subjects};baseline={}
    with torch.no_grad():
        for family,cp in checkpoints.items():
            c=cp['config'];model=build('gesture',c['memory'],c['theta'],100,family,c['readout']);model.load_state_dict(cp['model']);model.to(a.device).eval()
            if a.device=='cpu':enable_cpu_scan(model)
            else:
                for module in model.modules():
                    if isinstance(module,MultiStepLIFNode):module.backend='cupy'
            for case in cases:
                t=100+case['length']
                if family!='fsnn':
                    # Regenerate deterministic impulse buffers for the extended horizon;
                    # preserve trained parameters and original physical time constant.
                    model.fc1.seq_len=t;model.fc1.H,model.fc1.fft_H=model.fc1.impulse()
                outputs={k:[] for k in ('mean','last10','last')}
                for begin in range(0,len(xs),4):
                    batch=[corrupt(xs[i],case['position'],case['length'],case['kind'],rate*case['multiplier'],case['seed']+1009*i) for i in range(begin,min(begin+4,len(xs)))]
                    out=model(torch.from_numpy(np.stack(batch)).to(a.device))
                    seq=model.lif4.v_seq
                    assert torch.isfinite(out).all() and torch.isfinite(seq).all()
                    for key,z in (('mean',out),('last10',seq[-10:].mean(0)),('last',seq[-1])):outputs[key].extend(z.argmax(-1).cpu().tolist())
                for metric,values in outputs.items():
                    pred=np.array(values);predictions[f'{family}__{case["name"]}__{metric}']=pred
                    if case['length']==0:baseline[family,metric]=pred==y
                    clean_correct=baseline[family,metric]
                    rows.append(dict(family=family,metric=metric,**case,accuracy=float((pred==y).mean()),correct=int((pred==y).sum()),n=len(y),retained_clean_correct=float((pred[clean_correct]==y[clean_correct]).mean()) if clean_correct.any() else None))
                (a.output/'rows.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(a.output/'predictions.npz',**predictions)
                print(json.dumps(dict(family=family,case=case['name'],mean_accuracy=rows[-3]['accuracy'],last10_accuracy=rows[-2]['accuracy'],elapsed_seconds=time.time()-start)),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(completed=True,cases_per_model=len(cases),models=3,validation_samples=len(y),wall_seconds=time.time()-start)))
if __name__=='__main__':main()
