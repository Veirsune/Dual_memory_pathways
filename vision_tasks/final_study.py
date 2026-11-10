"""Frozen-protocol gesture fine-tuning and official-test evaluation; no gate labels."""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from .models import build
from .cpu_lif import enable_cpu_scan
from .run import atomic_save
from .corruption_eval import corrupt

FAMILIES=('fsnn','dmp','commit')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_samples(root,files):
    result=[]
    for name in files:
        with np.load(root/name) as z:result.append(z['frames'].astype(np.float32))
    return result

def make_model(cp,device):
    c=cp['config'];model=build('gesture',c['memory'],c['theta'],100,c['family'],c['readout']);model.load_state_dict(cp['model']);model.to(device).eval()
    if device=='cpu':enable_cpu_scan(model)
    else:
        for module in model.modules():
            if isinstance(module,MultiStepLIFNode):module.backend='cupy'
    return model

def horizon(model,t):
    if model.family!='fsnn' and model.fc1.seq_len!=t:
        model.fc1.seq_len=t;model.fc1.H,model.fc1.fft_H=model.fc1.impulse()

def training_batch(xs,ids,case,epoch,rate):
    return np.stack([corrupt(xs[int(i)],case['position'],case['length'],case['kind'],rate,70100000+epoch*10000+int(i)) for i in ids])

def plan(n,epochs):
    rng=np.random.default_rng(701);out=[]
    for epoch in range(epochs):
        order=rng.permutation(n);batches=[]
        for begin in range(0,n,4):
            kind=str(rng.choice(['clean','silence','poisson']));length=0 if kind=='clean' else int(rng.choice([20,40,80]));pos=int(rng.choice([50,100]))
            batches.append(dict(ids=order[begin:begin+4].tolist(),position=pos,length=length,kind=kind))
        out.append(batches)
    return out

def train(a):
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    cps={f:torch.load(a.weights/f/'best.pt',map_location='cpu') for f in FAMILIES};split=cps['dmp']['split'];assert all(c['split']==split for c in cps.values())
    manifest=json.loads((a.data/'manifest.json').read_text());assert sha(a.data/'manifest.json')==split['manifest_sha256']
    rows={r['file']:r for r in manifest['samples']};files=split['train_files'];xs=load_samples(a.data,files);labels=np.array([rows[f]['label'] for f in files]);rate=sum(x.sum(dtype=np.float64) for x in xs)/sum(x.size for x in xs)
    schedule=plan(len(xs),20);(a.output/'training-plan.json').write_text(json.dumps(schedule))
    protocol=dict(epochs=20,seed=701,optimizer='fresh Adam',lr=.0001,batch=4,train_n=len(xs),train_files=files,validation_files=split['validation_files'],noise_rate=rate,train_conditions='uniform clean/silence/poisson; L20/40/80; position50/100; Poisson1x only',loss='0.5 CE(mean membrane, class) + 0.5 CE(last20 mean membrane, class)',noise_position_labels=False,gate_loss=False,checkpoint_selection='fixed epoch20, no TEST or VALID selection',zero_rule='existing Commit raw-input==0 gate closure; original DMP/FSNN unchanged',checkpoint_sha256={f:sha(a.weights/f/'best.pt') for f in FAMILIES},plan_sha256=sha(a.output/'training-plan.json'),device=a.device,torch=torch.__version__)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));all_start=time.time()
    for family in FAMILIES:
        dest=a.output/family;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():continue
        torch.manual_seed(701);random.seed(701);np.random.seed(701)
        model=make_model(cps[family],a.device);opt=torch.optim.Adam(model.parameters(),lr=.0001);history=[];start=time.time();batch_hash=hashlib.sha256()
        for epoch,batches in enumerate(schedule):
            model.train();total=0.;correct=0;n=0;ts=time.time()
            for index,case in enumerate(batches):
                x=training_batch(xs,case['ids'],case,epoch,rate);batch_hash.update(x.tobytes());target=torch.as_tensor(labels[case['ids']],device=a.device);horizon(model,x.shape[1])
                opt.zero_grad(set_to_none=True);out=model(torch.from_numpy(x).to(a.device));tail=model.lif4.v_seq[-20:].mean(0);loss=(F.cross_entropy(out,target)+F.cross_entropy(tail,target))/2
                assert torch.isfinite(loss);loss.backward()
                if epoch==0 and index==0:
                    grads=[p.grad for p in model.parameters() if p.grad is not None];assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.square().sum()) for g in grads)>0
                opt.step();n+=len(target);total+=float(loss.detach())*len(target);correct+=int((out.argmax(1)==target).sum())
            row=dict(epoch=epoch+1,loss=total/n,accuracy=correct/n,seconds=time.time()-ts);history.append(row)
            # Restore canonical horizon for portable checkpoints; no learned weights change.
            horizon(model,100)
            atomic_save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},config=cps[family]['config'],split=split,epoch=epoch+1,finetune_protocol=protocol,optimizer=opt.state_dict()),dest/'last.pt')
            (dest/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(dict(family=family,**row)),flush=True)
        (dest/'complete.json').write_text(json.dumps(dict(epochs=20,wall_seconds=time.time()-start,input_sha256=batch_hash.hexdigest(),checkpoint_sha256=sha(dest/'last.pt'))))
    complete=[json.loads((a.output/f/'complete.json').read_text()) for f in FAMILIES];assert len(set(q['input_sha256'] for q in complete))==1
    (a.output/'complete.json').write_text(json.dumps(dict(completed=True,epochs_per_model=20,input_identical=True,wall_seconds=time.time()-all_start)))

def evaluate(a):
    torch.set_num_threads(1);a.output.mkdir(parents=True,exist_ok=True)
    train_protocol=json.loads((a.finetuned/'protocol.json').read_text());assert json.loads((a.finetuned/'complete.json').read_text())['completed']
    m=json.loads((a.test/'manifest.json').read_text());assert m['split']=='official TEST via Tonic mirror';rows=m['samples'];assert len(rows)==264
    assert not set(r['subject'] for r in rows)&set(range(1,24))
    xs=load_samples(a.test,[r['file'] for r in rows]);labels=np.array([r['label'] for r in rows]);subjects=np.array([r['subject'] for r in rows]);preds=dict(labels=labels,subjects=subjects);metrics=[];cases=[dict(name='clean',position=100,length=0,kind='silence',multiplier=0.,seed=0)]
    for pos in (50,100):
        for kind,mult in (('silence',0.),('poisson',1.),('poisson',3.)):
            for length in (20,40,80):
                for seed in ((0,) if kind=='silence' else (1101,2202,3303)):
                    cases.append(dict(name=f'p{pos}_{kind}{mult:g}_L{length}_s{seed}',position=pos,length=length,kind=kind,multiplier=mult,seed=seed))
    protocol=dict(test_manifest_sha256=sha(a.test/'manifest.json'),test_n=264,cases=cases,noise_rate=train_protocol['noise_rate'],official_test=True,selection='all six fixed checkpoints evaluated after all training; no retraining',readouts=['mean: primary','last20: endpoint diagnostic, trained equally for all fine-tuned models'],checkpoint_sha256={f'{stage}_{f}':sha((a.weights/f/'best.pt') if stage=='zero' else a.finetuned/f/'last.pt') for stage in ('zero','finetuned') for f in FAMILIES})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.time()
    with torch.no_grad():
        for stage in ('zero','finetuned'):
            for family in FAMILIES:
                cp=torch.load(a.weights/family/'best.pt' if stage=='zero' else a.finetuned/family/'last.pt',map_location='cpu');model=make_model(cp,a.device)
                for case in cases:
                    horizon(model,100+case['length']);outs={k:[] for k in ('mean','last20')}
                    for begin in range(0,264,4):
                        batch=np.stack([corrupt(xs[i],case['position'],case['length'],case['kind'],train_protocol['noise_rate']*case['multiplier'],case['seed']+1009*i) for i in range(begin,min(begin+4,264))])
                        logits=model(torch.from_numpy(batch).to(a.device));tail=model.lif4.v_seq[-20:].mean(0);assert torch.isfinite(logits).all() and torch.isfinite(tail).all()
                        for key,v in (('mean',logits),('last20',tail)):outs[key].extend(v.argmax(-1).cpu().tolist())
                    for key,v in outs.items():
                        v=np.array(v);preds[f'{stage}__{family}__{case["name"]}__{key}']=v;metrics.append(dict(stage=stage,family=family,metric=key,**case,n=264,correct=int((v==labels).sum()),accuracy=float((v==labels).mean())))
                    (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(a.output/'predictions.npz',**preds)
                    print(json.dumps(dict(stage=stage,family=family,case=case['name'],mean=metrics[-2]['accuracy'],last20=metrics[-1]['accuracy'])),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(completed=True,models=6,cases_per_model=43,samples=264,wall_seconds=time.time()-start)))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate']);p.add_argument('--data',type=Path);p.add_argument('--test',type=Path);p.add_argument('--weights',type=Path,required=True);p.add_argument('--finetuned',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    (train if a.mode=='train' else evaluate)(a)
if __name__=='__main__':main()
