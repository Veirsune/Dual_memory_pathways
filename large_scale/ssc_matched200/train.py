"""Matched SSC reproduction and memory extension. Run alongside commit_memory, large_scale, upstream_ssc.py."""
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from upstream_ssc import ConvLMU2
from large_scale.models import FSNN
from commit_memory.base_model import StreamModel
from commit_memory.vector_ops import matrix_kernel, vector_fft

BASE=Path(os.environ.get('SSC_ROOT','/content/ssc200'))
OUT=BASE/'results'
OUT.mkdir(parents=True,exist_ok=True)
PROTO=json.loads(Path(os.environ.get('SSC_PROTOCOL', Path(__file__).with_name('protocol.json'))).read_text())
SEED=PROTO['seed']
BATCH=PROTO['batch']
DEVICE=os.environ.get('SSC_DEVICE','cuda')
FAMILIES=PROTO['models']


def write(name, data, binary=False):
    path=OUT/name
    tmp=path.with_suffix('.partial')
    if binary: torch.save(data,tmp)
    else: tmp.write_text(json.dumps(data,indent=2))
    tmp.replace(path)


def load_split(split):
    data=BASE/'data'
    x=np.load(data/f'{split}-x.npy',mmap_mode='r')
    y=np.load(data/f'{split}-y.npy')
    assert len(y)=={'train':75466,'valid':9981,'test':20382}[split] and x.shape==(len(y),250,140)
    return torch.from_numpy(np.asarray(x)).to(DEVICE),torch.from_numpy(y).to(DEVICE)


def vector_candidate(cell,u):
    n=u.size(1)
    if n not in cell._kernels:
        cell._kernels[n]=matrix_kernel(cell.A,cell.vector_input_scale,n)
    return vector_fft(u,cell._kernels[n])


def make_models(scale):
    torch.manual_seed(SEED)
    width,second=PROTO['hidden']
    assert width==second
    dmp=ConvLMU2(PROTO['memory_dimension_per_state'],250,PROTO['theta'],hidden=width)
    fsnn=FSNN(hidden=width)
    fsnn.drop=nn.Identity()
    gdsm=StreamModel(variant='commit_dmp',d=10,gate_mode='zero',gate_threshold=.02,drive='vector',protect=True,zero_rule=True,hidden=width)
    gdsm.rnn_drop=nn.Identity()
    gdsm.dense=copy.deepcopy(dmp.dense)
    del gdsm.memory_head
    for c in gdsm.cells():
        del c.bn_x
        c.candidate=MethodType(vector_candidate,c)
    gdsm.warm_start(dmp.state_dict())
    gdsm.LMU1.feature_scale.fill_(scale)
    gdsm.permit.scale.fill_(scale)
    with torch.no_grad():
        gdsm.permit.net[-1].weight.zero_()
        gdsm.permit.net[-1].bias.fill_(2.)
    for name,cell in [('fc1',dmp.LMU1),('fc2',dmp.LMU2)]:
        getattr(fsnn,name).load_state_dict(cell.W_x.state_dict())
    fsnn.dense.load_state_dict(dmp.dense.state_dict())
    available=dict(fsnn=fsnn,dmp=dmp,gdsm=gdsm)
    models={k:available[k] for k in FAMILIES}
    for model in models.values():
        model.to(DEVICE)
        if DEVICE=='cuda':
            for mod in model.modules():
                if isinstance(mod,MultiStepLIFNode): mod.backend='cupy'
    return models


def scores(family,model,x):
    if family=='dmp':
        functional.reset_net(model)
        for c in (model.LMU1,model.LMU2):
            if c.seq_len!=x.shape[1]:
                c.seq_len=x.shape[1]
                c.H,c.fft_H=c.impulse()
        return model.dense(model.forward_features(x))
    if family=='gdsm': model.reset_state()
    return model(x)


def state_dict(model):
    # Upstream kernels are derived from A and B; checkpoints use the clean sequence length.
    if isinstance(model,ConvLMU2):
        for c in (model.LMU1,model.LMU2):
            if c.seq_len!=250:
                c.seq_len=250
                c.H,c.fft_H=c.impulse()
    return model.state_dict()


def stimulus(x,kind,pos,length,rate,seed):
    if kind=='clean':return x
    rng=torch.Generator(device=x.device).manual_seed(seed)
    if kind=='speech':return x+torch.poisson(torch.full_like(x,rate),generator=rng)
    gap=x.new_zeros(len(x),length,140)
    if kind=='noise':gap=torch.poisson(gap.fill_(rate),generator=rng)
    return torch.cat([x[:,:pos],gap,x[:,pos:]],dim=1)


@torch.no_grad()
def evaluate(fam,model,data,cases):
    model.eval()
    x,y=data
    result={}
    for kind,pos,L,rate in cases:
        hits=[0,0]
        for i in range(0,len(x),BATCH):
            v=stimulus(x[i:i+BATCH].float(),kind,pos,L,rate,120000+i)
            s=scores(fam,model,v)
            for k,read in enumerate((s.mean(1),s[:,-20:].mean(1))):
                assert torch.isfinite(read).all()
                hits[k]+=int((read.argmax(1)==y[i:i+BATCH]).sum())
        result[f'{kind}:{pos}:{L}:{rate}']=dict(n=len(y),whole=hits[0]/len(y),endpoint=hits[1]/len(y))
    return result


def main():
    torch.set_num_threads(1)
    x,y=load_split('train'); valid=load_split('valid')
    ids=np.random.default_rng(6501).choice(len(y),2048,replace=False)
    scale=max(.05,float(torch.quantile(x[ids].float().mean(-1).flatten(),.9)))
    models=make_models(scale)
    diagnose=PROTO.get('width_diagnostics',False)
    if diagnose:
        from width_diagnostics import drive_rates, paired_test
        idx=np.random.default_rng(123).choice(len(valid[1]),128,replace=False)
        probe=valid[0][idx].float()
    write('model-info.json',dict(parameters={k:sum(v.numel() for v in m.parameters() if v.requires_grad) for k,m in models.items()},gpu=torch.cuda.get_device_name() if DEVICE=='cuda' else 'cpu',torch=torch.__version__,feature_scale=scale))
    clean=[('clean',250,0,0)]
    retention=clean+[('silence',250,80,0),('noise',250,80,.25)]
    phases=[p for p in ['clean','mixed'] if PROTO[f'{p}_epochs']>0]
    for phase in phases:
        optimizers={}; schedules={}; history={}; best={}
        for fam,m in models.items():
            if phase=='mixed':m.load_state_dict(torch.load(OUT/f'{fam}-clean-best.pt',map_location=DEVICE))
            opt=(torch.optim.Adam(m.parameters(),lr=PROTO.get('clean_lr', .1)) if phase=='clean' else torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.05))
            sched=(torch.optim.lr_scheduler.StepLR(opt,80,.5) if phase=='clean' else torch.optim.lr_scheduler.CosineAnnealingLR(opt,12,eta_min=.00005))
            checkpoint=OUT/f'{fam}-{phase}-last.pt'
            rows=[]; top=-1.
            if checkpoint.exists():
                cp=torch.load(checkpoint,map_location=DEVICE)
                m.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);sched.load_state_dict(cp['scheduler']);rows=cp['history'];top=cp['best']
            optimizers[fam]=opt;schedules[fam]=sched;history[fam]=rows;best[fam]=top
        epochs=PROTO[f'{phase}_epochs']
        for epoch in range(epochs):
            epseed=SEED+1009*epoch+(0 if phase=='clean' else 1000000)
            order=np.random.default_rng(epseed).permutation(len(y))
            for fam in FAMILIES:
                if len(history[fam])>epoch:continue
                m=models[fam];opt=optimizers[fam];m.train();rng=np.random.default_rng(epseed+1)
                tick=time.monotonic()
                total=0.;hits=0
                for j,start in enumerate(range(0,len(y),BATCH)):
                    idx=order[start:start+BATCH];a=x[idx].float();target=y[idx]
                    if phase=='mixed':
                        a=stimulus(a,str(rng.choice(['clean','silence','noise'])),int(rng.choice([125,250])),int(rng.choice([40,80])),.25,epseed*1000+j)
                    opt.zero_grad(set_to_none=True)
                    z=scores(fam,m,a);average=z.mean(1)
                    loss=F.cross_entropy(average,target)
                    if phase=='mixed':loss=(loss+F.cross_entropy(z[:,-20:].mean(1),target))/2
                    if not torch.isfinite(loss):raise ValueError((fam,phase,epoch,j,'nonfinite loss'))
                    loss.backward()
                    if phase=='mixed':nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
                    opt.step();total+=float(loss.detach())*len(idx);hits+=int((average.argmax(1)==target).sum())
                    if j%100==0:write('progress.json',dict(phase=phase,family=fam,epoch=epoch+1,batch=j+1,updates_per_epoch=295))
                val=evaluate(fam,m,valid,clean if phase=='clean' else retention)
                metric=np.mean([r['whole' if phase=='clean' else 'endpoint'] for r in val.values()]).item()
                row=dict(epoch=epoch+1,train_accuracy=hits/len(y),loss=total/len(y),val=val,selection=metric,seconds=time.monotonic()-tick)
                if diagnose:
                    row['memory_drive_positive']=drive_rates(m,fam,probe,scores)
                history[fam].append(row);schedules[fam].step()
                if metric>best[fam]:
                    write(f'{fam}-{phase}-best.pt',state_dict(m),True);best[fam]=metric
                write(f'{fam}-{phase}-last.pt',dict(model=state_dict(m),optimizer=opt.state_dict(),scheduler=schedules[fam].state_dict(),history=history[fam],best=best[fam]),True)
                write(f'{fam}-{phase}-history.json',history[fam])
                print(f'{phase} {epoch+1}/{epochs} {fam} val={metric:.4f}',flush=True)
    test=load_split('test')
    cases=clean+[('speech',0,250,r) for r in [.0625,.125,.25]]+[(k,250,L,.25 if k=='noise' else 0) for L in [80,120] for k in ['silence','noise']]+[('noise',125,80,.25)]
    if PROTO.get("test_clean_only",False):cases=clean
    result={}
    for fam,m in models.items():
        for phase in phases:
            state_dict(m)  # Restore upstream FFT buffer shape after variable-length evaluation.
            m.load_state_dict(torch.load(OUT/f'{fam}-{phase}-best.pt',map_location=DEVICE))
            result[f'{fam}-{phase}']=evaluate(fam,m,test,cases)
            if diagnose:
                paired_test(m,fam,test,scores,stimulus,BATCH,OUT/f'{fam}-paired.npz')
    write('test.json',result);write('complete.json',{'completed':True})

if __name__=='__main__':main()
