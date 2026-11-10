"""Export a trained MNIST first-layer memory trace, not a full classifier test."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from mnist_study.models import initial_models,backend
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
cp=torch.load(a.checkpoint,map_location='cpu');model=backend(initial_models(cp['seed'])['commit'],'cpu');model.load_state_dict(cp['model']);model.eval();cell=model.cell1;cell.capture=True
z=np.load(a.data);image=z['test_x'][0].astype(np.float32)/255.;rng=np.random.default_rng(2701);noise=np.minimum(rng.poisson(.1,80),1).astype(np.float32);raw=np.concatenate((image,np.zeros(80,np.float32),noise));x=torch.from_numpy(raw)[None,:,None]
with torch.no_grad():
 cell.permission=model.permission(x)[...,:40];cell(x)
 floating={key:cell.trace[key][0].numpy() for key in ('p','m','gate')};floating['current']=(cell.weight(cell.trace['p'])+cell.read_m(cell.trace['m']))[0,:,:4].numpy()
Q=4096;bits=24;lo=-(1<<(bits-1));hi=(1<<(bits-1))-1
clipped={}
def quant(name,t):
 arr=t.detach().numpy() if isinstance(t,torch.Tensor) else np.asarray(t);v=np.rint(arr*Q);clipped[name]=int(((v<lo)|(v>hi)).sum());return np.clip(v,lo,hi).astype(np.int64)
w={
 'drive':quant('drive',cell.W_u.weight),'bias':quant('bias',cell.W_u.bias),
 'a':quant('a',cell.A),'b':quant('b',cell.vector_input_scale),
 'rp':quant('rp',cell.weight.weight[:4]),'rpbias':quant('rpbias',cell.weight.bias[:4]),'rm':quant('rm',cell.read_m.weight[:4]),
 'local':quant('local',cell.gate.weight),'localBias':quant('localBias',cell.gate.bias),
 'hidden':quant('hidden',model.permission.net[0].weight),'hiddenBias':quant('hiddenBias',model.permission.net[0].bias),
 'out':quant('out',model.permission.net[2].weight[:40]),'outBias':quant('outBias',model.permission.net[2].bias[:40])}
rawq=quant('inputs',raw);d=40;state=np.zeros(d,np.int64);stable=state.copy();ema=np.zeros(4,np.int64);rawema=ema.copy();prev=0;prevraw=0;traces=[];rho=round(.8*Q)
def sat(v):return np.clip(v,lo,hi).astype(np.int64)
def affine(w,x,b):return sat((w@x+b*Q)>>12)
def features(v,previous):
 mean=int(np.trunc(v.sum()/len(v)));return sat(np.array([mean,abs(mean-previous),int(((v-mean)**2).sum()//len(v))>>12,int((v>0).sum())*Q//len(v)],np.int64))
for value in rawq:
 v=np.array([value],np.int64);old=state.copy();f=features(v,prev);rf=features(v,prevraw);lf=np.r_[f,ema];pf=np.r_[rf,rawema]
 u=np.maximum(affine(w['drive'],v,w['bias']),0);state=sat((w['a']@old+w['b']*u)>>12)
 hidden=np.maximum(affine(w['hidden'],pf,w['hiddenBias']),0);allow=affine(w['out'],hidden,w['outBias'])>=0
 logit=affine(w['local'],lf,w['localBias']);idx=np.clip((logit+8*Q+Q//32)>>8,0,256)
 # Math.round on positive values = floor(value+.5), including ties.
 probability=np.floor(Q/(1+np.exp(-(-8+idx/16)))+.5).astype(np.int64)
 gate=np.where(allow & (value>0) & (probability>=82),probability,0)
 stable=sat(((Q-gate)*stable+gate*old)>>12);current=sat(affine(w['rp'],state,w['rpbias'])+sat((w['rm']@stable)>>12))
 traces.append(dict(p=state.tolist(),m=stable.tolist(),g=gate.tolist(),current=current.tolist()))
 ema=sat((rho*ema+(Q-rho)*f)>>12);rawema=sat((rho*rawema+(Q-rho)*rf)>>12);prev=int(f[0]);prevraw=int(value)
metrics={}
for key,target in [('p','p'),('m','m'),('g','gate'),('current','current')]:
 fixed=np.array([t[key] for t in traces],np.float64)/Q;ref=floating[target]
 metrics[key]=dict(mae=float(np.abs(fixed-ref).mean()),max_abs=float(np.abs(fixed-ref).max()))
metrics['gate_open_agreement']=float(((np.array([t['g'] for t in traces])>0)==(floating['gate']>0)).mean())
record=dict(config=dict(inputs=1,neurons=4,d=40,bits=24,frac=12,permissionHidden=32),weights={k:v.tolist() for k,v in w.items()},raw=rawq.tolist(),expected=traces,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),source=dict(task=cp['task'],seed=cp['seed'],phase=cp['phase'],test_index=0,label=int(z['test_y'][0]),layer='cell1; first4 memory readout coordinates; excludes fast path/LIF/classifier',sequence='784 pixels+80 silence+80 clipped Poisson(.1)'),quantization_clips=clipped,float_comparison=metrics)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record));a.output.with_suffix('.summary.json').write_text(json.dumps({k:v for k,v in record.items() if k not in ('weights','raw','expected')},indent=2));print(json.dumps(metrics,indent=2))
