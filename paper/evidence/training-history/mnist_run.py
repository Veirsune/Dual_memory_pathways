import argparse,json,hashlib,time,tarfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .models import initial_models,backend

FAMILIES=('fsnn','dmp','commit');TASKS=('smnist','psmnist');SEEDS=(42,43,44)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def save(p,v):
 q=p.with_suffix('.tmp');torch.save(v,q);q.replace(p)
def load(root,device):
 z=np.load(root/'mnist.npz');metadata=json.loads((root/'metadata.json').read_text());assert sha(root/'mnist.npz')==metadata['archive_sha256']
 for k in z.files:assert hashlib.sha256(z[k].tobytes()).hexdigest()==metadata[k]['sha256']
 return {k:torch.from_numpy(z[k].copy()).to(device) for k in z.files}
def scores(z):return z.mean(1),z[:,-20:].mean(1)
def batch_input(x,ids,perm):return x[ids][:,perm].float().unsqueeze(-1)/255.
def corrupt(x,kind,L,pos,seed):
 if L==0:return x
 gap=x.new_zeros(len(x),L,1)
 if kind=='poisson':
  gen=torch.Generator(device=x.device).manual_seed(seed);gap=torch.poisson(torch.full_like(gap,.1),generator=gen).clamp_max(1)
 return torch.cat((x[:,:pos],gap,x[:,pos:]),1)
@torch.no_grad()
def evaluate_clean(model,x,y,perm,batch):
 model.eval();count=[0,0]
 for i in range(0,len(x),batch):
  z=model(batch_input(x,slice(i,i+batch),perm))
  for k,v in enumerate(scores(z)):count[k]+=int((v.argmax(-1)==y[i:i+batch]).sum())
 return dict(mean=count[0]/len(x),last20=count[1]/len(x),n=len(x))
def snapshot(root,dest,key):
 p=root/f'snapshot-{key}.tar.gz'
 with tarfile.open(p.with_suffix('.tmp'),'w:gz') as t:t.add(dest,arcname=str(dest.relative_to(root)))
 p.with_suffix('.tmp').replace(p);write(p.with_suffix('.json'),dict(path=p.name,sha256=sha(p)))
def train(a):
 torch.set_num_threads(2);z=load(a.data,a.device);root=a.output;root.mkdir(parents=True,exist_ok=True);fingerprints={};started=time.time()
 protocol=dict(tasks=list(TASKS),seeds=list(SEEDS),families=list(FAMILIES),train_samples=54000,valid_samples=6000,test_samples=10000,steps=784,max_steps=944,d=40,hidden=200,layers=2,theta=300,lif_tau=2.,lif_threshold=1.,dropout=0.,batch=a.batch,epochs_clean=50,epochs_mixed=12,optimizer='Adam; clean lr .001 S-MNIST/.005 PS-MNIST, constant; mixed lr .0003 both tasks; clip norm1; no weight decay',initialization='paired PyTorch default Linear initialization (Kaiming uniform weights, fan-in uniform biases); NOT author trunc_normal std.02 zero-bias initialization; same fast path, output and scalar-expanded memory initial weights across families',clean_loss='CE(mean logits), original primary readout',mixed_loss='0.5 CE(mean logits)+0.5 CE(last20 logits), equal for all models',augmentation='none in clean phase; mixed uniform clean/silence/clipped Poisson(.1); L40/80; positions392/784; no noise labels',test='fixed final checkpoint of both stages; all10000; clean plus middle/tail silence/noise L40/80/160; 160 unseen in training',permutation_seed=1702,split_seed=1701,gate='unchanged delayed candidate commit, independent reads, threshold .02, EMA .8, permission initial bias+2, raw-zero closure; scalar pixel feature permission adapter',data_archive_sha256=sha(a.data/'mnist.npz'),torch=torch.__version__,gpu=torch.cuda.get_device_name() if a.device=='cuda' else 'cpu')
 write(root/'protocol.json',protocol)
 for task in TASKS:
  perm=z['permutation'] if task=='psmnist' else torch.arange(784,device=a.device)
  for seed in SEEDS:
   models=initial_models(seed)
   for family in FAMILIES:
    model=backend(models[family],a.device)
    for phase,epochs,lr,tag in [('clean',50,.005 if task=='psmnist' else .001,0),('mixed',12,.0003,1000000)]:
     dest=root/task/str(seed)/family/phase;dest.mkdir(parents=True,exist_ok=True)
     opt=torch.optim.Adam(model.parameters(),lr=lr);history=[];start=0
     if (dest/'last.pt').exists():
      cp=torch.load(dest/'last.pt',map_location=a.device);assert (cp['task'],cp['seed'],cp['family'],cp['phase'])==(task,seed,family,phase)
      model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);history=cp['history'];start=cp['epoch']
     for epoch in range(start,epochs):
      model.train();rng=np.random.default_rng(seed+tag+epoch*1009);order=rng.permutation(54000);loss_sum=correct=n=0;tick=time.time();digest=hashlib.sha256();first=None
      for j,begin in enumerate(range(0,54000,a.batch)):
       ids=order[begin:begin+a.batch];s=seed+tag+epoch*10000+j;x=batch_input(z['train_x'],ids,perm);y=z['train_y'][ids]
       kind='clean' if phase=='clean' else str(rng.choice(['clean','silence','poisson']));L=0 if kind=='clean' else int(rng.choice([40,80]));pos=int(rng.choice([392,784]));x=corrupt(x,kind,L,pos,s+30000000)
       digest.update(ids.tobytes());digest.update(f'{s},{kind},{L},{pos}'.encode())
       if j==0:first=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
       torch.manual_seed(s+40000000);opt.zero_grad(set_to_none=True);mean,tail=scores(model(x));loss=F.cross_entropy(mean,y) if phase=='clean' else (F.cross_entropy(mean,y)+F.cross_entropy(tail,y))/2
       assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
       n+=len(ids);loss_sum+=float(loss.detach())*len(ids);correct+=int((mean.argmax(-1)==y).sum())
       if j%25==0:write(root/'progress.json',dict(task=task,seed=seed,family=family,phase=phase,epoch=epoch+1,epochs=epochs,batch=j+1,batches=(54000+a.batch-1)//a.batch,seconds=time.time()-tick))
      row=dict(epoch=epoch+1,loss=loss_sum/n,train_accuracy=correct/n,seconds=time.time()-tick,plan_sha256=digest.hexdigest(),first_batch_sha256=first)
      if (epoch+1)%5==0 or epoch+1==epochs:row['validation']=evaluate_clean(model,z['valid_x'],z['valid_y'],perm,a.batch)
      history.append(row);write(dest/'history.json',history)
      cp=dict(model=model.state_dict(),optimizer=opt.state_dict(),history=history,epoch=epoch+1,task=task,seed=seed,family=family,phase=phase,protocol=protocol)
      save(dest/'last.pt',cp);print(json.dumps(dict(task=task,seed=seed,family=family,phase=phase,**row)),flush=True)
     save(dest/'final.pt',dict(model=model.state_dict(),task=task,seed=seed,family=family,phase=phase,epoch=epochs,protocol=protocol))
     write(dest/'complete.json',dict(epochs=epochs,sha256=sha(dest/'final.pt')))
     fingerprints[task,seed,family,phase]=[(r['plan_sha256'],r['first_batch_sha256']) for r in history]
     snapshot(root,dest,f'{task}-{seed}-{family}-{phase}')
    model.cpu()
   del models
 for task in TASKS:
  for seed in SEEDS:
   for phase in ('clean','mixed'):assert fingerprints[task,seed,'fsnn',phase]==fingerprints[task,seed,'dmp',phase]==fingerprints[task,seed,'commit',phase]
 write(root/'complete.json',dict(completed=True,tasks=2,models=3,seeds=3,epochs_each=62,input_pairing_verified=True,wall_seconds=time.time()-started))

def evaluate(a):
 torch.set_num_threads(2);assert json.loads((a.runs/'complete.json').read_text())['completed'];z=load(a.data,a.device);a.output.mkdir(parents=True,exist_ok=True)
 cases=[dict(name='clean',pos=784,L=0,kind='silence')]+[dict(name=f'p{pos}_{kind}_L{L}',pos=pos,L=L,kind=kind) for pos in (392,784) for kind in ('silence','poisson') for L in (40,80,160)]
 metrics=[];preds={'labels':z['test_y'].cpu().numpy()};target=preds['labels'];start=time.time()
 frozen={f'{t}/{s}/{f}/{p}':sha(a.runs/t/str(s)/f/p/'final.pt') for t in TASKS for s in SEEDS for f in FAMILIES for p in ('clean','mixed')}
 write(a.output/'protocol.json',dict(cases=cases,checkpoints=frozen,all_models_frozen_before_test=True,samples=10000,noise_seed=2701,noise_rate=.1,noise_clip=1,L160_unseen=True))
 with torch.no_grad():
  for task in TASKS:
   perm=z['permutation'] if task=='psmnist' else torch.arange(784,device=a.device)
   for seed in SEEDS:
    for family in FAMILIES:
     for phase in ('clean','mixed'):
      cp=torch.load(a.runs/task/str(seed)/family/phase/'final.pt',map_location='cpu');model=backend(initial_models(seed)[family],a.device);model.load_state_dict(cp['model']);model.eval()
      for case in cases:
       out=[[],[]];g=torch.Generator(device=a.device).manual_seed(2701)
       noise=torch.poisson(torch.full((10000,160,1),.1,device=a.device),generator=g).clamp_max(1) if case['kind']=='poisson' else None
       for i in range(0,10000,a.batch):
        x=batch_input(z['test_x'],slice(i,i+a.batch),perm);L=case['L'];pos=case['pos']
        if L:
         gap=x.new_zeros(len(x),L,1) if noise is None else noise[i:i+len(x),:L];x=torch.cat((x[:,:pos],gap,x[:,pos:]),1)
        for k,v in enumerate(scores(model(x))):assert torch.isfinite(v).all();out[k].extend(v.argmax(-1).cpu().tolist())
       for metric,v in zip(('mean','last20'),out):
        v=np.array(v,dtype=np.int16);key=f'{task}__{seed}__{family}__{phase}__{case["name"]}__{metric}';preds[key]=v;metrics.append(dict(task=task,seed=seed,family=family,phase=phase,metric=metric,**case,n=10000,correct=int((v==target).sum()),accuracy=float((v==target).mean())))
       write(a.output/'metrics.json',metrics);print(json.dumps(dict(task=task,seed=seed,family=family,phase=phase,case=case['name'],mean=metrics[-2]['accuracy'],last20=metrics[-1]['accuracy'])),flush=True)
      model.cpu();del model;np.savez_compressed(a.output/'predictions.npz',**preds)
 write(a.output/'complete.json',dict(completed=True,tasks=2,models=3,seeds=3,stages=2,cases=13,metrics=len(metrics),samples=10000,wall_seconds=time.time()-start))
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate']);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=Path);p.add_argument('--batch',type=int,default=512);p.add_argument('--device',default='cuda');a=p.parse_args();(train if a.mode=='train' else evaluate)(a)
if __name__=='__main__':main()
