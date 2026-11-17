import argparse,json,hashlib,time,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .models import initial_models,backend

FAMILIES=('fsnn','dmp','commit');SEED=801

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for q in iter(lambda:f.read(8*1024**2),b''):h.update(q)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def save(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def load_data(root,split,device):
 x=np.load(root/f'{split}-x.npy',mmap_mode='r');y=np.load(root/f'{split}-y.npy');return torch.from_numpy(np.array(x)).to(device),torch.from_numpy(y).to(device)
def dataset_hash(root,split):
 x=np.load(root/f'{split}-x.npy',mmap_mode='r');h=hashlib.sha256()
 for start in range(0,len(x),512):h.update(x[start:start+512].tobytes())
 assert h.hexdigest()==json.loads((root/f'{split}-metadata.json').read_text())['frame_sha256'];return h.hexdigest()
def insert(x,pos,L,kind,seed):
 if not L:return x
 gen=torch.Generator(device=x.device).manual_seed(seed);gap=x.new_zeros(len(x),L,140)
 if kind=='poisson':gap=torch.poisson(torch.full_like(gap,.25),generator=gen)
 return torch.cat((x[:,:pos],gap,x[:,pos:]),1)
def logits(model,x):
 z=model(x);return z.mean(1),z[:,-20:].mean(1)

@torch.no_grad()
def clean_score(model,x,y,batch=512):
 model.eval();correct=tail=0
 for i in range(0,len(x),batch):
  a,b=logits(model,x[i:i+batch].float());correct+=int((a.argmax(-1)==y[i:i+batch]).sum());tail+=int((b.argmax(-1)==y[i:i+batch]).sum())
 return dict(mean=correct/len(x),last20=tail/len(x),n=len(x))

def train(a):
 root=a.output;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);data_hash=dataset_hash(a.data,'train');x,y=load_data(a.data,'train',a.device);vx,vy=load_data(a.data,'valid',a.device)
 calibration=np.random.default_rng(6501).choice(len(x),2048,replace=False);scale=max(.05,float(torch.quantile(x[calibration].float().mean(-1).flatten(),.9)));models=initial_models(scale)
 protocol=dict(seed=SEED,classes=35,steps=250,channels=140,d=30,hidden=128,layers=2,theta=40,gate_threshold=.02,evidence_decay=.8,permission_initial_bias=2.,zero_input_rule=True,reader='fixed independent p/m',lif_tau=2.,lif_threshold=1.,dropout=.2,train_samples=len(x),valid_samples=len(vx),batch=a.batch,clean_epochs=40,mixed_epochs=12,initialization='from scratch; shared fast weights and classifier; vector drive expanded from same scalar initialization',objective='0.5 CE(full sequence mean logits) + 0.5 CE(last20 mean logits); same for all families, no auxiliary memory head loss',optimizer='AdamW wd=.05, grad_clip1; clean lr=.003 cosine40; mixed lr=.001 cosine12; floor5e-5',training_conditions='clean stage; then uniform clean/silence/poisson, positions125/250, L40/80 only, Poisson .25; channel shifts ±12',test_conditions='all official TEST; clean + middle/tail silence/poisson L40/80/120; L120 held out of training; noise seed901; mean andlast20',selection='fixed final epoch each stage; no TEST-driven choice',data_sha256=data_hash,parameters={f:sum(p.numel() for p in m.parameters() if p.requires_grad) for f,m in models.items()},feature_scale=scale,torch=torch.__version__,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None)
 write(root/'protocol.json',protocol);started=time.time();fingerprints={}
 for family in FAMILIES:
  model=backend(models[family],a.device)
  for phase,epochs,lr,tag in [('clean',40,.003,0),('mixed',12,.001,1000000)]:
   dest=root/family/phase;dest.mkdir(parents=True,exist_ok=True);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.05);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=5e-5);history=[];start_epoch=0
   if (dest/'last.pt').exists():
    cp=torch.load(dest/'last.pt',map_location=a.device);assert cp['family']==family and cp['phase']==phase;model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);sched.load_state_dict(cp['scheduler']);start_epoch=cp['epoch'];history=cp['history']
   for epoch in range(start_epoch,epochs):
    model.train();rng=np.random.default_rng(SEED+tag+epoch*1009);order=rng.permutation(len(x));total=correct=n=0;tick=time.time();digest=hashlib.sha256();first=None
    for j,begin in enumerate(range(0,len(x),a.batch)):
     ids=order[begin:begin+a.batch];seed=SEED+tag+epoch*10000+j;gen=torch.Generator(device=a.device).manual_seed(seed);batch=x[ids].float();target=y[ids]
     offset=torch.randint(-12,13,(len(ids),1,1),generator=gen,device=a.device);index=torch.arange(140,device=a.device).view(1,1,-1)-offset;batch=batch.gather(2,index.clamp(0,139).expand_as(batch))*((index>=0)&(index<140))
     kind='clean' if phase=='clean' else str(rng.choice(['clean','silence','poisson']));L=0 if kind=='clean' else int(rng.choice([40,80]));pos=int(rng.choice([125,250]));batch=insert(batch,pos,L,kind,seed+30000000)
     digest.update(ids.tobytes());digest.update(f'{seed},{kind},{L},{pos}'.encode())
     if j==0:first=hashlib.sha256(batch.cpu().numpy().tobytes()).hexdigest()
     torch.manual_seed(seed+40000000);opt.zero_grad(set_to_none=True);mean,tail=logits(model,batch);loss=(F.cross_entropy(mean,target)+F.cross_entropy(tail,target))/2;assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
     total+=float(loss.detach())*len(ids);correct+=int((mean.argmax(-1)==target).sum());n+=len(ids)
     if j%40==0:write(root/'progress.json',dict(family=family,phase=phase,epoch=epoch+1,batch=j+1,batches=(len(x)+a.batch-1)//a.batch,seconds=time.time()-tick))
    sched.step();row=dict(epoch=epoch+1,loss=total/n,train_accuracy=correct/n,seconds=time.time()-tick,plan_sha256=digest.hexdigest(),first_batch_sha256=first)
    if (epoch+1)%10==0 or epoch+1==epochs:row['validation']=clean_score(model,vx,vy)
    history.append(row);save(dest/'last.pt',dict(family=family,phase=phase,epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),history=history,scale=scale,protocol=protocol));write(dest/'history.json',history);print(json.dumps(dict(family=family,phase=phase,**row)),flush=True)
   save(dest/'final.pt',dict(family=family,phase=phase,epoch=epochs,model=model.state_dict(),scale=scale,protocol=protocol));write(dest/'complete.json',dict(epochs=epochs,sha256=sha(dest/'final.pt')));fingerprints[family,phase]=[(r['plan_sha256'],r['first_batch_sha256']) for r in history]
  model.cpu()
 for phase in ('clean','mixed'):assert fingerprints['fsnn',phase]==fingerprints['dmp',phase]==fingerprints['commit',phase]
 write(root/'complete.json',dict(completed=True,models=3,epochs_each=52,input_plan_and_first_batch_hashes_identical=True,wall_seconds=time.time()-started))

def evaluate(a):
 assert json.loads((a.runs/'complete.json').read_text())['completed'];a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);dataset_hash(a.data,'test');x,y=load_data(a.data,'test',a.device);assert len(x)==20382;target=y.cpu().numpy();speakers=np.load(a.data/'test-speaker.npy');predictions=dict(labels=target,speakers=speakers);metrics=[];cases=[dict(name='clean',pos=250,L=0,kind='silence')]+[dict(name=f'p{pos}_{kind}_L{L}',pos=pos,L=L,kind=kind) for pos in (125,250) for kind in ('silence','poisson') for L in (40,80,120)];start=time.time()
 write(a.output/'protocol.json',dict(samples=len(x),cases=cases,noise_seed=901,noise_rate=.25,all_models_frozen_before_test=True,historically_test_used=True,batch=a.batch,checkpoints={f'{f}_{s}':sha(a.runs/f/s/'final.pt') for f in FAMILIES for s in ('clean','mixed')}))
 with torch.no_grad():
  for phase in ('clean','mixed'):
   for family in FAMILIES:
    cp=torch.load(a.runs/family/phase/'final.pt',map_location='cpu');model=backend(initial_models(cp['scale'])[family],a.device);model.load_state_dict(cp['model']);model.eval()
    for case in cases:
     out={'mean':[],'last20':[]};gen=torch.Generator(device=a.device).manual_seed(901)
     # Full max-length noise generated once per condition; identical sample/length prefixes.
     noise=torch.poisson(torch.full((len(x),120,140),.25,device=a.device),generator=gen) if case['kind']=='poisson' else None
     for begin in range(0,len(x),a.batch):
      batch=x[begin:begin+a.batch].float();L=case['L'];pos=case['pos']
      if L:
       gap=noise[begin:begin+len(batch),:L] if noise is not None else batch.new_zeros(len(batch),L,140);batch=torch.cat((batch[:,:pos],gap,batch[:,pos:]),1)
      mean,tail=logits(model,batch);assert torch.isfinite(mean).all() and torch.isfinite(tail).all()
      out['mean'].extend(mean.argmax(-1).cpu().tolist());out['last20'].extend(tail.argmax(-1).cpu().tolist())
     for metric,v in out.items():
      v=np.asarray(v,dtype=np.int16);predictions[f'{phase}__{family}__{case["name"]}__{metric}']=v;metrics.append(dict(phase=phase,family=family,metric=metric,**case,n=len(x),correct=int((v==target).sum()),accuracy=float((v==target).mean()),macro_accuracy=float(np.mean([(v[target==c]==c).mean() for c in range(35)]))))
     np.savez_compressed(a.output/'predictions.npz',**predictions);write(a.output/'metrics.json',metrics);print(json.dumps(dict(phase=phase,family=family,case=case['name'],mean=metrics[-2]['accuracy'],last20=metrics[-1]['accuracy'])),flush=True)
    model.cpu();del model
 write(a.output/'complete.json',dict(completed=True,samples=20382,models=6,conditions=13,wall_seconds=time.time()-start))

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate']);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--runs',type=Path);p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=512);a=p.parse_args();(train if a.mode=='train' else evaluate)(a)
if __name__=='__main__':main()
