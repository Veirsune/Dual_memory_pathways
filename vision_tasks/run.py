"""Reproducible train/validation pilots and hardware benchmarks for visual tasks.

Default behavior never evaluates official test data. Validation uses held-out
training images or held-out training subjects. Checkpoints include permutation.
"""
import argparse, hashlib, json, math, random, resource, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from .models import build
from .cpu_lif import enable_cpu_scan

class Images(Dataset):
    def __init__(self, base, indices, perm): self.base,self.indices,self.perm=base,indices,perm
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        k=self.indices[i]
        return self.base.data[k].reshape(784,1)[self.perm].float()/255.,int(self.base.targets[k])

class Gestures(Dataset):
    def __init__(self,root,rows,steps):
        self.root,self.rows,self.steps=root,rows,steps;self.cache=None
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        if self.cache is not None:return self.cache[i]
        row=self.rows[i]
        with np.load(self.root/row['file'],allow_pickle=False) as z: x=z['frames'].astype(np.float32)
        if x.shape[0]!=self.steps:
            if x.shape[0]%self.steps: raise ValueError('Requested steps must divide cached steps')
            x=x.reshape(self.steps,x.shape[0]//self.steps,2,32,32).sum(1)
        return torch.from_numpy(x),row['label']

    def preload(self):self.cache=[self[i] for i in range(len(self))]

def rng_state(loader):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'loader':loader.generator.get_state()}

def restore_rng(state,loader):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])
    loader.generator.set_state(state['loader'])

def atomic_save(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def balanced_limit(indices,labels,limit,seed):
    if not limit or limit>=len(indices): return list(indices)
    rng=np.random.default_rng(seed)
    pools=[list(rng.permutation([i for i in indices if labels[i]==c])) for c in sorted(set(labels))]
    out=[]
    while len(out)<limit:
        for pool in pools:
            if pool and len(out)<limit: out.append(int(pool.pop()))
    return out

def datasets(args):
    if args.task!='gesture':
        base=MNIST(str(args.data),train=True,download=True)
        labels=base.targets.numpy();rng=np.random.default_rng(1701);tr=[];va=[]
        for c in range(10):
            ids=rng.permutation(np.flatnonzero(labels==c));va.extend(ids[:600]);tr.extend(ids[600:])
        tr=balanced_limit(tr,labels,args.train_limit,args.seed)
        va=balanced_limit(va,labels,args.val_limit,args.seed+1)
        perm=torch.randperm(784,generator=torch.Generator().manual_seed(args.permutation_seed)) if args.task=='psmnist' else torch.arange(784)
        return Images(base,tr,perm),Images(base,va,perm),{'train_indices':list(map(int,tr)),
             'validation_indices':list(map(int,va)),'permutation':perm.tolist(),
             'split':'54,000 / 6,000 from official TRAIN; official TEST unused'}
    manifest=json.loads((args.data/'manifest.json').read_text());rows=manifest['samples']
    labels=[r['label'] for r in rows]
    tr=[i for i,r in enumerate(rows) if r['subject']<20]
    va=[i for i,r in enumerate(rows) if r['subject']>=20]
    tr=balanced_limit(tr,labels,args.train_limit,args.seed);va=balanced_limit(va,labels,args.val_limit,args.seed+1)
    if not tr or not va: raise ValueError('Need both training subjects <20 and validation subjects >=20')
    return Gestures(args.data,[rows[i] for i in tr],args.steps),Gestures(args.data,[rows[i] for i in va],args.steps),{
        'train_files':[rows[i]['file'] for i in tr],'validation_files':[rows[i]['file'] for i in va],
        'train_subjects':sorted({rows[i]['subject'] for i in tr}),'validation_subjects':sorted({rows[i]['subject'] for i in va}),
        'manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
        'split':'official training mirror only; subjects <20 train, >=20 validation; official TEST unused'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['smnist','psmnist','gesture'],required=True)
    p.add_argument('--data',type=Path,default=Path('/workspace/data/vision'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--train-limit',type=int,default=500)
    p.add_argument('--val-limit',type=int,default=200)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--permutation-seed',type=int,default=1702)
    p.add_argument('--memory',type=int,default=40)
    p.add_argument('--theta',type=int,default=300)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--family',choices=['dmp','fsnn','commit'],default='dmp')
    p.add_argument('--readout',choices=['mean','last'],default='mean')
    p.add_argument('--lr',type=float)
    p.add_argument('--device',default='auto')
    p.add_argument('--backend',choices=['torch','cupy','cpu_scan'],default='torch')
    p.add_argument('--benchmark-steps',type=int,default=0)
    p.add_argument('--resume',type=Path)
    p.add_argument('--cache-data',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.threads,args.memory,args.theta,args.steps)<1:
        raise ValueError('Epochs, batch size, threads, memory, theta and steps must be positive')
    if min(args.train_limit,args.val_limit,args.benchmark_steps)<0:
        raise ValueError('Sample limits and benchmark steps must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    train,val,split=datasets(args)
    model=build(args.task,args.memory,args.theta,args.steps,args.family,args.readout).to(device)
    if args.cache_data:
        if args.task!='gesture':raise ValueError('cache-data currently applies to gestures')
        train.preload();val.preload()
    if args.backend=='cupy' and device.type!='cuda':raise ValueError('CuPy backend requires CUDA')
    if args.backend=='cupy':
        import cupy  # fail before training when the optional accelerator is absent
    if args.backend=='cpu_scan':
        if device.type!='cpu':raise ValueError('cpu_scan requires CPU')
        enable_cpu_scan(model)
    else:
        for module in model.modules():
            if isinstance(module,MultiStepLIFNode):module.backend=args.backend
    lr=args.lr or (0.005 if args.task=='psmnist' else 0.001)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=100 if args.task=='psmnist' else 150,gamma=.5)
    criterion=nn.CrossEntropyLoss()
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed+2))
    vloader=DataLoader(val,batch_size=args.batch_size,num_workers=0)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(device=str(device),actual_lr=lr,steps=784 if args.task!='gesture' else args.steps,
                  train_samples=len(train),validation_samples=len(val),torch=torch.__version__,
                  parameters=sum(p.numel() for p in model.parameters()),
                  original_threshold=1.0,paper_table_threshold=10.0,
                  reconstruction=args.task=='gesture')
    source_hashes={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in Path(__file__).parent.glob('*.py')}
    (args.output/'protocol.json').write_text(json.dumps({'config':config,'split':split,'source_sha256':source_hashes},indent=2)+'\n')
    times=[];history=[];start=time.time();best=-1;first_update=None;initial_epoch=0;best_model=None;best_epoch=0
    if args.resume:
        cp=torch.load(args.resume,map_location='cpu')
        if cp.get('format')!='visual-training-v1':raise ValueError('Resume requires a full last.pt checkpoint')
        for key in ('task','family','memory','theta','steps','seed','permutation_seed','batch_size','actual_lr','backend','readout'):
            if cp['config'][key]!=config[key]:raise ValueError('Resume config mismatch: '+key)
        if cp['split']!=split:raise ValueError('Resume data split mismatch')
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        restore_rng(cp['rng'],loader);initial_epoch=cp['epoch'];history=cp['history'];best=cp['best_accuracy']
        best_model=cp['best_model'];best_epoch=cp['best_epoch']
        atomic_save({'model':best_model,'config':config,'split':split,'epoch':best_epoch},args.output/'best.pt')
        if args.epochs<=initial_epoch:raise ValueError('epochs must exceed the checkpoint epoch')
    for epoch in range(initial_epoch,args.epochs):
        model.train();loss_sum=0;correct=0;count=0;epoch_start=time.time()
        for step,(x,y) in enumerate(loader):
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            if device.type=='cuda':torch.cuda.synchronize()
            ts=time.time();out=model(x);loss=criterion(out,y)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            if first_update is None:
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('Nonfinite gradient')
                first_update={'gradient_l2':float(sum(g.square().sum() for g in grads).sqrt()),
                              'loss':float(loss.detach())}
                if first_update['gradient_l2']==0:raise RuntimeError('Zero gradient')
            opt.step()
            if device.type=='cuda':torch.cuda.synchronize()
            times.append({'seconds':time.time()-ts,'batch':len(y)})
            (args.output/'progress.json').write_text(json.dumps({'epoch':epoch+1,'step':step+1,
                 'last_batch':times[-1],'elapsed_seconds':time.time()-start})+'\n')
            count+=len(y);correct+=int((out.argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
            if step==0 or (step+1)%10==0:print(json.dumps({'task':args.task,'epoch':epoch+1,'step':step+1,'loss':float(loss.detach()),'batch_seconds':times[-1]['seconds']}),flush=True)
            if args.benchmark_steps and len(times)>=args.benchmark_steps:break
        train_seconds=time.time()-epoch_start
        if args.benchmark_steps:break
        model.eval();vc=0;vn=0;vl=0;vstart=time.time()
        with torch.no_grad():
            for x,y in vloader:
                x=x.to(device);y=y.to(device);out=model(x);v=criterion(out,y)
                vc+=int((out.argmax(1)==y).sum());vn+=len(y);vl+=float(v)*len(y)
        row={'epoch':epoch+1,'train_loss':loss_sum/count,'train_accuracy':correct/count,
             'validation_loss':vl/vn,'validation_accuracy':vc/vn,'train_seconds':train_seconds,
             'validation_seconds':time.time()-vstart}
        history.append(row);scheduler.step();print(json.dumps(row),flush=True)
        if row['validation_accuracy']>best:
            best=row['validation_accuracy']
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
            atomic_save({'model':best_model,'config':config,'split':split,'epoch':best_epoch},args.output/'best.pt')
        atomic_save({'format':'visual-training-v1','model':model.state_dict(),'config':config,'split':split,
                     'epoch':epoch+1,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                     'rng':rng_state(loader),'history':history,'best_accuracy':best,
                     'best_model':best_model,'best_epoch':best_epoch},args.output/'last.pt')
        (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    steady=times[1:] if len(times)>1 else times
    throughput=sum(q['batch'] for q in steady)/sum(q['seconds'] for q in steady)
    full_train=54000 if args.task!='gesture' else None
    result={'config':config,'history':history,'first_gradient':first_update,'timings':times,
            'steady_samples_per_second':throughput,'process_peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else None,
            'cpu_full_training_epoch_hours_extrapolation':full_train/throughput/3600 if full_train and device.type=='cpu' else None,
            'wall_seconds':time.time()-start,'official_test_evaluated':False,
            'initial_epoch':initial_epoch,'completed_epoch':history[-1]['epoch'] if history else 0}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('timings','history','config')}),flush=True)

if __name__=='__main__':main()
