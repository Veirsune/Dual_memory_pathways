import copy,json,time,argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from .models import initial_models,backend
from commit_memory.vector_ops import VectorLMU

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(9001);models=initial_models(42);results={'parameters':{f:sum(p.numel() for p in m.parameters()) for f,m in models.items()}}
 for family in ('dmp','commit'):
  model=models[family].to(a.device)
  for ci,c in enumerate((model.cell1,model.cell2)):
   u=torch.rand(2,944,40 if family=='commit' else 1,device=a.device,requires_grad=True)
   ref=VectorLMU.apply(u if family=='commit' else u.expand(-1,-1,40),c.A,c.vector_input_scale if family=='commit' else c.B[:,0]);got=c.candidate(u)
   error=float((ref-got).abs().max());assert torch.allclose(ref,got,atol=3e-5,rtol=1e-4),(family,ci,error)
   v=torch.randn_like(got);g1=torch.autograd.grad((ref*v).sum(),u,retain_graph=True)[0];g2=torch.autograd.grad((got*v).sum(),u)[0];ge=float((g1-g2).abs().max());assert torch.allclose(g1,g2,atol=5e-5,rtol=2e-4),(family,ci,ge)
   results[f'{family}_cell{ci}_candidate_equivalence']=dict(state_max_error=error,gradient_max_error=ge)
  model.cpu()
 # Original author's scalar FFT cell has the same A/B, pre-spike drive and mean readout.
 from vision_tasks.models import SequentialClassifier
 original=SequentialClassifier('smnist',40,300).to(a.device);dmp=models['dmp'].to(a.device)
 for oc,dc in ((original.core.LMU1,dmp.cell1),(original.core.LMU2,dmp.cell2)):
  oc.W_u.load_state_dict(dc.W_u.state_dict());oc.W_x.load_state_dict(dc.W_x.state_dict());oc.weight.load_state_dict(dc.weight.state_dict());assert torch.equal(oc.A,dc.A) and torch.equal(oc.B,dc.B)
 original.core.dense.load_state_dict(dmp.dense.state_dict())
 # Use torch LIF for this small equivalence check, independent of fused backend.
 x=torch.rand(2,784,1,device=a.device)
 with torch.no_grad():
  ref=original(x);got=dmp(x).mean(1);assert torch.allclose(ref,got,atol=1e-6,rtol=1e-5),float((ref-got).abs().max())
 results['original_dmp_forward_equivalence']=True;original.cpu();dmp.cpu()
 for family in models:
  m=backend(models[family],a.device);x=torch.rand(2,944,1,device=a.device);x[:,784:]=0;y=torch.tensor([1,2],device=a.device);m.train();out=m(x);loss=F.cross_entropy(out.mean(1),y);loss.backward();grad=sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None);assert grad>0 and torch.isfinite(torch.tensor(grad))
  if family=='commit':
   assert not m.cell1.gate_for_loss[:,784:].count_nonzero() and not m.cell2.gate_for_loss[:,784:].count_nonzero()
   # Zero-read stable branch makes paired Commit initialization match DMP.
  if a.device=='cuda':
   opt=torch.optim.Adam(m.parameters(),lr=.001);batch=torch.rand(512,784,1,device=a.device);labels=torch.arange(512,device=a.device)%10;times=[]
   for _ in range(3):
    torch.cuda.synchronize();start=time.time();opt.zero_grad(set_to_none=True);z=m(batch);F.cross_entropy(z.mean(1),labels).backward();opt.step();torch.cuda.synchronize();times.append(time.time()-start)
   results[family]=dict(batch512_seconds=times,gradient_squared=grad,peak_gpu_bytes=torch.cuda.max_memory_allocated());m.cpu();del batch,z;torch.cuda.empty_cache()
  else:results[family]=dict(gradient_squared=grad)
 results['gpu']=torch.cuda.get_device_name() if a.device=='cuda' else None;results['torch']=torch.__version__;a.output.write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
