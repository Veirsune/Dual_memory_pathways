import argparse,json,time,copy
import torch
from pathlib import Path
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from .models import initial_models,backend
from commit_memory.vector_ops import VectorLMU

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('gpu-verification.json'));a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.manual_seed(931);device='cuda';results={}
 x=torch.randn(70,4,19,device=device,requires_grad=True);a=MultiStepLIFNode(tau=2.,detach_reset=True,backend='torch').to(device);b=MultiStepLIFNode(tau=2.,detach_reset=True,backend='cupy').to(device)
 y=a(x);(y.sum()+a.v_seq.sum()*.1).backward();g=x.grad.clone();x.grad=None;z=b(x);(z.sum()+b.v_seq.sum()*.1).backward();assert torch.equal(y,z);assert torch.allclose(g,x.grad,atol=2e-5,rtol=2e-5);results['lif_spike_gradient_equivalence']=True
 models=initial_models(.05)
 for family,model in models.items():
  model=backend(model,device);model.train();inp=torch.poisson(torch.full((8,370,140),.08,device=device));inp[:,250:]=0;target=torch.arange(8,device=device)%35
  out=model(inp);loss=torch.nn.functional.cross_entropy(out.mean(1),target);loss.backward();grad=sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None);assert grad>0 and torch.isfinite(torch.tensor(grad))
  if family=='commit':assert not model.core.LMU1.gate_for_loss[:,250:].count_nonzero() and not model.core.LMU2.gate_for_loss[:,250:].count_nonzero()
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003);batch=torch.poisson(torch.full((512,250,140),.08,device=device));labels=torch.arange(512,device=device)%35;times=[]
  for k in range(3):
   torch.cuda.synchronize();t=time.time();opt.zero_grad(set_to_none=True);out=model(batch);loss=(torch.nn.functional.cross_entropy(out.mean(1),labels)+torch.nn.functional.cross_entropy(out[:,-20:].mean(1),labels))/2;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize();times.append(time.time()-t)
  results[family]=dict(gradient_squared=grad,batch512_seconds=times,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad));model.cpu();print(family,results[family],flush=True)
 results['gpu']=torch.cuda.get_device_name(0);results['torch']=torch.__version__;a.output.write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
