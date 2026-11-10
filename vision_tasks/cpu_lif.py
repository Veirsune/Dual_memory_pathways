"""Linear-time CPU backward for the fixed LIF neurons used in these experiments.

The reference per-time-step tensor indexing creates expensive slice-backward
operations. This explicit reverse recurrence preserves its sigmoid surrogate and
detached hard reset, while writing each input gradient once. No truncated BPTT.
"""
from typing import Optional
import torch
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven.surrogate import Sigmoid

@torch.jit.script
def _forward(x:torch.Tensor,v:torch.Tensor,tau:float,threshold:float):
    spikes=torch.empty_like(x);states=torch.empty_like(x);pre=torch.empty_like(x)
    for t in range(x.shape[0]):
        v=v+(x[t]-v)/tau
        pre[t]=v
        spike=(v>=threshold).to(x.dtype)
        v=(1.-spike)*v
        spikes[t]=spike;states[t]=v
    return spikes,states,pre

@torch.jit.script
def _backward(pre:torch.Tensor,spikes:torch.Tensor,ds:Optional[torch.Tensor],
              dv:Optional[torch.Tensor],tau:float,threshold:float,alpha:float):
    dx=torch.empty_like(pre);carry=torch.zeros_like(pre[0])
    for k in range(pre.shape[0]):
        t=pre.shape[0]-1-k
        total=carry
        if dv is not None:total=total+dv[t]
        dh=total*(1.-spikes[t])
        if ds is not None:
            sigmoid=((pre[t]-threshold)*alpha).sigmoid()
            dh=dh+ds[t]*(1.-sigmoid)*sigmoid*alpha
        dx[t]=dh/tau
        carry=dh-dh/tau
    return dx,carry

class _LIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,v,tau,threshold,alpha):
        spikes,states,pre=_forward(x,v,tau,threshold)
        ctx.save_for_backward(pre,spikes);ctx.parameters=(tau,threshold,alpha)
        ctx.set_materialize_grads(False)
        return spikes,states
    @staticmethod
    def backward(ctx,ds,dv):
        pre,spikes=ctx.saved_tensors
        dx,initial=_backward(pre,spikes,ds,dv,*ctx.parameters)
        return dx,initial,None,None,None

class CpuLIF(MultiStepLIFNode):
    def forward(self,x):
        if x.device.type!='cpu':raise ValueError('cpu_scan is a CPU backend')
        if x.dtype not in (torch.float32,torch.float64):raise ValueError('Expected float32/float64')
        initial=torch.full_like(x[0],self.v) if isinstance(self.v,float) else self.v
        spikes,self.v_seq=_LIF.apply(x,initial,self.tau,self.v_threshold,self.surrogate_function.alpha)
        self.v=self.v_seq[-1]
        return spikes

def enable_cpu_scan(model):
    count=0
    for name,child in list(model.named_children()):
        if type(child) is MultiStepLIFNode:
            if not (child.decay_input and child.detach_reset and child.v_reset==0.
                    and type(child.surrogate_function) is Sigmoid and child.surrogate_function.spiking):
                raise ValueError('cpu_scan requires decay-input, detached zero reset and spiking Sigmoid')
            new=CpuLIF(tau=child.tau,v_threshold=child.v_threshold,detach_reset=True,
                       surrogate_function=Sigmoid(alpha=child.surrogate_function.alpha))
            new.train(child.training);setattr(model,name,new);count+=1
        else:count+=enable_cpu_scan(child)
    return count
