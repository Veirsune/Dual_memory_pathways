"""Causal, variable-length SHD models with explicit streaming state.

The original LMU/LIF parameters and names are preserved for paired warm starts.
No model receives corruption masks, query boundaries, task IDs, or gap lengths.
"""
import math
import torch
from torch import nn
from spikingjelly.clock_driven import functional
from .reference_model import SpikingLMUFFTCell
from .vector_ops import VectorLMU

VARIANTS=('dmp','instant_gate_dmp','commit_dmp','dual_lmu_control')

def delay(x):return torch.cat((torch.zeros_like(x[:,:1]),x[:,:-1]),1)

def affine_scan(a,b):
    offset=1
    while offset<b.shape[1]:
        b=torch.cat((b[:,:offset],b[:,offset:]+a[:,offset:]*b[:,:-offset]),1)
        a=torch.cat((a[:,:offset],a[:,offset:]*a[:,:-offset]),1)
        offset*=2
    return b

@torch.jit.script
def selective_lmu(u:torch.Tensor,g:torch.Tensor,A:torch.Tensor,B:torch.Tensor):
    h=torch.zeros((u.shape[0],A.shape[0]),dtype=u.dtype,device=u.device)
    states=torch.jit.annotate(list[torch.Tensor],[])
    for t in range(u.shape[1]):
        proposal=torch.nn.functional.linear(h,A)+u[:,t]*B[:,0]
        h=h+g[:,t]*(proposal-h)
        states.append(h)
    return torch.stack(states,1)

@torch.jit.script
def selective_backward(gradient:torch.Tensor,u:torch.Tensor,g:torch.Tensor,A:torch.Tensor,B:torch.Tensor,h:torch.Tensor):
    du=torch.zeros_like(u);dg=torch.zeros_like(g);carry=torch.zeros_like(h[:,0]);zero=torch.zeros_like(carry)
    for k in range(u.shape[1]):
        t=u.shape[1]-1-k;total=gradient[:,t]+carry
        previous=h[:,t-1] if t>0 else zero
        difference=torch.nn.functional.linear(previous,A)+u[:,t]*B[:,0]-previous
        dg[:,t]=total*difference
        written=total*g[:,t]
        du[:,t]=torch.sum(written*B[:,0],dim=-1,keepdim=True)
        carry=total*(1-g[:,t])+torch.nn.functional.linear(written,A.t())
    return du,dg

class SelectiveLMU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,g,A,B):
        h=selective_lmu(u,g,A,B);ctx.save_for_backward(u,g,A,B,h);return h
    @staticmethod
    def backward(ctx,gradient):
        u,g,A,B,h=ctx.saved_tensors;du,dg=selective_backward(gradient,u,g,A,B,h)
        return du,dg,None,None

def features(x,previous,scale):
    rate=x.mean(-1,keepdim=True);scale=scale.clamp_min(.05)
    return torch.cat((rate/scale,(rate-previous.mean(-1,keepdim=True)).abs()/scale,
                      (x-rate).square().mean(-1,keepdim=True)/scale.square(),(x>0).to(x.dtype).mean(-1,keepdim=True)),-1)

class StreamCell(SpikingLMUFFTCell):
    def __init__(self,inputs,hidden,d,variant,gate_mode='soft',gate_threshold=.02,drive='scalar'):
        super().__init__(inputs,hidden,d,100,40)
        self.variant=variant
        self.drive=drive
        if drive in ('vector','wide_scalar'):
            self.W_u=nn.Linear(inputs,d)
            self.register_buffer('initial_row_scale',torch.linspace(.5,1.5,d))
            if drive=='wide_scalar':self.reduce_u=nn.Linear(d,1)
            else:self.register_buffer('vector_input_scale',self.B[:,0]/self.initial_row_scale)
        self.permission=None;self.permission_step=None;self.gate_for_loss=None
        self.gate_mode=gate_mode;self.gate_threshold=gate_threshold;self.memory_for_aux=None
        self.register_buffer('feature_scale',torch.tensor(1.))
        self.register_buffer('evidence_decay',torch.tensor(.8))
        if variant in ('instant_gate_dmp','commit_dmp'):
            self.gate=nn.Linear(8,d)
            nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,math.log(.95/.05))
        if variant in ('commit_dmp','dual_lmu_control'):
            self.read_m=nn.Linear(d,hidden,bias=False);nn.init.zeros_(self.read_m.weight)
        if variant=='dual_lmu_control':self.drive_q=nn.Linear(inputs,1)
        self._kernels={};self.capture=False;self.trace={};self.gate_override=None;self.stream={}

    def write_gate(self,logits):
        probability=logits.sigmoid()
        if self.gate_mode=='zero':
            actual=probability*(probability>=self.gate_threshold).to(probability.dtype)
            # Exact zeros in the forward pass; sigmoid surrogate in backward.
            return actual.detach()+probability-probability.detach()
        return probability

    def candidate(self,u):
        if self.drive=='vector':return VectorLMU.apply(u,self.A,self.vector_input_scale)
        # Shared recurrence keeps scalar/vector numerical paths aligned.
        return VectorLMU.apply(u.expand(-1,-1,self.memory_size),self.A,self.B[:,0])

    def drive_input(self,x):
        u=self.f_u(self.W_u(x))
        return self.f_u(self.reduce_u(u)) if self.drive=='wide_scalar' else u

    def forward(self,x):
        u=self.drive_input(x);g=None;m=None
        if self.variant in ('instant_gate_dmp','commit_dmp'):
            f=features(x,delay(x),self.feature_scale)
            context=delay(f)
            if self.variant=='commit_dmp':
                rho=self.evidence_decay
                context=delay(affine_scan(torch.ones_like(f)*rho,(1-rho)*f))
            g=self.write_gate(self.gate(torch.cat((f,context),-1)))
            if self.gate_override is not None:g=torch.ones_like(g)*self.gate_override
            if self.permission is not None:g=g*self.permission
            self.gate_for_loss=g
        if self.variant=='instant_gate_dmp':p=SelectiveLMU.apply(u,g,self.A,self.B)
        else:p=self.candidate(u)
        current=self.W_x(x)+self.weight(p)
        if self.variant=='commit_dmp':
            m=affine_scan(1-g,g*delay(p));current=current+self.read_m(m)
        elif self.variant=='dual_lmu_control':
            m=self.candidate(self.f_u(self.drive_q(x)));current=current+self.read_m(m)
        self.memory_for_aux=m if m is not None else p
        spikes=self.f_x(current.permute(1,0,2)).permute(1,0,2).contiguous()
        if self.capture:self.trace={'p':p.detach(),'m':m.detach() if m is not None else None,'gate':g.detach() if g is not None else None,'u':u.detach()}
        return spikes

    def reset_stream(self):
        self.stream={};self.memory_for_aux=None;self.gate_for_loss=None;self.permission=None;self.permission_step=None

    def step(self,x):
        if not self.stream:
            zero=x.new_zeros(x.shape[0],self.memory_size)
            self.stream={'p':zero}
            if self.variant in ('commit_dmp','dual_lmu_control'):self.stream['m']=zero.clone()
            if self.variant in ('instant_gate_dmp','commit_dmp'):
                self.stream['previous_rate']=x.new_zeros(x.shape[0],1)
                self.stream['e' if self.variant=='commit_dmp' else 'previous_f']=x.new_zeros(x.shape[0],4)
        s=self.stream;u=self.drive_input(x);g=None
        proposal=torch.nn.functional.linear(s['p'],self.A)+u*(self.vector_input_scale if self.drive=='vector' else self.B[:,0])
        if self.variant in ('instant_gate_dmp','commit_dmp'):
            f=features(x,s['previous_rate'],self.feature_scale)
            context=s['e'] if self.variant=='commit_dmp' else s['previous_f']
            g=self.write_gate(self.gate(torch.cat((f,context),-1)))
            if self.gate_override is not None:g=torch.ones_like(g)*self.gate_override
            if self.permission_step is not None:g=g*self.permission_step
            if self.variant=='commit_dmp':s['e']=self.evidence_decay*s['e']+(1-self.evidence_decay)*f
            else:s['previous_f']=f
            s['previous_rate']=x.mean(-1,keepdim=True)
        p=s['p']+g*(proposal-s['p']) if self.variant=='instant_gate_dmp' else proposal
        current=self.W_x(x)+self.weight(p)
        if self.variant=='commit_dmp':
            s['m']=(1-g)*s['m']+g*s['p'];current=current+self.read_m(s['m'])
        elif self.variant=='dual_lmu_control':
            s['m']=torch.nn.functional.linear(s['m'],self.A)+self.f_u(self.drive_q(x))*self.B[:,0]
            current=current+self.read_m(s['m'])
        s['p']=p
        return self.f_x(current.unsqueeze(0))[0]

class CausalPermission(nn.Module):
    def __init__(self,d,zero_rule=False):
        super().__init__();self.zero_rule=zero_rule;self.net=nn.Sequential(nn.Linear(36,32),nn.ReLU(),nn.Linear(32,2*d))
        nn.init.constant_(self.net[-1].bias,-3.)
        self.register_buffer('scale',torch.tensor(1.));self.register_buffer('decay',torch.tensor(.8));self.stream={}
    def current_features(self,x,previous):
        pooled=x.reshape(*x.shape[:-1],14,10).mean(-1)/self.scale.clamp_min(.05)
        return torch.cat((features(x,previous,self.scale),pooled),-1)
    def feature_sequence(self,x):
        f=self.current_features(x,delay(x));context=delay(affine_scan(torch.ones_like(f)*self.decay,(1-self.decay)*f))
        return torch.cat((f,context),-1)
    def binary(self,logits,x):
        q=logits.sigmoid();actual=(logits>=0).to(q.dtype)
        permission=actual.detach()+q-q.detach()
        return permission*(x.sum(-1,keepdim=True)>0).to(q.dtype) if self.zero_rule else permission
    def forward(self,x):
        logits=self.net(self.feature_sequence(x));return self.binary(logits,x),logits
    def step(self,x):
        if not self.stream:self.stream={'rate':x.new_zeros(len(x),1),'e':x.new_zeros(len(x),18)}
        f=self.current_features(x,self.stream['rate']);logits=self.net(torch.cat((f,self.stream['e']),-1))
        self.stream['e']=self.decay*self.stream['e']+(1-self.decay)*f;self.stream['rate']=x.mean(-1,keepdim=True)
        return self.binary(logits,x)
    def reset_stream(self):self.stream={}

class StreamModel(nn.Module):
    def __init__(self,variant='commit_dmp',d=10,gate_mode='zero',aux_enabled=True,gate_threshold=.02,drive='scalar',protect=False,zero_rule=False):
        super().__init__();assert variant in VARIANTS
        assert drive=='scalar' or variant=='commit_dmp','Vector and wide-scalar drives are defined for commit only'
        self.variant=variant;self.d=d;self.gate_mode=gate_mode;self.aux_enabled=aux_enabled;self.drive=drive;self.protect=protect;self.zero_rule=zero_rule
        self.LMU1=StreamCell(140,128,d,variant,gate_mode,gate_threshold,drive);self.LMU2=StreamCell(128,128,d,variant,gate_mode,gate_threshold,drive)
        self.rnn_drop=nn.Dropout(.2);self.dense=nn.Linear(128,20)
        self.memory_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,20))
        self.permit=CausalPermission(d,zero_rule);self.permission_logits=None
        if not protect:
            for parameter in self.permit.parameters():parameter.requires_grad_(False)
        if not aux_enabled:
            for parameter in self.memory_head.parameters():parameter.requires_grad_(False)

    def cells(self):return [self.LMU1,self.LMU2]
    def reset_state(self):
        functional.reset_net(self)
        for c in self.cells():c.reset_stream()
        self.permit.reset_stream();self.permission_logits=None
    def forward(self,x):
        if self.protect:
            permission,self.permission_logits=self.permit(x)
            self.LMU1.permission=permission[:,:,:self.d];self.LMU2.permission=permission[:,:,self.d:]
        return self.dense(self.LMU2(self.rnn_drop(self.LMU1(x))))
    def stream_chunk(self,x):
        outputs=[]
        for t in range(x.shape[1]):
            if self.protect:
                permission=self.permit.step(x[:,t]);self.LMU1.permission_step=permission[:,:self.d];self.LMU2.permission_step=permission[:,self.d:]
            outputs.append(self.dense(self.LMU2.step(self.rnn_drop(self.LMU1.step(x[:,t])))))
        return torch.stack(outputs,1)
    def memory_logits(self,ends):
        # Scoring annotation only; this head never feeds neural updates.
        indices=[end-1 for _,end in ends]
        return self.memory_head(self.LMU2.memory_for_aux[:,indices])
    def warm_start(self,state):
        own=self.state_dict();copied={k:v for k,v in state.items() if k in own and v.shape==own[k].shape}
        missing,unexpected=self.load_state_dict(copied,strict=False);assert not unexpected
        for cell in self.cells():
            if self.variant=='dual_lmu_control':cell.drive_q.load_state_dict(cell.W_u.state_dict())
        with torch.no_grad():
            for name,cell in (('LMU1',self.LMU1),('LMU2',self.LMU2)):
                if self.drive in ('vector','wide_scalar'):
                    scale=cell.initial_row_scale
                    cell.W_u.weight.copy_(state[name+'.W_u.weight'].repeat(self.d,1)*scale[:,None])
                    cell.W_u.bias.copy_(state[name+'.W_u.bias'].repeat(self.d)*scale)
                    if self.drive=='wide_scalar':
                        cell.reduce_u.weight.copy_((1/(self.d*scale))[None]);cell.reduce_u.bias.zero_()
                    else:cell.vector_input_scale.copy_(cell.B[:,0]/scale)
        return missing

def query_logits(sequence,ends,window=20):
    values=[]
    for item in ends:
        start,end=item if isinstance(item,(tuple,list)) else (max(0,item-window),item)
        values.append(sequence[:,start:end].mean(1))
    return torch.stack(values,1)
