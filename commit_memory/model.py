"""Portable fixed architecture; read adaptation cannot change upstream writes.

The LMU, LIF, vector drive, local gate, permission features and final read gate
are inherited from the sealed SHD study. This module consolidates their public
configuration, training stages and batch/chunk interfaces; it adds no new gate.
"""
from dataclasses import dataclass,asdict
from contextlib import nullcontext
import math
import torch
from torch import nn
from spikingjelly.clock_driven import functional
from .base_model import StreamCell,CausalPermission,features,delay,affine_scan,query_logits

@dataclass(frozen=True)
class MemoryConfig:
    classes:int=35
    family:str='commit'
    memory:int=30
    hidden:int=128
    inputs:int=140
    reader:str='fixed'
    dropout:float=.2
    def __post_init__(self):
        if self.family not in ('dmp','commit'):raise ValueError('family must be dmp or commit')
        if self.reader not in ('fixed','static','dynamic'):raise ValueError('Unknown reader')
        if self.family=='dmp' and self.reader!='fixed':raise ValueError('DMP has two paths; three-path readers require Commit')
        if self.inputs!=140:raise ValueError('This fixed auditory architecture expects140channels; changing the feature extractor is a separate study')
        if min(self.classes,self.hidden,self.memory)<1 or not 0<=self.dropout<1:raise ValueError('Invalid dimensions/dropout')
    def to_dict(self):return asdict(self)

class ReadAdapter(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        if mode=='dynamic':
            self.net=nn.Sequential(nn.Linear(12,16),nn.ReLU(),nn.Linear(16,3))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        elif mode=='static':self.logits=nn.Parameter(torch.zeros(3))
        else:self.register_buffer('logits',torch.zeros(3))
    def forward(self,z):
        return 3*(self.net(z) if self.mode=='dynamic' else self.logits.expand(*z.shape[:-1],3)).softmax(-1)

def read_features(f,context,p,m):
    pn=(p.square().mean(-1,keepdim=True)+1e-6).sqrt();mn=(m.square().mean(-1,keepdim=True)+1e-6).sqrt()
    similarity=(p*m).mean(-1,keepdim=True)/(pn*mn+1e-4)
    distance=((p-m).square().mean(-1,keepdim=True)+1e-6).sqrt()/(1+pn+mn)
    return torch.cat((f,context,pn.log1p(),mn.log1p(),similarity,distance),-1)

class MemoryClassifier(nn.Module):
    def __init__(self,config=MemoryConfig()):
        super().__init__();self.config=config;d=config.memory
        variant='dmp' if config.family=='dmp' else 'commit_dmp';drive='scalar' if config.family=='dmp' else 'vector'
        self.LMU1=StreamCell(config.inputs,config.hidden,d,variant,'zero',.02,drive)
        self.LMU2=StreamCell(config.hidden,config.hidden,d,variant,'zero',.02,drive)
        # The inherited BatchNorm is never called. Remove its unused parameters.
        for cell in self.cells():del cell.bn_x
        self.rnn_drop=nn.Dropout(config.dropout);self.dense=nn.Linear(config.hidden,config.classes)
        self.memory_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,config.classes))
        self.permit=CausalPermission(d,True);self.adapter=ReadAdapter(config.reader)
        with torch.no_grad():self.permit.net[-1].weight.zero_();self.permit.net[-1].bias.fill_(2.)
        self.stage='content';self.read_only=False;self.read_weights=None;self.online_last={};self._stream_batch_size=None
        if config.family=='dmp':
            for p in self.permit.parameters():p.requires_grad_(False)

    def cells(self):return (self.LMU1,self.LMU2)

    def set_stage(self,stage):
        if stage not in ('content','robust','reader'):raise ValueError(stage)
        if stage=='reader' and (self.config.family!='commit' or self.config.reader=='fixed'):
            raise ValueError('Reader stage requires a trainable Commit reader')
        self.stage=stage;self.read_only=stage=='reader'
        for name,p in self.named_parameters():
            p.requires_grad_(name.startswith('adapter.') if self.read_only else not (self.config.family=='dmp' and name.startswith('permit.')))
        return self.train(self.training)

    def train(self,mode=True):
        if getattr(self,'read_only',False):
            super().train(False);self.adapter.train(mode);self.LMU2.f_x.train(mode)
            return self
        return super().train(mode)

    def reset_state(self):
        functional.reset_net(self)
        for cell in self.cells():cell.reset_stream();cell.trace={}
        self.permit.reset_stream();self.read_weights=None;self.online_last={};self._stream_batch_size=None

    def _check(self,x):
        if x.ndim!=3 or x.shape[-1]!=self.config.inputs or x.shape[1]<1:
            raise ValueError('Expected nonempty[batch,time,140]float input')
        if not x.is_floating_point():raise TypeError('Frame counts must be floating point')

    def encode(self,x):
        self._check(x)
        permission=self.permit(x)[0] if self.stage!='content' and self.config.family=='commit' else None
        self.LMU1.permission=permission[:,:,:self.config.memory] if permission is not None else None
        self.LMU2.permission=permission[:,:,self.config.memory:] if permission is not None else None
        hidden=self.rnn_drop(self.LMU1(x));c=self.LMU2
        if self.config.family=='dmp':return {'spikes':c(hidden)}
        u=c.drive_input(hidden);f=features(hidden,delay(hidden),c.feature_scale)
        context=delay(affine_scan(torch.ones_like(f)*c.evidence_decay,(1-c.evidence_decay)*f))
        g=c.write_gate(c.gate(torch.cat((f,context),-1)))
        if c.permission is not None:g=g*c.permission
        p=c.candidate(u);m=affine_scan(1-g,g*delay(p));c.memory_for_aux=m
        if c.capture:c.trace={'p':p.detach(),'m':m.detach(),'gate':g.detach(),'u':u.detach()}
        return {'paths':(c.W_x(hidden),c.weight(p),c.read_m(m)),
                'features':read_features(f,context,p,m),'p':p,'m':m,'gate':g}

    def decode(self,encoded):
        if self.config.family=='dmp':return self.dense(encoded['spikes'])
        r=self.adapter(encoded['features']);self.read_weights=r.detach();x,p,m=encoded['paths']
        current=r[...,:1]*x+r[...,1:2]*p+r[...,2:]*m
        spikes=self.LMU2.f_x(current.permute(1,0,2)).permute(1,0,2).contiguous()
        return self.dense(spikes)

    def forward(self,x):
        """Classify a fresh sequence; stream_chunk is the stateful alternative."""
        self.reset_state()
        # Freeze upstream activations as well as parameters in reader training.
        with torch.no_grad() if self.read_only else nullcontext():encoded=self.encode(x)
        return self.decode(encoded)

    def memory_logits(self,queries):
        return self.memory_head(self.LMU2.memory_for_aux[:,[end-1 for _,end in queries]])

    @torch.no_grad()
    def stream_chunk(self,x):
        self._check(x)
        if self.training:raise RuntimeError('Call eval() before online inference')
        if self._stream_batch_size is not None and self._stream_batch_size!=len(x):
            raise ValueError('Online batch size changed; reset_state() before a new stream')
        self._stream_batch_size=len(x)
        output=[];d=self.config.memory
        for t in range(x.shape[1]):
            raw=x[:,t]
            permission=self.permit.step(raw) if self.stage!='content' and self.config.family=='commit' else None
            self.LMU1.permission_step=permission[:,:d] if permission is not None else None
            self.LMU2.permission_step=permission[:,d:] if permission is not None else None
            hidden=self.LMU1.step(raw);c=self.LMU2
            if self.config.family=='dmp':output.append(self.dense(c.step(hidden)));continue
            if not c.stream:
                c.stream={'p':hidden.new_zeros(len(hidden),d),'m':hidden.new_zeros(len(hidden),d),
                          'previous_rate':hidden.new_zeros(len(hidden),1),'e':hidden.new_zeros(len(hidden),4)}
            s=c.stream;u=c.drive_input(hidden);p=torch.nn.functional.linear(s['p'],c.A)+u*c.vector_input_scale
            f=features(hidden,s['previous_rate'],c.feature_scale);context=s['e']
            g=c.write_gate(c.gate(torch.cat((f,context),-1)))
            if c.permission_step is not None:g=g*c.permission_step
            m=(1-g)*s['m']+g*s['p'];r=self.adapter(read_features(f,context,p,m))
            current=r[:,:1]*c.W_x(hidden)+r[:,1:2]*c.weight(p)+r[:,2:]*c.read_m(m)
            output.append(self.dense(c.f_x(current.unsqueeze(0))[0]))
            s.update(p=p,m=m,e=c.evidence_decay*context+(1-c.evidence_decay)*f,previous_rate=hidden.mean(-1,keepdim=True))
            self.online_last={'p':p,'m':m,'gate':g,'read':r}
        return torch.stack(output,1)

    def transfer_common(self,source):
        """Copy matching weights; expand a scalar LMU input into vector drive."""
        own=self.state_dict();compatible={k:v for k,v in source.items() if k in own and v.shape==own[k].shape
                                         and not k.startswith(('dense.','memory_head.','permit.','adapter.'))}
        self.load_state_dict(compatible,strict=False)
        if self.config.family=='commit':
            with torch.no_grad():
                for name,cell in (('LMU1',self.LMU1),('LMU2',self.LMU2)):
                    w=source[name+'.W_u.weight'];b=source[name+'.W_u.bias']
                    if w.shape[0]==1:
                        scale=cell.initial_row_scale;cell.W_u.weight.copy_(w.repeat(self.config.memory,1)*scale[:,None])
                        cell.W_u.bias.copy_(b.repeat(self.config.memory)*scale)
                        cell.vector_input_scale.copy_(cell.B[:,0]/scale)
        return sorted(compatible)

    def checkpoint(self):
        return {'format_version':1,'config':self.config.to_dict(),'stage':self.stage,'model_state_dict':self.state_dict()}

    @classmethod
    def from_checkpoint(cls,checkpoint,device='cpu'):
        if checkpoint.get('format_version')!=1:raise ValueError('Unsupported checkpoint version')
        model=cls(MemoryConfig(**checkpoint['config']));model.load_state_dict(checkpoint['model_state_dict'])
        model.set_stage(checkpoint['stage']);model.to(device);model.eval();model.reset_state();return model
