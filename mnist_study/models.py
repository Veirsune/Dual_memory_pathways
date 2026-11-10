"""Fixed Commit equations adapted to a one-channel 784-step pixel stream.

No timestep, permutation index, corruption mask, label or segment boundary is
available to the model. FFT computes the same fixed full-A candidate recurrence.
"""
import torch
from torch import nn
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from commit_memory.base_model import StreamCell, features, delay, affine_scan
from commit_memory.vector_ops import matrix_kernel, vector_fft, VectorLMU

D=40; HIDDEN=200; THETA=300

class PixelPermission(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,2*D))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,2.)
    def forward(self,x):
        f=features(x,delay(x),x.new_tensor(1.))
        past=delay(affine_scan(torch.full_like(f,.8),.2*f))
        logits=self.net(torch.cat((f,past),-1));q=logits.sigmoid()
        return ((logits>=0).to(q.dtype).detach()+q-q.detach())*(x.sum(-1,keepdim=True)>0).to(q.dtype)

class PixelCell(StreamCell):
    def __init__(self,inputs,family):
        super().__init__(inputs,HIDDEN,D,'commit_dmp' if family=='commit' else 'dmp','zero',.02,'vector' if family=='commit' else 'scalar')
        del self.bn_x
        self.theta=THETA;self.seq_len=784
        self.A,self.B=self.stateSpaceMatrices()
        if self.drive=='vector':self.vector_input_scale.copy_(self.B[:,0]/self.initial_row_scale)
        self.H,self.fft_H=self.impulse();self.kernel_cache={};self.recurrent=False
    def candidate(self,u):
        if self.recurrent:return super().candidate(u)
        n=u.shape[1];key=(n,str(u.device),u.dtype)
        if key not in self.kernel_cache:
            with torch.no_grad():
                # CPU kernel preparation avoids hundreds of tiny CUDA launches.
                A=self.A.detach().cpu()
                if self.drive=='vector':k=matrix_kernel(A,self.vector_input_scale.detach().cpu(),n)
                else:
                    h=self.B.detach().cpu();values=[]
                    for _ in range(n):values.append(h);h=A@h
                    k=torch.fft.rfft(torch.cat(values,-1),n=2*n,dim=-1)
                self.kernel_cache[key]=k.to(u.device)
        k=self.kernel_cache[key]
        if self.drive=='vector':return vector_fft(u,k)
        return torch.fft.irfft(torch.fft.rfft(u.transpose(1,2),n=2*n)*k[None],n=2*n)[...,:n].transpose(1,2).contiguous()

class Classifier(nn.Module):
    def __init__(self,family):
        super().__init__();self.family=family
        if family=='fsnn':
            self.fc1=nn.Linear(1,HIDDEN);self.fc2=nn.Linear(HIDDEN,HIDDEN)
            self.lif1=MultiStepLIFNode(tau=2.,detach_reset=True);self.lif2=MultiStepLIFNode(tau=2.,detach_reset=True)
        else:
            self.cell1=PixelCell(1,family);self.cell2=PixelCell(HIDDEN,family)
            if family=='commit':self.permission=PixelPermission()
        self.dense=nn.Linear(HIDDEN,10)
    def forward(self,x):
        functional.reset_net(self)
        if self.family=='fsnn':
            h=self.lif1(self.fc1(x).transpose(0,1)).transpose(0,1)
            h=self.lif2(self.fc2(h).transpose(0,1)).transpose(0,1)
        else:
            if self.family=='commit':
                q=self.permission(x);self.cell1.permission=q[...,:D];self.cell2.permission=q[...,D:]
            h=self.cell2(self.cell1(x))
        return self.dense(h)

def initial_models(seed):
    torch.manual_seed(seed);dmp=Classifier('dmp')
    torch.manual_seed(seed+10000);commit=Classifier('commit')
    torch.manual_seed(seed+20000);fsnn=Classifier('fsnn')
    with torch.no_grad():
        for i in (1,2):
            a=getattr(dmp,f'cell{i}');c=getattr(commit,f'cell{i}')
            getattr(fsnn,f'fc{i}').load_state_dict(a.W_x.state_dict());c.W_x.load_state_dict(a.W_x.state_dict());c.weight.load_state_dict(a.weight.state_dict())
            c.W_u.weight.copy_(a.W_u.weight.expand(D,-1)*c.initial_row_scale[:,None]);c.W_u.bias.copy_(a.W_u.bias.expand(D)*c.initial_row_scale)
            c.read_m.weight.zero_()
        commit.dense.load_state_dict(dmp.dense.state_dict());fsnn.dense.load_state_dict(dmp.dense.state_dict())
    return dict(fsnn=fsnn,dmp=dmp,commit=commit)

def backend(model,device):
    model.to(device)
    if device=='cpu':
        from vision_tasks.cpu_lif import enable_cpu_scan
        enable_cpu_scan(model)
    else:
        for m in model.modules():
            if isinstance(m,MultiStepLIFNode):m.backend='cupy'
    return model
