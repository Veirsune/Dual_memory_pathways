"""Visual adaptation of the sealed Commit cell (fixed-read variant).

Only the permission input is adapted: 14 auditory bands become 2x4x4 spatial
polarity bins. Candidate, delayed commit, local gate and independent reads retain
the existing equations. No corruption masks or labels enter the cell.
"""
import torch
from torch import nn
from torch.nn import functional as F
from commit_memory.base_model import StreamCell,features,delay,affine_scan

class VisualPermission(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(72,32),nn.ReLU(),nn.Linear(32,d))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,2.)
    def forward(self,raw):
        b,t=raw.shape[:2]
        x=raw.flatten(2)/16.
        spatial=F.adaptive_avg_pool2d(raw.flatten(0,1)/16.,(4,4)).reshape(b,t,32)
        f=torch.cat((features(x,delay(x),x.new_tensor(1.)),spatial),-1)
        past=delay(affine_scan(torch.full_like(f,.8),.2*f))
        logits=self.net(torch.cat((f,past),-1));q=logits.sigmoid()
        hard=(logits>=0).to(q.dtype)
        return (hard.detach()+q-q.detach())*(x.sum(-1,keepdim=True)>0).to(q.dtype)

class VisualCommitCell(StreamCell):
    def __init__(self,inputs,hidden,d,steps,theta):
        super().__init__(inputs,hidden,d,'commit_dmp','zero',.02,'vector')
        del self.bn_x
        self.theta=theta;self.seq_len=steps
        self.A,self.B=self.stateSpaceMatrices()
        self.vector_input_scale.copy_(self.B[:,0]/self.initial_row_scale)
        self.H,self.fft_H=self.impulse()
        self.permit=VisualPermission(d)
    def forward(self,x,raw):
        self.permission=self.permit(raw)
        return super().forward(x)
