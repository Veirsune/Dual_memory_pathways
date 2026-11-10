import torch
from torch import nn
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from commit_memory.base_model import StreamModel,CausalPermission

class FSNN(nn.Module):
    def __init__(self):
        super().__init__();self.fc1=nn.Linear(140,128);self.lif1=MultiStepLIFNode(tau=2.,detach_reset=True);self.fc2=nn.Linear(128,128);self.lif2=MultiStepLIFNode(tau=2.,detach_reset=True);self.drop=nn.Dropout(.2);self.dense=nn.Linear(128,35)
    def forward(self,x):
        functional.reset_net(self);h=self.lif1(self.fc1(x).transpose(0,1)).transpose(0,1);h=self.drop(h);return self.dense(self.lif2(self.fc2(h).transpose(0,1)).transpose(0,1))

class FixedMemory(nn.Module):
    def __init__(self,family):
        super().__init__();self.core=StreamModel(variant='commit_dmp' if family=='commit' else 'dmp',d=30,gate_mode='zero',aux_enabled=False,gate_threshold=.02,drive='vector' if family=='commit' else 'scalar',protect=family=='commit',zero_rule=family=='commit');self.core.dense=nn.Linear(128,35)
        for c in self.core.cells():del c.bn_x
        del self.core.memory_head
        if family=='commit':
            with torch.no_grad():self.core.permit.net[-1].weight.zero_();self.core.permit.net[-1].bias.fill_(2.)
    def forward(self,x):self.core.reset_state();return self.core(x)

def initial_models(scale,seed=801):
    torch.manual_seed(seed);dmp=FixedMemory('dmp');torch.manual_seed(seed);commit=FixedMemory('commit');torch.manual_seed(seed);fsnn=FSNN()
    # Common fast weights and classifier; vector initialization expands the SAME scalar drive.
    commit.core.warm_start(dmp.core.state_dict());commit.core.dense.load_state_dict(dmp.core.dense.state_dict())
    torch.manual_seed(seed+23117);commit.core.permit=CausalPermission(30,True)
    with torch.no_grad():commit.core.permit.net[-1].weight.zero_();commit.core.permit.net[-1].bias.fill_(2.)
    fsnn.fc1.load_state_dict(dmp.core.LMU1.W_x.state_dict());fsnn.fc2.load_state_dict(dmp.core.LMU2.W_x.state_dict());fsnn.dense.load_state_dict(dmp.core.dense.state_dict())
    for c in (dmp.core,commit.core):c.LMU1.feature_scale.fill_(scale);c.permit.scale.fill_(scale)
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
