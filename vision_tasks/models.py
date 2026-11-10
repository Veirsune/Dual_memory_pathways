"""Original sequential-MNIST model and a documented DVS reconstruction."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

ROOT = Path(__file__).resolve().parents[1]

def original_module(task='smnist'):
    spec = importlib.util.spec_from_file_location('original_'+task, ROOT/task/'src/dual_pathway.py')
    module = importlib.util.module_from_spec(spec)
    # Author files seed torch at import time. Preserve the caller's experiment RNG.
    with torch.random.fork_rng():
        spec.loader.exec_module(module)
    return module

class SequentialClassifier(nn.Module):
    def __init__(self, task='smnist', memory=40, theta=300, readout='mean'):
        super().__init__()
        self.core = original_module(task).ConvLMU2(memory, 784, theta)
        self.readout = readout

    def forward(self, x):
        functional.reset_net(self)
        # Original core uses squeeze(), which removes the batch axis at B=1.
        # Preserve B while retaining the same layers, initialization and equations.
        z = self.core.dense(self.core.forward_features(x))
        return z.mean(1) if self.readout == 'mean' else z[:, -1]

class GestureClassifier(nn.Module):
    """Paper-description reconstruction, not unpublished author code.

    Original-resolution event counts are pooled 4x4 while loading (division by
    16 is equivalent to average pooling). Then Conv32-LIF-pool2, Conv64-LIF-pool2,
    DMP-FC256-LIF, FC11-LIF, mean output membrane potential. Channels, pooling,
    optimizer and d are explicit reconstruction choices, not paper hyperparameters.
    """
    def __init__(self, steps=1000, memory=40, theta=300, family='dmp'):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.lif1 = MultiStepLIFNode(tau=2., detach_reset=True)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.lif2 = MultiStepLIFNode(tau=2., detach_reset=True)
        self.family = family
        if family == 'dmp':
            self.fc1 = original_module().SpikingLMUFFTCell(64*8*8, 256, memory, steps, theta)
        elif family == 'commit':
            with torch.random.fork_rng():
                from .commit_cell import VisualCommitCell
            self.fc1 = VisualCommitCell(64*8*8,256,memory,steps,theta)
        else:
            self.fc1 = nn.Linear(64*8*8, 256)
            self.lif3 = MultiStepLIFNode(tau=2., detach_reset=True)
        self.fc2 = nn.Linear(256, 11)
        self.lif4 = MultiStepLIFNode(tau=2., detach_reset=True)

    def forward(self, x):
        functional.reset_net(self)
        b,t,c,h,w = x.shape
        if (c,h,w) != (2,32,32): raise ValueError('Expected [B,T,2,32,32] pooled counts')
        z = self.conv1(x.flatten(0,1)/16.)
        z = self.lif1(z.reshape(b,t,32,32,32).transpose(0,1)).transpose(0,1)
        z = F.avg_pool2d(z.flatten(0,1),2)
        z = self.conv2(z)
        z = self.lif2(z.reshape(b,t,64,16,16).transpose(0,1)).transpose(0,1)
        z = F.avg_pool2d(z.flatten(0,1),2).reshape(b,t,-1)
        if self.family == 'dmp': z,_ = self.fc1(z)
        elif self.family == 'commit': z = self.fc1(z,x)
        else: z = self.lif3(self.fc1(z).transpose(0,1)).transpose(0,1)
        self.lif4(self.fc2(z).transpose(0,1))
        return self.lif4.v_seq.mean(0)

def build(task, memory=40, theta=300, steps=1000, family='dmp', readout='mean'):
    if task in ('smnist','psmnist'):
        if family != 'dmp': raise ValueError('Sequential runner currently uses original DMP')
        return SequentialClassifier(task,memory,theta,readout)
    return GestureClassifier(steps,memory,theta,family)
