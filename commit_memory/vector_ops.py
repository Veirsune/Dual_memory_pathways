"""Fixed full-A LMU driven by a diagonal, full-rank input map."""
import torch
@torch.jit.script
def vector_forward(u:torch.Tensor,A:torch.Tensor,b:torch.Tensor):
    h=torch.zeros_like(u[:,0]);states=torch.jit.annotate(list[torch.Tensor],[])
    for t in range(u.shape[1]):
        h=torch.nn.functional.linear(h,A)+u[:,t]*b
        states.append(h)
    return torch.stack(states,1)
@torch.jit.script
def vector_backward(gradient:torch.Tensor,A:torch.Tensor,b:torch.Tensor):
    carry=torch.zeros_like(gradient[:,0]);out=torch.zeros_like(gradient)
    for k in range(gradient.shape[1]):
        t=gradient.shape[1]-1-k;total=gradient[:,t]+carry
        out[:,t]=total*b;carry=torch.nn.functional.linear(total,A.t())
    return out
class VectorLMU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,A,b):ctx.save_for_backward(A,b);return vector_forward(u,A,b)
    @staticmethod
    def backward(ctx,gradient):
        A,b=ctx.saved_tensors;return vector_backward(gradient,A,b),None,None

def matrix_kernel(A,b,steps):
    h=torch.diag(b);values=[]
    for _ in range(steps):values.append(h);h=A@h
    return torch.fft.rfft(torch.stack(values,-1),n=2*steps,dim=-1)
def vector_fft(u,kernel):
    n=u.shape[1];uf=torch.fft.rfft(u.permute(0,2,1),n=2*n,dim=-1)
    z=torch.einsum('bif,oif->bof',uf,kernel)
    return torch.fft.irfft(z,n=2*n,dim=-1)[...,:n].permute(0,2,1).contiguous()
