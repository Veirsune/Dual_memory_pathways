"""Inference-only trace accounting and executable stable-read cache verification.

Counters are logical operations/accesses for a documented causal mapping, not
physical GPU transactions or measured silicon power. Original eager kernels do
not branch on zero gates. Run only after final-study training/evaluation complete.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from spikingjelly.clock_driven import functional
from .final_study import make_model,horizon,load_samples,sha
from .corruption_eval import corrupt

def cached_read(p,g,weight):
    b,t,d=p.shape;n=weight.shape[0];m=p.new_zeros(b,d);cache=p.new_zeros(b,n);outs=[];states=[];calls=0;active_dims=0
    for k in range(t):
        previous=p[:,k-1] if k else torch.zeros_like(m);mask=g[:,k]!=0;active=mask.any(-1)
        if mask.any():
            old=m[mask];m[mask]=old+g[:,k][mask]*(previous[mask]-old);active_dims+=int(mask.sum())
        if active.any():
            cache[active]=F.linear(m[active],weight);calls+=int(active.sum())
        # These clones are verification outputs, not persistent streaming storage.
        outs.append(cache.clone());states.append(m.clone())
    return torch.stack(outs,1),torch.stack(states,1),calls,active_dims

def mapping(family,frames,nnz,active_steps=0,active_dims=0):
    M,N,d=4096,256,40;neurons=49419
    # Shared front-end excluded from core energy; its measured counts are separate.
    fast=nnz*N;drive=nnz*(d if family=='commit' else 1);candidate=frames*d*d;readp=frames*d*N
    gate=frames*(8*d+72*32+32*d) if family=='commit' else 0
    common_mac=fast+drive+candidate+readp+gate
    # Input-stationary/output-stationary logical 32-bit element-access model.
    common_read=frames*(M+2*d+N)
    common_write=frames*(d+N)
    if family=='commit':
        common_read+=frames*(2048+42);common_write+=frames*42
    def record(steps,dims,cache):
        stable=steps*d*N if family=='commit' else 0
        # One weight fetch per effective matrix MAC; scale B fetches d/frame.
        wr=common_mac+stable+frames*d
        sr=common_read;sw=common_write
        if family=='commit':
            sr+=2*dims+steps*d;sw+=dims
            if cache:sr+=(frames-steps)*N;sw+=steps*N
        return dict(mac=common_mac+stable,weight_reads=wr,state_reads=sr,state_writes=sw,scalar_mul=frames*d+(dims if family=='commit' else 0),scalar_add=2*dims if family=='commit' else 0,stable_read_mac=stable,stable_update_coordinates=dims)
    if family=='commit':return dict(dense_causal=record(frames,frames*d,False),cached_causal=record(active_steps,active_dims,True))
    return dict(dense_causal=record(0,0,False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--batch-size',type=int,default=16);a=p.parse_args();b=a.base
    assert json.loads((b/'official-evaluation/complete.json').read_text())['completed'],'Block3 must finish first'
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);start=time.time();protocol=json.loads((b/'finetuned/protocol.json').read_text());manifest=json.loads((b/'data/gesture-test-t100/manifest.json').read_text());rows=manifest['samples'][:4] if a.smoke else manifest['samples'];xs=load_samples(b/'data/gesture-test-t100',[r['file'] for r in rows]);rate=protocol['noise_rate'];reports=[];params={}
    cases=[dict(name='clean',length=0,position=100,kind='silence',multiplier=0.)]+[dict(name=f'p{pos}_{kind}{mult:g}_L{L}',length=L,position=pos,kind=kind,multiplier=mult) for pos in (50,100) for kind,mult in [('silence',0.),('poisson',1.),('poisson',3.)] for L in (20,40,80)]
    if a.smoke:cases=[cases[0],cases[-1]]
    for family in ('dmp','commit'):
        cp=torch.load(b/'finetuned'/family/'last.pt',map_location='cpu');model=make_model(cp,a.device)
        params[family]=dict(trainable_parameters=sum(p.numel() for p in model.parameters()),parameter_bytes_fp32=sum(p.numel()*p.element_size() for p in model.parameters()),core_persistent_floats=256+40 if family=='dmp' else 256+80+42,core_cached_floats=256+40 if family=='dmp' else 256+80+42+256,full_persistent_floats=49419+40 if family=='dmp' else 49419+80+42,full_cached_floats=49419+40 if family=='dmp' else 49419+80+42+256,checkpoint_sha256=sha(b/'finetuned'/family/'last.pt'))
        if family=='commit':model.fc1.capture=True
        cache={};conv_counts={}
        def core_hook(module,args):cache['z']=args[0].detach()
        def out_hook(module,args):cache['spikes']=args[0].detach()
        def conv_hook(name):
            def hook(module,args):
                x=args[0];patch=F.unfold(x,kernel_size=module.kernel_size,padding=module.padding,stride=module.stride)
                conv_counts[name]=(patch.numel()*module.out_channels,int(torch.count_nonzero(patch))*module.out_channels)
            return hook
        handles=[model.fc1.register_forward_pre_hook(core_hook),model.fc2.register_forward_pre_hook(out_hook),model.conv1.register_forward_pre_hook(conv_hook('conv1')),model.conv2.register_forward_pre_hook(conv_hook('conv2'))]
        with torch.no_grad():
            for case in cases:
                T=100+case['length'];horizon(model,T);totals=dict(frames=0,nnz=0,active_steps=0,active_dims=0,gap_frames=0,gap_active_steps=0,gap_active_dims=0,conv_dense_mac=0,conv_effective_mac=0,output_sop=0,max_memory_error=0.,max_read_error=0.,classification_disagreements=0,spike_disagreements=0,eager_stable_read_mac=0,eager_stable_scan_mul=0,eager_stable_scan_add=0)
                for begin in range(0,len(xs),a.batch_size):
                    ids=list(range(begin,min(begin+a.batch_size,len(xs))));xx=np.stack([corrupt(xs[i],case['position'],case['length'],case['kind'],rate*case['multiplier'],1101+1009*i) for i in ids]);x=torch.from_numpy(xx).to(a.device);out=model(x);original_spikes=cache['spikes'].clone();original_tail=model.lif4.v_seq[-20:].mean(0).clone();z=cache['z'];B=len(ids);totals['frames']+=B*T;totals['nnz']+=int(torch.count_nonzero(z));totals['conv_dense_mac']+=sum(v[0] for v in conv_counts.values());totals['conv_effective_mac']+=sum(v[1] for v in conv_counts.values());totals['output_sop']+=int(torch.count_nonzero(original_spikes))*11
                    if family=='commit':
                        tr=model.fc1.trace;p_,m_,g=tr['p'],tr['m'],tr['gate'];cached,states,calls,dims=cached_read(p_,g,model.fc1.read_m.weight);dense=F.linear(m_,model.fc1.read_m.weight)
                        merr=float((states-m_).abs().max());rerr=float((cached-dense).abs().max());assert torch.allclose(states,m_,atol=2e-4,rtol=2e-4);assert torch.allclose(cached,dense,atol=3e-4,rtol=3e-4)
                        totals['max_memory_error']=max(totals['max_memory_error'],merr);totals['max_read_error']=max(totals['max_read_error'],rerr);totals['active_steps']+=calls;totals['active_dims']+=dims
                        pos,L=case['position'],case['length'];gap=g[:,pos:pos+L];totals['gap_frames']+=B*L;totals['gap_active_steps']+=int((gap!=0).any(-1).sum());totals['gap_active_dims']+=int((gap!=0).sum())
                        if case['kind']=='silence' and L:assert not torch.count_nonzero(gap)
                        # Replay downstream layers with the executable cached stable read.
                        current=model.fc1.W_x(z)+model.fc1.weight(p_)+cached;functional.reset_net(model.fc1.f_x);sp=model.fc1.f_x(current.transpose(0,1)).transpose(0,1);functional.reset_net(model.lif4);model.lif4(model.fc2(sp).transpose(0,1));replay=model.lif4.v_seq.mean(0);tail=model.lif4.v_seq[-20:].mean(0)
                        totals['classification_disagreements']+=int((replay.argmax(-1)!=out.argmax(-1)).sum())+int((tail.argmax(-1)!=original_tail.argmax(-1)).sum());totals['spike_disagreements']+=int((sp!=original_spikes).sum())
                        scan_sum=sum(T-2**k for k in range((T-1).bit_length()));totals['eager_stable_read_mac']+=B*T*40*256;totals['eager_stable_scan_mul']+=B*40*(T+2*scan_sum);totals['eager_stable_scan_add']+=B*40*(T+scan_sum)
                mapped=mapping(family,totals['frames'],totals['nnz'],totals['active_steps'],totals['active_dims']);report=dict(family=family,**case,samples=len(xs),**totals,mapping=mapped);reports.append(report);(a.output/'counts.json').write_text(json.dumps(reports,indent=2));print(json.dumps(dict(family=family,case=case['name'],cached_mac=mapped.get('cached_causal',{}).get('mac'),gap_closed=1-totals['gap_active_steps']/totals['gap_frames'] if totals['gap_frames'] else None)),flush=True)
        for handle in handles:handle.remove()
    (a.output/'parameters.json').write_text(json.dumps(params,indent=2));(a.output/'complete.json').write_text(json.dumps(dict(completed=True,smoke=a.smoke,batch_size=a.batch_size,samples=len(xs),cases_per_model=len(cases),wall_seconds=time.time()-start,physical_energy_measured=False,physical_sram_transactions_measured=False)))
if __name__=='__main__':main()
