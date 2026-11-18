"""Evaluate the one-step stable-memory reset without changing the trained model."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn
from commit_memory import base_model
from commit_memory.vector_ops import matrix_kernel, vector_fft

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT/'protocol.json').read_text())
MID = PROTOCOL['boundary']


def fft_candidate(cell,u):
    n=u.size(1)
    if n not in cell._kernels:
        cell._kernels[n]=matrix_kernel(cell.A,cell.vector_input_scale,n)
    return vector_fft(u,cell._kernels[n])


def load_model(checkpoint):
    if checkpoint.suffix=='.pt':
        raw=checkpoint.read_bytes()
    else:
        with tarfile.open(checkpoint) as archive:
            raw=archive.extractfile('results/gdsm-mixed-best.pt').read()
    assert hashlib.sha256(raw).hexdigest()==PROTOCOL['weights_sha256']
    m=base_model.StreamModel(variant='commit_dmp',d=10,gate_mode='zero',gate_threshold=.02,drive='vector',protect=True,zero_rule=True)
    m.rnn_drop=nn.Identity()
    m.dense=nn.Linear(128,35)
    del m.memory_head
    for c in m.cells():
        del c.bn_x
        c.candidate=MethodType(fft_candidate,c)
    m.load_state_dict(torch.load(io.BytesIO(raw),map_location='cpu'))
    return m.eval()


def forward(model,x,clear):
    model.reset_state()
    if not clear:
        return model(x)
    # m[t] = a[t]*m[t-1] + b[t]. Cutting a[MID] once is exactly a reset
    # before processing frame 126. b[MID] still uses the original p[125].
    scan=base_model.affine_scan
    calls=[]
    def cut_stable_edge(a,b):
        if a.shape[-1]==model.d:
            # Other affine scans in this checkpoint compute 4- or 18-D evidence.
            assert len(calls)<2
            calls.append(True)
            a=a.clone()
            a[:,MID]=0
        return scan(a,b)
    with patch.object(base_model,'affine_scan',cut_stable_edge):
        z=model(x)
    assert len(calls)==2
    return z


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    x=np.load(args.data/'test-x.npy',mmap_mode='r')
    y=np.load(args.data/'test-y.npy')
    assert x.shape==(20382,250,140) and hashlib.sha256(x).hexdigest()==PROTOCOL['input_sha256']
    with h5py.File(args.raw) as h:
        assert np.array_equal(y,h['labels'][:])
        speakers=h['extra/speaker'][:].astype('S')
    model=load_model(args.checkpoint)
    rng=np.random.default_rng(4917)
    preds={}
    checks={}
    for i in range(0,len(x),args.batch):
        v=torch.tensor(x[i:i+args.batch],dtype=torch.float32)
        noisy=v.clone()
        noisy[:,MID:] += torch.tensor(rng.poisson(.125,size=noisy[:,MID:].shape),dtype=v.dtype)
        for cond,batch in [('clean',v),('noise',noisy)]:
            model.reset_state()
            model.stream_chunk(batch[:,:MID])
            # Save the common prefix once. All suffix steps overwrite permission_step
            # and v_seq; only the last membrane voltage is a recurrent LIF state.
            permit={k:v.clone() for k,v in model.permit.stream.items()}
            states=[({k:v.clone() for k,v in cell.stream.items()},cell.f_x.v.clone())
                    for cell in model.cells()]
            for clear in [False,True]:
                key=f'{cond}_'+('reset' if clear else 'keep')
                model.permit.stream={k:v.clone() for k,v in permit.items()}
                for cell,(state,voltage) in zip(model.cells(),states):
                    cell.stream={k:v.clone() for k,v in state.items()}
                    cell.f_x.v=voltage.clone()
                    if clear:cell.stream['m'].zero_()
                scores=model.stream_chunk(batch[:,MID:]).mean(1)
                assert torch.isfinite(scores).all()
                if i==0:
                    reference=forward(model,batch[:16],clear)[:,MID:].mean(1)
                    checks[key]=dict(probe_n=16,equal_predictions=int((reference.argmax(1)==scores[:16].argmax(1)).sum()),max_score_difference=float((reference-scores[:16]).abs().max()))
                    assert checks[key]['equal_predictions']==16,checks
                preds.setdefault(key,[]).append(scores.argmax(1).numpy())
        if i%(args.batch*32)==0:
            print('Scored',min(i+args.batch,len(x)),flush=True)
    preds={k:np.concatenate(v) for k,v in preds.items()}
    np.savez_compressed(ROOT/'predictions.npz',labels=y,speakers=speakers,**preds)
    _,groups=np.unique(speakers,return_inverse=True)
    counts=np.bincount(groups)
    draws=np.random.default_rng(413).integers(len(counts),size=(10000,len(counts)))
    result=dict(n=len(y),speakers=len(counts),implementation_checks=checks)
    diffs={}
    for cond in ['clean','noise']:
        keep=preds[cond+'_keep']==y
        reset=preds[cond+'_reset']==y
        diffs[cond]=keep.astype(int)-reset.astype(int)
        result[cond]=dict(keep_correct=int(keep.sum()),reset_correct=int(reset.sum()),keep_pct=100*keep.mean(),reset_pct=100*reset.mean(),keep_only=int((keep&~reset).sum()),reset_only=int((reset&~keep).sum()))
    for cond,diff in list(diffs.items())+[('noise_minus_clean',diffs['noise']-diffs['clean'])]:
        samples=100*np.bincount(groups,weights=diff)[draws].sum(1)/counts[draws].sum(1)
        result.setdefault(cond,{}).update(delta_pp=100*diff.mean(),ci95=np.percentile(samples,[2.5,97.5]).tolist())
    (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ['data','raw','checkpoint']:p.add_argument(arg,type=Path)
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--batch',type=int,default=64)
    with torch.no_grad():run(p.parse_args())
