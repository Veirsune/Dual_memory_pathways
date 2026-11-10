import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import MNIST
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--download',action='store_true');a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
train=MNIST(a.root,train=True,download=a.download);test=MNIST(a.root,train=False,download=a.download)
y=train.targets.numpy();rng=np.random.default_rng(1701);tr=[];va=[]
for c in range(10):
 ids=rng.permutation(np.flatnonzero(y==c));va.extend(ids[:600]);tr.extend(ids[600:])
assert len(tr)==54000 and len(va)==6000 and not set(tr)&set(va)
perm=torch.randperm(784,generator=torch.Generator().manual_seed(1702)).numpy()
arrays=dict(train_x=train.data.numpy()[tr].reshape(-1,784),train_y=y[tr],valid_x=train.data.numpy()[va].reshape(-1,784),valid_y=y[va],test_x=test.data.numpy().reshape(-1,784),test_y=test.targets.numpy(),train_indices=np.array(tr),valid_indices=np.array(va),permutation=perm)
np.savez_compressed(out/'mnist.npz',**arrays)
metadata={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
metadata['archive_sha256']=hashlib.sha256((out/'mnist.npz').read_bytes()).hexdigest();(out/'metadata.json').write_text(json.dumps(metadata,indent=2));print('DATA_READY',metadata['archive_sha256'],flush=True)
