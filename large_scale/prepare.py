import argparse
from pathlib import Path
import numpy as np,h5py,json,hashlib,time,tarfile
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
for split in ('train','valid','test'):
 p=out/f'{split}-x.npy';raw=a.raw/f'ssc_{split}.h5'
 if (out/f'{split}-metadata.json').exists():continue
 start=time.time();digest=hashlib.sha256();max_count=0
 with h5py.File(raw,'r') as f:
  y=f['labels'][:].astype(np.int64);s=f['extra/speaker'][:].astype('S8');x=np.lib.format.open_memmap(p.with_suffix('.partial'),mode='w+',dtype=np.uint8,shape=(len(y),250,140))
  for i in range(len(y)):
   t=f['spikes/times'][i].astype(np.float64);u=f['spikes/units'][i].astype(np.int64);assert len(t)>0 and len(t)==len(u) and np.all((u>=0)&(u<700)) and np.isfinite(t).all()
   bins=np.minimum(((t-t.min())/(t.max()-t.min())*250).astype(np.int64),249) if t.max()>t.min() else np.zeros(len(t),np.int64)
   c=np.bincount(bins*700+u,minlength=250*700).reshape(250,140,5).max(2);assert c.max()<256;x[i]=c;max_count=max(max_count,int(c.max()));digest.update(x[i].tobytes())
   if (i+1)%10000==0:print(split,i+1,len(y),round(time.time()-start),flush=True)
  x.flush();del x;p.with_suffix('.partial').rename(p);np.save(out/f'{split}-y.npy',y);np.save(out/f'{split}-speaker.npy',s)
 meta=dict(split=split,samples=len(y),classes=35,steps=250,channels=140,frame_sha256=digest.hexdigest(),max_count=max_count,encoding='first-to-last event equal-duration250bins; contiguous5channel max pooling; lossless uint8 storage',raw=str(raw),seconds=time.time()-start)
 (out/f'{split}-metadata.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
sets=[set(np.load(out/f'{split}-speaker.npy').tolist()) for split in ('train','valid','test')];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
print('DATA_READY',str(out),flush=True)
