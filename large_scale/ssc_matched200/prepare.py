"""Bin SSC events from first to last event and sum each five adjacent cochlear channels."""
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np

def convert(raw, dest, split):
    with h5py.File(raw / f'ssc_{split}.h5') as h:
        y = h['labels'][:].astype(np.int64)
        out = np.lib.format.open_memmap(dest / f'{split}-x.npy', mode='w+', dtype=np.uint8, shape=(len(y),250,140))
        for start in range(0,len(y),256):
            times=h['spikes/times'][start:start+256]
            units=h['spikes/units'][start:start+256]
            for j,(t,u) in enumerate(zip(times,units)):
                t=t.astype(np.float64)
                bins=np.minimum(((t-t.min())/(t.max()-t.min())*250).astype(np.int64),249) if t.max()>t.min() else np.zeros(len(t),np.int64)
                counts=np.bincount(bins*140+u.astype(np.int64)//5,minlength=35000)
                assert counts.max()<256 and counts.sum()==len(t)
                out[start+j]=counts.reshape(250,140)
        out.flush()
        np.save(dest/f'{split}-y.npy',y)
        print(split,len(y),hashlib.sha256(out).hexdigest(),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('dest',type=Path);p.add_argument('split');a=p.parse_args();convert(a.raw,a.dest,a.split)
