"""Convert the documented Tonic DVS Gesture mirror without extracting raw data.

No Tonic dependency. Verify the mirror checksum and stream the tar archive; save
selected subjects as compressed integer event counts. Cache remains reproducible.
"""
import argparse, hashlib, io, json, re, tarfile, time
from pathlib import Path
import numpy as np

URL = 'https://api.figshare.com/v2/file/download/38022171'
MD5 = '3a8f0d4120a166bac7591f77409cb105'

def integrate(events, steps):
    if events.ndim != 2 or events.shape[1] != 4: raise ValueError('Expected [events, x/y/p/t]')
    x,y,p,t = (events[:,i].astype(np.int64) for i in range(4))
    if len(t)==0: raise ValueError('Empty recording')
    if min(x.min(),y.min())<0 or max(x.max(),y.max())>=128: raise ValueError('Sensor bounds')
    if not np.isin(p,[0,1]).all(): raise ValueError('Polarity bounds')
    bins = np.minimum((t-t.min())*steps//(t.max()-t.min()+1),steps-1)
    flat = ((bins*2+p)*32+y//4)*32+x//4
    counts = np.bincount(flat,minlength=steps*2*32*32).reshape(steps,2,32,32)
    if counts.max()>65535: raise ValueError('uint16 event count overflow')
    assert int(counts.sum())==len(events)
    return counts.astype(np.uint16)

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--archive',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--steps',type=int,default=1000)
    a.add_argument('--subjects',type=int,nargs='*',default=[1,2,3,20])
    args=a.parse_args()
    if args.steps<1:raise ValueError('steps must be positive')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use a fresh cache directory')
    args.output.mkdir(parents=True,exist_ok=True)
    h=hashlib.md5()
    with args.archive.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''): h.update(data)
    if h.hexdigest()!=MD5: raise RuntimeError('Incorrect archive MD5')
    rows=[]; start=time.time(); seen=0
    with tarfile.open(args.archive,'r|gz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.npy'): continue
            seen+=1
            match=re.search(r'user(\d+)_',member.name)
            if not match: raise ValueError('Cannot identify subject: '+member.name)
            subject=int(match.group(1))
            if args.subjects and subject not in args.subjects: continue
            label=int(Path(member.name).stem)
            if not 0<=label<11: raise ValueError('Expected zero-based class label')
            source=tar.extractfile(member)
            events=np.load(io.BytesIO(source.read()),allow_pickle=False)
            frames=integrate(events,args.steps)
            identity=hashlib.sha256(member.name.encode()).hexdigest()[:20]
            target=args.output/f'{identity}.npz'
            np.savez_compressed(target,frames=frames,label=label,subject=subject)
            rows.append({'file':target.name,'label':label,'subject':subject,'source':member.name,
                         'events':len(events),'duration_ms':int(events[:,3].max()-events[:,3].min())})
            if len(rows)%20==0: print(json.dumps({'converted':len(rows),'seconds':round(time.time()-start)}),flush=True)
    manifest={'source_url':URL,'archive_md5':MD5,'source_format':'Tonic mirror; presegmented; ms timestamps',
              'archive_samples':seen,'steps':args.steps,'spatial_pool':4,'subjects':args.subjects,
              'binning':'equal-duration across each full clip; no event drop; sum counts',
              'samples':rows,'seconds':time.time()-start}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'saved':len(rows),'archive_samples':seen,'seconds':time.time()-start}),flush=True)

if __name__=='__main__': main()
