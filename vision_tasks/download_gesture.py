"""Download the Tonic-referenced public training mirror with checksum validation."""
import argparse,hashlib,time
from pathlib import Path
import requests
from .prepare_gesture import URL,MD5

def digest(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if digest(a.output)!=MD5:raise RuntimeError('Existing file checksum mismatch; use a new path')
        print('Existing archive verified');return
    partial=a.output.with_name(a.output.name+'.partial')
    if partial.exists():raise FileExistsError('Partial file exists; inspect/remove it or use a new path')
    start=time.time();last=start;size=0;h=hashlib.md5()
    with requests.get(URL,stream=True,timeout=(20,60)) as r:
        r.raise_for_status()
        with partial.open('xb') as f:
            for chunk in r.iter_content(4*1024*1024):
                if not chunk:continue
                f.write(chunk);h.update(chunk);size+=len(chunk)
                if time.time()-last>=20:
                    print(f'{size/1e6:.0f} MB downloaded, {time.time()-start:.0f} s',flush=True);last=time.time()
    if h.hexdigest()!=MD5:raise RuntimeError('Downloaded checksum mismatch')
    partial.rename(a.output);print('Verified',MD5,'bytes',size)

if __name__=='__main__':main()
