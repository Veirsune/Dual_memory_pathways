"""Offline prediction from a saved visual-task checkpoint and a local sample."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .models import build

GESTURES=['Hand clapping','Right hand wave','Left hand wave','Right arm clockwise',
          'Right arm counterclockwise','Left arm clockwise','Left arm counterclockwise',
          'Arm roll','Air drums','Air guitar','Other gestures']

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(1)
    checkpoint=torch.load(a.checkpoint,map_location='cpu');c=checkpoint['config']
    model=build(c['task'],c['memory'],c['theta'],c['steps'],c['family'],c['readout'])
    model.load_state_dict(checkpoint['model']);model.to(a.device).eval()
    loaded=np.load(a.input,allow_pickle=False)
    if isinstance(loaded,np.lib.npyio.NpzFile):
        with loaded:x=loaded['frames'].astype(np.float32)
    else:x=loaded
    if c['task']=='gesture':
        if x.ndim!=4 or x.shape[1:]!=(2,32,32):raise ValueError('Expected [T,2,32,32] pooled event counts')
        if len(x)!=c['steps']:
            if len(x)%c['steps']:raise ValueError('Cannot aggregate to checkpoint time steps')
            x=x.reshape(c['steps'],len(x)//c['steps'],2,32,32).sum(1)
        names=GESTURES
    else:
        if x.shape not in ((28,28),(784,),(784,1)):raise ValueError('Expected one MNIST image or 784 pixels')
        x=x.reshape(784,1)
        if x.dtype==np.uint8:x=x.astype(np.float32)/255.
        if x.min()<0 or x.max()>1:raise ValueError('Float pixels must be in [0,1]')
        x=x[np.array(checkpoint['split']['permutation'])];names=list(map(str,range(10)))
    with torch.no_grad():prob=model(torch.as_tensor(x,dtype=torch.float32,device=a.device).unsqueeze(0)).softmax(-1)[0].cpu()
    result={'task':c['task'],'predicted_index':int(prob.argmax()),'predicted_label':names[int(prob.argmax())],
            'probabilities':dict(zip(names,prob.tolist())),'checkpoint_epoch':checkpoint['epoch']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
