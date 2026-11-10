"""Run a trained fixed-memory checkpoint on encoded auditory frames."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from commit_memory import MemoryClassifier

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True,help='NPY float/count array[time,140]or[batch,time,140]')
    parser.add_argument('--chunk',type=int,default=0,help='Positive size uses persistent online state across chunks;0uses optimized offline batch')
    parser.add_argument('--window',type=int,default=20,help='Average the last N frames for classification;0averages the whole sequence')
    parser.add_argument('--output',type=Path,required=True,help='NPZ predictions,probabilities and per-frame logits')
    parser.add_argument('--labels',type=Path,help='Optional JSON array of class names in classifier order')
    parser.add_argument('--threads',type=int,default=2)
    a=parser.parse_args()
    if a.chunk<0 or a.window<0:parser.error('chunk and window must be nonnegative')
    torch.set_num_threads(a.threads)
    data=np.load(a.input,allow_pickle=False)
    if data.ndim==2:data=data[None]
    if data.ndim!=3 or data.shape[-1]!=140 or min(data.shape[:2])<1 or not np.isfinite(data).all() or (data<0).any():
        parser.error('Expected finite,nonnegative nonempty auditory counts[batch,time,140]')
    x=torch.from_numpy(data.astype(np.float32));cp=torch.load(a.checkpoint,map_location='cpu')
    model=MemoryClassifier.from_checkpoint(cp)
    with torch.no_grad():
        if a.chunk:logits=torch.cat([model.stream_chunk(x[:,i:i+a.chunk]) for i in range(0,x.shape[1],a.chunk)],1)
        else:logits=model(x)
        pooled=logits[:,-a.window:].mean(1) if a.window else logits.mean(1)
        probability=pooled.softmax(-1);prediction=probability.argmax(-1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,predictions=prediction.numpy(),probabilities=probability.numpy(),logits=logits.numpy())
    result={'predictions':prediction.tolist(),'output':str(a.output),'mode':'online' if a.chunk else 'offline',
            'classes':model.config.classes,'frames':x.shape[1],'window':min(a.window,x.shape[1]) if a.window else x.shape[1]}
    if a.labels:
        labels=json.loads(a.labels.read_text())
        if not isinstance(labels,list) or len(labels)!=model.config.classes:parser.error('labels must contain exactly one name per class')
        result['class_names']=[labels[i] for i in prediction.tolist()]
    print(json.dumps(result))

if __name__=='__main__':main()
