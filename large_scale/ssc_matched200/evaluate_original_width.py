"""Score the archived original-width-protocol DMP/GDSM pair on identical stimuli."""
import argparse
import json
from pathlib import Path
import torch
import train
from width_diagnostics import paired_test

def main(checkpoints,out):
    torch.set_num_threads(1)
    test=train.load_split('test')
    scale=json.loads((checkpoints/'model-info.json').read_text())['feature_scale']
    out.mkdir(exist_ok=True,parents=True)
    for family,model in train.make_models(scale).items():
        state=torch.load(checkpoints/(family+'-clean-best.pt'),map_location=train.DEVICE)
        model.load_state_dict(state)
        model.eval()
        paired_test(model,family,test,train.scores,train.stimulus,train.BATCH,
                    out/(family+'-original.npz'),full_only=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('checkpoints',type=Path)
    p.add_argument('out',type=Path)
    args=p.parse_args()
    main(args.checkpoints,args.out)
