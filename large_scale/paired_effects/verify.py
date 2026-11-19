"""Cross-check label order, archived clean predictions, and the new sample denominators."""
import argparse
import json
from pathlib import Path
import numpy as np

def check(exp,pred):
    agreements={}
    def compare(name,fresh,old,y,labels):
        assert np.array_equal(y,labels)
        fraction=float((fresh==old).mean())
        agreements[name]=fraction
        assert fraction>.998, (name,fraction)
    for s in (301,302,303):
        for model in ('dmp','commit_dmp'):
            key=f'{model}-d30-s{s}'
            a=np.load(pred/f'shd-{s}-{model}.npz')
            b=np.load(exp/f'shd-streaming/evaluation/{key}/clean.npz')
            compare(key,a['clean'],b['predictions'][:,0],a['labels'],b['targets'][:,0])
    for s in (501,502,503):
        a=np.load(pred/f'shd-vector-{s}-full.npz')
        b=np.load(exp/f'shd-supervision-ablation/evaluation/zero_rule-d30-s{s}/official_test/clean.npz')
        compare(f'vector-{s}',a['clean'],b['predictions'][:,0],a['labels'],b['targets'][:,0])
    b=np.load(exp/'final-study/results/official-evaluation/predictions.npz')
    for model in ('dmp','commit'):
        a=np.load(pred/f'gesture-{model}.npz')
        compare('gesture-'+model,a['clean'],b[f'finetuned__{model}__clean__mean'],a['labels'],b['labels'])
    return agreements

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('experiments',type=Path)
    p.add_argument('pred',type=Path)
    a=p.parse_args()
    print(json.dumps(check(a.experiments,a.pred),indent=2))
