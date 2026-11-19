"""Fit the vector GDSM once on the archived seed-301 DMP fine-tuning schedule."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import torch
from paired_eval import save

def main(exp,out,repo):
    source=exp/'shd-streaming'
    vector=exp/'shd-supervision-ablation'
    # Reuse the archived sampling, augmentation, optimizer and loss implementation.
    sys.path.insert(0,str(source))
    import train
    from data import arrays
    ref=json.loads((source/'evaluation/dmp-d30-s301/results.json').read_text())
    arch=json.loads((Path(ref['checkpoint']).parent/'summary.json').read_text())
    assert arch['config']['seed']==301 and arch['actual_epochs']==30
    sys.path.insert(0,str(vector))
    spec=importlib.util.spec_from_file_location('vector_model',vector/'model.py')
    vm=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vm)
    sys.path.insert(0,str(repo))
    from vision_tasks.cpu_lif import enable_cpu_scan
    class MatchedVector(vm.StreamModel):
        def __init__(self,variant,d):
            super().__init__(variant,d,'zero',False,.02,'vector',True,True)
            with torch.no_grad():
                self.permit.net[-1].weight.zero_()
                self.permit.net[-1].bias.fill_(2.)
            enable_cpu_scan(self)
        def forward(self,x):
            self.permit.scale.copy_(self.LMU1.feature_scale)
            return super().forward(x)
    out.mkdir(parents=True,exist_ok=True)
    train.StreamModel=MatchedVector
    train.ROOT=out
    for name in ('reference_model.py','data.py','evaluation.py','train.py','pretrained-sources.json'):
        shutil.copy2(source/name,out/name)
    for name in ('model.py','vector_ops.py'):shutil.copy2(vector/name,out/name)
    if not list(out.glob('runs/*/final.pt')):
        sys.argv=['train.py','--variant','commit_dmp','--d','30','--phase','final','--seed','301','--epochs','30','--gate-init','.95']
        train.main()
    [cp_path]=out.glob('runs/*/final.pt')
    cp=torch.load(cp_path,map_location='cpu')
    hist=lambda p:[json.loads(z) for z in p.read_text().splitlines()]
    history=hist(cp_path.parent/'metrics.jsonl')
    baseline=hist(Path(ref['checkpoint']).parent/'metrics.jsonl')
    assert len(history)==len(baseline)==30
    for a,b in zip(history,baseline):
        for k in ('epoch','lr','task_sample_counts','query_count'):assert a[k]==b[k]
    assert cp['config']['source']['sha256']==ref['config']['source']['sha256']
    m=MatchedVector('commit_dmp',30)
    m.load_state_dict(cp['model_state_dict']);m.eval()
    def predict(batch,tail):
        m.reset_state()
        z=m(batch)
        return z[:,-20:].mean(1) if tail else z.mean(1)
    x,y,_=arrays('test')
    save(out,'shd-matched-301-gdsm',y,x,predict,[.0625,.125,.25],.25,48)
    (out/'matched.json').write_text(json.dumps(dict(seed=301,epochs=30,n=len(y),
        train_n=cp['config']['train_samples'],loss='same classification queries as DMP; no auxiliary or gate loss',
        batch=128,matched_schedule=True,dmp_sha256=ref['checkpoint_sha256'],
        source_sha256=cp['config']['source']['sha256'],gdsm_sha256=hashlib.sha256(cp_path.read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('experiments','out','repo'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();main(a.experiments,a.out,a.repo)
