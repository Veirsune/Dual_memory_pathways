import torch
from pathlib import Path
import train
from width_diagnostics import paired_test
torch.set_num_threads(1)
test=train.load_split('test')
models=train.make_models(.8142856955528259)
for f,m in models.items():
    m.load_state_dict(torch.load('original-'+f+'-clean-best.pt',map_location=train.DEVICE))
    m.eval()
    paired_test(m,f,test,train.scores,train.stimulus,train.BATCH,Path('/content')/(f+'-original.npz'),full_only=True)
print('BASELINE COMPLETE',flush=True)
