"""Observe memory entries on fixed validation items and score paired interventions."""
import numpy as np
import torch

@torch.no_grad()
def drive_rates(model, family, inputs, scores):
    results = []
    def record(_module, _args, output):
        results.append(float((output > 0).float().mean()))
    hooks = [c.f_u.register_forward_hook(record) for c in (model.LMU1, model.LMU2)]
    try:
        model.eval()
        scores(family, model, inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return results

@torch.no_grad()
def paired_test(model, family, data, scores, stimulus, batch_size, outfile, full_only=False):
    x, y = data
    saved = {'labels': y.cpu().numpy(), 'ids': np.arange(len(y))}
    cells = (model.LMU1, model.LMU2)
    weights = [c.weight.weight for c in cells]
    stable = [c.read_m.weight for c in cells] if family == 'gdsm' else []
    modes=['full'] if full_only else (['full','all-off','stable-off'] if stable else ['full','all-off'])
    for mode in modes:
        selected = {'full': [], 'all-off': weights + stable, 'stable-off': stable}[mode]
        originals = [w.clone() for w in selected]
        try:
            for w in selected:
                w.zero_()
            for name,kind,length,rate in [('clean','clean',0,0),('overlay-0.0625','speech',0,.0625),('overlay-0.125','speech',0,.125),('overlay-0.25','speech',0,.25),('tail80','noise',80,.25)]:
                labels = []
                for i in range(0,len(y),batch_size):
                    v = stimulus(x[i:i+batch_size].float(),kind,250,length,rate,120000+i)
                    s = scores(family,model,v)
                    read = s[:,-20:].mean(1) if name=='tail80' else s.mean(1)
                    assert torch.isfinite(read).all()
                    labels.append(read.argmax(1).cpu().numpy())
                saved[mode+'_'+name] = np.concatenate(labels).astype(np.uint8)
        finally:
            for w, original in zip(selected, originals):
                w.copy_(original)
    np.savez_compressed(outfile,**saved)
