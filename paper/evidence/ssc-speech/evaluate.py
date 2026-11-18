"""Run with the archived SSC runner on PYTHONPATH; inputs and final weights in one directory."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from large_scale.models import initial_models


def evaluate(root, dest, protocol, device, batch_size):
    x = np.load(root / 'test-x.npy', mmap_mode='r')
    labels_path = root / 'test-y.npy'
    labels = np.load(labels_path)
    assert x.shape == (protocol['samples'], protocol['frames'], protocol['channels'])
    assert hashlib.sha256(x).hexdigest() == protocol['data_sha256']
    assert hashlib.sha256(labels_path.read_bytes()).hexdigest() == protocol['labels_file_sha256']
    models = {}
    for family, digest in protocol['checkpoints'].items():
        path = root / (family + '.pt')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = initial_models(ckpt['scale'])[family]
        model.load_state_dict(ckpt['model'])
        models[family] = model.to(device).eval()

    rates = protocol['poisson_rates']
    increments = np.diff(rates)
    generators = [np.random.default_rng(seed) for seed in protocol['increment_seeds']]
    assert len(generators) == len(increments) and rates[0] == 0 and all(increments > 0)
    predictions = {}
    example = {'speech': x[0], 'label': labels[0]}
    dest.mkdir(exist_ok=True, parents=True)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            speech = torch.tensor(x[start:start + batch_size], device=device, dtype=torch.float32)
            counts = np.zeros(tuple(speech.shape), dtype=np.uint16)
            for level, rate in enumerate(rates):
                if level:
                    extra = generators[level-1].poisson(increments[level-1], size=counts.shape)
                    counts += extra.astype(counts.dtype)
                if start == 0:
                    example[f'noise_{level}'] = counts[0].copy()
                stim = speech + torch.as_tensor(counts.astype(np.float32), device=device)
                for family, model in models.items():
                    scores = model(stim).mean(1)
                    assert torch.isfinite(scores).all()
                    key = f'{family}_{level}'
                    if key not in predictions:
                        predictions[key] = np.empty(len(x), dtype=np.uint8)
                    predictions[key][start:start + len(speech)] = scores.argmax(1).cpu().numpy()
            if start % (10 * batch_size) == 0:
                print(f'{start + len(speech)}/{len(x)}', flush=True)
    np.savez_compressed(dest / 'example.npz', **example)
    np.savez_compressed(dest / 'predictions.npz', labels=labels, **predictions)
    print(json.dumps({k: int((v == labels).sum()) for k,v in predictions.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('dest', type=Path)
    p.add_argument('--protocol', type=Path, default=Path(__file__).with_name('protocol.json'))
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    evaluate(args.root, args.dest, json.loads(args.protocol.read_text()), args.device, args.batch)
