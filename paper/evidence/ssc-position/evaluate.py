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

    rng = np.random.Generator(np.random.PCG64(protocol['noise_seed']))
    length = protocol['noise_frames']
    frames = x.shape[1]
    predictions = {}
    dest.mkdir(exist_ok=True, parents=True)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            speech = torch.tensor(x[start:start + batch_size], device=device, dtype=torch.float32)
            counts = rng.poisson(protocol['poisson_rate'], (len(speech), length, x.shape[2])).astype(np.uint8)
            noise = torch.as_tensor(counts, device=device)
            if start == 0:
                np.savez_compressed(dest / 'example.npz', speech=x[0], noise=counts[0], label=labels[0])
            for overlap in [None, *protocol['overlap_frames']]:
                if overlap is None:
                    stim = speech
                    case = 'clean'
                else:
                    onset = frames - overlap
                    stim = torch.nn.functional.pad(speech, (0, 0, 0, length - overlap))
                    stim[:, onset:onset + length] += noise
                    case = str(overlap)
                for family, model in models.items():
                    scores = model(stim)[:, -20:].mean(1)
                    assert torch.isfinite(scores).all()
                    key = family + '_' + case
                    if key not in predictions:
                        predictions[key] = np.empty(len(x), dtype=np.uint8)
                    predictions[key][start:start + len(speech)] = scores.argmax(1).cpu().numpy()
            if start % (10 * batch_size) == 0:
                print(f'{start + len(speech)}/{len(x)}', flush=True)
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
