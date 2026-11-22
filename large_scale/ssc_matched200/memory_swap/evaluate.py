"""Exchange only stable memory across words at the SSC midpoint."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from commit_memory import base_model
from large_scale.ssc_matched200.midpoint.evaluate import MID, PROTOCOL, load_model

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "midpoint"


def pick_donors(x, y, speaker):
    features = []
    for i in range(0, len(y), 256):
        frames = x[i:i+256, MID:].reshape(-1, 5, 25, 14, 10)
        features.append(np.log1p(frames.sum(axis=(2, 4))).reshape(len(frames), -1))
    features = np.concatenate(features)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    donors = np.full((len(y), 2), -1, dtype=int)
    for who in np.unique(speaker):
        group = np.flatnonzero(speaker == who)
        if len(np.unique(y[group])) < 3:
            continue
        sim = features[group] @ features[group].T
        for row, idx in enumerate(group):
            order = group[np.argsort(-sim[row], kind="stable")]
            other = order[y[order] != y[idx]]
            first = other[0]
            donors[idx] = first, other[y[other] != y[first]][0]
    return donors


def clone_dict(state):
    return {k: v.clone() for k, v in state.items()}


def snapshot(model):
    return (clone_dict(model.permit.stream),
            [(clone_dict(cell.stream), cell.f_x.v.clone()) for cell in model.cells()])


def restore(model, saved, injected=None):
    permit, cells = saved
    model.permit.stream = clone_dict(permit)
    for layer, (cell, (state, voltage)) in enumerate(zip(model.cells(), cells)):
        cell.stream = clone_dict(state)
        cell.f_x.v = voltage.clone()
        if injected is not None:
            cell.stream["m"] = injected[:, layer].clone()


def independent_swap(model, x, donor_state):
    # Unroll the same recurrence in parallel. At the boundary replace a*m_prev
    # with a*m_donor, retaining the recipient's b = gate * previous_candidate.
    original_scan = base_model.affine_scan
    layers = []
    def substitute(a, b):
        if a.shape[-1] == model.d:
            idx = len(layers)
            layers.append(idx)
            a, b = a.clone(), b.clone()
            b[:, MID] += a[:, MID] * donor_state[:, idx]
            a[:, MID] = 0
        return original_scan(a, b)
    model.reset_state()
    with patch.object(base_model, "affine_scan", substitute):
        scores = model(x)[:, MID:].mean(1)
    assert len(layers) == 2
    return scores


@torch.inference_mode()
def evaluate(data, output, batch):
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    x = np.load(data / "test-x.npy", mmap_mode="r")
    y = np.load(data / "test-y.npy")
    assert hashlib.sha256(x).hexdigest() == PROTOCOL["input_sha256"]
    with np.load(PRIOR / "predictions.npz") as prior:
        speakers, archived = prior["speakers"], prior["clean_keep"]
        assert np.array_equal(y, prior["labels"])
    donor = pick_donors(x, y, speakers)
    valid = donor[:, 0] >= 0
    assert valid.any()
    pairs = donor[valid]
    assert (speakers[pairs] == speakers[valid, None]).all()
    assert (y[pairs] != y[valid, None]).all() and (y[pairs[:,0]] != y[pairs[:,1]]).all()
    model = load_model(PRIOR / "gdsm-mixed-best.pt")
    saved, pool = [], []
    for i in range(0, len(y), batch):
        model.reset_state()
        model.stream_chunk(torch.tensor(x[i:i+batch, :MID], dtype=torch.float32))
        saved.append(snapshot(model))
        pool.append(torch.stack([cell.stream["m"] for cell in model.cells()], dim=1))
    pool = torch.cat(pool)
    print("Prefix states cached:", len(y), "Eligible recipients:", int(valid.sum()), flush=True)
    preds = np.zeros((len(y), 3), dtype=np.uint8)
    margins = np.zeros((len(y), 2), dtype=np.float32)
    for k, state in enumerate(saved):
        i = k * batch
        idx = np.arange(i, min(i+batch, len(y)))
        # Ineligible rows take their own state as a placeholder and are excluded from every comparison.
        partners = np.where(valid[idx, None], donor[idx], idx[:, None])
        frames = torch.tensor(x[idx, MID:], dtype=torch.float32)
        for arm in range(3):
            injected = pool[partners[:, arm-1]] if arm else None
            restore(model, state, injected)
            scores = model.stream_chunk(frames).mean(1)
            assert torch.isfinite(scores).all()
            preds[idx, arm] = scores.argmax(1).numpy()
            if arm:
                margins[idx, arm-1] = (scores[np.arange(len(idx)), y[partners[:,0]]] -
                                       scores[np.arange(len(idx)), y[partners[:,1]]]).numpy()
            if k == 0 and arm == 1:
                take = np.flatnonzero(valid[idx])[:16]
                probe = independent_swap(model, torch.tensor(x[idx[take]], dtype=torch.float32), injected[take])
                assert len(take) == 16 and torch.equal(scores[take].argmax(1), probe.argmax(1))
                probe_max = float((scores[take]-probe).abs().max())
        if k % 20 == 0:
            print("Scored", idx[-1]+1, flush=True)
    assert np.array_equal(archived, preds[:,0]), int((archived != preds[:,0]).sum())
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", index=np.flatnonzero(valid).astype(np.uint16),
                        labels=y[valid].astype(np.uint8), speakers=speakers[valid],
                        donor=donor[valid].astype(np.uint16), donor_labels=y[pairs].astype(np.uint8),
                        predictions=preds[valid], margin=margins[valid])
    return dict(input_n=len(y), eligible=int(valid.sum()), speakers_total=len(np.unique(speakers)),
                own_predictions_match_archive=len(y), independent_probe_n=16, probe_max_logit_difference=probe_max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    metrics = evaluate(args.data, args.output, args.batch)
    (args.output / "checks.json").write_text(json.dumps(metrics, indent=2)+"\n")
