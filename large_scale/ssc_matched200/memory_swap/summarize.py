"""Compute paired donor-specific effects from the archived predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def summarize(folder):
    with np.load(folder / "predictions.npz") as rows:
        y = rows["labels"]
        speakers, group = np.unique(rows["speakers"], return_inverse=True)
        own, swap_a, swap_b = rows["predictions"].T
        a, b = rows["donor_labels"].T
        margins = rows["margin"]
        donors = rows["donor"]
        index = rows["index"]
    assert np.all((a != b) & (a != y) & (b != y))
    actual = np.stack((swap_a == a, swap_b == b))
    alternate = np.stack((swap_a == b, swap_b == a))
    base = np.stack((own == a, own == b))
    recipient = np.stack((swap_a == y, swap_b == y)).mean(axis=0)
    primary = actual.mean(axis=0) - alternate.mean(axis=0)
    n = len(y)
    counts = np.bincount(group)
    draw = np.random.default_rng(413).integers(len(counts), size=(10000,len(counts)))
    denom = counts[draw].sum(axis=1)

    def estimate(obs):
        total = np.bincount(group, weights=obs)
        sample = total[draw].sum(axis=1) / denom
        return dict(mean=float(np.mean(obs)), ci95=np.quantile(sample,[.025,.975]).tolist())

    data = dict(n=n, speakers=len(speakers), classes=len(np.unique(y)), swaps=2*n,
                distinct_donors=len(np.unique(donors)), actual_donor_hits=int(actual.sum()),
                alternate_donor_hits=int(alternate.sum()), baseline_donor_hits=int(base.sum()),
                own_recipient_hits=int((own==y).sum()), swapped_recipient_hits=int((recipient*2).sum()),
                new_donor_hits=int((actual & ~base).sum()), lost_donor_hits=int((~actual & base).sum()),
                actual_donor_pct=estimate(100*actual.mean(axis=0)),
                alternate_donor_pct=estimate(100*alternate.mean(axis=0)),
                own_state_donor_pct=estimate(100*base.mean(axis=0)),
                donor_specific_pp=estimate(100*primary),
                donor_change_from_own_pp=estimate(100*(actual.mean(axis=0)-base.mean(axis=0))),
                own_recipient_pct=estimate(100*(own==y)),
                swapped_recipient_pct=estimate(100*recipient),
                recipient_change_pp=estimate(100*(recipient-(own==y))),
                donor_logit_contrast=estimate(.5*(margins[:,0]-margins[:,1]).astype(float)))
    # Illustration: count all eligible three-way top-1 changes and show the first in test order.
    three_way = np.flatnonzero((own == y) & (swap_a == a) & (swap_b == b))
    data['three_way_examples'] = len(three_way)
    if len(three_way):
        j = three_way[0]
        names = json.loads((HERE/'class_names.json').read_text())
        data['first_three_way'] = dict(recipient_index=int(index[j]), recipient=names[y[j]],
            donor_indices=donors[j].tolist(), donor_words=[names[a[j]], names[b[j]]])
    checks = json.loads((folder / 'checks.json').read_text())
    assert n == checks['eligible'] and checks['input_n'] == 20382
    (folder / 'results.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', type=Path, default=HERE)
    summarize(parser.parse_args().folder)
