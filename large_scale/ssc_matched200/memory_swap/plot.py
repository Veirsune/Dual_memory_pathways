"""Show the paired donor-specific effect across all recipient word classes."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def plot(folder):
    names = json.loads((HERE/'class_names.json').read_text())
    with np.load(folder/'predictions.npz') as d:
        y = d['labels']
        a, b = d['donor_labels'].T
        pa, pb = d['predictions'][:,1:].T
    contrast = 50*((pa==a).astype(int) - (pa==b) + (pb==b) - (pb==a))
    counts = np.bincount(y)
    by_word = np.bincount(y, weights=contrast)/counts
    order = sorted(range(len(names)), key=lambda i: names[i])
    with (folder/'by_word.csv').open('w',newline='') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['recipient_word','utterances','donor_specific_pp'])
        writer.writerows((names[i],counts[i],by_word[i]) for i in order)
    result = json.loads((folder/'results.json').read_text())['donor_specific_pp']
    assert np.isclose(contrast.mean(), result['mean'])
    lo, hi = result['ci95']
    fig, axes = plt.subplots(1, 2, figsize=(7.4,5.1), sharex=True, layout='constrained')
    for ax, group in zip(axes,(order[:18],order[18:])):
        vals = by_word[group]
        ax.axvline(0,color='#576069',lw=.8)
        ax.axvspan(lo, hi,color='#d9e4ea')
        ax.axvline(result['mean'],ls='--',color='#31627c',lw=.9)
        ax.scatter(vals,np.arange(len(group)),color='#254f69',s=19)
        ax.set_yticks(np.arange(len(group)),[names[i] for i in group],fontsize=8)
        ax.set_ylim(17.7,-.7)
        ax.grid(axis='x',alpha=.15)
        ax.tick_params(length=0)
        ax.spines[['top','right','left']].set_visible(False)
    fig.suptitle('Does the prediction follow the transplanted word?')
    fig.supxlabel('Actual donor minus alternate donor (percentage points)',fontsize=10)
    for ext in ('png','pdf'):
        fig.savefig(folder/f'donor-effect.{ext}',dpi=170)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path,nargs='?',default=HERE)
    plot(p.parse_args().folder)
