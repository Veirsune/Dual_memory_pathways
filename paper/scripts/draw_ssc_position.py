"""Draw both panels from the saved spike counts and official-test predictions."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'evidence' / 'ssc-position'
STYLES = [('fsnn', 'FSNN', '#939DA6', 'o'), ('dmp', 'DMP', '#576E84', 's'),
          ('commit', 'GDSM', '#184F7A', '^')]


def draw():
    example = np.load(DATA / 'example.npz')
    speech, noise = example['speech'], example['noise']
    frames, channels = speech.shape
    length = len(noise)
    overlaps = np.array([0, 20, 40, 60, 80])
    percent = overlaps * 100 // length
    observations = np.load(DATA / 'predictions.npz')
    labels = observations['labels']
    plt.rcParams.update({'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.right': False, 'axes.spines.top': False})
    fig = plt.figure(figsize=(7.25, 2.7))
    grid = fig.add_gridspec(3, 2, left=.082, right=.976, bottom=.20, top=.82,
                            wspace=.38, hspace=.24, width_ratios=(1.12, 1.0))
    for row, overlap in enumerate(overlaps[::2]):
        ax = fig.add_subplot(grid[row, 0])
        start = frames - overlap
        rgb = np.ones((frames + length - overlap, channels, 3))
        rgb[start:start + length][noise > 0] = [.65, .77, .85]
        rgb[:frames][speech > 0] = [.09, .23, .35]
        ax.imshow(rgb.transpose(1, 0, 2), origin='lower', extent=(0,len(rgb),0,channels),
                  interpolation='nearest', aspect='auto', rasterized=True)
        ax.axvline(frames, ls=':', lw=.8, c='#414C55')
        ax.set_xlim(0, frames + length)
        ax.set_yticks([])
        ax.set_ylabel(f'{100 * overlap // length}%', rotation=0, labelpad=14, va='center')
        ax.spines['left'].set_visible(False)
        ax.set_xticks([0, 100, 200, 250, 330])
        if row < 2:
            ax.tick_params(axis='x', labelbottom=False, bottom=False)
        else:
            ax.set_xlabel('Time (frames)')
        if row == 0:
            ax.legend(handles=[Patch(color='#173B59',label='Speech'),
                               Patch(color='#A6C4D9',label='Noise')],
                      loc='lower left', bbox_to_anchor=(.0,1.00), borderaxespad=0,
                      ncol=2, frameon=False, handlelength=1)
    ax = fig.add_subplot(grid[:, 1])
    rows = []
    for key, name, color, marker in STYLES:
        points = []
        clean = 100 * np.mean(observations[key + '_clean'] == labels)
        for case in ['clean', *overlaps]:
            correct = np.count_nonzero(observations[f'{key}_{case}'] == labels)
            acc = 100 * correct / len(labels)
            rows.append([name, case, len(labels), correct, f'{acc:.6f}', f'{acc-clean:.6f}'])
            if case != 'clean':
                points.append(acc)
        ax.plot(percent, points, marker=marker, color=color, lw=1.55, ms=4, label=name)
    ax.set(xlim=(-4,104), xticks=percent, ylim=(0,65),
           xlabel='Noise burst overlapping speech (%)', ylabel='Endpoint accuracy (%)')
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='lower left', bbox_to_anchor=(0,1.01), ncol=3, frameon=False,
              handlelength=1.2, columnspacing=.9, borderaxespad=0)
    fig.text(.018,.94,'a',fontweight='bold',fontsize=11)
    fig.text(.515,.94,'b',fontweight='bold',fontsize=11)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(ROOT / 'figures' / ('ssc-position.' + ext), dpi=300)
    plt.close(fig)
    with (DATA / 'accuracy.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model','overlap_frames','n','correct','accuracy_pct','change_from_clean_pp'])
        writer.writerows(rows)


if __name__ == '__main__':
    draw()
