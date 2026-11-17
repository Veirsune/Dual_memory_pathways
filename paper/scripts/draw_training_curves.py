"""Plot only recorded clean-validation evaluations; never train or infer missing epochs."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

TASKS = ('SSC', 'S-MNIST', 'PS-MNIST')
FAMILIES = ('fsnn', 'dmp', 'commit')
BOUNDARIES = {'SSC': 40, 'S-MNIST': 50, 'PS-MNIST': 50}
COLORS = {'fsnn': '#9AA3AC', 'dmp': '#64788C', 'commit': '#245781'}
MARKERS = {'fsnn': 'o', 'dmp': 's', 'commit': '^'}


def training_points(root):
    manifest = json.loads((root / 'evidence/training-history-manifest.json').read_text())
    expected = {(t, s, f, p) for t in TASKS for s in ([801] if t == 'SSC' else [42, 43, 44])
                for f in FAMILIES for p in ('clean', 'mixed')}
    assert len(manifest) == len(expected) == 42
    assert {(m['task'], m['seed'], m['family'], m['stage']) for m in manifest} == expected
    grouped = {}
    for m in manifest:
        path = root / m['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m['sha256'], path
        history = json.loads(path.read_text())
        boundary = BOUNDARIES[m['task']]
        end = boundary if m['stage'] == 'clean' else 12
        assert [row['epoch'] for row in history] == list(range(1, end + 1))
        interval = 10 if m['task'] == 'SSC' else 5
        valid = [row for row in history if 'validation' in row]
        assert [row['epoch'] for row in valid] == [ep for ep in range(1, end + 1) if ep % interval == 0 or ep == end]
        for row in valid:
            val = row['validation']
            assert val['n'] == (9981 if m['task'] == 'SSC' else 6000)
            assert 0 <= val['mean'] <= 1
            assert abs(val['mean'] * val['n'] - round(val['mean'] * val['n'])) < 1e-7
            epoch = row['epoch'] + (boundary if m['stage'] == 'mixed' else 0)
            key = (m['task'], m['family'], m['stage'], epoch)
            grouped.setdefault(key, {})[m['seed']] = 100 * val['mean']
    points = []
    for (task, family, stage, epoch), by_seed in sorted(grouped.items()):
        assert set(by_seed) == ({801} if task == 'SSC' else {42, 43, 44})
        values = list(by_seed.values())
        points.append(dict(task=task, family=family, stage=stage, epoch=epoch,
                           mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else 0.,
                           seeds=len(values)))
    assert len(points) == 96
    return points


def draw_training_curves(root, colors=None, markers=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = COLORS if colors is None else colors
    markers = MARKERS if markers is None else markers
    points = training_points(root)
    with (root / 'evidence/training-curve-points.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader(); writer.writerows(points)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.75), sharey=True)
        for ax, task in zip(axes, TASKS):
            boundary = BOUNDARIES[task]
            for family in FAMILIES:
                rows = sorted((p for p in points if p['task'] == task and p['family'] == family), key=lambda p:p['epoch'])
                x = np.array([p['epoch'] for p in rows]); y = np.array([p['mean'] for p in rows]); sd = np.array([p['sd'] for p in rows])
                ax.plot(x, y, color=colors[family], marker=markers[family], lw=1.6, ms=3.5,
                        label={'fsnn':'FSNN','dmp':'DMP','commit':'GDSM'}[family])
                if task != 'SSC':
                    ax.fill_between(x, np.clip(y-sd, 0, 100), np.clip(y+sd, 0, 100), color=colors[family], alpha=.13, lw=0)
            ax.axvline(boundary, color='black', ls='--', lw=.8)
            ax.set_title(task, fontsize=10, fontweight='bold', pad=20)
            ax.text(boundary/2, 1.03, 'Clean', ha='center', transform=ax.get_xaxis_transform(), fontsize=8)
            ax.text(boundary+6, 1.03, 'Mixed', ha='center', transform=ax.get_xaxis_transform(), fontsize=8)
            ax.set_xlim(0, boundary+13); ax.set_ylim(0, 103)
            ax.set_xticks([0, 10, 20, 30, 40, 52] if task == 'SSC' else [0, 10, 20, 30, 40, 50, 62])
            ax.set_xlabel('Epoch'); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        axes[0].set_ylabel('Validation accuracy (%)')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.025))
        fig.tight_layout(rect=(0, .09, 1, 1), w_pad=1.2)
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(root / f'figures/training-curves.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    return points


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    points = draw_training_curves(root)
    print(f'Plotted {len(points)} aggregates from 42 archived histories; no new training.')
