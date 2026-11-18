"""Plot paired whole-utterance SSC predictions and additive input spike rasters."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'evidence' / 'ssc-speech'
STYLES = [('fsnn', 'FSNN', '#939DA6', 'o'), ('dmp', 'DMP', '#576E84', 's'),
          ('commit', 'GDSM', '#184F7A', '^')]


def draw():
    rates = json.loads((DATA / 'protocol.json').read_text())['poisson_rates']
    example = np.load(DATA / 'example.npz')
    predictions = np.load(DATA / 'predictions.npz')
    labels = predictions['labels']
    speech = example['speech']
    frames, channels = speech.shape
    plt.rcParams.update({'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.right': False, 'axes.spines.top': False})
    fig = plt.figure(figsize=(7.25, 2.85))
    grid = fig.add_gridspec(2, 2, left=.088, right=.975, bottom=.20, top=.82,
                            width_ratios=(1.03, 1), wspace=.34, hspace=.32)
    for row, level in enumerate([0, len(rates)-1]):
        ax = fig.add_subplot(grid[row, 0])
        rgb = np.ones((frames, channels, 3))
        rgb[example[f'noise_{level}'] > 0] = [.64,.76,.85]
        rgb[speech > 0] = [.09,.23,.35]
        ax.imshow(rgb.transpose(1,0,2), origin='lower', extent=(0,frames,0,channels),
                  aspect='auto', interpolation='nearest')
        ax.set_yticks([0,140])
        ax.set_title('Clean' if level == 0 else 'Background throughout speech', loc='left', fontsize=8, pad=2)
        if row == 0:
            ax.tick_params(bottom=False,labelbottom=False)
        else:
            ax.set_xlabel('Time (frames)')
            ax.set_xticks([0,50,100,150,200,250])
    fig.text(.013, .50, 'Channel', rotation=90, va='center')
    fig.text(.034,.94,'a',fontweight='bold',fontsize=11)
    fig.text(.515,.94,'b',fontweight='bold',fontsize=11)
    fig.legend(handles=[Patch(color='#173B59',label='Speech'),Patch(color='#A6C2D9',label='Noise')],
               bbox_to_anchor=(.47,.98),loc='upper right',ncol=2,frameon=False,handlelength=1)
    ax = fig.add_subplot(grid[:,1])
    rows = []
    for key,name,color,marker in STYLES:
        values=[]
        for level,rate in enumerate(rates):
            correct=int((predictions[f'{key}_{level}']==labels).sum())
            values.append(100*correct/len(labels))
            rows.append([name,rate,len(labels),correct,f'{values[-1]:.6f}',f'{values[-1]-values[0]:.6f}'])
        ax.plot(rates,values,color=color,marker=marker,ms=4,lw=1.6,label=name)
    ax.set(xticks=rates,xticklabels=[str(r) for r in rates],ylim=(0,65),
           xlabel='Noise rate (events / channel / frame)',ylabel='Word accuracy (%)')
    ax.grid(axis='y',alpha=.18)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,
              handlelength=1.2,columnspacing=.8,borderaxespad=0)
    for ext in ['pdf','svg','png']:
        out = ROOT/'figures'/f'ssc-speech.{ext}'
        fig.savefig(out,dpi=300)
        if ext == 'svg':
            out.write_text('\n'.join(line.rstrip() for line in out.read_text().splitlines())+'\n')
    plt.close(fig)
    with (DATA/'accuracy.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n')
        w.writerow(['model','poisson_rate','n','correct','accuracy_pct','change_from_clean_pp'])
        w.writerows(rows)


if __name__=='__main__':
    draw()
