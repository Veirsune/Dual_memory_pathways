"""Plot dataset-specific paired contrasts with a shared percentage-point axis."""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def plot(csv_path, dest):
    rows=list(csv.DictReader(Path(csv_path).open()))
    tasks=[('shd','SHD\nScalar candidate–stable vs DMP'),
           ('gesture','DVS Gesture\nGDSM vs DMP'),
           ('ssc-repair','SSC\nGDSM vs DMP, 200 epochs')]
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),sharey=True)
    for ax,(task,title) in zip(axes,tasks):
        available=[r for r in rows if r['dataset']==task and r['comparison'].startswith('DMP /')]
        assert available
        groups=defaultdict(list)
        for row in available:groups[row['condition']].append(row)
        keys=sorted((k for k in groups if k.startswith('overlay-')),key=lambda s:float(s.split('-')[1]))
        ticks=['0.5×','1×','3×'] if task=='gesture' else ['0.0625','0.125','0.25']
        for key,tag,offset,color in [('delta_pp','All test examples',-.07,'#617180'),('shared_delta_pp','Both correct when clean',.07,'#205d83')]:
            values=[[float(r[key]) for r in groups[c]] for c in keys]
            pos=np.arange(len(keys))+offset
            for xp,v in zip(pos,values):
                if len(v)>1:ax.plot([xp,xp],[min(v),max(v)],color=color,lw=1)
            ax.plot(pos,[np.mean(v) for v in values],color=color,marker='o',ms=4,label=tag)
        ax.axhline(0,color='black',lw=.65)
        ax.set_xticks(range(len(keys)),ticks)
        ax.set_title(title,fontsize=10)
        ax.set_xlabel('Noise / training density' if task=='gesture' else 'Poisson rate per bin')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Architecture difference (pp)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.10,1,1))
    fig.savefig(dest,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':
    import sys
    plot(sys.argv[1],sys.argv[2])
