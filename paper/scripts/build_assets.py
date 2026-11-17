"""Regenerate manuscript tables/figures from archived results; no model execution."""
from pathlib import Path
import csv,json,hashlib,shutil,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT.parent
for name in ['figures','tables','evidence']:(ROOT/name).mkdir(exist_ok=True)
sources={
 'ssc.csv':'block5-ssc/metrics.csv','mnist.csv':'block6-mnist/metrics.csv','gesture.csv':'final-study/block3-summary.csv',
 'shd-capacity.csv':'shd-vector-memory/metrics.csv','shd-supervision.csv':'shd-supervision-ablation/seed-metrics.csv',
 'mnist-statistics.json':'block6-mnist/paired-statistics.json','ssc-statistics.json':'block5-ssc/paired-statistics.json',
 'hardware-cycles.csv':'block7-hardware/cycles.csv','hardware-operations.csv':'final-study/hardware-summary.csv',
 'hardware-replay.json':'block7-hardware/results/trained-replay.json',
 'shd-capacity-report.md':'shd-vector-memory/REPORT.md','shd-supervision-report.md':'shd-supervision-ablation/REPORT.md',
 'shd-readout-report.md':'shd-final-readout/REPORT.md','shd-clean-report.md':'shd-accuracy/REPORT.md',
 'ssc-report.md':'block5-ssc/REPORT.md','mnist-report.md':'block6-mnist/REPORT.md',
 'gesture-report.md':'final-study/BLOCK3_REPORT.md','operations-report.md':'final-study/BLOCK4_REPORT.md',
 'rtl-report.md':'block7-hardware/REPORT.md','ssc-protocol.json':'block5-ssc/LOCKED_PROTOCOL.json',
 'mnist-protocol.json':'block6-mnist/LOCKED_PROTOCOL.json','gesture-protocol.json':'final-study/LOCKED_PROTOCOL.json',
 'shd-protocol.json':'shd-supervision-ablation/protocol.json'}
manifest=[]
for dest,src in sources.items():
 p=EXP/src;target=ROOT/'evidence'/dest
 if p.exists():shutil.copy2(p,target)
 assert target.exists(),str(target)
 manifest.append({'file':'evidence/'+dest,'source':src,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
(ROOT/'evidence/source-manifest.json').write_text(json.dumps(manifest,indent=2))
def read(n):return list(csv.DictReader((ROOT/'evidence'/n).open()))
data={k:read(k+'.csv') for k in ['ssc','mnist','gesture']}
families=['fsnn','dmp','commit']; labels={'fsnn':'FSNN','dmp':'DMP','commit':'GDSM'}
tasks=['SSC','DVS Gesture','S-MNIST','PS-MNIST']; steps={'SSC':250,'DVS Gesture':100,'S-MNIST':784,'PS-MNIST':784}
lengths={'SSC':[40,80,120],'DVS Gesture':[20,40,80],'S-MNIST':[40,80,160],'PS-MNIST':[40,80,160]}
claims=[]
def get(task,family,phase='mixed',metric='last20',L=0,kind='silence',middle=False,mult=1):
 T=steps[task];pos=T//2 if middle else T
 if task=='DVS Gesture':
  rs=[r for r in data['gesture'] if r['family']==family and r['stage']==('finetuned' if phase=='mixed' else 'zero_shot') and r['metric']==metric and int(r['length'])==L and int(r['position'])==(T if L==0 else pos) and r['kind']==('silence' if L==0 else kind) and float(r['multiplier'])==(0 if L==0 or kind=='silence' else mult)]
 else:
  rs=[r for r in data['ssc' if task=='SSC' else 'mnist'] if r['family']==family and r['phase']==phase and r['metric']==metric and int(r['L'])==L and int(r['pos'])==(T if L==0 else pos) and r['kind']==('silence' if L==0 else kind) and (task=='SSC' or r['task']==('smnist' if task=='S-MNIST' else 'psmnist'))]
 assert len(rs)==(3 if 'MNIST' in task else 1),(task,family,phase,metric,L,kind,middle,rs)
 v=np.array([float(r['accuracy'])*100 for r in rs]);a=float(v.mean());s=float(v.std(ddof=1)) if len(v)>1 else 0
 claims.append(dict(task=task,family=family,phase=phase,metric=metric,L=L,kind=kind,middle=middle,mult=mult,mean=a,sd=s,n_training_seeds=len(v),values=v.tolist()))
 return a,s

def fmt(v,sd=False):return f'{v[0]:.2f}'+(r' $\pm$ '+f'{v[1]:.2f}' if sd else '')
# Complete compact main tables.
s=[]
for task in tasks:
 for ph in ['clean','mixed']:
  if task=='DVS Gesture' and ph=='clean':continue
  s.append(task+' & '+('Clean' if ph=='clean' else 'Mixed')+' & '+' & '.join(fmt(get(task,f,ph,'mean'), 'MNIST' in task) for f in families)+r' \\')
(ROOT/'tables/clean.tex').write_text('\n'.join(s)+'\n')
s=[]
for task in tasks:
 L=lengths[task][-1]
 for f in families:
  s.append(task+' & '+labels[f]+' & '+str(L)+' & '+' & '.join(fmt(get(task,f,L=l,kind=k),'MNIST' in task) for l,k in [(0,'silence'),(L,'silence'),(L,'poisson')])+r' \\')
 s.append(r'\addlinespace')
(ROOT/'tables/retention.tex').write_text('\n'.join(s)+'\n')
s=[]
for task in tasks:
 for k in ['silence','poisson']:
  for L in lengths[task]:
   s.append(task+' & '+('Silent' if k=='silence' else 'Poisson')+' & '+str(L)+' & '+' & '.join(fmt(get(task,f,metric='mean',L=L,kind=k,middle=True),'MNIST' in task) for f in families)+r' \\')
 s.append(r'\addlinespace')
(ROOT/'tables/middle.tex').write_text('\n'.join(s)+'\n')
longest=[row for row in s if '&' in row and int(row.split('&')[2])==lengths[row.split('&')[0].strip()][-1]]
(ROOT/'tables/middle-longest.tex').write_text('\n'.join(longest)+'\n')
# Retention figure: matched endpoint at L=0; seed SD only for MNIST.
plt.style.use('default')
# Blue-gray palette: neutral baseline, slate DMP, and deep-blue GDSM.
colors={'fsnn':'#9AA3AC','dmp':'#64788C','commit':'#245781'}
markers={'fsnn':'o','dmp':'s','commit':'^'}
# Keep DMP and GDSM colors consistent across curves and bars.
bar_colors={'dmp':colors['dmp'],'dense':'#A5B9CC','cached':colors['commit']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,4,figsize=(11.5,5.0),sharey=True)
for col,task in enumerate(tasks):
 for row,kind in enumerate(['silence','poisson']):
  ax=axes[row,col];ls=[0]+lengths[task]
  for f in families:
   vals=np.array([get(task,f,L=L,kind=kind) for L in ls]);ax.plot(ls,vals[:,0],linestyle='-',marker=markers[f],c=colors[f],lw=1.8,ms=3.5,label=labels[f])
   if 'MNIST' in task:ax.fill_between(ls,np.clip(vals[:,0]-vals[:,1],0,100),np.clip(vals[:,0]+vals[:,1],0,100),color=colors[f],alpha=.13,lw=0)
  ax.set_ylim(0,101);ax.set_xticks(ls);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  if row==0:ax.set_title(task,fontweight='bold',fontsize=10)
  if row==1:ax.set_xlabel('Appended frames, $L$')
  if col==0:ax.set_ylabel(('Silence' if row==0 else 'Poisson')+'\nEndpoint accuracy (%)')
  if task!='DVS Gesture':ax.axvline(lengths[task][-2],ls=':',c='0.6',lw=.8);ax.text(.98,.95,'last length unseen',ha='right',va='top',transform=ax.transAxes,fontsize=7,color='black')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.015))
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(ROOT/'figures/retention.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/retention.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Architecture schematic: vector circuits and shared-state mechanism.
from draw_architecture import draw_architecture
draw_architecture(ROOT)
# Full-network operation accounting, in MACs/sample (not RTL cycles).
ops={(x['family'],x['condition']):x for x in read('hardware-operations.csv')}
conditions=['clean','p100_silence0_L80']; full=[]
for condition in conditions:
 a=ops['dmp',condition];b=ops['commit',condition]
 dmp=float(a['whole_mapped_mac']);cached=float(b['whole_mapped_mac'])
 saved=float(b['stable_read_mac_saved']);dense=cached+saved
 assert abs(dense-(float(b['conv_effective_mac'])+float(b['mac_dense_core'])))<1e-6
 full.append(dict(condition=condition,dmp_macs=dmp,commit_dense_macs=dense,commit_cached_macs=cached,
                  cached_overhead_percent=100*(cached/dmp-1),cache_saving_percent=100*saved/dense))
(ROOT/'evidence/full-network-mac-claims.json').write_text(json.dumps(full,indent=2))
fig,ax=plt.subplots(figsize=(5.3,3.1));xs=np.arange(2);width=.23
for j,(key,label,color) in enumerate([('dmp_macs','DMP',bar_colors['dmp']),('commit_dense_macs','GDSM dense',bar_colors['dense']),('commit_cached_macs','GDSM cached',bar_colors['cached'])]):
 vals=[x[key]/1e6 for x in full];bars=ax.bar(xs+(j-1)*width,vals,width,color=color,label=label)
 for b,v in zip(bars,vals):ax.text(b.get_x()+width/2,v+.8,f'{v:.2f}',ha='center',fontsize=8)
ax.set_xticks(xs,['Clean (100 frames)','Tail silence L80 (180 frames)'])
ax.set_ylabel('Full-network MACs / sample (M)',fontsize=9);ax.set_ylim(0,84)
ax.text(0,79.2,f"GDSM: +{full[0]['cached_overhead_percent']:.2f}%",ha='center',fontsize=8)
ax.text(1,79.2,f"Cache saves {full[1]['cache_saving_percent']:.2f}%",ha='center',fontsize=8)
ax.legend(fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3)
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(ROOT/'figures/full-network-macs.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/full-network-macs.png',dpi=180,bbox_inches='tight');plt.close(fig)

# Hardware schedule measured from CSV, rather than hard-coded cycles.
rows=read('hardware-cycles.csv');fig,ax=plt.subplots(figsize=(5.3,2.8));xs=np.arange(2);width=.24
for j,(variant,condition,label,color) in enumerate([('dmp','signal','Matched scalar DMP',bar_colors['dmp']),('commit_dense','signal','GDSM dense',bar_colors['dense']),('commit_cached','silence','GDSM, closed + cache',bar_colors['cached'])]):
 vals=[]
 for d in [10,30]:
  rr=[r for r in rows if r['scenario']==f'capacity{d}' and r['variant']==variant and (r['condition']==condition)]
  if variant=='commit_cached':rr=[r for r in rr if int(r['open_coordinates'])==0]
  assert rr,(d,variant,condition)
  v=sorted(set(int(r['cycles']) for r in rr));assert len(v)==1,v;vals.append(v[0])
 bars=ax.bar(xs+(j-1)*width,vals,width,color=color,label=label)
 for b,v in zip(bars,vals):ax.text(b.get_x()+width/2,v+3,str(v),ha='center',fontsize=8)
ax.set_xticks(xs,['$d=10$','$d=30$']);ax.set_ylabel('Processing cycles / frame');ax.set_ylim(0,260);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=7,frameon=False,loc='upper left');fig.tight_layout();fig.savefig(ROOT/'figures/hardware.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/hardware.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Deduplicated, machine-readable claims used by tables and figures.
unique={json.dumps({k:v for k,v in c.items() if k not in ['mean','sd','values']},sort_keys=True):c for c in claims}
(ROOT/'evidence/generated-claims.json').write_text(json.dumps(list(unique.values()),indent=2))
from draw_training_curves import draw_training_curves
draw_training_curves(ROOT, colors=colors, markers=markers)
print('Generated 5 figure assets, 4 tables,',len(unique),'numerical aggregates,',len(manifest),'source copies.')
