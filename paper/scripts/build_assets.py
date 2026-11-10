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
# Retention figure: matched endpoint at L=0; seed SD only for MNIST.
colors={'fsnn':'#7b8492','dmp':'#c45a35','commit':'#007e87'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,4,figsize=(11.5,5.0),sharey=True)
for col,task in enumerate(tasks):
 for row,kind in enumerate(['silence','poisson']):
  ax=axes[row,col];ls=[0]+lengths[task]
  for f in families:
   vals=np.array([get(task,f,L=L,kind=kind) for L in ls]);ax.plot(ls,vals[:,0],'-o',c=colors[f],lw=1.8,ms=3.5,label=labels[f])
   if 'MNIST' in task:ax.fill_between(ls,np.clip(vals[:,0]-vals[:,1],0,100),np.clip(vals[:,0]+vals[:,1],0,100),color=colors[f],alpha=.13,lw=0)
  ax.set_ylim(0,101);ax.set_xticks(ls);ax.grid(axis='y',alpha=.18)
  if row==0:ax.set_title(task,fontweight='bold',fontsize=10)
  if row==1:ax.set_xlabel('Appended frames, $L$')
  if col==0:ax.set_ylabel(('Silence' if row==0 else 'Poisson')+'\nEndpoint accuracy (%)')
  if task!='DVS Gesture':ax.axvline(lengths[task][-2],ls=':',c='#aeb3ba',lw=.8);ax.text(.98,.95,'last length unseen',ha='right',va='top',transform=ax.transAxes,fontsize=7,color='#444444')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.015))
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(ROOT/'figures/retention.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/retention.png',dpi=180,bbox_inches='tight');plt.close(fig)
# Architecture schematic with separate fast, candidate, and stable reads.
fig,ax=plt.subplots(figsize=(12,5.2));ax.set_xlim(0,12);ax.set_ylim(0,5.2);ax.axis('off')
def box(x,y,w,h,text,color='#edf4f7',ec='#557283',fs=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.08',fc=color,ec=ec,lw=1.2));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
def arrow(a,b,color='#4b5a66',style='-',rad=0):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.3,color=color,linestyle=style,connectionstyle=f'arc3,rad={rad}'))
box(.15,2.1,1.1,.8,'Layer input\n$z_t$');box(2.1,2.1,1.35,.8,'Vector drive\n$u_t\\in\\mathbb{R}^d$')
box(4.25,2.0,1.65,1.0,'Candidate $p_t$\n$\\bar A p_{t-1}+D u_t$',fs=10.5)
box(6.5,2.0,1.6,1.0,'Stable $m_t$\ngated update',color='#e4f3ed',ec='#3d8571',fs=10.5)
box(4.15,3.85,3.95,.8,'Causal gate $g_t$: current features + past EMA',color='#f7eedb',ec='#af8a35',fs=10)
box(8.8,2.1,1.1,.8,'Sum\n$I_t$');box(10.55,2.1,1.1,.8,'LIF\nspikes')
arrow((1.32,2.5),(2.02,2.5));arrow((3.53,2.5),(4.17,2.5));arrow((5.98,2.5),(6.42,2.5));ax.text(6.2,2.75,'$p_{t-1}$',ha='center',fontsize=10)
arrow((8.18,2.5),(8.72,2.5),color='#007e87');ax.text(8.45,2.75,'$W_m$',ha='center')
arrow((9.98,2.5),(10.47,2.5))
# Routed paths; arrows only at final segment.
ax.plot([.7,.7,9.35],[2.98,4.95,4.95],c='#4b5a66',lw=1.3);arrow((9.35,4.95),(9.35,2.98));ax.text(2.4,5.04,'Fast path $W_x z_t$',ha='center',fontsize=10)
ax.plot([5.07,5.07,9.35],[1.92,1.1,1.1],c='#007e87',lw=1.3);arrow((9.35,1.1),(9.35,2.02),color='#007e87');ax.text(7.15,1.26,'Candidate read $W_p p_t$',ha='center',fontsize=10)
arrow((7.3,3.77),(7.3,3.08),style='--',color='#af8a35');ax.text(7.7,3.42,'write',fontsize=9,color='#806421')
ax.text(.2,3.7,'Raw input\nstatistics',fontsize=10,ha='left');arrow((1.3,4.1),(4.07,4.22),style='--',color='#af8a35')
ax.text(5.95,.5,r'$m_t=(1-g_t)\odot m_{t-1}+g_t\odot p_{t-1}$',ha='center',fontsize=13)
ax.text(5.95,.08,'When the write gate is closed, stable memory is held; candidate and fast paths keep running.',ha='center',fontsize=10,color='#44515b')
fig.savefig(ROOT/'figures/architecture.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/architecture.png',dpi=180,bbox_inches='tight');plt.close(fig)
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
for j,(key,label,color) in enumerate([('dmp_macs','DMP','#c45a35'),('commit_dense_macs','GDSM dense','#8ebfc0'),('commit_cached_macs','GDSM cached','#007e87')]):
 vals=[x[key]/1e6 for x in full];bars=ax.bar(xs+(j-1)*width,vals,width,color=color,label=label)
 for b,v in zip(bars,vals):ax.text(b.get_x()+width/2,v+.8,f'{v:.2f}',ha='center',fontsize=8)
ax.set_xticks(xs,['Clean (100 frames)','Tail silence L80 (180 frames)'])
ax.set_ylabel('Full-network MACs / sample (M)',fontsize=9);ax.set_ylim(0,84)
ax.text(0,79.2,f"GDSM: +{full[0]['cached_overhead_percent']:.2f}%",ha='center',fontsize=8)
ax.text(1,79.2,f"Cache saves {full[1]['cache_saving_percent']:.2f}%",ha='center',fontsize=8)
ax.legend(fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3)
ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(ROOT/'figures/full-network-macs.pdf',bbox_inches='tight');fig.savefig(ROOT/'figures/full-network-macs.png',dpi=180,bbox_inches='tight');plt.close(fig)

# Hardware schedule measured from CSV, rather than hard-coded cycles.
rows=read('hardware-cycles.csv');fig,ax=plt.subplots(figsize=(5.3,2.8));xs=np.arange(2);width=.24
for j,(variant,condition,label,color) in enumerate([('dmp','signal','Matched scalar DMP','#c45a35'),('commit_dense','signal','GDSM dense','#007e87'),('commit_cached','silence','GDSM, closed + cache','#71bdaf')]):
 vals=[]
 for d in [10,30]:
  rr=[r for r in rows if r['scenario']==f'capacity{d}' and r['variant']==variant and (r['condition']==condition)]
  if variant=='commit_cached':rr=[r for r in rr if int(r['open_coordinates'])==0]
  assert rr,(d,variant,condition)
  v=sorted(set(int(r['cycles']) for r in rr));assert len(v)==1,v;vals.append(v[0])
 bars=ax.bar(xs+(j-1)*width,vals,width,color=color,label=label)
 for b,v in zip(bars,vals):ax.text(b.get_x()+width/2,v+3,str(v),ha='center',fontsize=8)
ax.set_xticks(xs,['$d=10$','$d=30$']);ax.set_ylabel('Processing cycles / frame');ax.set_ylim(0,260);ax.legend(fontsize=7,frameon=False,loc='upper left');fig.tight_layout();fig.savefig(ROOT/'figures/hardware.pdf',bbox_inches='tight');plt.close(fig)
# Deduplicated, machine-readable claims used by tables and figures.
unique={json.dumps({k:v for k,v in c.items() if k not in ['mean','sd','values']},sort_keys=True):c for c in claims}
(ROOT/'evidence/generated-claims.json').write_text(json.dumps(list(unique.values()),indent=2))
print('Generated 4 figures, 3 tables,',len(unique),'numerical aggregates,',len(manifest),'source copies.')
