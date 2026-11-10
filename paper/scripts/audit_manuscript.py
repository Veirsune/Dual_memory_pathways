"""Check manuscript quantities against copied results; never train or alter models."""
import csv,json,re,hashlib,statistics,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
checks=[]
def check(name,ok,details=None):
 checks.append({'check':name,'passed':bool(ok),'details':details});assert ok,(name,details)
def csvread(name):return list(csv.DictReader((R/'evidence'/name).open()))
# Source files must match archived hashes.
for row in json.loads((R/'evidence/source-manifest.json').read_text()):
 check('source checksum: '+row['file'],hashlib.sha256((R/row['file']).read_bytes()).hexdigest()==row['sha256'])
# Explicitly check all values in manually typeset SHD ablation tables.
a=csvread('shd-supervision.csv')
expected={
(10,'task_only'):[85.62,2.06,31.38,3.19,41.64,9.83,73.28,2.51],
(10,'zero_rule'):[86.31,1.11,58.42,4.47,46.29,8.23,72.97,1.60],
(10,'joint_supervised'):[86.10,1.85,60.38,4.03,53.47,4.17,69.05,1.82],
(30,'task_only'):[87.38,.98,29.52,6.19,56.96,12.77,75.89,1.27],
(30,'zero_rule'):[87.99,.87,67.82,5.48,60.16,8.76,75.63,1.59],
(30,'joint_supervised'):[87.99,.19,66.84,7.17,60.81,13.50,73.29,.27]}
for (d,mode),want in expected.items():
 rs=[r for r in a if r['split']=='official_test' and int(r['d'])==d and r['mode']==mode];got=[]
 for metric in ['clean','silence','noise','switch']:
  v=[float(r[metric])*100 for r in rs];got.extend([round(statistics.mean(v),2),round(statistics.stdev(v),2)])
 check(f'SHD supervision d{d} {mode}',got==want,{'actual':got,'expected':want})
b=csvread('shd-capacity.csv')
for d,drive,aux,ret in [(10,'scalar',48.19,40.15),(10,'wide_scalar',56.67,47.46),(10,'vector',68.24,51.88),(30,'scalar',68.82,52.65),(30,'wide_scalar',69.57,59.76),(30,'vector',81.05,66.70)]:
 rs={r['condition']:r for r in b if r['key']==f'robust-{drive}_protected-d{d}-s501'}
 actual=[round(float(rs['clean']['aux_endpoint_accuracy_single'])*100,2),round(statistics.mean(float(rs[k]['accuracy_retained'])*100 for k in ['pair-silence-L80','pair-poisson-L80']),2)]
 check(f'SHD capacity d{d} {drive}',actual==[aux,ret],actual)
# All headline quantities derive from raw csv accuracy (not copied rounded text).
claims=json.loads((R/'evidence/generated-claims.json').read_text())
def value(task,family,L,kind):
 c=[v for v in claims if v['task']==task and v['family']==family and v['phase']=='mixed' and v['metric']=='last20' and v['L']==L and v['kind']==kind and not v['middle']]
 assert len(c)==1
 return round(c[0]['mean'],2)
for t,f,L,k,want in [('SSC','commit',120,'poisson',52.79),('SSC','dmp',120,'poisson',3.91),('DVS Gesture','commit',80,'silence',70.83),('DVS Gesture','dmp',80,'silence',19.70),('S-MNIST','commit',160,'silence',78.68),('S-MNIST','dmp',160,'silence',8.69)]:
 check('abstract '+str((t,f,L,k)),value(t,f,L,k)==want)
# Float schedule arithmetic and exact-integer scope.
replay=json.loads((R/'evidence/hardware-replay.json').read_text())
check('native replay archived outcome',replay['passed'] and replay['steps']==944 and replay['checked_values']==117056 and replay['cycles']==185528 and not replay['full_classifier'])
check('real trace cached cycles',188800-818*4==185528)
check('real trace cycle saving',round((188800-185528)/188800*100,2)==1.73)
check('gate coordinate count',944*40==37760)
check('scalar-DMP cycles',[(1+d+32) for d in (10,30)]==[43,63])
check('dense Commit cycles',[(4*d+32+64) for d in (10,30)]==[136,216])
check('closed Commit cycles',[(4*d+32+32) for d in (10,30)]==[104,184])
# Revised Figure 3: compare network MAC estimates using the selected tail condition.
ops={(v['family'],v['condition']):v for v in csvread('hardware-operations.csv')}
full=json.loads((R/'evidence/full-network-mac-claims.json').read_text())
for v in full:
 d=ops['dmp',v['condition']];c=ops['commit',v['condition']]
 check('network MAC scope '+v['condition'],abs(v['dmp_macs']-float(d['whole_mapped_mac']))<1e-6 and abs(v['commit_cached_macs']-float(c['whole_mapped_mac']))<1e-6)
 check('dense versus cached '+v['condition'],abs(v['commit_dense_macs']-v['commit_cached_macs']-float(c['stable_read_mac_saved']))<1e-6)
check('clean whole-network overhead',round(full[0]['cached_overhead_percent'],2)==6.30)
check('tail whole-network overhead',round(full[1]['cached_overhead_percent'],2)==6.70)
check('tail cache saving',round(full[1]['cache_saving_percent'],3)==1.183)
text=(R/'main.tex').read_text();log=(R/'main.log').read_text()
check('no unfinished manuscript markers',not re.search(r'\b(TODO|FIXME|TBD|INSERT HERE)\b',text))
check('no unresolved citations or references','undefined' not in log.lower())
check('no overflowing boxes','Overfull' not in log)
check('no LaTeX errors','! LaTeX Error' not in log)
info=subprocess.check_output(['pdfinfo',str(R/'main.pdf')],text=True)
pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));check('complete PDF',pages>=8,{'pages':pages})
report={'status':'passed','checks':checks,'pages':pages,'new_training_runs':0,'note':'Numerical/source and compilation audit; figure pages are also visually inspected. Scientific interpretation is described in the manuscript.'}
(R/'evidence/manuscript-audit.json').write_text(json.dumps(report,indent=2))
print(f'Passed {len(checks)} checks; {pages} PDF pages; zero new training runs.')
