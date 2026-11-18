"""Check every requested epoch and selected checkpoint, then report the paired scores."""
import argparse,csv,hashlib,json,tarfile
from pathlib import Path

def report(archive,output):
    output.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as t:
        def read(name):return json.load(t.extractfile('results/'+name))
        assert read('complete.json')['completed']
        test=read('test.json');model=read('model-info.json')
        expected={f+'-'+phase for f in ('fsnn','dmp','gdsm') for phase in ('clean','mixed')}
        assert set(test)==expected
        checks={};rows=[]
        for key,conditions in test.items():
            family,phase=key.split('-')
            h=read(key+'-history.json')
            count=200 if phase=='clean' else 12
            assert [v['epoch'] for v in h]==list(range(1,count+1))
            best=max(h,key=lambda r:r['selection'])
            checks[key]=dict(epochs=count,updates=295*count,selected_epoch=best['epoch'],validation=best['selection'],checkpoint_sha256=hashlib.sha256(t.extractfile('results/'+key+'-best.pt').read()).hexdigest())
            assert len(conditions)==9
            for name,c in conditions.items():
                assert c['n']==20382 and all(0<=c[m]<=1 for m in ('whole','endpoint'))
                rows.append(dict(model=family,phase=phase,condition=name,selected_epoch=best['epoch'],**c))
    with (output/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (output/'audit.json').write_text(json.dumps(dict(models=model,runs=checks,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))
    lines=['# SSC 完整训练对照','', '每个模型从头训练 200 轮，每批 256 条。以干净整词验证准确率选模。下表使用官方测试集全部 20,382 条语音。','', '| 模型 | 轮数 | 验证集选中的轮次 | 干净整词准确率 |','|---|---:|---:|---:|']
    for fam in ('fsnn','dmp','gdsm'):
        key=fam+'-clean'
        lines.append(f"| {fam.upper()} | 200 | {checks[key]['selected_epoch']} | {100*test[key]['clean:250:0:0']['whole']:.2f}% |")
    lines+=['','原论文双层 DMP 报告 72.90%。本次采用论文表 4 的记忆窗口 40；原始事件到 250 帧的分箱规则不能从公开训练入口独立确认。','', 'metrics.csv 同时列出各自再训练 12 轮后的干净、整词叠加噪声及延迟读取结果。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args();report(a.archive,a.output)
