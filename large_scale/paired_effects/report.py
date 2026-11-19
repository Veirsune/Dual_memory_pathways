"""Render the cross-dataset contrast table and the paired SSC repair from archived results."""
import argparse
import csv
import json
import statistics
from pathlib import Path

def mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)

def build(effects, results, dest):
    rows=list(csv.DictReader(effects.open()))
    groups=[
        ('shd-matched','DMP / GDSM','SHD，向量 GDSM 对 DMP，同配方','overlay-0.125'),
        ('gesture','DMP / GDSM','DVS Gesture，GDSM 对 DMP','overlay-1.81265'),
        ('ssc-repair','DMP / GDSM','SSC，200 轮，初始学习率 0.01','overlay-0.125')]
    lines=['# 记忆架构对识别和保持的影响','',
      '各列的百分点差值均为新结构减去对应参照。SHD 最终架构对 DMP 使用预先固定的种子 301；标量版本和向量版本的读取消融分别是三组种子的等权平均；手势和 SSC 每个配方各一组训练种子。各测试集均使用全部官方样本。各任务独立训练。','',
      '| 数据集和对照 | 干净，参照 → 新结构（Δ） | 事件内中档叠加，参照 → 新结构 | 事件内 Δ | 共同干净正确样本，每百个净多保住 | 事件后 L80，参照 → 新结构 | 事件后 Δ |',
      '|---|---:|---:|---:|---:|---:|---:|']
    def add_contrast(task,cmp,label,mid):
        g=[r for r in rows if r['dataset']==task and r['comparison']==cmp]
        clean,over,tail=[[r for r in g if r['condition']==c] for c in ('clean',mid,'tail80')]
        lines.append(f"| {label} | {mean(clean,'acc_a'):.2f}% → {mean(clean,'acc_b'):.2f}% ({mean(clean,'delta_pp'):+.2f}) | {mean(over,'acc_a'):.2f}% → {mean(over,'acc_b'):.2f}% | {mean(over,'delta_pp'):+.2f} | {mean(over,'shared_delta_pp'):+.2f} | {mean(tail,'acc_a'):.2f}% → {mean(tail,'acc_b'):.2f}% | {mean(tail,'delta_pp'):+.2f} |")
    for group in groups:
        add_contrast(*group)
    images = list(csv.DictReader((Path(__file__).resolve().parents[2]/'paper/evidence/mnist.csv').open()))
    for task, label in [('smnist','S-MNIST'), ('psmnist','PS-MNIST')]:
        def accuracies(condition, readout):
            values = []
            for family in ('dmp', 'commit'):
                selected = [r for r in images if r['task']==task and r['family']==family
                            and r['phase']=='mixed' and r['name']==condition and r['metric']==readout]
                assert sorted(int(r['seed']) for r in selected)==[42,43,44]
                values.append(100*mean(selected,'accuracy'))
            return values
        ca,cb=accuracies('clean','mean')
        ta,tb=accuracies('p784_poisson_L80','last20')
        lines.append(f'| {label}，三种子均值 | {ca:.2f}% → {cb:.2f}% ({cb-ca:+.2f}) | — | — | — | {ta:.2f}% → {tb:.2f}% | {tb-ta:+.2f} |')
    lines+=['','S-MNIST 和 PS-MNIST 使用全部 10,000 张测试图像，数据来自[逐种子评估](../../paper/evidence/mnist.csv)。两项图像任务评估了插入和追加干扰，尚未测量逐像素同时叠加噪声。各任务的输入单位、训练方式及噪声强度见实验协议。','',
            '## 机制与训练配方对照','',lines[4],lines[5]]
    for group in [
        ('shd','DMP / scalar candidate-stable','SHD，标量候选–稳定对 DMP','overlay-0.125'),
        ('shd-vector','GDSM stable read off / full','SHD，全模型对关闭稳定读取','overlay-0.125'),
        ('ssc-archived','DMP / GDSM','SSC，40+12 混合训练','overlay-0.125'),
        ('ssc-256-original','DMP / GDSM','SSC，200 轮，初始学习率 0.1','overlay-0.125')]:
        add_contrast(*group)
    lines+=['','SHD 向量 GDSM 与 DMP 从同一黄金检查点起步，经历相同的 30 轮微调、类别损失、增强、优化器和样本顺序。另两组 SHD 对照分别检验早期标量版本的收益和稳定读取对已训练向量模型的推理贡献。','',
           '手势模型在干净集均为 86.74%。对 224 个双方原本都认对的手势，在整个动作上叠加 1 倍训练密度的 Poisson 噪声，GDSM 相比 DMP 净少保住 25 段；动作后追加 80 帧同强度噪声，GDSM 净多保住 58 段。','',
           '旧 SSC 混合训练模型的强项是延迟保持。全程叠加噪声时，强度越高，GDSM 的干净优势越小；最高档 DMP 领先。与 200 轮配方相比，40+12 轮配方同时改变了分箱聚合、宽度、记忆容量和训练目标。','',
           '## SSC 的学习率对照','',
           '| 模型 | 参数 | 更新次数 | 最佳轮 | 验证 | 官方测试 | 相对该模型 0.1 测试变化 |','|---|---:|---:|---:|---:|---:|---:|']
    audit=json.loads((results/'audit.json').read_text())
    old=[r for r in rows if r['dataset']=='ssc-256-original' and r['condition']=='clean'][0]
    for fam,key in [('dmp','acc_a'),('gdsm','acc_b')]:
        a=audit[fam]
        lines.append(f"| {fam.upper()} | {a['parameters']:,} | {a['updates']:,} | {a['best_epoch']} | {a['valid']:.2%} | {a['test']:.2%} | {a['test']*100-float(old[key]):+.2f} |")
    lines+=['','每轮用固定 128 条验证语音观察记忆入口。下表给出该层记忆驱动中非零值占比（%，第一层 / 第二层）。','',
            '| 模型 | 第 1 轮 | 第 20 轮 | 第 100 轮 | 第 200 轮 |','|---|---:|---:|---:|---:|']
    for fam in ('dmp','gdsm'):
        h=json.loads((results/f'{fam}-clean-history.json').read_text())
        def entry(i):
            return ' / '.join(f'{v*100:.2f}' for v in h[i-1]['memory_drive_positive'])
        lines.append('| '+fam.upper()+' | '+' | '.join(entry(i) for i in (1,20,100,200))+' |')
    lines+=['','用同一个最佳模型，在官方测试时关闭动态记忆读取并保留偏置。数值为完整模型减去关闭之后的准确率，单位为百分点。','',
            '| 干预 | 干净 | 事件内中档 | 事件后 L80 |','|---|---:|---:|---:|']
    for name,label in [('DMP all memory off / full','DMP 全部动态记忆'),('GDSM all memory off / full','GDSM 全部动态记忆'),('GDSM stable read off / full','GDSM 仅稳定读取')]:
        sub={r['condition']:r for r in rows if r['dataset']=='ssc-repair' and r['comparison']==name}
        lines.append('| '+label+' | '+' | '.join(f"{float(sub[c]['delta_pp']):+.2f}" for c in ('clean','overlay-0.125','tail80'))+' |')
    lines+=['','分子分母与所有强度见[逐样本差值汇总](paired.csv)。SSC 按说话人、手势按受试者重采样的 95% 区间见[区间表](uncertainty.csv)。数据编码、训练与复核方法见[实验协议](protocol.md)。']
    dest.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('effects','results','dest'):p.add_argument(name,type=Path)
    a=p.parse_args();build(a.effects,a.results,a.dest)
