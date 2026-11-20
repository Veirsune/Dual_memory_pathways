# GDSM：门控双状态记忆

**面向鲁棒时序处理的脉冲神经网络门控双状态记忆**

*Gated Dual-State Memory for Robust Temporal Processing in Spiking Neural Networks*

DMP 让一层脉冲神经元共享一份输入历史。GDSM 让候选状态继续跟踪新输入，写入门决定何时更新稳定状态。神经元同时读取两份状态。

**[下载论文 PDF](https://raw.githubusercontent.com/Veirsune/Dual_memory_pathways/main/paper/GDSM_Paper.pdf)** · **[GitHub 在线查看 PDF](paper/GDSM_Paper.pdf)** · [LaTeX 源码](paper/main.tex) · [源码包](paper/GDSM_LaTeX.zip) · [实验协议与结果](paper/evidence/) · [论文中文导读](paper/README.md)


在 iPad 上可用 Safari 打开 PDF 下载链接，再通过“分享 → 存储到文件”保存。

本仓库基于 Sun 等人的 [Dual Memory Pathways](https://github.com/sunpengfei1122/Dual_memory_pathways) 扩展。原作者介绍、安装说明和引用信息保存在 [README.upstream.md](README.upstream.md)，原始数据集示例仍在各自目录中。GDSM 的代码配置键沿用 `commit`，以兼容历史实验和检查点。

## 代码导航

| 目录 | 内容 |
|---|---|
| `commit_memory/` | 向量驱动、候选/稳定状态、因果门控、固定/静态/动态读取、分块推理 |
| `infer_memory.py` | 听觉序列分类命令行接口 |
| `large_scale/` | 全量 SSC 的 FSNN / DMP / GDSM 对照、数据准备与评估 |
| `mnist_study/` | S-MNIST / PS-MNIST 三模型、三种子完整对照 |
| `vision_tasks/` | DVS Gesture 数据预处理、训练、干扰评估、操作计数和 CPU 加速 |
| `tests/` | 状态、因果性、梯度、恢复训练及视觉模型功能测试 |
| `dmp-hw-chisel/` | 原 DMP 硬件工程、GDSM 记忆核、定点参考和 Verilator 重放 |
| `paper/` | 论文 PDF、LaTeX、图表脚本、指标 CSV、协议和证据快照 |

## 前面的输入怎样参与分类

同一套骨干和训练安排下，原事件结束时的干净分类准确率如下。S-MNIST 和 PS-MNIST 为三个训练种子的平均值，其余各行是一组配对模型。

| 数据集 | DMP | GDSM |
|---|---:|---:|
| SHD | 89.80% | 90.33% |
| SSC（200 轮干净训练） | 72.34% | 75.29% |
| DVS Gesture | 86.74% | 86.74% |
| S-MNIST | 94.39% | 96.42% |
| PS-MNIST | 89.00% | 89.63% |

在另一个固定的 SSC 模型上，我们只在一个词的中点清空一次稳定状态。两份副本听到相同的后半段，只用后半段分数分类。保留状态认对 6,692 个词，清零认对 5,879 个词，净差 813 个。[逐例结果与实验方法](large_scale/ssc_matched200/midpoint/)

干扰实验进一步比较噪声伴随动作和动作结束后出现的情况。手势的中档噪声贯穿动作时 DMP 更准，接在动作之后时 GDSM 更准。[跨任务结果](large_scale/paired_effects/REPORT.md)

## 环境与功能检查

已验证的研究环境为 Python 3.10、PyTorch 2.0.0、torchvision 0.15.1、SpikingJelly 0.0.0.0.14。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r commit_memory/requirements.txt
python -m unittest discover -s tests -v
python -m mnist_study.verify --device cpu --output /tmp/mnist-verification.json
```

GPU 环境请安装对应 CUDA 的 PyTorch/torchvision；完整 SSC/MNIST 入口的 CUDA LIF 后端还使用 CuPy（CUDA 11.x 环境可安装 `cupy-cuda11x`）。CPU 功能检查不需要 GPU。

## 顺序与乱序 MNIST

数据准备保留固定的训练/验证划分种子 1701 和像素排列种子 1702。

```bash
python -m mnist_study.prepare --root data/raw-mnist --output data/mnist --download
python -m mnist_study.run train --data data/mnist --output runs/mnist --device cuda
python -m mnist_study.run evaluate --data data/mnist --runs runs/mnist --output runs/mnist-evaluation --device cuda
```

入口依次运行两种任务、三个模型、三个种子，每次固定 50 轮干净训练 + 12 轮混合训练。可指定 `--batch`；`--device cpu` 可用于 CPU。训练按轮数运行，运行时间由设备决定。

## 全量 SSC

从 [Heidelberg spiking datasets](https://zenkelab.org/datasets/) 获取并解压 `ssc_train.h5`、`ssc_valid.h5`、`ssc_test.h5` 到 `data/ssc-raw/`。

```bash
python -m large_scale.prepare --raw data/ssc-raw --output data/ssc250
python -m large_scale.ssc_study train --data data/ssc250 --output runs/ssc --device cuda
python -m large_scale.ssc_study evaluate --data data/ssc250 --runs runs/ssc --output runs/ssc-evaluation --device cuda
```

三个模型各固定 40 轮干净训练 + 12 轮混合训练；SSC 每词直接重建为 250 帧、140 通道。

## 手势与硬件

DVS Gesture 的完整准备、干净训练和混合训练命令见 [vision_tasks/README.md](vision_tasks/README.md)。硬件配置和精度契约见 [dmp-hw-chisel/COMMIT_README.md](dmp-hw-chisel/COMMIT_README.md)。

已附带真实训练权重导出的定点轨迹，可在安装 Java、sbt、Verilator 和 firtool 后运行：

```bash
cd dmp-hw-chisel
COMMIT_TRAINED_TRACE="$PWD/fixtures/trained-trace.json" bash tools/run_commit_checks.sh
```

记录中的 944 步原生 Verilator 重放，117,056 个定点值与独立参考一致。该核实现的是记忆分支；全网 MAC 统计与局部 RTL 周期分别报告。

## 编译论文

```bash
cd paper
make all
make chinese
# 可选：从随附 CSV 重新生成图表，然后再次编译
python scripts/build_assets.py
```

PDF 和所有图表已经包含在仓库中；在 iPad 上直接使用页首下载链接即可。
