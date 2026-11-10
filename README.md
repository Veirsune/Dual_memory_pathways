# GDSM：门控双状态记忆

**Gated Dual-State Memory for Robust Temporal Processing in Spiking Neural Networks**

GDSM 将持续更新的候选状态与门控更新的稳定状态分开，用小型共享记忆增强脉冲神经网络在静音和噪声干扰后的信息保持。

**[下载论文 PDF](https://raw.githubusercontent.com/Veirsune/Dual_memory_pathways/main/paper/GDSM_Paper.pdf)** · **[GitHub 在线查看 PDF](paper/GDSM_Paper.pdf)** · [LaTeX 源码](paper/main.tex) · [实验协议与结果](paper/evidence/) · [论文中文导读](paper/README.md)

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

## 主要实验结果

以下是混合训练后，干扰结束时最后 20 帧的分类准确率；MNIST 为三个训练种子的均值，完整标准差见论文。

| 测试条件 | DMP | GDSM |
|---|---:|---:|
| SSC：尾部 Poisson 噪声 L120 | 3.91% | 52.79% |
| DVS Gesture：尾部静音 L80 | 19.70% | 70.83% |
| S-MNIST：尾部静音 L160 | 8.69% | 78.68% |
| PS-MNIST：尾部静音 L160 | 14.12% | 56.43% |

SSC 的 L120 和 MNIST 的 L160 未用于相应混合训练。数据划分、损失函数、种子、容量对照、噪声分布偏移和切换实验见论文及 `paper/evidence/`。训练权重和原始数据不在本次源码发布中；论文图表可直接从随附指标重建。

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
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
# 可选：从随附 CSV 重新生成图表，然后再次编译
python scripts/build_assets.py
```

PDF 和所有图表已经包含在仓库中；在 iPad 上直接使用页首下载链接即可。
