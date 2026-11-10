# 当前训练规则

按用户要求，仅限制训练轮数，不设置时间预算、不按耗时提前停止。Commit 视觉固定读取版本及 DMP/FSNN 均以50轮为目标，进度见 `/workspace/experiments/gesture-commit-50/status.json`。下文旧实验的3小时记录属于历史设置，现已取消。

# 手势与长序列实验入口

此目录是独立的新任务入口，不修改论文示例或已封板的 SHD/SSC 实验。
使用 Python 3.10、PyTorch 2.0.0、torchvision 0.15.1、SpikingJelly 0.0.0.0.14。

## CPU 优化更新（2026-11-02）

新增 `--backend cpu_scan`：保留原 LIF 方程、Sigmoid 代理梯度和 detached hard reset，用显式逆序递推计算输入梯度，避免逐时间片索引带来的昂贵反向图。完整 BPTT，不截断时间长度，不降时间步。

输出、膜电位、跨块状态、输入梯度、整网参数梯度已与 SpikingJelly torch 后端对照。1000 步真实手势、batch=1 的稳定批次由约 145.6 秒降为约 2.52 秒。因此下文基于旧后端提出的 GPU 建议不再是开展 CPU 实验的前提。原始测量保留在 vision-start，不覆写。

新增每轮 `last.pt` 完整恢复：模型、优化器、调度器、最佳模型、随机数、DataLoader 顺序和训练历史。连续两轮与一轮后恢复到第二轮的模型权重、损失和随机状态逐项一致。恢复需新 output 目录，`--epochs` 表示累计目标轮数；数据划分和关键超参数必须一致。

```bash
python -m vision_tasks.run --task gesture --family dmp \
  --data /workspace/data/vision/gesture-all-t100 --output /path/to/fresh-run \
  --steps 100 --epochs 50 --train-limit 0 --val-limit 0 --batch-size 4 \
  --threads 1 --backend cpu_scan --cache-data
# 续训时在同一组参数后追加：--resume /path/to/previous-run/last.pt
```

完整训练归档已缓存为100时间步：901段TRAIN（受试者1–19），176段VALID（20–23）。DMP与FSNN的50轮CPU对照已启动，每个进程预算约3小时，超时在轮末保存后停止。进度见 `/workspace/experiments/gesture-cpu-research/status.json`；完成后监督进程生成 `TRAINING_REPORT.md` 和验证曲线。该实验仍是100步设置，不能写成论文1000步复现。

## 论文与代码中可核实的训练设置

来源：[DMP 论文 v3](https://arxiv.org/abs/2512.07602v3)，Methods 第 15–16 页、扩展表 1（34 页）和表 4（37 页）；本仓库 `smnist/src/`、`psmnist/src/`。

| 项目 | S-MNIST / PS-MNIST |
|---|---|
| 输入 | 28×28 图像展开为 784×1；PS-MNIST 使用所有样本共用的固定像素排列 |
| 结构 | 两层、各 200 个 LIF 神经元；10 类输出 |
| 记忆 | 表 4 中 d=40，窗口 θ=300；公开脚本默认 d=100 |
| 训练 | 离线监督，沿时间反向传播；SpikingJelly 脉冲函数使用代理梯度 |
| 优化器 / 损失 | 公开脚本：Adam、CrossEntropyLoss |
| 学习率 | S-MNIST 0.001；PS-MNIST 0.005 |
| batch / epochs | 公开脚本实际 DataLoader batch=128、实际循环 50 轮；脚本的部分变量/参数未真正生效 |
| 调度器 | StepLR：S-MNIST 150 轮，PS-MNIST 100 轮，gamma=0.5；在 50 轮循环内没有下降 |
| 慢状态 | LMU A/B 矩阵固定，输入和读取投影可训练；公开实现用 FFT 计算线性记忆 |
| 输出 | 论文 Methods 说末层膜电位均值；公开长序列代码是最后 LIF 后线性分类 logit 的时间均值 |
| 阈值 | 论文表 4 写 10，公开代码没有覆盖 SpikingJelly 默认值，因此实际为 1 |

这些不一致需要记录，不能把论文表格、仓库默认值拼成一个未经说明的“完全复现”。本入口使用原模型及其实际阈值，显式指定 d=40。额外提供 `--readout last` 作为严格末帧监督实验，不能与默认均值读取混称同一协议。

原始训练脚本每轮用 MNIST 官方 TEST 作所谓 validation。本入口从 60,000 个官方训练样本按类别固定划分 54,000 TRAIN / 6,000 VALID，日常训练不评估官方 TEST。torchvision 下载时同时下载 TEST 文件并不代表使用 TEST 选模型。PS-MNIST 的排列独立设种子并存入 protocol 和 checkpoint，避免原脚本在设全局种子之前产生排列的问题。

## 当前环境运行

从仓库根目录执行：

```bash
source /workspace/.onboarding/dual-memory/env.sh
python -m vision_tasks.run --task smnist --output /workspace/experiments/my-smnist \
  --epochs 2 --train-limit 500 --val-limit 200 --batch-size 64
python -m vision_tasks.run --task psmnist --output /workspace/experiments/my-psmnist \
  --epochs 2 --train-limit 200 --val-limit 100 --batch-size 32 --threads 1
```

每次使用全新的 output 路径；已有实验不会被覆盖。`best.pt` 保存验证最优权重、模型配置及数据划分。日志保留有限梯度检查、实际时间步、批耗时、CPU 峰值 RSS 和 CUDA 实测峰值（若有 GPU）。小样本/少轮训练属于通路验证，不能比较论文最终准确率。

对单个已编码样本生成预测（此轮试跑权重尚不具有实用识别准确率）：

```bash
python -m vision_tasks.predict --checkpoint /path/to/best.pt \
  --input /path/to/sample.npz --output /path/to/prediction.json
```

手势输入是缓存格式 `[T,2,32,32]` 的事件计数 `.npz`，不是普通 RGB 视频；MNIST 输入可以是 uint8 的 28×28 `.npy` 图片或已归一化像素。预测时自动恢复检查点中保存的 PS-MNIST 固定排列。

## DVS Gesture 数据

使用 [Tonic 官方加载器所列的公开镜像](https://github.com/neuromorphs/tonic/blob/develop/tonic/datasets/dvsgesture.py)。该镜像已按类别切分，并把时间精度降为毫秒，共 1,077 TRAIN 样本；与 SpikingJelly 原始 AEDAT 预处理的 1,176 TRAIN 样本并不相同，结果不能直接视为同预处理复现。

训练归档下载地址：`https://api.figshare.com/v2/file/download/38022171`。
文件大小 2,443,675,558 字节，MD5 `3a8f0d4120a166bac7591f77409cb105`。
此轮归档位于 `/tmp/dmp-gesture/ibmGestureTrain.tar.gz`；/tmp 可被回收，丢失后从上述地址重新下载。

```bash
python -m vision_tasks.download_gesture --output /path/to/ibmGestureTrain.tar.gz
```

预处理从归档逐项读取，不展开整个原始数据目录：

```bash
python -m vision_tasks.prepare_gesture \
  --archive /tmp/dmp-gesture/ibmGestureTrain.tar.gz \
  --output /workspace/data/vision/gesture-t1000 \
  --steps 1000 --subjects 1 2 3 20
```

每段按完整手势持续时间均匀分 1000 个时间箱，空间按 4×4 汇总为 32×32 双极性计数，网络内除以 16 对应平均池化。检查原始事件数等于累计帧计数，保存 uint16 压缩帧。1000 步并不表示固定 1 秒。这一编码需要完整片段边界，并非无边界在线输入。

此轮准备用户 1/2/3 用于 TRAIN，用户 20 用于 VALID，共 186 段，严格分开受试者。准备全部训练归档时用 `--subjects` 不跟编号，并使用新的 output 目录；训练入口将用户 1–19 用于 TRAIN，20–23 用于 VALID。官方 TEST 不参与此次试跑。

```bash
python -m vision_tasks.run --task gesture \
  --data /workspace/data/vision/gesture-t1000 \
  --output /workspace/experiments/my-gesture-pilot \
  --steps 100 --epochs 2 --train-limit 44 --val-limit 22 --batch-size 2 --threads 1

# 保留 1000 步，测真实数据上的前向、反向和参数更新；不输出“准确率结果”。
python -m vision_tasks.run --task gesture \
  --data /workspace/data/vision/gesture-t1000 \
  --output /workspace/experiments/my-gesture-t1000-benchmark \
  --steps 1000 --epochs 1 --train-limit 22 --val-limit 11 \
  --batch-size 1 --threads 1 --benchmark-steps 3
```

`--steps 100` 将已缓存的 1000 箱每 10 箱相加，事件总数不变；它是较短序列的试跑，不是论文 1000 步实验。

## 手势模型的复现边界

论文给出“两层脉冲卷积 + 两层脉冲全连接；输入及每层卷积后池化；DMP 放在第一层全连接；1000 步”，没有提供对应训练脚本及完整通道数/优化器细节。此处重建为：

`pool4 → conv32-LIF-pool2 → conv64-LIF-pool2 → DMP-FC256-LIF → FC11-LIF → mean membrane`

卷积通道、池化尺度、d=40、θ=300、阈值=1、Adam lr=0.001 都是显式重建选择。FSNN/DMP 参数规模接近原文表中的 1.07M/1.09M，但规模接近不能证明结构与作者未公开实现完全相同。用 `--family fsnn` 可在相同前端下去除慢记忆作对照。

本入口先建立原版 DMP 在视觉任务上的基线。当前听觉 Commit 的特征提取及许可网络固定在 140 通道；尚未把它不加区分地接到视觉输入，也未声称这些试跑属于 Commit 迁移结果。

“手势预测”目前指给已有事件片段预测类别；如果研究动作尚未完成时的提前识别，需要另设 20%/40%/60% 前缀评价与相应训练协议。

## GPU 迁移与完整训练

论文没有报告训练 GPU 型号、最低显存或训练耗时。硬件章节的 22FDX 神经形态芯片是推理设计，训练这些模型不需要该芯片。

工程建议（不是论文实测最低要求）：

- S/PS-MNIST：单张 8–12 GB NVIDIA GPU 通常足以作为起点；batch 64/128，按实测显存调整。
- 手势 1000 步卷积训练：建议 24 GB 显存、32–64 GB 系统内存；先 batch 1/2 测显存再增加。12–16 GB 可尝试小 batch，不能保证任意配置都装得下。
- 两个方向共用：RTX 4090 24 GB / RTX A5000 24 GB 或同档单卡即可作为合理起点，无需先购买多卡或 A100。预留 50–100 GB 磁盘以容纳数据、缓存和重复实验。
- 论文只有约百万参数不代表训练显存小：长序列训练需要保留多个时间步的激活和代理梯度。推理状态开销不能代替训练显存。

示例安装（独立 Python 3.10 环境，不替换当前 CPU 环境）：

```bash
python -m venv .venv-vision
source .venv-vision/bin/activate
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r vision_tasks/requirements.txt
# 可选：SpikingJelly 多步 CUDA 融合内核后端；CUDA 11.x 环境
python -m pip install 'cupy-cuda11x>=12,<13'
```

GPU 环境和 CuPy 后端当前未实机验证，迁移后先执行 3 个批次基准。`--backend torch` 是通用入口；确认 CuPy/驱动兼容后可用 `--backend cupy` 加速 LIF 时序循环。

```bash
python -m vision_tasks.run --task psmnist --data /path/to/mnist \
  --output /path/to/runs/psmnist-seed42 --device cuda --backend cupy \
  --epochs 50 --train-limit 0 --val-limit 0 --batch-size 128 --memory 40 --theta 300

python -m vision_tasks.run --task gesture --data /path/to/all-gesture-t1000 \
  --output /path/to/runs/gesture-dmp-seed42 --device cuda --backend cupy \
  --steps 1000 --epochs 50 --train-limit 0 --val-limit 0 --batch-size 2
```

手势 50 轮是可执行的起始训练预算，原文没有报告该任务训练轮数；不能把它写成作者设置。正式对照建议先定验证协议，再锁定配置及种子重复，最后一次评估官方 TEST。
