# GDSM 论文草稿

**论文目标：说明“候选状态持续接收、稳定状态门控更新”能让共享记忆 SNN 在长时间干扰后保留可用的分类信息。** 写作定位为面向 *Neural Networks* 类期刊的算法与实证研究论文；当前采用通用双栏格式。

英文标题：**Gated Dual-State Memory for Robust Temporal Processing in Spiking Neural Networks**

中文标题：**门控双状态记忆：提升脉冲神经网络在时间干扰下的信息保持**

## 直接阅读

- `main.pdf`：完整英文论文，11 页，含正文、参考文献及附录。
- `main.tex`：可继续编辑的 LaTeX 主文件。
- `references.bib`：14 篇参考文献。
- `figures/`：可直接用于论文的矢量架构图、保持曲线和硬件周期图。
- `tables/`：从已有实验 CSV 自动生成的表格。
- `evidence/`：对应指标、协议、报告及数值核验记录。

这是已经写成的研究草稿，不是提纲。作者和单位暂未填写，PDF 使用 Anonymous authors；没有代为投稿或上传至公开平台。

## 这篇论文讲什么

普通 DMP 的共享记忆持续更新，遇到长空白或背景干扰后，已经识别出的类别可能不再能从当前网络状态中读出。GDSM 把记忆拆成持续工作的候选状态 p 和选择性更新的稳定状态 m，并让两者分别向快通路提供信息。

论文围绕三个实验结论展开：

1. **确实保得更久。** SSC 追加 120 帧 Poisson 噪声后，GDSM 的末端分类准确率为 52.79%，DMP 为 3.91%；DVS Gesture 追加 80 帧静音后为 70.83% 对 19.70%。S-MNIST 在未训练过的 160 帧静音后为 78.68% 对 8.69%（三训练种子均值）。
2. **存入的内容和保护机制缺一不可。** SHD 的向量输入、宽标量容量对照及监督来源消融解释了收益来源。最终跨数据集实验不使用噪声位置标签训练门控。
3. **可以实现，但成本要算清。** 真实训练权重的 944 步定点 RTL 重放中，117,056 个数值与参考完全一致。稳定读取可以缓存，但 GDSM 增加了运算和参数，本文不把功能仿真写成超越原版的能效证明。

全文使用四个来源数据集：SHD、SSC、DVS Gesture、MNIST。MNIST 有顺序和固定乱序两个任务，不当成两个独立来源数据集。

## 阅读顺序

- 先读摘要和图 1：理解架构及论文问题。
- 看表 3、图 2：比较干扰前与干扰后的同口径末端分类准确率。
- 看第 5.4–5.5 节：理解向量驱动、门控监督和切换代价。
- 看第 6 节：了解已完成的硬件验证及实际计算成本。
- 附录给出训练协议、完整中途中断结果、统计说明和证据来源。

已有证据最支持“干扰后的保持能力”，因此没有把论文包装成普遍提高所有任务准确率。强噪声失败、切换较慢和种子差异集中在结果与讨论中说明，不需要为了成文继续无限调参。

## 编译与复现图表

上传本目录的源码包到 Overleaf，选择 `main.tex` 为主文件、pdfLaTeX 编译即可。所有表格和 PDF 图已经包含在包内，不需要训练环境。

本地编译：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

若未安装 latexmk：

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

重建数据图表需要 Python、NumPy 和 Matplotlib：

```bash
python scripts/build_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
python scripts/audit_manuscript.py
```

图表脚本优先读取工作区实验源文件；独立解压时自动使用包内 `evidence/` 快照，不依赖模型检查点。参考文献已包含；`verify_references.py` 仅供联网重新核对 DOI 元数据，不是编译步骤。

## 本次完成范围

确定研究主张、整理现有证据、撰写全文、绘制四幅论文图、整理七张表、核对文献、编译 PDF、检查数值与版面，并打包可独立编译的源码。本轮新增训练数为 **0**，未改变任何封存实验或模型。

`evidence/manuscript-audit.json` 记录数值及编译检查；`EVIDENCE_MAP.md` 提供原始研究归档的定位与校验信息。源码包以小体积成文材料为主，完整训练权重和预测数组保留在原实验归档中。

## 本次修订（v2）

- 主文图 3 改为 Gesture 全网有效 MAC 对照；局部周期图移入附录，明确单位及范围。干净输入下增加 6.30%，尾部静音 L80 下缓存版增加 6.70%；该尾部条件的缓存节省为 1.183%。
- Discussion 的两段限制讨论压缩为两句 Future directions；具体分布偏移结果放入附录。
- 测试集评估历史移至附录的 Evaluation chronology，保留实验协议信息。

## 命名更新（v3）

模型统一称为 **门控双状态记忆（Gated Dual-State Memory，GDSM）**，直接描述候选状态、稳定状态及门控写入。论文标题、正文、表格和图例已同步更新；新源码包为 `gdsm-manuscript.zip`。原始实验记录中的模型键 `commit` 保留，以便与已封存的结果逐项对应。

## GitHub 发布

最新版 PDF 为 [GDSM_Paper.pdf](GDSM_Paper.pdf)。论文名称 GDSM 对应源码中的 `commit` 模型键。工作区历史目录仅用于标识实验来源；重建图表可直接使用本目录自带的 evidence 快照。
