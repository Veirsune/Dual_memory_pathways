# 门控双状态记忆（GDSM）论文

论文共 10 页，包含正文、参考文献及附录。

[下载论文 PDF](https://raw.githubusercontent.com/Veirsune/Dual_memory_pathways/main/paper/GDSM_Paper.pdf) · [GitHub 查看](https://github.com/Veirsune/Dual_memory_pathways/blob/main/paper/GDSM_Paper.pdf) · [LaTeX 主文件](main.tex)

iPad 可用 Safari 打开 PDF，然后选择“分享 → 存储到文件”。论文文件统一为 `GDSM_Paper.pdf`，源码包为 [GDSM_LaTeX.zip](GDSM_LaTeX.zip)。

## 论文讲的科学故事

**识别对了，不代表记住了。** 一段语音或手势结束后，如果后面还有停顿或背景噪声，分类器仍可能需要保留刚才的判断。对整段序列取平均的准确率，会把前面正确的输出算进去，因此未必能反映最后还记得多少。

DMP 为脉冲网络增加一个全层共享的小记忆，但它始终接收输入、始终更新。我们把它分成两个状态：候选状态持续跟踪新输入，稳定状态只在门控允许时接收候选内容。向量输入让不同记忆维度接收不同的通道组合；独立读取让快通路、候选状态和稳定状态共同参与分类。

实验直接在完整事件后追加干扰，只看干扰末尾的输出，检验“事件已经结束后还记不记得”。结果是：

| 任务与干扰 | DMP | GDSM |
|---|---:|---:|
| SSC，追加 120 帧噪声 | 3.91% | 52.79% |
| DVS Gesture，追加 80 帧静音 | 19.70% | 70.83% |
| S-MNIST，追加 160 帧静音，三训练种子均值 | 8.69% | 78.68% |

Gesture 两个模型在干扰开始前的末端准确率接近（85.61% 与 86.36%），干扰后却明显分开，尤其清楚地展示了保持能力的差异。SHD 的容量和组件对照进一步表明：**存入更有用的内容，与避免内容被覆盖，是两个互补的作用。**

这项研究的成果是一个能跨语音、事件视觉和像素序列改善干扰后记忆保持的共享记忆架构，以及相应的对照证据和定点实现。真实权重重放的 117,056 个定点数值与整数参考完全一致；Gesture 干净输入下，全网有效 MAC 增加 6.30%。这是运算量对照，物理能耗仍需工艺级测量。

## 论文结构与记号

英文标题：*Gated Dual-State Memory for Robust Temporal Processing in Spiking Neural Networks*。

| 符号 | 固定含义 |
|---|---|
| `s_t` / `s_t^(0)` | 当前层输入 / 原始传感器帧 |
| `x_t` / `h_t` | 原版 DMP 的标量驱动 / 共享状态 |
| `u_t` | 新模型的向量驱动 |
| `p_t` / `m_t` | 候选状态 / 稳定状态 |
| `g_t` | 最终逐维写入门 |
| `W_f` / `W_p` / `W_m` | 快通路 / 候选 / 稳定读取权重 |
| `T_0` / `T` / `L` / `K` | 原序列长度 / 扩展后长度 / 干扰长度 / 末端读取窗口 |

第 3 节是方法，第 5 节是主要结果，第 6 节是实现成本。附录 A 给训练设置，B 给 SHD 组件与监督对照，C 给完整补充结果，D 给硬件细节。主文表 2 和图 2 可直接阅读核心结果。

图 1 并列对照 DMP 与 GDSM，各结构只画一次。`figures/architecture.pdf` 与 `figures/architecture.svg` 是矢量文件，`scripts/draw_architecture.py` 提供独立绘图源码。图采用纯黑线条、纯黑文字和纯白背景；具体修改方法见 [图 1 编辑说明](FIGURE_EDITING.md)。

## 编译与数值核验

上传本目录到 Overleaf，选择 `main.tex`，用 pdfLaTeX 编译。图表与参考文献已随附，无需训练环境。

本地编译为 `GDSM_Paper.pdf`：

```bash
make
```

从封存数据重建图表并核验：

```bash
python scripts/build_assets.py
make audit
```

需要 TeX Live、latexmk、Python、NumPy 和 Matplotlib。`build_assets.py` 在研究工作区使用实验源文件，独立运行时使用 `evidence/` 中的快照。`audit_manuscript.py` 核对来源校验和、主要数值、计算成本、记号和编译结果。

`EVIDENCE_MAP.md` 给出证据对应关系。代码与图表生成所需指标已公开；原始数据和大型训练检查点不包含在源码仓库内。实验记录中的 `commit` 是 GDSM 的历史代码键，保留它是为了保持与封存结果的对应关系。
