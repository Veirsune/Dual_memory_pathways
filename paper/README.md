# 门控双状态记忆（GDSM）论文

论文共 8 页，包含正文、参考文献及附录。

[下载论文 PDF](https://raw.githubusercontent.com/Veirsune/Dual_memory_pathways/main/paper/GDSM_Paper.pdf) · [GitHub 查看](https://github.com/Veirsune/Dual_memory_pathways/blob/main/paper/GDSM_Paper.pdf) · [LaTeX 主文件](main.tex)

iPad 可用 Safari 打开 PDF，然后选择“分享 → 存储到文件”。论文文件统一为 `GDSM_Paper.pdf`，源码包为 [GDSM_LaTeX.zip](GDSM_LaTeX.zip)。

## 中文对照与段落写作

中文稿覆盖摘要、正文、附录、图注和表头，按英文顺序保留 54 个段落，并加上 `1-P2`、`4.2-P3` 等定位编号。公式、图表和参考文献编号与英文稿一致；SNN、DMP、GDSM、LIF 等术语及图内英文标签保留。中文 PDF 采用便于阅读的单栏排版。

- `GDSM_Paper_ZH.pdf`：中文阅读与批注文档。
- `GDSM_Chinese.docx`：用于逐段修改的 Word 文档，含可编辑公式和表格。
- `GDSM_Chinese.md`：用于文本编辑器的中文稿，图片通过相对路径引用。
- `main_zh.tex`：中文 PDF 的 LaTeX 源码；使用 XeLaTeX 和 Noto CJK 字体，运行 `make chinese`。
- `CHINESE_PARAGRAPHS.json`：对应的英文原段、中文译段和英文源码行号，记录此次翻译依据的源文件校验和。

Word、Markdown 和 LaTeX 是独立文件，直接修改其中一个不会自动同步另两个。后续按段落改稿时，请明确以哪个文件中的修改为准。中文译稿的核对记录见 `evidence/chinese-translation-audit.json`。

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

第 1 节引言合并问题背景与相关研究，第 2 节是方法，第 4 节是主要结果，第 5 节是实现成本。附录 A 汇总复现配置与硬件范围，B 给 SHD 监督消融，C 给关键补充结果。详细实现、完整结果表与探索记录见 [实验细节](EXPERIMENT_DETAILS.md)。主文表 2 和图 2 可直接阅读核心结果。

图 1 并列对照 DMP 与 GDSM，各结构只画一次。`figures/architecture.pdf` 与 `figures/architecture.svg` 是矢量文件，`scripts/draw_architecture.py` 提供独立绘图源码。图采用纯黑线条、纯黑文字和纯白背景；具体修改方法见 [图 1 编辑说明](FIGURE_EDITING.md)。

图 2–4 使用蓝灰主调：FSNN 为灰色，DMP 为灰蓝色，GDSM 为深蓝色，GDSM dense 柱为浅蓝灰色。配色在 `scripts/build_assets.py` 中统一定义。图 1 单独使用纯黑白。

图 3 展示 SSC、S-MNIST、PS-MNIST 的干净输入验证准确率随训练轮数的变化，标出干净训练与混合训练的切换。MNIST 使用三个种子的均值与标准差。图来自 42 份原始训练日志，仅绘制实际记录的验证点；原日志没有逐轮干扰分项验证，图中不将训练准确率或最终测试结果作为替代。单独重画可运行 `make training-curves`，脚本为 `scripts/draw_training_curves.py`。

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
