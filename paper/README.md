# 面向鲁棒时序处理的脉冲神经网络门控双状态记忆

*Gated Dual-State Memory for Robust Temporal Processing in Spiking Neural Networks*

[中文 PDF](GDSM_Paper_ZH.pdf) · [中文 Word](GDSM_Chinese.docx) · [中文 Markdown](GDSM_Chinese.md) · [英文 PDF](GDSM_Paper.pdf) · [LaTeX 源码包](GDSM_LaTeX.zip)

一只手顺时针或逆时针画圆，都要经过圆的最高点。此时要判断方向，还需要知道手刚才从哪边过来。原 DMP 给脉冲网络的每一层一份共享记忆，让过去的输入参与当前判断。GDSM 延续这个思路，让候选状态逐帧跟踪输入，由写入门控制稳定状态的更新。

## 文章从哪里读起

第 3 节说明各模型的训练条件和读出方式。第 4.3 节表 4 并列给出 SHD、SSC、DVS Gesture、S-MNIST 和 PS-MNIST 在原事件结束时的识别结果。词中清零实验取一个固定的 SSC 模型，在词的正中间复制两份，只清空其中一份的稳定状态，再给两份模型完全相同的后半段。保留前半段的状态，后半段输出净多认对 813 个词。[逐例证据](../large_scale/ssc_matched200/midpoint/)

第 4 节讨论噪声出现的时机。干净手势上两种模型同为 86.74%。中档噪声贯穿动作时 DMP 更准，接在动作后持续 80 帧时 GDSM 更准。第 5 节给出全网乘加数及记忆核的定点核验。[跨任务对照](../large_scale/paired_effects/REPORT.md) · [证据索引](EVIDENCE_MAP.md)

## 逐段写作与编译

中文 PDF 的段落编号与 Markdown、Word 及英文段落对照表一致。例如 `3.2-P3` 说明词中清零的做法，`4.3-P3` 和 `4.3-P4` 给出结果。Word 中的公式、表格和正文可以编辑。段落对照表记录英文源码行号。[段落对照](CHINESE_PARAGRAPHS.json)

英文 PDF 从 `main.tex` 编译，中文 PDF 从 `main_zh.tex` 编译，使用 XeLaTeX 和 Noto CJK 字体。在本目录运行以下命令。Word 和 Markdown 是独立的写作文件。

```bash
make all
make chinese
make audit
```

记号沿用 DMP。第 2 节给出 DMP 更新、候选与稳定状态更新、记忆读取、门控及训练目标的六个公式。第 4.2 节比较输入投影和写入保护，第 5 节给出定点验证与整网乘加开销。复现参数与完整结果可查阅仓库记录。[记号对照](NOTATION.md) · [实验细节](EXPERIMENT_DETAILS.md)

## 图表

| 图 | 内容 | 可编辑来源 |
|---|---|---|
| 1 | DMP 与 GDSM 的共享记忆结构 | [绘图脚本](scripts/draw_architecture.py) · [矢量 SVG](figures/architecture.svg) |
| 2 | 随间隔长度变化的末段分类准确率 | [结果图脚本](scripts/build_assets.py) |
| 3 | 干净验证集上的训练曲线 | [训练曲线脚本](scripts/draw_training_curves.py) |
| 4 | 全程叠加背景脉冲的语音和识别结果 | [语音图脚本](scripts/draw_ssc_speech.py) |
| 5 | 五项任务的架构差值及噪声出现的时机 | [差值图脚本](../large_scale/paired_effects/plot.py) |
| 6 | 手势识别的整网乘加数 | [结果图脚本](scripts/build_assets.py) |

图 1 保留黑白线稿，结果图沿用蓝灰色。用 `make architecture`、`make training-curves`、`make ssc-speech` 可分别重画图 1、3、4。图 5 的重画命令见[配图编辑说明](FIGURE_EDITING.md)。从源码包编译时可直接使用随附的矢量图。
