# Figure 1 编辑说明

图 1 使用 **Python + Matplotlib** 按坐标绘制。圆形神经元、矩形状态、菱形门、连线和文字由脚本分别生成，公式使用 Matplotlib MathText 排版。输出包含 PDF、SVG 和 PNG。

## 文件在哪里

| 文件 | 用途 |
|---|---|
| `scripts/draw_architecture.py` | 绘图源代码，修改结构、文字、公式和样式 |
| `figures/architecture.svg` | 在 Inkscape、Illustrator 等软件中直接编辑 |
| `figures/architecture.pdf` | 论文实际使用的矢量图 |
| `figures/architecture.png` | 300 dpi 预览图 |
| `main.tex` 中的 `fig:architecture` | 图注及插图宽度 |

## 用代码修改

脚本顶部集中定义配色、画布大小、字体、输出分辨率和三个分图的标题位置。当前采用黑白灰：主要文字和通路 `#222222`，次要文字 `#666666`，分隔线 `#CCCCCC`，填充 `#F5F5F5`，背景白色。快通路、候选状态、稳定状态和门控使用相同墨色，通过标签与实线/虚线区分。

- **改颜色：** 修改 `PALETTE`。
- **改整体大小：** 修改 `FIGURE_SIZE`，单位为英寸。
- **改字体：** 修改 `FONT_FAMILY`；局部字号在 `text(..., size=...)` 或调用中的字号参数里。
- **改分图标题和位置：** 修改 `PANEL_HEADINGS`。
- **改节点、文字、连线：** 在代码的 `# a:`、`# b:`、`# c:` 三个分区中编辑对应调用。

坐标范围为横向 `0–100`、纵向 `0–79`，原点在左下角。`x` 增大向右，`y` 增大向上。常用函数：

```python
# 文字：位置、内容、字号
text(41, 36.9, r'$W_p p_t$', 8)

# 状态框：左下角 x/y、宽、高、文字、墨色
state(28.6, 19.7, 13.2, 9, 'Candidate\n' + r'$p_t\in\mathbb{R}^{d}$', CAND)

# 折线箭头：依次列出起点、拐点、终点
arrow([(35.2, 28.9), (35.2, 35.3), (58.7, 35.3), (59.5, 37.4)], CAND, 1.2)

# 虚线控制箭头
arrow([(10.7, 4.6), (18.9, 4.6)], GATE, 1, ls='--')
```

移动节点后，相连箭头的端点和附近标签需要同步移动。公式写在原始字符串的 `$...$` 中，例如 `r'$m_t=m_{t-1}$'`。

在仓库根目录运行：

```bash
cd paper
python -m pip install matplotlib
python scripts/draw_architecture.py
make
```

绘图命令同时生成 `architecture.pdf`、`architecture.svg`、`architecture.png`；`make` 编译论文为 `GDSM_Paper.pdf`，需要安装 TeX Live 和 latexmk。也可用 `make architecture` 生成图。全量图表重建命令为 `python scripts/build_assets.py`，它同样调用这份绘图脚本。

## 用绘图软件修改

用 Inkscape 或 Illustrator 打开 `figures/architecture.svg`。SVG 保留文字对象，节点和连线可以选中、移动、改样式；公式通常由多个文字片段组成，大幅修改公式时用脚本更方便。默认字体为 DejaVu Sans。

保存 SVG 后，将画布导出成 `figures/architecture.pdf`，替换同名文件，再运行 `make` 即可更新论文。此时不要运行绘图脚本，否则它会按代码覆盖手工修改的图。长期维护时，建议将最终调整同步到脚本。

## 图注与图宽

在 `main.tex` 中搜索 `fig:architecture`，其前面的 `\caption{...}` 为图注。当前插图命令为：

```tex
\includegraphics[width=.94\textwidth]{figures/architecture.pdf}
```

修改 `.94` 可以调整图在论文中的整体宽度。图中的状态格子是结构示意，不是实验数据。
