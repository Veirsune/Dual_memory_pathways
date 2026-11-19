# Figure 1 编辑说明

图使用 **Python + Matplotlib** 按坐标绘制，公式用 MathText 排版。只保留左右两个分图：DMP、GDSM。左侧 Memory 与右侧 Candidate、Stable 均为 12 × 6 的状态框，底边统一位于 y = 17；右图不单独画驱动模块、门控计算模块或外围分组框。所有绘图对象均使用纯黑 `#000000` 描边/文字与纯白 `#FFFFFF` 底色，无灰色填充、透明度色块或渐变。

## 文件

- `scripts/draw_architecture.py`：绘图源代码。
- `figures/architecture.svg`：可用 Inkscape、Illustrator 编辑的矢量文件，保留文字对象。
- `figures/architecture.pdf`：论文实际使用的矢量图。
- `figures/architecture.png`：300 dpi 预览。
- `main.tex`：搜索 `fig:architecture` 修改图注或插图宽度。

## 修改代码

脚本顶部的三个表集中管理布局，坐标原点在左下角，横坐标向右增加，纵坐标向上增加。

| 配置 | 内容 |
|---|---|
| `NODES` | 每个模块的左下角坐标、宽度、高度、文字 |
| `ROUTES` | 每条箭头的起点、拐点、终点、线宽、是否虚线 |
| `LABELS` | 各标签的坐标、内容、字号 |
| `FIGURE_SIZE` | 整体尺寸，单位英寸 |
| `CANVAS` | 坐标范围，横向至 101，纵向显示 10–45 |
| `FONT_FAMILY` | 字体，当前为 DejaVu Sans |

例如：

```python
# NODES：左下角 x/y，宽，高，文字
'candidate': (59, 17, 12, 6, 'Candidate\n' + r'$p[k]$'),

# ROUTES：依次连接起点、拐点和终点；最后一项控制是否虚线
'candidate_read': ([(65,23.1), (65,30), (92.5,30), (93.5,31.5)], 1, False),

# LABELS：x，y，公式，字号
(77, 32, r'$W_p$', 9),
```

移动模块时，相连箭头的端点和附近标签需要同步移动。LIF 圆、写入门菱形和回路在 `draw_architecture()` 中绘制；修改这些对象可搜索 `Circle`、`Polygon`、`recurrence`。

在仓库根目录运行：

```bash
cd paper
python -m pip install matplotlib
python scripts/draw_architecture.py
make
```

绘图命令同时生成 PDF、SVG、PNG。`make` 编译论文 `GDSM_Paper.pdf`，需要 TeX Live 和 latexmk。也可以用 `make architecture` 只生成图。

## 用绘图软件修改

打开 `figures/architecture.svg`，直接选中文字、节点和连线修改。公式可能由多个文字片段组成；修改复杂公式时，编辑代码中的 `$...$` 更方便。

保存 SVG，并将画布导出为 `figures/architecture.pdf` 后，运行 `make` 更新论文。再次运行绘图脚本会覆盖手工修改，因此建议将最终布局同步到代码。

## 图与模型的对应

- `p[k-1]` 标签表示候选写入有一步延迟，图中不再另画延迟框。
- 输入驱动直接从 `s[k]` 主干引出。门控只保留菱形和 `g[k]` 标签，完整计算方式见正文公式 (5)。
- 左侧读取权重标为 `W_m`，右侧分别为 `W_p` 和 `W_m`。
- `bar B ⊙ x[k]` 与代码中的固定对角缩放是等价表示：将正的行缩放吸收进输入投影即可。具体映射在论文附录 A，模型与实验结果没有变化。
- 回路、实线和虚线表达结构，解释性语句放在正文和图注中。

图中沿用原 DMP 的离散时间写法 `[k]`；为避免拥挤，统一省略层上标。左侧状态为 `m[k]`，右侧为 `p[k]` 与 `m[k]`；输入驱动的粗体 `x[k]` 表示向量，膜电位保留符号 `u`。

## 跨数据集架构差值（图 3）

绘图脚本为 [plot.py](../large_scale/paired_effects/plot.py)。上排两条线分别表示全部测试样本与干净条件下双方共同认对的样本；下排菱形为整个测试集上的准确率差值，圆点是图像任务的各个种子。语音与手势使用各自输入单位定义的噪声强度。修改脚本中的蓝灰色、线型或布局后，在仓库根目录运行以下命令更新矢量图和中文写作稿预览图。

```bash
python large_scale/paired_effects/plot.py large_scale/paired_effects/paired.csv paper/figures/paired-effects.pdf
pdftoppm -scale-to 1450 -singlefile -png paper/figures/paired-effects.pdf paper/figures/paired-effects
```
