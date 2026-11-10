# GDSM / DMP / FSNN：DVS Gesture

论文中 GDSM 对应命令行参数 `--family commit`。当前完整研究使用 100 帧手势、901 段训练、176 段验证、264 段官方测试；训练受试者 1–19，验证 20–23，测试 24–29。

在根目录安装 PyTorch 2.0.0、torchvision 0.15.1 和 `vision_tasks/requirements.txt`。CPU 可使用 `--backend cpu_scan`，GPU 可使用 `--backend cupy`（需要对应 CuPy）。

## 数据

```bash
python -m vision_tasks.download_gesture --output data/ibmGestureTrain.tar.gz
python -m vision_tasks.prepare_gesture --archive data/ibmGestureTrain.tar.gz --output data/gesture-train --steps 100 --subjects
```

测试归档为 Tonic 公共镜像 `https://api.figshare.com/v2/file/download/38020584`，使用相同准备程序和 `--steps 100 --subjects` 转换到 `data/gesture-test`。

## 干净训练与混合训练

```bash
for family in fsnn dmp commit; do
  python -m vision_tasks.run --task gesture --family "$family" \
    --data data/gesture-train --output "runs/gesture-clean/$family" \
    --steps 100 --epochs 50 --train-limit 0 --val-limit 0 \
    --batch-size 4 --threads 1 --backend cpu_scan --cache-data
done
python -m vision_tasks.final_study train \
  --data data/gesture-train --weights runs/gesture-clean \
  --output runs/gesture-mixed --device cpu
python -m vision_tasks.final_study evaluate \
  --test data/gesture-test --weights runs/gesture-clean \
  --finetuned runs/gesture-mixed --output runs/gesture-evaluation --device cpu
```

混合训练固定 20 轮，三模型接收相同增强输入与分类损失。静音/Poisson 间隔长度为 20/40/80，位置在中途或尾部；评估还包括 3 倍噪声强度。编码依赖完整事件片段边界。命令行支持恢复训练，`--epochs` 表示累计目标轮数。

CPU 优化、原论文实现差异和早期探索记录保留在 [HISTORY.md](HISTORY.md)；其中的试跑进度及时间预算属于历史记录。完整最终协议与结果以 [论文证据](../paper/evidence/gesture-report.md) 为准。
