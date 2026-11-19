"""Plot matched architecture contrasts in whole-event noise and after a noisy delay."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def plot(csv_path, dest):
    csv_path = Path(csv_path).resolve()
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    comparisons = {
        (row["dataset"], row["condition"]): row for row in rows
        if row["comparison"] == "DMP / GDSM"
    }
    tasks = [
        ("shd-matched", "SHD", ["0.0625", "0.125", "0.25"]),
        ("gesture", "DVS Gesture", ["0.5×", "1×", "3×"]),
        ("ssc-repair", "SSC", ["0.0625", "0.125", "0.25"]),
    ]
    fig = plt.figure(figsize=(10, 6), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.2, 1))
    axes = []
    for column, (task, title, ticks) in enumerate(tasks):
        ax = fig.add_subplot(grid[0, column], sharey=axes[0] if axes else None)
        axes.append(ax)
        conditions = sorted(
            (c for ds, c in comparisons if ds == task and c.startswith("overlay-")),
            key=lambda c: float(c.split("-")[1])
        )
        assert len(conditions) == len(ticks)
        for key, label, marker, color in [
            ("delta_pp", "All examples", "o", "#205d83"),
            ("shared_delta_pp", "Both correct when clean", "s", "#71808d")
        ]:
            values = [float(comparisons[task, c][key]) for c in conditions]
            ax.plot(range(len(ticks)), values, color=color, marker=marker, ms=4,
                    lw=1.5, label=label)
        ax.axhline(0, color="black", lw=.65)
        ax.set_xticks(range(len(ticks)), ticks)
        ax.set_title(title)
        ax.set_xlabel("Multiple of training density" if task == "gesture" else "Poisson rate per bin")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("GDSM − DMP (pp)")
    axes[-1].legend(fontsize=8, frameon=False)
    axes[0].text(0, 1.19, "a   Noise throughout the event", transform=axes[0].transAxes,
                 weight="bold", ha="left")

    with (Path(__file__).resolve().parents[2] / "paper/evidence/mnist.csv").open() as handle:
        images = list(csv.DictReader(handle))
    tail = [(name, [float(comparisons[key, "tail80"]["delta_pp"])]) for key, name in [
        ("shd-matched", "SHD"), ("gesture", "DVS Gesture"),
        ("ssc-repair", "SSC\nclean training"), ("ssc-archived", "SSC\nmixed training")
    ]]
    for task, title in [("smnist", "S-MNIST"), ("psmnist", "PS-MNIST")]:
        points = {
            (r["family"], int(r["seed"])): float(r["accuracy"]) for r in images
            if r["task"] == task and r["phase"] == "mixed" and r["metric"] == "last20"
            and r["name"] == "p784_poisson_L80"
        }
        tail.append((title, [100 * (points["commit", s] - points["dmp", s]) for s in (42, 43, 44)]))
    ax = fig.add_subplot(grid[1, :])
    ax.axhline(0, color="black", lw=.65)
    for i, (_, values) in enumerate(tail):
        if len(values) > 1:
            ax.scatter(i + np.linspace(-.13, .13, len(values)), values, s=22, color="#778795", zorder=2)
        ax.scatter(i, np.mean(values), marker="D", s=38, color="#205d83", zorder=3)
    ax.set_xlim(-.5, len(tail)-.5)
    ax.set_xticks(range(len(tail)), [name for name, _ in tail])
    ax.set_ylabel("GDSM − DMP (pp)")
    ax.set_title("b   Classification after 80 frames of background noise", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], marker="D", color="#205d83", label="Mean architecture difference")
    ax.scatter([], [], s=22, color="#778795", label="Individual MNIST training seeds")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -.25))
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    plot(sys.argv[1], sys.argv[2])
