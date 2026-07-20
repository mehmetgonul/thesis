from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import parse_config
from src.hash import make_hash

ARTIFACTS_DIR = ROOT / "artifacts"
SEEDS = [40, 41, 42]
INTERNAL_UNITS = (3, 4, 10)
DEFAULT_CONFIGS = [
    ROOT / "configs" / "erf",
    ROOT / "configs" / "relu",
    ROOT / "configs" / "gelu",
    ROOT / "configs" / "h1",
    ROOT / "configs" / "h3",
    ROOT / "configs" / "h5",
]

ACTIVATION_LABELS = {
    "erf": "erf",
    "relu": "ReLU",
    "gelu": "GELU",
    "h1": r"$H_1$",
    "h3": r"$H_3$",
    "h5": r"$H_5$",
}

ACTIVATION_MARKERS = {
    "erf": "o",
    "relu": "s",
    "gelu": "^",
    "h1": "D",
    "h3": "v",
    "h5": "P",
}

ACTIVATION_COLORS = {
    "erf": "tab:blue",
    "relu": "tab:orange",
    "gelu": "tab:green",
    "h1": "tab:red",
    "h3": "tab:purple",
    "h5": "tab:brown",
}


@lru_cache(maxsize=None)
def read_metrics(path: Path) -> dict | None:
    if not path.exists():
        print(f"missing: {path}")
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_curves(config_path: Path, metric: str):
    config_dir = config_path if config_path.is_dir() else config_path.parent
    xs = []
    ys = []

    if config_path.is_dir():
        internal_config_paths = [
            config_dir / "internal_unit" / f"k={k}.yaml"
            for k in INTERNAL_UNITS
        ]
    else:
        internal_config_paths = [config_path]

    for internal_config_path in internal_config_paths:
        if not internal_config_path.exists():
            continue

        run_config = parse_config(internal_config_path)
        run_hash = make_hash(run_config=run_config)
        seed_values = []

        for seed in SEEDS:
            metrics_path = (
                ARTIFACTS_DIR
                / run_hash
                / "probes"
                / "linear"
                / f"seed_{seed}"
                / "metrics.json"
            )
            metrics = read_metrics(metrics_path)
            if metrics is None:
                continue

            value = metrics.get(metric)
            if value is None and metric == "test_accuracy_at_1":
                value = metrics.get("test_accuracy")
            if value is None:
                print(f"missing metric '{metric}': {metrics_path}")
                continue

            seed_values.append(float(value))

        if seed_values:
            xs.append(run_config.K)
            ys.append(seed_values)

    if not ys:
        return config_dir.name, None

    order = np.argsort(xs)
    x = np.asarray(xs, dtype=float)[order]
    mean = np.asarray([np.mean(y) for y in ys], dtype=float)[order]
    std = np.asarray([np.std(y) for y in ys], dtype=float)[order]
    return run_config.activation_function, (x, mean, std, max(len(y) for y in ys))


def plot_metric(
    ax,
    config_paths: list[Path],
    metric: str,
    title: str,
    *,
    show_legend: bool,
) -> int:
    plotted = 0
    plotted_ks = set()

    for config_path in config_paths:
        activation, curve = load_curves(config_path, metric)
        if curve is None:
            continue

        x, mean, std, _seed_count = curve
        plotted_ks.update(int(k) for k in x)
        ax.errorbar(
            x,
            mean,
            yerr=std,
            color=ACTIVATION_COLORS.get(activation),
            marker=ACTIVATION_MARKERS.get(activation, "o"),
            markersize=5,
            linewidth=1.8,
            elinewidth=1.1,
            capsize=3,
            capthick=1.1,
            label=ACTIVATION_LABELS.get(activation, activation),
        )
        plotted += 1

    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "No classifier metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title(title, fontweight="semibold")
    ax.set_xlabel(r"Internal units, $K$")
    if plotted_ks:
        ax.set_xticks(sorted(plotted_ks))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(y=0.08)
    if show_legend and plotted:
        ax.legend(ncols=2, frameon=True, framealpha=0.9)
    return plotted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "plots" / "classifier_compare.png",
    )
    args = parser.parse_args()

    style = {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8, 3.1),
            sharex=True,
            layout="constrained",
        )
        acc1_plotted = plot_metric(
            axes[0],
            args.configs,
            "test_accuracy_at_1",
            "(a) Top-1 accuracy",
            show_legend=True,
        )
        acc3_plotted = plot_metric(
            axes[1],
            args.configs,
            "test_accuracy_at_3",
            "(b) Top-3 accuracy",
            show_legend=False,
        )
        fig.supylabel("Final test accuracy")

        if not acc1_plotted and not acc3_plotted:
            plt.close(fig)
            raise SystemExit("No classifier accuracy metrics found.")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300)
        plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
