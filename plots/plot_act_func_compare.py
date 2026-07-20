from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import parse_config
from src.data_manager import DataManager
from src.hash import make_hash

ARTIFACTS_DIR = ROOT / "artifacts"
SEEDS = [40, 41, 42]
ACTIVATIONS = ["erf", "relu", "gelu", "h1", "h2", "h3", "h4", "h5"]
ACTIVATION_LABELS = {
    "erf": "erf",
    "relu": "ReLU",
    "gelu": "GELU",
    "h1": r"$H_1$",
    "h2": r"$H_2$",
    "h3": r"$H_3$",
    "h4": r"$H_4$",
    "h5": r"$H_5$",
}
KS = [3, 4, 10]
MAX_PLOT_POINTS = 5000 # arbitrary, increasing doesn't change resoultion much but decreases sped of gen.
Y_AXIS_MAX = 0.22


def read_metric_series(path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    metric_key = f'"{metric}"'

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if metric_key not in line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"malformed JSON skipped: {path}:{line_number}")
                continue

            value = row.get(metric)
            if value is not None:
                xs.append(row["alpha"])
                ys.append(value)

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def load_curves(config_path: Path, metric: str):
    run_config = parse_config(config_path)
    run_hash = make_hash(run_config=run_config)
    xs = []
    ys = []

    for seed in SEEDS:
        loss_path = ARTIFACTS_DIR / run_hash / "autoencoder" / f"seed_{seed}" / "loss.jsonl"
        if not loss_path.exists():
            print(f"missing: {loss_path}")
            continue

        x, y = read_metric_series(loss_path, metric)
        if not len(y):
            print(f"missing metric '{metric}': {loss_path}")
            continue

        xs.append(x)
        ys.append(y)

    if not ys:
        return run_config.activation_function, None

    n = min(len(y) for y in ys)
    x = xs[0][:n]
    y = np.vstack([curve[:n] for curve in ys])
    return run_config.activation_function, (x, y.mean(axis=0), y.std(axis=0), len(ys))


def downsample_curve(x, mean, std):
    if len(x) <= MAX_PLOT_POINTS:
        return x, mean, std

    stride = int(np.ceil(len(x) / MAX_PLOT_POINTS))
    return x[::stride], mean[::stride], std[::stride]


def plot_pca_baselines(ax, run_configs):
    if not run_configs:
        return

    run_config = run_configs[0]
    for rank, error in enumerate(DataManager(SEEDS[0], run_config).pca_errors(run_config.K), start=1):
        ax.axhline(
            error,
            color="0.55",
            linestyle="--",
            linewidth=1,
            alpha=0.55,
            zorder=0,
            label=rf"PCA ranks $1,\ldots,{run_config.K}$" if rank == 1 else None,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="*", default=None)
    parser.add_argument("--metric", default="generalization_error")
    parser.add_argument("--output", type=Path, default=ROOT / "plots" / "act_func_compare.png")
    args = parser.parse_args()

    config_groups = []
    if args.configs:
        config_groups.append((None, args.configs))
    else:
        for k in KS:
            config_paths = []
            for activation in ACTIVATIONS:
                config_paths.append(ROOT / "configs" / activation / "internal_unit" / f"k={k}.yaml")
            config_groups.append((k, config_paths))

    metric_labels = {
        "generalization_error": "Generalization error",
        "test_loss": "Test loss",
        "loss": "Training loss",
    }
    metric_label = metric_labels.get(args.metric, args.metric.replace("_", " ").title())
    style = {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }

    with plt.rc_context(style):
        fig, axes = plt.subplots(
            len(config_groups),
            1,
            figsize=(8, 5 * len(config_groups)),
            sharex=False,
            sharey=False,
            layout="constrained",
            squeeze=False,
        )
        axes = axes[:, 0]
        total_plotted = 0

        for panel_index, (k, config_paths) in enumerate(config_groups):
            ax = axes[panel_index]
            run_configs = [parse_config(config_path) for config_path in config_paths]
            plotted = 0

            for config_path in config_paths:
                activation, curve = load_curves(config_path, args.metric)
                if curve is None:
                    continue

                x, mean, std, _seed_count = curve
                x, mean, std = downsample_curve(x, mean, std)
                (line,) = ax.plot(
                    x,
                    mean,
                    linewidth=1.8,
                    label=ACTIVATION_LABELS.get(activation, activation),
                    zorder=2,
                )
                ax.fill_between(
                    x,
                    np.clip(mean - std, np.finfo(float).tiny, None),
                    mean + std,
                    color=line.get_color(),
                    alpha=0.14,
                    linewidth=0,
                    zorder=1,
                )
                plotted += 1
                total_plotted += 1

            if plotted:
                plot_pca_baselines(ax, run_configs)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No generalization-error data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            if len(config_groups) == 1:
                title = "Activation comparison"
                if k is not None:
                    title = rf"{title}, $K={k}$"
            else:
                panel_letter = chr(ord("a") + panel_index)
                title = rf"({panel_letter}) $K={k}$"

            ax.set_title(title, fontweight="semibold")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(metric_label)
            ax.set_yscale("log")
            ax.set_ylim(top=Y_AXIS_MAX)
            ax.grid(True, alpha=0.25)
            ax.set_axisbelow(True)

            if plotted:
                ax.legend(
                    loc="upper left",
                    ncols=1,
                    frameon=True,
                    framealpha=0.9,
                    handlelength=2.0,
                )

        if total_plotted == 0:
            plt.close(fig)
            raise SystemExit("No activation-comparison curves found.")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300)
        plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
