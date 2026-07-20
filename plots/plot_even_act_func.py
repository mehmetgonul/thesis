from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RunConfig, parse_config
from src.data_manager import DataManager
from src.hash import make_hash

ARTIFACTS_DIR = ROOT / "artifacts"
SEEDS = [42]
PREPROCESS_CONFIGS = [
    ROOT / "configs" / "h2" / "preprocess" / "k=3.yaml",
    ROOT / "configs" / "h4" / "preprocess" / "k=3.yaml",
    ROOT / "configs" / "h6" / "preprocess" / "k=3.yaml",
]
NO_PREPROCESS_CONFIGS = [
    ROOT / "configs" / "h2" / "no_preprocess" / "k=3.yaml",
    ROOT / "configs" / "h4" / "no_preprocess" / "k=3.yaml",
    ROOT / "configs" / "h6" / "no_preprocess" / "k=3.yaml",
]
MAX_PLOT_POINTS = 5000


def curve_label(run_config: RunConfig) -> str:
    degree = run_config.activation_function.removeprefix("h")
    return rf"$H_{degree}$"


def curve_color(run_config: RunConfig) -> str:
    colors = {
        "h2": "tab:blue",
        "h4": "tab:orange",
        "h6": "tab:green",
    }
    return colors[run_config.activation_function]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_curve(config_path: Path, metric: str):
    run_config = parse_config(config_path)
    run_hash = make_hash(run_config=run_config)
    xs = []
    ys = []

    for seed in SEEDS:
        loss_path = ARTIFACTS_DIR / run_hash / "autoencoder" / f"seed_{seed}" / "loss.jsonl"
        if not loss_path.exists():
            print(f"missing: {loss_path}")
            continue

        rows = read_jsonl(loss_path)
        metric_rows = [row for row in rows if row.get(metric) is not None]
        if not metric_rows:
            print(f"missing metric '{metric}': {loss_path}")
            continue

        xs.append(np.asarray([row["alpha"] for row in metric_rows], dtype=float))
        ys.append(np.asarray([row[metric] for row in metric_rows], dtype=float))

    if not ys:
        return run_config, None

    n = min(len(y) for y in ys)
    x = xs[0][:n]
    y = np.vstack([curve[:n] for curve in ys])
    return run_config, (x, y.mean(axis=0), y.std(axis=0))


def plot_pca_baselines(ax, run_config: RunConfig) -> None:
    for rank, error in enumerate(DataManager(SEEDS[0], run_config).pca_errors(run_config.K), start=1):
        ax.axhline(
            error,
            color="0.55",
            linestyle="--",
            linewidth=1,
            alpha=0.35,
            label=f"PCA ranks 1..{run_config.K}" if rank == 1 else None,
        )


# done to speed up generation
def downsample_curve(x, mean, std):
    if len(x) <= MAX_PLOT_POINTS:
        return x, mean, std

    stride = int(np.ceil(len(x) / MAX_PLOT_POINTS))
    return x[::stride], mean[::stride], std[::stride]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="test_loss")
    parser.add_argument("--output", type=Path, default=ROOT / "plots" / "h2_h4_no_preprocess_k3.png")
    args = parser.parse_args()

    panel_groups = [
        ("(a) Preprocessed", PREPROCESS_CONFIGS),
        ("(b) Not preprocessed", NO_PREPROCESS_CONFIGS),
    ]
    style = {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }

    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(6.5, 2.8),
            sharex=True,
            layout="constrained",
        )

        total_plotted = 0
        for panel_index, (title, config_paths) in enumerate(panel_groups):
            ax = axes[panel_index]
            plotted = 0
            first_run_config = None

            for config_path in config_paths:
                run_config, curve = load_curve(config_path, args.metric)
                if curve is None:
                    continue

                x, mean, std = curve
                x, mean, std = downsample_curve(x, mean, std)
                color = curve_color(run_config)
                is_h6 = run_config.activation_function == "h6"

                ax.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=1.4 if is_h6 else 1.8,
                    alpha=0.85 if is_h6 else 1.0,
                    zorder=2 if is_h6 else 3,
                    label=curve_label(run_config),
                )
                ax.fill_between(
                    x,
                    np.clip(mean - std, np.finfo(float).tiny, None),
                    mean + std,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                    zorder=1,
                )
                plotted += 1
                total_plotted += 1
                if first_run_config is None:
                    first_run_config = run_config

            if first_run_config is not None:
                plot_pca_baselines(ax, first_run_config)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No test-loss data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            ax.set_title(title, fontweight="semibold")
            ax.set_xlabel(r"$\alpha$")
            ax.grid(True, alpha=0.25)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

            if panel_index == 0 and plotted:
                ax.legend(frameon=True, framealpha=0.9)

        if total_plotted == 0:
            plt.close(fig)
            raise SystemExit("No H2/H4/H6 K=3 test-loss curves found.")

        fig.supylabel("Test loss")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300)
        plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
