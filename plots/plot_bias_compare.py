from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RunConfig, parse_config
from src.data_manager import DataManager
from src.hash import make_hash

ARTIFACTS_DIR = ROOT / "artifacts"
SEEDS = [40, 41, 42]
DEFAULT_CONFIGS = [
    ROOT / "configs" / "relu" / "bias" / "relu_cifar_bias.yaml",
    ROOT / "configs" / "relu" / "bias" / "no_bias.yaml",
    ROOT / "configs" / "gelu" / "bias" / "gelu_cifar_bias.yaml",
    ROOT / "configs" / "gelu" / "bias" / "no_bias.yaml",
]
MAX_PLOT_POINTS = 5000


def curve_label(run_config: RunConfig) -> str:
    bias_label = "with bias" if run_config.bias else "without bias"
    return f"{activation_label(run_config)}, {bias_label}"


def activation_label(run_config: RunConfig) -> str:
    labels = {
        "relu": "ReLU",
        "gelu": "GELU",
        "erf": "erf",
    }
    return labels.get(run_config.activation_function, run_config.activation_function)


def curve_style(run_config: RunConfig) -> dict:
    colors = {
        "relu": "tab:blue",
        "gelu": "tab:orange",
    }
    return {
        "color": colors.get(run_config.activation_function),
        "linestyle": "-" if run_config.bias else "--",
    }


def uncertainty_band_style(run_config: RunConfig) -> dict:
    styles = {
        ("relu", True): ("#1f77b4", 0.22),
        ("relu", False): ("#9ecae1", 0.30),
        ("gelu", True): ("#d95f02", 0.22),
        ("gelu", False): ("#7f2704", 0.36),
    }
    color, alpha = styles.get(
        (run_config.activation_function, run_config.bias),
        (curve_style(run_config)["color"], 0.22 if run_config.bias else 0.30),
    )
    return {
        "color": color,
        "alpha": alpha,
    }


def read_metric_series(path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    metric_key = f'"{metric}"'

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if metric_key not in line:
                continue

            row = json.loads(line)
            value = row.get(metric)
            if value is not None:
                xs.append(row["alpha"])
                ys.append(value)

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


@lru_cache(maxsize=None)
def read_bias_data(
    path: Path,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
    xs = []
    magnitudes = []
    last_line = None

    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue

            last_line = line
            if row_index % stride != 0:
                continue

            row = json.loads(line)
            bias_values = row.get("bias_values")
            if bias_values is not None:
                xs.append(row["alpha"])
                magnitudes.append(np.linalg.norm(bias_values))

    if last_line is None:
        return None

    last_row = json.loads(last_line)
    bias_values = last_row.get("bias_values")
    if bias_values is None:
        return None

    final_alpha = float(last_row["alpha"])
    final_values = np.asarray(bias_values, dtype=float)
    if not xs or final_alpha != xs[-1]:
        xs.append(final_alpha)
        magnitudes.append(np.linalg.norm(final_values))

    return (
        np.asarray(xs, dtype=float),
        np.asarray(magnitudes, dtype=float),
        final_alpha,
        final_values,
    )


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
        return run_config, None

    n = min(len(y) for y in ys)
    x = xs[0][:n]
    y = np.vstack([curve[:n] for curve in ys])
    return run_config, (x, y.mean(axis=0), y.std(axis=0), len(ys))


def bias_sample_stride(run_config: RunConfig) -> int:
    expected_steps = max(1, int(run_config.alpha * run_config.D))
    return max(1, int(np.ceil(expected_steps / MAX_PLOT_POINTS)))


def load_bias_magnitudes(config_path: Path):
    run_config = parse_config(config_path)
    if not run_config.bias:
        return run_config, None

    run_hash = make_hash(run_config=run_config)
    stride = bias_sample_stride(run_config)
    xs = []
    magnitudes = []

    for seed in SEEDS:
        loss_path = ARTIFACTS_DIR / run_hash / "autoencoder" / f"seed_{seed}" / "loss.jsonl"
        if not loss_path.exists():
            print(f"missing: {loss_path}")
            continue

        bias_data = read_bias_data(loss_path, stride)
        if bias_data is None:
            print(f"missing bias values: {loss_path}")
            continue

        x, magnitude, _final_alpha, _final_values = bias_data
        xs.append(x)
        magnitudes.append(magnitude)

    if not magnitudes:
        return run_config, None

    n = min(len(values) for values in magnitudes)
    x = xs[0][:n]
    values = np.vstack([seed_values[:n] for seed_values in magnitudes])
    return run_config, (x, values.mean(axis=0), values.std(axis=0), len(values))


def load_final_bias_values(config_path: Path):
    run_config = parse_config(config_path)
    if not run_config.bias:
        return run_config, None

    run_hash = make_hash(run_config=run_config)
    stride = bias_sample_stride(run_config)
    final_alphas = []
    final_values = []

    for seed in SEEDS:
        loss_path = ARTIFACTS_DIR / run_hash / "autoencoder" / f"seed_{seed}" / "loss.jsonl"
        if not loss_path.exists():
            print(f"missing: {loss_path}")
            continue

        bias_data = read_bias_data(loss_path, stride)
        if bias_data is None:
            print(f"missing bias values: {loss_path}")
            continue

        _x, _magnitude, final_alpha, bias_values = bias_data
        final_alphas.append(final_alpha)
        final_values.append(bias_values)

    if not final_values:
        return run_config, None

    n = min(len(values) for values in final_values)
    values = np.vstack([seed_values[:n] for seed_values in final_values])
    return run_config, (
        values.mean(axis=0),
        values.std(axis=0),
        min(final_alphas),
        len(final_values),
    )


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
            label=f"PCA ranks 1..{run_config.K}" if rank == 1 else None,
        )


def downsample_curve(x, mean, std):
    if len(x) <= MAX_PLOT_POINTS:
        return x, mean, std

    stride = int(np.ceil(len(x) / MAX_PLOT_POINTS))
    return x[::stride], mean[::stride], std[::stride]


def plot_bias_magnitudes(ax, config_paths: list[Path]) -> None:
    plotted = 0
    for config_path in config_paths:
        run_config, curve = load_bias_magnitudes(config_path)
        if curve is None:
            continue

        x, mean, std, _seed_count = curve
        x, mean, std = downsample_curve(x, mean, std)

        ax.plot(
            x,
            mean,
            linewidth=1.8,
            label=activation_label(run_config),
        )
        ax.fill_between(
            x,
            np.clip(mean - std, 0.0, None),
            mean + std,
            alpha=0.18,
            linewidth=0,
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit("No encoder bias values found.")

    ax.set_title(r"(a) Bias magnitude", fontweight="semibold")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"Encoder bias norm, $\Vert b\Vert_2$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def plot_final_bias_values(ax, config_paths: list[Path]) -> None:
    series = []
    for config_path in config_paths:
        run_config, final_bias = load_final_bias_values(config_path)
        if final_bias is None:
            continue

        mean, std, final_alpha, seed_count = final_bias
        series.append((run_config, mean, std, final_alpha, seed_count))

    if not series:
        raise SystemExit("No encoder bias values found.")

    bar_width = 0.8 / len(series)
    max_units = max(len(mean) for _, mean, _, _, _ in series)

    for series_index, (run_config, mean, std, final_alpha, seed_count) in enumerate(series):
        unit_indices = np.arange(1, len(mean) + 1)
        offset = (series_index - (len(series) - 1) / 2) * bar_width
        ax.bar(
            unit_indices + offset,
            mean,
            width=bar_width,
            yerr=std,
            capsize=3,
            label=activation_label(run_config),
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )

    final_alphas = {final_alpha for _, _, _, final_alpha, _ in series}
    title = "(b) Final bias values"
    if len(final_alphas) == 1:
        final_alpha = next(iter(final_alphas))
        title = rf"{title} ($\alpha={final_alpha:g}$)"

    ax.axhline(0.0, color="0.35", linewidth=1)
    ax.set_title(title, fontweight="semibold")
    ax.set_xlabel(r"Hidden unit, $i$")
    ax.set_ylabel(r"Encoder bias, $b_i$")
    ax.set_xticks(np.arange(1, max_units + 1))
    ax.grid(axis="y", alpha=0.25)


def save_bias_summary_plot(config_paths: list[Path], output_path: Path) -> None:
    style = {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(6.5, 2.8),
            layout="constrained",
        )
        plot_bias_magnitudes(axes[0], config_paths)
        plot_final_bias_values(axes[1], config_paths)
        for ax in axes:
            ax.set_axisbelow(True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    print(f"saved: {output_path}")


def save_learning_curve_plot(
    config_paths: list[Path],
    metric: str,
    output_path: Path,
) -> None:
    style = {
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(
            figsize=(6.5, 3.8),
            layout="constrained",
        )

        plotted = 0
        plotted_run_configs = []
        for config_path in config_paths:
            run_config, curve = load_curves(config_path, metric)
            if curve is None:
                continue

            x, mean, std, _seed_count = curve
            x, mean, std = downsample_curve(x, mean, std)
            plot_style = curve_style(run_config)

            ax.plot(
                x,
                mean,
                linewidth=1.8,
                label=curve_label(run_config),
                **plot_style,
            )
            ax.fill_between(
                x,
                np.clip(mean - std, np.finfo(float).tiny, None),
                mean + std,
                linewidth=0,
                **uncertainty_band_style(run_config),
            )
            plotted += 1
            plotted_run_configs.append(run_config)

        if plotted == 0:
            plt.close(fig)
            raise SystemExit("No autoencoder curves found.")

        if metric in {"generalization_error", "test_loss"}:
            plot_pca_baselines(ax, plotted_run_configs)

        metric_labels = {
            "generalization_error": "Generalization error",
            "test_loss": "Test loss",
            "loss": "Training loss",
        }
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(metric_labels.get(metric, metric.replace("_", " ").title()))
        ax.set_yscale("log")
        if metric == "test_loss":
            decimal_formatter = FormatStrFormatter("%.2f")
            ax.yaxis.set_major_formatter(decimal_formatter)
            ax.yaxis.set_minor_formatter(decimal_formatter)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper right",
            ncol=2,
            frameon=False,
            columnspacing=1.2,
            handlelength=2.6,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--metric", default="test_loss")
    parser.add_argument("--output", type=Path, default=ROOT / "plots" / "bias_compare.png")
    parser.add_argument("--bias-summary-output", type=Path)
    args = parser.parse_args()

    output_suffix = args.output.suffix or ".png"
    bias_summary_output = args.bias_summary_output or args.output.with_name(
        f"{args.output.stem}_bias_summary{output_suffix}"
    )

    save_learning_curve_plot(args.configs, args.metric, args.output)
    save_bias_summary_plot(args.configs, bias_summary_output)


if __name__ == "__main__":
    main()
