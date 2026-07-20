from __future__ import annotations

import argparse
import json
import re
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


# The ranges to show the sharp specializaiton in each activation functions
PLOT_SETTINGS = {
    "erf": {
        "x_interval": (500, 750),
        "y_interval": (0.061, 0.063),
    },
    "relu": {
        "x_interval": (450, 800),
        "y_interval": (0.075, 0.081),
    },
    "gelu": {
        "x_interval": (850, 1000),
        "y_interval": (0.062, 0.070),
    },
    "h1": {
        "x_interval": (400, 700),
        "y_interval": (0.061, 0.063),
    },
    "h3": {
        "x_interval": (150, 200),
        "y_interval": (0.061, 0.063),
    },
}

K = 3
SEEDS = [40, 41, 42]
DEFAULT_METRIC = "generalization_error"
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "init_k_compare_paired"
MAX_PLOT_POINTS = 5000

INITIALIZATION_STYLES = {
    "default": ("Default", "tab:blue"),
    "std": ("Std. 0.5", "tab:orange"),
    "uniform": ("Uniform", "tab:green"),
    "sphere_0.1": ("Unit sphere 0.1", "tab:red"),
    "sphere_0.5": ("Unit sphere 0.5", "tab:purple"),
    "sphere_1": ("Unit sphere 1", "tab:brown"),
    "sphere_2": ("Unit sphere 2", "tab:pink"),
    "sphere_5": ("Unit sphere 5", "tab:gray"),
}

ACTIVATION_LABELS = {
    "erf": "erf",
    "relu": "ReLU",
    "gelu": "GELU",
    "h1": r"$H_1$",
    "h3": r"$H_3$",
}

METRIC_LABELS = {
    "generalization_error": "Generalization error",
    "test_loss": "Test loss",
    "loss": "Training loss",
}


def initialization_key(config_path: Path, run_config: RunConfig) -> str:
    if config_path.parent.name == "internal_unit":
        return "default"
    if run_config.init_unit_sphere:
        return f"sphere_{run_config.unit_sphere_scale:g}"
    if run_config.init_uniform:
        return "uniform"
    return "std"


def initialization_style(config_path: Path, run_config: RunConfig) -> tuple[str, str]:
    key = initialization_key(config_path, run_config)
    if key in INITIALIZATION_STYLES:
        return INITIALIZATION_STYLES[key]
    return key.replace("_", " "), "tab:cyan"


def metric_sample_stride(run_config: RunConfig) -> int:
    expected_steps = max(1, int(run_config.alpha * run_config.D))
    return max(1, int(np.ceil(expected_steps / MAX_PLOT_POINTS)))


def read_metric_series(
    path: Path,
    metric: str,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    metric_key = f'"{metric}"'
    metric_index = 0
    last_metric_line = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if metric_key not in line:
                continue

            last_metric_line = line
            if metric_index % stride == 0:
                row = json.loads(line)
                value = row.get(metric)
                if value is not None:
                    xs.append(row["alpha"])
                    ys.append(value)
            metric_index += 1

    if last_metric_line is not None:
        row = json.loads(last_metric_line)
        value = row.get(metric)
        final_alpha = float(row["alpha"])
        if value is not None and (not xs or not np.isclose(xs[-1], final_alpha)):
            xs.append(final_alpha)
            ys.append(value)

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def load_curve(config_path: Path, metric: str):
    run_config = parse_config(config_path)
    run_hash = make_hash(run_config=run_config)
    stride = metric_sample_stride(run_config)
    xs = []
    ys = []

    for seed in SEEDS:
        loss_path = ARTIFACTS_DIR / run_hash / "autoencoder" / f"seed_{seed}" / "loss.jsonl"
        if not loss_path.exists():
            print(f"missing: {loss_path}")
            continue

        x, y = read_metric_series(loss_path, metric, stride)
        if not len(y):
            print(f"missing metric '{metric}': {loss_path}")
            continue

        xs.append(x)
        ys.append(y)

    if not ys:
        return run_config, None

    common_start = max(x[0] for x in xs)
    common_end = min(x[-1] for x in xs)
    reference_x = xs[0][(xs[0] >= common_start) & (xs[0] <= common_end)]
    if not len(reference_x):
        print(f"incompatible alpha ranges: {config_path}")
        return run_config, None

    aligned_ys = []
    for x, y in zip(xs, ys):
        if len(x) == len(reference_x) and np.allclose(x, reference_x):
            aligned_ys.append(y)
        else:
            aligned_ys.append(np.interp(reference_x, x, y))

    values = np.vstack(aligned_ys)
    return run_config, (
        reference_x,
        values.mean(axis=0),
        values.std(axis=0),
        len(values),
    )


def plot_pca_baselines(
    ax,
    run_config: RunConfig,
    *,
    include_label: bool,
) -> None:
    for rank, error in enumerate(
        DataManager(SEEDS[0], run_config).pca_errors(run_config.K),
        start=1,
    ):
        ax.axhline(
            error,
            color="0.55",
            linestyle="--",
            linewidth=1,
            alpha=0.55,
            zorder=0,
            label=(
                f"PCA ranks 1–{run_config.K}"
                if include_label and rank == 1
                else None
            ),
        )


def sphere_scale(config_path: Path) -> float:
    match = re.search(r"scale=([^_]+)_k=", config_path.name)
    if match is None:
        return float("inf")
    return float(match.group(1))


def config_paths_for(activation: str) -> list[Path]:
    config_dir = ROOT / "configs" / activation
    initialization_dir = config_dir / "initialization"
    paths = []
    candidate_paths = [
        config_dir / "internal_unit" / f"k={K}.yaml",
        initialization_dir / f"{activation}_cifar_init_std=0.5_k={K}.yaml",
        initialization_dir / f"{activation}_cifar_init_uniform_k={K}.yaml",
    ]

    for config_path in candidate_paths:
        if config_path.exists():
            paths.append(config_path)
        else:
            print(f"missing config: {config_path}")

    paths.extend(
        sorted(
            initialization_dir.glob(
                f"{activation}_cifar_init_unit_sphere_scale=*_k={K}.yaml"
            ),
            key=sphere_scale,
        )
    )
    return paths


def validate_plot_settings() -> None:
    for activation, settings in PLOT_SETTINGS.items():
        for interval_name in ("x_interval", "y_interval"):
            interval = settings[interval_name]
            if len(interval) != 2 or interval[0] >= interval[1]:
                raise ValueError(
                    f"{activation} {interval_name} must contain two increasing values"
                )


def right_axis_decimals(y_interval: tuple[float, float]) -> int:
    return 4 if y_interval[1] - y_interval[0] < 0.01 else 3


def save_activation_plot(
    activation: str,
    metric: str,
    output_dir: Path,
) -> Path | None:
    curves = []
    for config_path in config_paths_for(activation):
        run_config, curve = load_curve(config_path, metric)
        if curve is None:
            continue

        label, color = initialization_style(config_path, run_config)
        x, mean, std, seed_count = curve
        curves.append((run_config, label, color, x, mean, std, seed_count))

    if not curves:
        print(f"no curves found: {activation}, K={K}")
        return None

    settings = PLOT_SETTINGS[activation]
    x_interval = settings["x_interval"]
    y_interval = settings["y_interval"]
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    style = {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    }

    with plt.rc_context(style):
        fig, (full_ax, interval_ax) = plt.subplots(
            1,
            2,
            figsize=(8.0, 3.1),
            layout="constrained",
        )

        first_run_config = curves[0][0]
        if metric in {"generalization_error", "test_loss"}:
            plot_pca_baselines(full_ax, first_run_config, include_label=True)
            plot_pca_baselines(interval_ax, first_run_config, include_label=False)

        for _run_config, label, color, x, mean, std, _seed_count in curves:
            full_ax.plot(
                x,
                mean,
                color=color,
                linewidth=1.8,
                label=label,
                zorder=2,
            )
            full_ax.fill_between(
                x,
                np.clip(mean - std, np.finfo(float).tiny, None),
                mean + std,
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )
            interval_ax.plot(
                x,
                mean,
                color=color,
                linewidth=1.8,
                zorder=2,
            )

        full_ax.set_title("(a) Full range", fontweight="semibold")
        interval_ax.set_title("(b) Selected interval", fontweight="semibold")
        for ax in (full_ax, interval_ax):
            ax.set_xlabel(r"$\alpha$")
            ax.grid(True, alpha=0.25)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        full_ax.set_xlim(left=0)
        full_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        full_ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        interval_ax.set_xlim(*x_interval)
        interval_ax.set_ylim(*y_interval)
        interval_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        interval_ax.yaxis.set_major_formatter(
            FormatStrFormatter(f"%.{right_axis_decimals(y_interval)}f")
        )

        full_ax.legend(
            loc="upper right",
            ncols=2,
            frameon=True,
            framealpha=0.92,
            columnspacing=0.8,
            handlelength=1.8,
            labelspacing=0.35,
            borderpad=0.4,
        )
        fig.supylabel(metric_label)
        fig.suptitle(
            f"Initialization comparison: {ACTIVATION_LABELS[activation]}, $K={K}$",
            fontsize=10,
            fontweight="semibold",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"init_k_compare_{activation}_k{K}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    print(f"saved: {output_path}")
    return output_path


def main() -> None:
    validate_plot_settings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        choices=tuple(PLOT_SETTINGS),
        default=list(PLOT_SETTINGS),
    )
    args = parser.parse_args()

    for activation in args.activations:
        save_activation_plot(activation, args.metric, args.output_dir)


if __name__ == "__main__":
    main()
