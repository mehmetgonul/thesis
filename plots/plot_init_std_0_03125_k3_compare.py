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

from src.config import RunConfig, parse_config
from src.data_manager import DataManager
from src.hash import make_hash

ARTIFACTS_DIR = ROOT / "artifacts"
SEEDS = [40, 41, 42]
DEFAULT_CONFIGS = [
    ROOT / "configs" / "erf" / "initialization" / "erf_cifar_init_std=0.03125_k=3.yaml",
    ROOT / "configs" / "erf" / "initialization" / "erf_cifar_init_unit_sphere_scale=1_k=3.yaml",
    ROOT / "configs" / "relu" / "initialization" / "relu_cifar_init_std=0.03125_k=3.yaml",
    ROOT / "configs" / "relu" / "initialization" / "relu_cifar_init_unit_sphere_scale=1_k=3.yaml",
    ROOT / "configs" / "gelu" / "initialization" / "gelu_cifar_init_std=0.03125_k=3.yaml",
    ROOT / "configs" / "gelu" / "initialization" / "gelu_cifar_init_unit_sphere_scale=1_k=3.yaml",
    ROOT / "configs" / "h1" / "initialization" / "h1_cifar_init_std=0.03125_k=3.yaml",
    ROOT / "configs" / "h1" / "initialization" / "h1_cifar_init_unit_sphere_scale=1_k=3.yaml",
    ROOT / "configs" / "h3" / "initialization" / "h3_cifar_init_std=0.03125_k=3.yaml",
    ROOT / "configs" / "h3" / "initialization" / "h3_cifar_init_unit_sphere_scale=1_k=3.yaml",
]
MAX_PLOT_POINTS = 5000


def curve_label(run_config: RunConfig) -> str:
    if run_config.init_unit_sphere:
        return f"{run_config.activation_function} unit sphere scale={run_config.unit_sphere_scale:g}"
    if run_config.init_uniform:
        return f"{run_config.activation_function} uniform"
    return f"{run_config.activation_function} std={run_config.init_std:g}"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_curves(config_path: Path, metric: str):
    run_config = parse_config(config_path)
    run_hash = make_hash(run_config=run_config)
    xs = []
    ys = []

    for seed in SEEDS:
        loss_path = ARTIFACTS_DIR / run_hash / "autoencoder" / f"seed_{seed}" / "loss.jsonl"
        if not loss_path.exists():
            print(f"missing: {loss_path}")
            print(run_config)
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
    return run_config, (x, y.mean(axis=0), y.std(axis=0), len(ys))


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
            alpha=0.35,
            label=f"PCA ranks 1..{run_config.K}" if rank == 1 else None,
        )


def downsample_curve(x, mean, std):
    if len(x) <= MAX_PLOT_POINTS:
        return x, mean, std

    stride = int(np.ceil(len(x) / MAX_PLOT_POINTS))
    return x[::stride], mean[::stride], std[::stride]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--metric", default="generalization_error")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "plots" / "init_std_0_03125_k3_compare.png",
    )
    args = parser.parse_args()

    run_configs = [parse_config(config_path) for config_path in args.configs]

    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = 0
    for config_path in args.configs:
        run_config, curve = load_curves(config_path, args.metric)
        if curve is None:
            continue

        x, mean, std, _seed_count = curve
        x, mean, std = downsample_curve(x, mean, std)

        ax.plot(x, mean, linewidth=2, label=curve_label(run_config))
        ax.fill_between(
            x,
            np.clip(mean - std, np.finfo(float).tiny, None),
            mean + std,
            alpha=0.18,
            linewidth=0,
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit("No init_std=0.03125 or unit sphere K=3 autoencoder curves found.")

    plot_pca_baselines(ax, run_configs)

    ax.set_xlabel(r"$\alpha$", fontsize=14)
    ax.set_ylabel("Generalization error", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
