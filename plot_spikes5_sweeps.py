"""Save five-spike width sweeps.

Run from the project environment with: python plot_spikes5_sweeps.py
Add --order-parameters to also save six-panel histories for each activation/width.
Outputs are grouped by activation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.config import parse_config, resolve_run_modes
from src.hash import make_hash


ROOT = Path(__file__).resolve().parent
# ACTIVATIONS = {
#     "erf": "ERF", "relu": "ReLU", "linear": "Linear", "gelu": "GELU",
# }
# WIDTHS = (3, 4, 5, 6, 7)
ACTIVATIONS = {
    "gelu": "GELU",
}
WIDTHS = (3, 4, 5, 6, 7)
COLORS = dict(zip(WIDTHS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")))
ORDER_PARAMETERS = ("Q0", "R0", "T0", "Q1", "R1", "T1")
SYMMETRIC_PARAMETERS = {"Q0", "Q1", "T0", "T1"}


def sweep_config_path(activation: str, width: int) -> Path:
    return ROOT / "configs" / activation / f"spiked_wishart_5_k={width}.yaml"


def read_curve(path: Path) -> tuple[list[float], list[float]]:
    """Read recorded GE points, skipping intermediate training-loss rows."""
    alphas: list[float] = []
    errors: list[float] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if '"generalization_error"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                # A running trainer may not have finished writing its last row.
                if not line.endswith("\n"):
                    print(f"Skipping unfinished final row: {path}")
                    break
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from error

            if row.get("generalization_error") is None:
                continue
            alpha = float(row["alpha"])
            error = float(row["generalization_error"])
            if not math.isfinite(alpha) or not math.isfinite(error):
                raise ValueError(f"Non-finite measurement at {path}:{line_number}")
            if alphas and alpha <= alphas[-1]:
                raise ValueError(f"Alpha must increase at {path}:{line_number}")
            alphas.append(alpha)
            errors.append(error)
    return alphas, errors


def read_eigenvalues(run_dir: Path, dimension: int) -> np.ndarray | None:
    """Read the covariance spectrum from JSON metadata or a legacy final state."""
    metadata_path = run_dir / "ode_meta.json"
    eigenvalues = None
    source = metadata_path
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "eigenvalues" in metadata:
            eigenvalues = np.asarray(metadata["eigenvalues"], dtype=float)

    # Existing runs retain their spectra in the old final-state artifact.
    state_path = run_dir / "ode_state.npz"
    if eigenvalues is None and state_path.is_file():
        source = state_path
        with np.load(state_path) as state:
            eigenvalues = np.asarray(state["eigenvalues"], dtype=float)

    if eigenvalues is None:
        return None
    if eigenvalues.shape != (dimension,) or not np.isfinite(eigenvalues).all():
        raise ValueError(f"Invalid covariance spectrum in {source}")
    return np.sort(eigenvalues)


def pca_reference_errors(eigenvalues: np.ndarray, spike_count: int) -> dict[int, float]:
    """Reference ranks follow the spike count, independently of hidden width K."""
    dimension = len(eigenvalues)
    return {
        rank: 0.5 * float(eigenvalues[:-rank].sum()) / dimension
        for rank in range(1, min(spike_count, dimension) + 1)
    }


def draw_pca_references(ax, errors: dict[int, float]) -> None:
    for index, error in enumerate(errors.values()):
        ax.axhline(
            error, color="gray", linestyle=":", linewidth=1, alpha=0.7,
            zorder=1, label="PCA reference errors" if index == 0 else "_nolegend_",
        )


def read_order_parameters(
    path: Path, width: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read whole-alpha matrix histories, ignoring loss-only and legacy rows."""
    alphas: list[float] = []
    histories: dict[str, list[np.ndarray]] = {name: [] for name in ORDER_PARAMETERS}
    markers = tuple(f'"{name}"' for name in ORDER_PARAMETERS)
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not any(marker in line for marker in markers):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                if not line.endswith("\n"):
                    print(f"Skipping unfinished final row: {path}")
                    break
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from error

            missing = [name for name in ORDER_PARAMETERS if name not in row]
            if missing:
                raise ValueError(
                    f"Missing order parameters {', '.join(missing)} at {path}:{line_number}"
                )
            alpha = float(row["alpha"])
            if not math.isfinite(alpha) or (alphas and alpha <= alphas[-1]):
                raise ValueError(f"Alpha must be finite and increase at {path}:{line_number}")
            for name in ORDER_PARAMETERS:
                values = np.asarray(row[name], dtype=float)
                if values.shape != (width, width) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {name} matrix at {path}:{line_number}")
                histories[name].append(values)
            alphas.append(alpha)

    return np.asarray(alphas), {
        name: np.asarray(values).reshape(-1, width, width)
        for name, values in histories.items()
    }


def plot_order_parameter_evolution(
    activation: str,
    width: int,
    *,
    artifacts_dir: Path,
    output_dir: Path,
    seed: int,
) -> Path | None:
    """Plot matrix components from any available ODE and SGD trajectories."""
    config_path = sweep_config_path(activation, width)
    config = parse_config(config_path)
    activation_label = ACTIVATIONS[activation]
    run_sgd, run_ode = resolve_run_modes(config)
    run_dir = artifacts_dir / make_hash(run_config=config) / "autoencoder" / f"seed_{seed}"
    curves: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    missing: list[str] = []
    mode_labels: dict[str, str] = {}
    for mode, filename, enabled in (
        ("ODE", "ode_loss.jsonl", run_ode),
        ("SGD", "loss.jsonl", run_sgd),
    ):
        if not enabled:
            continue
        path = run_dir / filename
        if not path.is_file():
            missing.append(f"{mode}: no log")
            continue
        alphas, values = read_order_parameters(path, width)
        if not len(alphas):
            detail = (
                "rerun the ODE config to record history"
                if mode == "ODE"
                else "legacy SGD log; completed SGD runs are still skipped"
            )
            missing.append(f"{mode}: no recorded order parameters ({detail})")
            continue
        curves[mode] = (alphas, values)
        mode_labels[mode] = mode
        if alphas[-1] < float(config.alpha) - 1e-9:
            mode_labels[mode] += f" (through alpha={alphas[-1]:g})"

    description = f"{activation_label}, K={width} order parameters"
    if missing:
        print(f"{description}: {'; '.join(missing)}")
    if not curves:
        print(f"Skipping {description}: no trajectories available")
        return None

    unit_colors = plt.get_cmap("tab10")(np.arange(width) % 10)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    try:
        for ax, name in zip(axes.flat, ORDER_PARAMETERS):
            for mode, (alphas, values) in curves.items():
                linestyle = "--" if mode == "ODE" else "-"
                matrix_history = values[name]
                for i in range(width):
                    for j in range(width):
                        if i == j or (name in SYMMETRIC_PARAMETERS and j < i):
                            continue
                        ax.plot(
                            alphas,
                            matrix_history[:, i, j],
                            color="gray",
                            linewidth=0.7,
                            alpha=0.4,
                            linestyle=linestyle,
                            zorder=1,
                            marker="." if len(alphas) == 1 else None,
                        )
                for unit in range(width):
                    ax.plot(
                        alphas,
                        matrix_history[:, unit, unit],
                        color=unit_colors[unit],
                        linewidth=1.7 if mode == "ODE" else 1.2,
                        linestyle=linestyle,
                        alpha=1.0 if mode == "ODE" else 0.8,
                        zorder=3 if mode == "ODE" else 2,
                        marker="." if len(alphas) == 1 else None,
                    )
            ax.set_title(rf"${name[0]}^{{{name[1]}}}$")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel("Matrix entry")
            ax.set_xlim(left=0)
            ax.grid(False)

        handles = [
            Line2D([], [], color=color, linewidth=2, label=f"Unit {unit + 1} diagonal")
            for unit, color in enumerate(unit_colors)
        ]
        handles.append(Line2D([], [], color="gray", linewidth=1, label="Off-diagonal entries"))
        handles.extend(
            Line2D(
                [], [], color="black", linestyle="--" if mode == "ODE" else "-",
                label=label,
            )
            for mode, label in mode_labels.items()
        )
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
        fig.suptitle(
            f"{activation_label} | {len(config.spike_multipliers)} "
            f"{'spike' if len(config.spike_multipliers) == 1 else 'spikes'} | "
            f"K={width} | Order parameter evolution",
            fontweight="semibold",
        )
        fig.tight_layout(rect=(0, 0.11, 1, 0.95))
        output_dir = output_dir / activation
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{activation}_spikes5_k{width}_order_parameters.png"
        fig.savefig(output_path, dpi=300)
    finally:
        plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_activation(
    activation: str,
    *,
    artifacts_dir: Path,
    output_dir: Path,
    seed: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.8), layout="constrained")
    missing: list[str] = []
    partial: list[str] = []
    plotted_curves = 0
    pca_errors: dict[int, float] = {}
    enabled_widths: dict[str, list[int]] = {"ODE": [], "SGD": []}

    try:
        for width in WIDTHS:
            config_path = ROOT / "configs" / activation / f"spiked_wishart_5_k={width}.yaml"
            config = parse_config(config_path)
            run_sgd, run_ode = resolve_run_modes(config)
            run_dir = (
                artifacts_dir
                / make_hash(run_config=config)
                / "autoencoder"
                / f"seed_{seed}"
            )

            eigenvalues = read_eigenvalues(run_dir, config.D) if not pca_errors else None
            if eigenvalues is not None:
                pca_errors = pca_reference_errors(eigenvalues, len(config.spike_multipliers))

            modes = (
                ("ODE", "ode_loss.jsonl", run_ode),
                ("SGD", "loss.jsonl", run_sgd),
            )
            for mode, filename, enabled in modes:
                if not enabled:
                    continue
                enabled_widths[mode].append(width)
                label = f"{mode}, K={width}"
                path = run_dir / filename
                if not path.is_file():
                    missing.append(label)
                    continue
                alphas, errors = read_curve(path)
                if not alphas:
                    missing.append(label)
                    continue

                if alphas[-1] < float(config.alpha) - 1e-9:
                    partial.append(f"{label} through alpha={alphas[-1]:g}")
                    label += " (partial)"

                ax.plot(
                    alphas,
                    errors,
                    color=COLORS[width],
                    linestyle="--" if mode == "ODE" else "-",
                    linewidth=2 if mode == "ODE" else 1.2,
                    marker="o" if mode == "SGD" or len(alphas) == 1 else None,
                    markevery=max(1, len(alphas) // 20),
                    markersize=3,
                    markerfacecolor="white",
                    alpha=1.0 if mode == "ODE" else 0.8,
                    zorder=3 if mode == "ODE" else 2,
                    label=label,
                )
                plotted_curves += 1

        draw_pca_references(ax, pca_errors)

        mode_summary = "; ".join(
            f"{mode}: K={', '.join(map(str, widths))}"
            for mode, widths in enabled_widths.items() if widths
        )
        spike_count = len(config.spike_multipliers)
        ax.set_title(
            f"{ACTIVATIONS[activation]} | {spike_count} "
            f"{'spike' if spike_count == 1 else 'spikes'}\n{mode_summary}",
            fontweight="semibold",
        )
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Generalization error")
        ax.set_xlim(left=0)
        ax.grid(False)
        if plotted_curves:
            ax.legend(ncol=3, fontsize=9, framealpha=0.95)
        else:
            ax.text(
                0.5,
                0.5,
                f"No results available yet.\nRun bash scripts/run_exp_{activation}_spikes5.sh",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        output_dir = output_dir / activation
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{activation}_spikes5_sweep.png"
        fig.savefig(output_path, dpi=300)
    finally:
        plt.close(fig)

    if missing:
        print(f"{ACTIVATIONS[activation]} missing: {', '.join(missing)}")
    if partial:
        print(f"{ACTIVATIONS[activation]} incomplete: {'; '.join(partial)}")
    if not pca_errors:
        print(f"{ACTIVATIONS[activation]}: PCA references unavailable; no saved ODE spectrum")
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "plots",
        help="root output folder; plots are grouped into activation subfolders",
    )
    parser.add_argument(
        "--activations", nargs="+", choices=tuple(ACTIVATIONS), default=list(ACTIVATIONS),
        help="activation families to plot (default: all configured activations)",
    )
    parser.add_argument(
        "--order-parameters",
        action="store_true",
        help="also plot Q0, R0, T0, Q1, R1, T1 histories for each activation and width",
    )
    args = parser.parse_args()

    with plt.rc_context({"font.size": 11, "axes.titlesize": 13}):
        for activation in dict.fromkeys(args.activations):
            plot_options = {
                "artifacts_dir": args.artifacts_dir.resolve(),
                "output_dir": args.output_dir.resolve(),
                "seed": args.seed,
            }
            plot_activation(activation, **plot_options)
            if args.order_parameters:
                for width in WIDTHS:
                    plot_order_parameter_evolution(activation, width, **plot_options)


if __name__ == "__main__":
    main()
