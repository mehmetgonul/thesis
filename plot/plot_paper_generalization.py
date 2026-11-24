import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot._data import read_curve, read_last_point
from src.config import parse_config
from src.hash import make_hash


SEED = 43
CONFIGS_DIR = ROOT / "configs"
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "plot" / "output" / "paper_generalization"
OUTPUT_NAME = "generalization"
FORMATS = ["png", "pdf"]
DPI = 600
FIGSIZE = (12.2 / 2.54, 4.8 / 2.54)
FONT_SIZE = 7
Y_TICK_STEP = 0.01
Y_PADDING = (0.002, 0.003)
LEARNING_LEGEND_ANCHOR = (0.99, 0.99)
SWEEP_LEGEND_ANCHOR = (0.99, 0.91)
LEARNING_LEGEND_TITLE = "ODE (lines), SGD (markers)"

LEARNING_ACTIVATIONS = ["erf", "gelu"]
LEARNING_K_VALUES = [3, 6]
LEARNING_MODES = ["ODE", "SGD"]
LOG_FILES = {"ODE": "ode_loss.jsonl", "SGD": "loss.jsonl"}
TIME_FACTORS = {"erf": np.sqrt(2 / np.pi), "gelu": 0.5}
X_MAX = 5000
LEARNING_LINESTYLES = {3: "-", 6: "--"}
SGD_MARKERS = {3: "o", 6: "s"}
SGD_MARKER_SIZE = 3.2
SGD_MARKER_EDGEWIDTH = 0.7
# Marker spacing as a fraction of the axes diagonal.
SGD_MARKER_SPACING = 0.12

ACTIVATIONS = ["erf", "gelu", "h2", "h3_h2", "relu", "softplus", "swish"]
SPIKE_COUNT = 3
K_VALUES = list(range(1, 11))
# Evaluation times used by the existing paper figure.
SWEEP_ALPHA = {
    "erf": 5000, "gelu": 20000, "h2": 5000, "h3_h2": 5000,
    "relu": 5000, "softplus": 20000, "swish": 20000,
}
LABELS = {
    "erf": "erf", "gelu": "GELU", "h2": r"$H_2$", "h3_h2": r"$H_2+H_3$",
    "relu": "ReLU", "softplus": "softplus", "swish": "swish",
}
COLORS = {
    "erf": "#2878B5", "gelu": "#E47728", "h2": "#777777", "h3_h2": "#AA8D24",
    "relu": "#33925B", "softplus": "#CC6677", "swish": "#AA4499",
}
PC_RANKS = list(range(1, 11))
LINE_WIDTH = 0.9
SWEEP_MARKER = "o"
SWEEP_MARKER_SIZE = 2.7


def load_config(activation, width):
    path = CONFIGS_DIR / activation / f"spiked_wishart_{SPIKE_COUNT}_k={width}.yaml"
    config = parse_config(path)
    directory = ARTIFACTS_DIR / make_hash(run_config=config) / "autoencoder" / f"seed_{SEED}"
    return config, directory


def read_endpoint(path, alpha):
    """Use the requested recorded evaluation from a run of any saved duration."""
    if not path.is_file():
        return np.nan
    last = read_last_point(path)
    if not last or last[0]["alpha"] < alpha - 1e-8:
        return np.nan
    if np.isclose(last[0]["alpha"], alpha, rtol=0, atol=1e-8):
        return last[0]["generalization_error"]
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if row.get("generalization_error") is None:
                continue
            if np.isclose(row["alpha"], alpha, rtol=0, atol=1e-8):
                error = float(row["generalization_error"])
                if not np.isfinite(error):
                    raise ValueError(f"{path}: nonfinite evaluation")
                return error
            if row["alpha"] > alpha:
                break
    return np.nan


def main():
    plt.rcParams.update({"font.size": FONT_SIZE, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, (learning_axis, sweep_axis) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    eigenvalues = None

    for activation in ACTIVATIONS:
        errors = []
        for width in K_VALUES:
            config, directory = load_config(activation, width)
            metadata_path = directory / "ode_meta.json"
            if eigenvalues is None and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                spectrum = np.asarray(metadata["eigenvalues"], dtype=float)
                if spectrum.shape != (config.D,) or not np.isfinite(spectrum).all():
                    raise ValueError(f"{metadata_path}: invalid covariance spectrum")
                eigenvalues = np.sort(spectrum)
            error = read_endpoint(directory / LOG_FILES["ODE"], SWEEP_ALPHA[activation])
            if not np.isfinite(error):
                print(f"{activation}, K={width}: evaluation at alpha={SWEEP_ALPHA[activation]} missing")
            errors.append(error)
        sweep_axis.plot(K_VALUES, errors, color=COLORS[activation], label=LABELS[activation],
                        marker=SWEEP_MARKER, markersize=SWEEP_MARKER_SIZE,
                        markerfacecolor="white", markeredgewidth=0.65, linewidth=LINE_WIDTH)

    learning_handles = []
    for activation in LEARNING_ACTIVATIONS:
        for width in LEARNING_K_VALUES:
            _, directory = load_config(activation, width)
            available_modes = set()
            for mode in LEARNING_MODES:
                path = directory / LOG_FILES[mode]
                if not path.is_file():
                    print(f"{activation}, K={width}: {mode} learning curve missing")
                    continue
                alpha, error = read_curve(path)
                scaled_alpha = alpha * TIME_FACTORS[activation]
                visible = (scaled_alpha >= 0) & (scaled_alpha <= X_MAX)
                if not visible.any():
                    continue
                available_modes.add(mode)
                if mode == "SGD":
                    learning_axis.plot(scaled_alpha[visible], error[visible], linestyle="none",
                                       marker=SGD_MARKERS[width], markersize=SGD_MARKER_SIZE,
                                       markerfacecolor="white", markeredgecolor=COLORS[activation],
                                       markeredgewidth=SGD_MARKER_EDGEWIDTH,
                                       markevery=SGD_MARKER_SPACING, zorder=4)
                else:
                    learning_axis.plot(scaled_alpha, error, color=COLORS[activation],
                                       linestyle=LEARNING_LINESTYLES[width], linewidth=LINE_WIDTH)
            if available_modes:
                learning_handles.append(Line2D(
                    [], [], color=COLORS[activation], linewidth=LINE_WIDTH,
                    linestyle=LEARNING_LINESTYLES[width] if "ODE" in available_modes else "none",
                    marker=SGD_MARKERS[width] if "SGD" in available_modes else None,
                    markersize=SGD_MARKER_SIZE, markerfacecolor="white",
                    markeredgewidth=SGD_MARKER_EDGEWIDTH,
                    label=rf"{LABELS[activation]}, $K={width}$"))

    if eigenvalues is None:
        plt.close(figure)
        raise FileNotFoundError("No saved covariance spectrum for the paper PC references")
    for rank in PC_RANKS:
        pc_error = 0.5 * eigenvalues[:-rank].sum() / len(eigenvalues)
        for axis in (learning_axis, sweep_axis):
            if axis is learning_axis and rank > max(LEARNING_K_VALUES):
                continue
            axis.axhline(pc_error, color="0.40" if rank == SPIKE_COUNT else "0.74",
                         linestyle=":" if rank <= SPIKE_COUNT else "--",
                         linewidth=0.8 if rank == SPIKE_COUNT else 0.45, zorder=0,
                         label="PCA" if axis is sweep_axis and rank == SPIKE_COUNT else "_nolegend_")

    learning_axis.set(xlabel=r"$\alpha\,|g'(0)|$", ylabel="Reconstruction error",
                      xlim=(0, X_MAX))
    sweep_axis.set(xlabel=r"Number of internal units ($K$)", ylabel="Asymptotic\nreconstruction error",
                   xticks=K_VALUES, xlim=(min(K_VALUES) - 0.3, max(K_VALUES) + 0.3))
    lower = 0.5 * eigenvalues[:-max(PC_RANKS)].sum() / len(eigenvalues) - Y_PADDING[0]
    upper = 0.5 * eigenvalues.mean() + Y_PADDING[1]
    learning_axis.set_ylim(lower, upper)
    learning_axis.set_yticks(np.arange(np.ceil(lower / Y_TICK_STEP) * Y_TICK_STEP, upper, Y_TICK_STEP))
    learning_axis.legend(handles=learning_handles, loc="upper right", bbox_to_anchor=LEARNING_LEGEND_ANCHOR,
                         title=LEARNING_LEGEND_TITLE, title_fontsize=6, fontsize=6.5,
                         frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    sweep_axis.legend(loc="upper right", bbox_to_anchor=SWEEP_LEGEND_ANCHOR, ncol=2, fontsize=6.2,
                      frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    for axis in (learning_axis, sweep_axis):
        axis.tick_params(direction="in", top=True, right=True, width=0.6, pad=2)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    figure.subplots_adjust(left=0.105, right=0.98, bottom=0.25, top=0.98, wspace=0.30)
    for axis, label in zip((learning_axis, sweep_axis), ("(a)", "(b)")):
        bounds = axis.get_position()
        figure.text((bounds.x0 + bounds.x1) / 2, 0.025, label, ha="center", fontsize=7.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        path = OUTPUT_DIR / f"{OUTPUT_NAME}.{extension}"
        figure.savefig(path, dpi=DPI)
        print(f"Saved {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
