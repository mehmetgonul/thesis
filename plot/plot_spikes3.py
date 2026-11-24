import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot._data import read_curve
from src.config import parse_config
from src.hash import make_hash


ACTIVATIONS = ["erf", "gelu", "h2", "h3_h2", "relu", "softplus", "swish"]
SPIKE_COUNT = 3
K_VALUES = range(1, 11)
SEED = 43
CONFIGS_DIR = ROOT / "configs"
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "plot" / "output" / "spikes3"

MODE_FILES = {"ODE": "ode_loss.jsonl", "SGD": "loss.jsonl"}
MODE_STYLES = {
    "ODE": {"linestyle": "--", "linewidth": 2.0, "zorder": 3},
    "SGD": {"linestyle": "-", "linewidth": 0.8, "marker": "o",
            "markersize": 5, "markerfacecolor": "white", "markeredgewidth": 1.1,
            "zorder": 4},
}
COLORS = plt.get_cmap("tab10").colors
SGD_MARKER_COUNT = 30
PC_RANKS = range(1, 11)
PC_COLORS = {"spike": "0.55", "bulk": "#7099b5"}
PC_ALPHA = {"spike": 0.55, "bulk": 0.28}
PC_LINEWIDTH = 0.8
FIGSIZE = (11, 7)
DPI = 200
FONT_SIZE = 10
LEGEND_COLUMNS = 5


def plot_activation(activation):
    figure, axis = plt.subplots(figsize=FIGSIZE)
    available_widths, available_modes = set(), set()
    eigenvalues = None
    title = activation
    for k in K_VALUES:
        config_path = CONFIGS_DIR / activation / f"spiked_wishart_{SPIKE_COUNT}_k={k}.yaml"
        if not config_path.is_file():
            print(f"Missing config: {config_path}")
            continue
        config = parse_config(config_path)
        title = config.activation_function
        directory = ARTIFACTS_DIR / make_hash(run_config=config) / "autoencoder" / f"seed_{SEED}"
        metadata_path = directory / "ode_meta.json"
        if eigenvalues is None and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "eigenvalues" in metadata:
                eigenvalues = np.sort(np.asarray(metadata["eigenvalues"], dtype=float))
                if eigenvalues.shape != (config.D,) or not np.isfinite(eigenvalues).all():
                    raise ValueError(f"Invalid covariance spectrum: {metadata_path}")
        for mode, filename in MODE_FILES.items():
            path = directory / filename
            if not path.is_file():
                if mode == "ODE":
                    print(f"Missing ODE log: {activation}, K={k}")
                continue
            alpha, error = read_curve(path)
            if not alpha.size:
                continue
            axis.plot(alpha, error, color=COLORS[(k - 1) % len(COLORS)],
                      markevery=max(1, len(alpha) // SGD_MARKER_COUNT), **MODE_STYLES[mode])
            available_widths.add(k)
            available_modes.add(mode)
    if not available_widths:
        plt.close(figure)
        print(f"No saved curves: {activation}")
        return None

    handles = [Line2D([], [], color=COLORS[(k - 1) % len(COLORS)], linewidth=2, label=f"K={k}")
               for k in sorted(available_widths)]
    handles.extend(Line2D([], [], color="0.2", label=mode, **MODE_STYLES[mode])
                   for mode in MODE_FILES if mode in available_modes)
    if eigenvalues is None:
        print(f"No saved covariance spectrum: {activation}")
    else:
        for rank in PC_RANKS:
            group = "spike" if rank <= SPIKE_COUNT else "bulk"
            error = 0.5 * eigenvalues[:-rank].sum() / len(eigenvalues)
            line = axis.axhline(error, color=PC_COLORS[group],
                                linestyle=":" if group == "spike" else "--",
                                linewidth=PC_LINEWIDTH, alpha=PC_ALPHA[group], zorder=0)
            if rank <= SPIKE_COUNT:
                line.set_label(f"PC {rank} error")
                handles.append(line)
        if max(PC_RANKS) > SPIKE_COUNT:
            handles.append(Line2D([], [], color=PC_COLORS["bulk"], linestyle="--",
                                  linewidth=PC_LINEWIDTH,
                                  label=f"PC {SPIKE_COUNT + 1}-{max(PC_RANKS)} (bulk)"))
    axis.set(xlabel=r"$\alpha$", ylabel="Generalization error", title=f"{title} | {SPIKE_COUNT} spikes")
    figure.legend(handles=handles, loc="lower center", ncol=LEGEND_COLUMNS,
                  fontsize=FONT_SIZE, handlelength=2.8, columnspacing=1.8)
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    output = OUTPUT_DIR / activation / "generalization_error.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DPI)
    plt.close(figure)
    print(f"Saved {output}", flush=True)
    return output


def main():
    plt.rcParams.update({"font.size": FONT_SIZE})
    for activation in ACTIVATIONS:
        plot_activation(activation)


if __name__ == "__main__":
    main()
