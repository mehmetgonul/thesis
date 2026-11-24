import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot._data import read_curve
from src.config import parse_config
from src.hash import make_hash


ACTIVATIONS = ["h3_h2", "softplus"]
ACTIVATION_LABELS = {"h3_h2": r"$H_2+H_3$", "softplus": "Softplus"}
SPIKE_COUNTS = [1, 2, 3, 4]
K_OFFSETS = [0, 1, 2]
SEED = 43
CONFIGS_DIR = ROOT / "configs"
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "plot" / "output" / "spike_counts"

FIGSIZE = (10, 6)
DPI = 200
FONT_SIZE = 11
COLORS = plt.get_cmap("tab10").colors
CURVE_LINESTYLE = "-"
CURVE_LINEWIDTH = 1.8
PC_COLOR = "0.5"
PC_LINESTYLE = "--"
PC_LINEWIDTH = 0.8
PC_ALPHA = 0.55


def main():
    plt.rcParams.update({"font.size": FONT_SIZE})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for activation in ACTIVATIONS:
        for spikes in SPIKE_COUNTS:
            figure, axis = plt.subplots(figsize=FIGSIZE)
            spectrum = None
            for offset in K_OFFSETS:
                width = spikes + offset
                config_path = CONFIGS_DIR / activation / f"spiked_wishart_{spikes}_k={width}.yaml"
                if not config_path.is_file():
                    print(f"Missing config: {config_path}", flush=True)
                    continue
                config = parse_config(config_path)
                directory = ARTIFACTS_DIR / make_hash(config) / "autoencoder" / f"seed_{SEED}"
                metadata_path = directory / "ode_meta.json"
                if spectrum is None and metadata_path.is_file():
                    spectrum = json.loads(metadata_path.read_text(encoding="utf-8")).get("eigenvalues")
                log = directory / "ode_loss.jsonl"
                if not log.is_file():
                    print(f"Missing ODE log: {activation}, M={spikes}, K={width}", flush=True)
                    continue
                alpha, error = read_curve(log)
                if not alpha.size:
                    print(f"No evaluations: {activation}, M={spikes}, K={width}", flush=True)
                    continue
                axis.plot(alpha, error, color=COLORS[(width - 1) % len(COLORS)],
                          linestyle=CURVE_LINESTYLE, linewidth=CURVE_LINEWIDTH, label=f"K={width}")

            if not axis.lines:
                plt.close(figure)
                continue
            if spectrum is not None:
                eigenvalues = np.sort(np.asarray(spectrum, dtype=float))
                if eigenvalues.shape != (config.D,) or not np.isfinite(eigenvalues).all():
                    plt.close(figure)
                    raise ValueError(f"{activation}, M={spikes}: invalid saved covariance spectrum")
                for rank in range(1, spikes + max(K_OFFSETS) + 1):
                    error = 0.5 * eigenvalues[:-rank].sum() / len(eigenvalues)
                    axis.axhline(error, color=PC_COLOR, linestyle=PC_LINESTYLE,
                                 linewidth=PC_LINEWIDTH, alpha=PC_ALPHA, label=f"PC {rank} error")
            else:
                print(f"No saved PC spectrum: {activation}, M={spikes}", flush=True)
            axis.set(xlabel=r"$\alpha$", ylabel="Generalization error",
                     title=f"{ACTIVATION_LABELS[activation]} | M={spikes}")
            axis.legend(ncol=3, fontsize=9)
            figure.tight_layout()
            path = OUTPUT_DIR / activation / f"m={spikes}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=DPI)
            plt.close(figure)
            print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
