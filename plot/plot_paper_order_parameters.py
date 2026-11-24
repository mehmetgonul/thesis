import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot._data import instantaneous_correlations, load_run


SEED = 43
RELU_CONFIG = ROOT / "configs/relu/spiked_wishart_2_k=4.yaml"
GELU_CONFIG = ROOT / "configs/gelu/spiked_wishart_2_k=2.yaml"
ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "plot/output/paper_order_parameters"
OUTPUT_NAME = "relu_gelu_order_parameters"
FORMATS = ("png", "pdf")
DPI = 600

FIGSIZE = (6.4, 2.8)
COLORS = plt.get_cmap("tab10").colors
LINEWIDTH = 1.4
OFF_DIAGONAL_COLOR = "0.25"
OFF_DIAGONAL_STYLE = (0, (3, 2))
FONT_SIZE = 8
TICK_FONT_SIZE = 7
LEGEND_FONT_SIZE = 7
STYLE = {
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def plot_order_parameters(relu, gelu):
    figure, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex="col")
    pairs = [(i, j) for i in range(relu.config.K) for j in range(i + 1, relu.config.K)]
    for axis, name, matrices in zip(axes[:, 0], ("Q", "T"), (relu.encoder, relu.decoder)):
        correlations = instantaneous_correlations(matrices)
        for index, (i, j) in enumerate(pairs):
            axis.plot(relu.alpha, correlations[:, i, j], color=COLORS[index % len(COLORS)],
                      linewidth=LINEWIDTH, label=rf"$({i + 1},{j + 1})$")
        for value in (-1, 1):
            axis.axhline(value, color="0.6", linestyle=":", linewidth=0.65, zorder=0)
        axis.axhline(0, color="0.88", linewidth=0.6, zorder=0)
        row = "Encoder" if name == "Q" else "Decoder"
        axis.set(ylabel=row + "\n" + rf"${name}^0_{{ij}}/\sqrt{{{name}^0_{{ii}}{name}^0_{{jj}}}}$",
                 ylim=(-1.08, 1.08), yticks=(-1, 0, 1))

    for axis, name, matrices in zip(axes[:, 1], ("Q", "T"), (gelu.encoder, gelu.decoder)):
        diagonal = np.diagonal(matrices, axis1=1, axis2=2)
        for unit in range(gelu.config.K):
            axis.plot(gelu.alpha, diagonal[:, unit], color=COLORS[unit % len(COLORS)],
                      linewidth=LINEWIDTH, label=f"Unit {unit + 1}")
        axis.plot(gelu.alpha, matrices[:, 0, 1], color=OFF_DIAGONAL_COLOR,
                  linestyle=OFF_DIAGONAL_STYLE, linewidth=1.2, label="Off-diag.")
        row = "Encoder" if name == "Q" else "Decoder"
        axis.set_ylabel(row + rf" ${name}^0_{{ij}}$")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        axis.yaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)

    legend_options = dict(frameon=False, borderaxespad=0.2, borderpad=0.15,
                          handlelength=1.2, handletextpad=0.35, labelspacing=0.15,
                          columnspacing=0.7)
    axes[0, 0].legend(loc="upper right", ncol=2, **legend_options)
    axes[0, 1].legend(loc="upper right", ncol=3, **legend_options)
    for column, run in enumerate((relu, gelu)):
        for axis in axes[:, column]:
            axis.set(xscale="linear", xlim=(0, run.alpha.max()))
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    for axis in axes.flat:
        axis.tick_params(which="both", direction="in", top=True, right=True,
                         width=0.6, pad=2)
        axis.tick_params(which="major", length=3)
        axis.tick_params(which="minor", length=1.5)
    for axis in axes[1]:
        axis.set_xlabel(r"$\alpha$", labelpad=1)
    figure.subplots_adjust(left=0.105, right=0.975, bottom=0.16, top=0.94,
                           hspace=0.11, wspace=0.32)
    for axis, label in zip(axes[1], ("(a) ReLU", "(b) GELU")):
        bounds = axis.get_position()
        figure.text((bounds.x0 + bounds.x1) / 2, 0.025, label,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in FORMATS:
        path = OUTPUT_DIR / f"{OUTPUT_NAME}.{extension}"
        figure.savefig(path, dpi=DPI)
        print(f"Saved {path}")
    plt.close(figure)


def main():
    for path in (RELU_CONFIG, GELU_CONFIG):
        if not path.is_file():
            raise SystemExit(f"Missing paper configuration {path}")
    relu = load_run(RELU_CONFIG, ARTIFACTS_DIR, SEED)
    gelu = load_run(GELU_CONFIG, ARTIFACTS_DIR, SEED)
    if relu is None or gelu is None:
        raise SystemExit("Paper order-parameter plots require the completed ReLU and GELU ODE runs.")
    if relu.config.activation_function != "relu" or gelu.config.activation_function != "gelu":
        raise ValueError("Expected ReLU and GELU experiment configurations")
    if relu.config.K < 2 or gelu.config.K != 2:
        raise ValueError("Expected ReLU K>=2 and GELU K=2")
    with plt.rc_context(STYLE):
        plot_order_parameters(relu, gelu)


if __name__ == "__main__":
    main()
