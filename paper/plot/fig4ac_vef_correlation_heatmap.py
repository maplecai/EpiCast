"""fig4A / fig4C: pairwise correlations among the four VEFs, absolute and residual.

One PDF per manuscript panel. Each holds two heatmaps side by side, titled with their VEF
source: Sei on the left and AlphaGenome on the right, the order every figure that shows
both sources uses.

The assay order is DNase, H3K4me3, H3K27ac, CTCF. Only the lower triangle is drawn,
diagonal included, in the same style as fig1D/fig1F: black annotations, cell borders drawn
by hand so the empty upper right carries no grid lines, and a black outline around the
triangle. The diagonal is a VEF against itself, so it reads 1.00 with no SEM; every other
cell carries the mean PCC across the five assayed cell types with its SEM underneath.

fig4A is the absolute setting, fig4C the residual one; all four heatmaps share the 0-1
colour scale, so the loss of collinearity after removing the shared-across-cell-types
component is readable straight off the colours, and one colour bar file serves both. Single cell types do go negative in the
residual setting, down to -0.26, but every mean plotted here is positive.

Font sizes come from seaborn's "talk" context and are never set here: every figure in this
bundle ends up as one panel among several and gets scaled down, so the text has to start
out large. To make a figure look finer, grow its figsize; the text then reads smaller
relative to the whole. Margins are the same everywhere and do not need tuning because the
figure is saved with a tight bounding box.

Reads results/vef_pairwise_correlation, written by analysis/11_vef_pairwise_correlation.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params, warm_cmap
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import assays, figures_dir, results_dir

metrics_path = results_dir / "vef_pairwise_correlation" / "vef_pairwise_correlation.csv"

# (setting in the metrics table, output figure)
panels = [
    ("absolute", "fig4a_vef_correlation.pdf"),
    ("residual", "fig4c_vef_residual_correlation.pdf"),
]
colorbar_name = "fig4ac_colorbar.pdf"

# left heatmap, right heatmap
sources = [("sei", "Sei"), ("alphagenome", "AlphaGenome")]

color_limits = (0.0, 1.0)
color_ticks = [0.0, 0.5, 1.0]
cbar_fraction = 0.6
cell_edge = "gray"
outline = "black"


def summarize(table):
    """(source, setting, assay pair) -> mean and SEM of the PCC over cell types."""
    grouped = table.groupby(["vef_source", "setting", "assay_a", "assay_b"])["pcc"]
    return grouped.mean(), grouped.sem()


def build_matrix(mean, sem, source, setting):
    """Lower-triangle values and their 'mean / +- sem' labels, NaN above the diagonal."""
    values = pd.DataFrame(np.nan, index=assays, columns=assays, dtype=float)
    labels = pd.DataFrame("", index=assays, columns=assays)

    for row, assay_row in enumerate(assays):
        values.iloc[row, row] = 1.0
        labels.iloc[row, row] = "1.00"
        for col in range(row):
            key = (source, setting, assays[col], assay_row)
            values.iloc[row, col] = mean[key]
            labels.iloc[row, col] = f"{mean[key]:.2f}\n$\\pm$ {sem[key]:.2f}"
    return values, labels


def staircase(n):
    """Outline of the lower triangle of an n x n heatmap, in cell coordinates."""
    points = [(0, 0)]
    for i in range(n):
        points += [(i + 1, i), (i + 1, i + 1)]
    return points + [(0, n)]


def plot_panel(mean, sem, setting, save_path):
    n = len(assays)
    fig, axes = plt.subplots(1, len(sources), figsize=(12, 6), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.5)

    for ax, (source, source_label) in zip(axes, sources):
        values, labels = build_matrix(mean, sem, source, setting)
        print(f"  [{setting}] {source_label}")
        print(values.round(3).to_string())

        sns.heatmap(
            values,
            cmap=warm_cmap,
            vmin=color_limits[0],
            vmax=color_limits[1],
            square=True,
            annot=labels,
            fmt="",
            annot_kws={"color": "black"},
            linewidths=0,
            cbar=False,
            ax=ax,
        )
        ax.set_title(source_label)
        ax.set_xticklabels(assays, rotation=90)
        ax.set_yticklabels(assays, rotation=0)

        # borders drawn by hand so the empty upper right stays free of grid lines
        for row in range(n):
            for col in range(row + 1):
                ax.add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor=cell_edge, lw=0.5))
        ax.add_patch(Polygon(staircase(n), closed=True, fill=False, edgecolor=outline, lw=1.0))

    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_colorbar(save_path):
    """The bar is 0.6 of the heatmap side, i.e. 0.6 of the shared panel height of 6."""
    fig = plt.figure(figsize=(2, 6), dpi=100)
    ax = fig.add_axes([0.1, 0.5 - cbar_fraction / 2, 0.12, cbar_fraction])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(*color_limits), warm_cmap),
        cax=ax,
        ticks=color_ticks,
        label="PCC",
    )
    colorbar.outline.set(edgecolor=outline, linewidth=1.0)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(metrics_path)
    print(f"[load] {metrics_path} {table.shape}")
    mean, sem = summarize(table)

    for setting, fig_name in panels:
        out_path = figures_dir / fig_name
        plot_panel(mean, sem, setting, out_path)
        print(f"[save] {out_path.resolve()}")

    colorbar_path = figures_dir / colorbar_name
    plot_colorbar(colorbar_path)
    print(f"[save] {colorbar_path.resolve()}")


if __name__ == "__main__":
    main()
