"""fig3A: model comparison on residual activity inside the union CTS set.

A 4 (metric) x 2 (held-out cell type) grid of bars, with the two panels of a metric
sharing the y axis and the top row titled with the cell type of its column. Only the ten
VEF-based models are shown: on a held-out cell type
the sequence-only prediction is the mean of the three training cells, so its predicted
residual is identically zero and there is nothing to correlate.

Font sizes come from seaborn's "talk" context and are never set here: every figure in this
bundle ends up as one panel among several and gets scaled down, so the text has to start
out large. To make a figure look finer, grow its figsize; the text then reads smaller
relative to the whole. Margins are the same everywhere and do not need tuning because the
figure is saved with a tight bounding box.

Reads results/figure_metrics, written by analysis/15.
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import figure_metrics_dir, figures_dir, residual_model_blocks, test_cell_types

metrics_table = figure_metrics_dir / "residual_cts.tsv"
fig_name = "fig3a_residual_cts.pdf"
legend_name = "fig3a_legend.pdf"

# (metric column, axis label, y ticks); the axis stretches if the data exceeds the top tick
metric_rows = [
    ("pearson", "PCC", [0.0, 0.2, 0.4]),
    ("spearman", "SCC", [0.0, 0.2, 0.4]),
    ("mae", "MAE", [0.0, 1.0, 2.0]),
    ("rmse", "RMSE", [0.0, 1.0, 2.0]),
]

colormap_range = (0.28, 0.92)
bar_width = 0.86
block_gap = 0.9

model_names = [name for _, _, models in residual_model_blocks for name, _ in models]


def block_colors(colormap, n):
    return plt.get_cmap(colormap)(np.linspace(*colormap_range, n))


colors = np.concatenate(
    [block_colors(colormap, len(models)) for _, colormap, models in residual_model_blocks]
)


def bar_positions():
    positions = []
    offset = 0.0
    for _, _, models in residual_model_blocks:
        positions.append(offset + np.arange(len(models), dtype=float))
        offset += len(models) + block_gap
    return np.concatenate(positions)


def plot_panel(table, save_path):
    x = bar_positions()
    fig, axes = plt.subplots(len(metric_rows), len(test_cell_types), figsize=(8, 12), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, hspace=0.3, wspace=0.15)

    for row, (metric, ylabel, yticks) in enumerate(metric_rows):
        values = table.pivot(index="model", columns="cell_type", values=metric).loc[model_names]
        ymax = max(float(np.nanmax(values[test_cell_types].to_numpy())) * 1.12, max(yticks))

        for col, cell_type in enumerate(test_cell_types):
            ax = axes[row, col]
            ax.bar(
                x,
                values[cell_type].to_numpy(dtype=float),
                color=colors,
                width=bar_width,
                linewidth=0,
            )
            ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
            ax.set_ylim(0, ymax)
            ax.set_xticks([])
            ax.set_yticks(yticks)
            ax.tick_params(axis="y", which="major", left=True, length=3.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(cell_type)
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(axis="y", labelleft=False)

    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_legend(save_path):
    """One block per VEF source, each headed by the source name.

    The two blocks happen to be the same length here, but the axes are still sized by
    their row count so the entries cannot run past their axes into the next title.
    """
    rows = [len(models) + 1 for _, _, models in residual_model_blocks]
    fig, axes = plt.subplots(
        len(residual_model_blocks), 1, figsize=(4, 6), dpi=100, gridspec_kw={"height_ratios": rows}
    )
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, hspace=0.0)

    for ax, (block, colormap, models) in zip(axes, residual_model_blocks):
        ax.set_axis_off()
        handles = [
            mpatches.Patch(facecolor=color, linewidth=0)
            for color in block_colors(colormap, len(models))
        ]
        ax.legend(
            handles,
            [label for _, label in models],
            title=block,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            alignment="left",
            frameon=False,
            handlelength=1.0,
            handleheight=1.0,
            handletextpad=0.7,
            labelspacing=0.55,
        )

    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(metrics_table, sep="\t")
    print(f"[load] {metrics_table} {table.shape}")

    fig_path = figures_dir / fig_name
    plot_panel(table, fig_path)
    legend_path = figures_dir / legend_name
    plot_legend(legend_path)

    for p in [fig_path, legend_path]:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
