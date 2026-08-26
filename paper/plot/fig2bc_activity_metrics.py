"""fig2B / fig2C: model comparison on measured activity in the two held-out cell types.

fig2B evaluates the whole test-chromosome CRE set, fig2C the union CTS subset of it;
both are a 4 (metric) x 2 (held-out cell type) grid of bars over the 11 figure models,
drawn as a Sei block, an AlphaGenome block and the sequence-only baseline, separated by
gaps. The top row is titled with the cell type of its column. Corresponding rows of the
two panels share a y axis so 2B and 2C can be read against each other.

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
from config import figure_metrics_dir, figure_model_blocks, figures_dir, test_cell_types

# (metrics table, output figure) in manuscript panel order
panels = [
    ("activity_test.tsv", "fig2b_activity_whole.pdf"),
    ("activity_cts.tsv", "fig2c_activity_cts.pdf"),
]
legend_name = "fig2bc_legend.pdf"

# (metric column, axis label, y ticks); the axis stretches if the data exceeds the top tick
metric_rows = [
    ("pearson", "PCC", [0.0, 0.4, 0.8]),
    ("spearman", "SCC", [0.0, 0.4, 0.8]),
    ("mae", "MAE", [0.0, 0.5, 1.0]),
    ("rmse", "RMSE", [0.0, 0.5, 1.0]),
]

colormap_range = (0.28, 0.92)
bar_width = 0.86
block_gap = 0.9

model_names = [name for _, _, models in figure_model_blocks for name, _ in models]


def block_colors(colormap, n):
    return plt.get_cmap(colormap)(np.linspace(*colormap_range, n))


colors = np.concatenate(
    [block_colors(colormap, len(models)) for _, colormap, models in figure_model_blocks]
)


def bar_positions():
    positions = []
    offset = 0.0
    for _, _, models in figure_model_blocks:
        positions.append(offset + np.arange(len(models), dtype=float))
        offset += len(models) + block_gap
    return np.concatenate(positions)


def shared_ylims(tables):
    """Top of each metric row, taken over both panels so 2B and 2C stay comparable."""
    ylims = {}
    for metric, _, yticks in metric_rows:
        peak = max(
            float(np.nanmax(t.pivot(index="model", columns="cell_type", values=metric)[test_cell_types].to_numpy()))
            for t in tables
        )
        ylims[metric] = max(peak * 1.12, max(yticks))
    return ylims


def plot_panel(table, ylims, save_path):
    x = bar_positions()
    fig, axes = plt.subplots(len(metric_rows), len(test_cell_types), figsize=(8, 12), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, hspace=0.3, wspace=0.15)

    for row, (metric, ylabel, yticks) in enumerate(metric_rows):
        values = table.pivot(index="model", columns="cell_type", values=metric).loc[model_names]

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
            ax.set_ylim(0, ylims[metric])
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

    The blocks hold different numbers of models, so equal-height axes would let a long
    block's entries run past its axes and collide with the next block's title. Each axes
    gets the share of the height its own rows need, the title counting as one row.
    """
    rows = [len(models) + 1 for _, _, models in figure_model_blocks]
    fig, axes = plt.subplots(
        len(figure_model_blocks), 1, figsize=(4, 6), dpi=100, gridspec_kw={"height_ratios": rows}
    )
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, hspace=0.0)

    for ax, (block, colormap, models) in zip(axes, figure_model_blocks):
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

    tables = []
    for table_name, _ in panels:
        path = figure_metrics_dir / table_name
        table = pd.read_csv(path, sep="\t")
        print(f"[load] {path} {table.shape}")
        tables.append(table)

    ylims = shared_ylims(tables)
    saved = []
    for table, (_, fig_name) in zip(tables, panels):
        out_path = figures_dir / fig_name
        plot_panel(table, ylims, out_path)
        saved.append(out_path)

    legend_path = figures_dir / legend_name
    plot_legend(legend_path)
    saved.append(legend_path)

    for p in saved:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
