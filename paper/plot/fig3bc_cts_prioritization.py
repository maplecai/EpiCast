"""fig3B / fig3C: prioritization of CTS-high and CTS-low CREs in the held-out cell types.

Each panel is a 2 (metric) x 2 (held-out cell type) grid of bars over the 11 figure
models, the top row titled with the cell type of its column. Positives are the target cell
type's CTS-high (3B) or CTS-low (3C) CREs; the ranking score is the predicted residual
activity, negated for CTS-low.

AUROC shares a fixed 0-1 axis with a random line at 0.5. AUPRC is scaled per cell type,
because its random expectation is the positive prevalence of that cell type and the two
differ; the prevalence is drawn as the dashed line. In CTS-high/A549 one model sits an
order of magnitude above the rest, so that cell is drawn as two stacked axes with
different ranges rather than letting the outlier flatten every other bar; the diagonal
break marks are added by hand.

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

# (metrics table, output figure, {(metric, cell type): (top of lower segment, bottom of
# upper segment)}); a break hides the empty span between the bulk of the models and a
# single outlier.
panels = [
    ("cts_high.tsv", "fig3b_cts_high.pdf", {("auprc", "A549"): (0.09, 0.19)}),
    ("cts_low.tsv", "fig3c_cts_low.pdf", {}),
]
legend_name = "fig3bc_legend.pdf"

metric_rows = [("auroc", "AUROC"), ("auprc", "AUPRC")]

colormap_range = (0.28, 0.92)
bar_width = 0.86
block_gap = 0.9
break_ratio = (1.0, 2.6)  # height of the upper vs lower segment of a broken axis

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


x = bar_positions()


def draw_bars(ax, heights):
    ax.bar(x, heights, color=colors, width=bar_width, linewidth=0)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", which="major", left=True, length=3.5)


def plot_panel(table, breaks, save_path):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    grid = fig.add_gridspec(
        len(metric_rows),
        len(test_cell_types),
        left=0.15,
        bottom=0.15,
        right=0.9,
        top=0.9,
        hspace=0.3,
        wspace=0.25,
    )

    for row, (metric, ylabel) in enumerate(metric_rows):
        values = table.pivot(index="model", columns="cell_type", values=metric).loc[model_names]
        prevalence = table.groupby("cell_type", observed=True)["prevalence"].first()

        for col, cell_type in enumerate(test_cell_types):
            heights = values[cell_type].to_numpy(dtype=float)
            random_line = 0.5 if metric == "auroc" else float(prevalence[cell_type])
            cell_break = breaks.get((metric, cell_type))

            if cell_break is None:
                ax = fig.add_subplot(grid[row, col])
                draw_bars(ax, heights)
                ax.axhline(random_line, ls="--", lw=1.0, color="gray")
                if metric == "auroc":
                    ax.set_ylim(0, 1)
                    ax.set_yticks([0.0, 0.5, 1.0])
                else:
                    ax.set_ylim(0, float(np.nanmax(heights)) * 1.15)
                ax.set_ylabel(ylabel if col == 0 else "")
                if row == 0:
                    ax.set_title(cell_type)
                continue

            low_top, high_bottom = cell_break
            sub = grid[row, col].subgridspec(2, 1, hspace=0.12, height_ratios=break_ratio)
            upper = fig.add_subplot(sub[0])
            lower = fig.add_subplot(sub[1])

            draw_bars(upper, heights)
            draw_bars(lower, heights)
            if row == 0:
                upper.set_title(cell_type)
            upper.set_ylim(high_bottom, float(np.nanmax(heights)) * 1.08)
            lower.set_ylim(0, low_top)
            lower.axhline(random_line, ls="--", lw=1.0, color="gray")
            upper.spines["bottom"].set_visible(False)
            lower.spines["top"].set_visible(False)
            lower.set_ylabel(ylabel if col == 0 else "")

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

    saved = []
    for table_name, fig_name, breaks in panels:
        path = figure_metrics_dir / table_name
        table = pd.read_csv(path, sep="\t")
        print(f"[load] {path} {table.shape}")

        out_path = figures_dir / fig_name
        plot_panel(table, breaks, out_path)
        saved.append(out_path)

    legend_path = figures_dir / legend_name
    plot_legend(legend_path)
    saved.append(legend_path)

    for p in saved:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
