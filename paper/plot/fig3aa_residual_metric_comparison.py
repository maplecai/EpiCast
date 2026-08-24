"""fig3aa: model comparison on residual activity inside the union CTS set.

Reads results/figure_metrics, written by analysis/15. One figure of 4 (metric) x 2 (held-out
cell line) panels, with the two panels of a metric sharing the y axis; the legend is a
separate file. Titles are left out on purpose and are added by hand afterwards.
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
from config import figure_metrics_dir, figures_dir, test_cell_types

metrics_table = figure_metrics_dir / "fig3b_residual_cts.tsv"
fig_name = "fig3aa_residual_cts_metrics.pdf"
legend_name = "fig3aa_legend.pdf"

# (group header, colormap, [(model name, legend label), ...]) in left-to-right bar order.
# Each group gets its own colormap, sampled evenly so the bars run light to dark in the
# order they are drawn: Sei yellow to red, AlphaGenome green to blue.
model_groups = [
    (
        "Sei",
        "YlOrRd",
        [
            ("linear_sei_dnase", "DNase"),
            ("linear_sei_vef", "VEF-only (linear)"),
            ("mlp_sei_vef", "VEF-only (MLP)"),
            ("xgb_sei_vef", "VEF-only (XGBoost)"),
            ("epicast_sei_vef", "EpiCast"),
        ],
    ),
    (
        "AlphaGenome",
        "GnBu",
        [
            ("linear_ag_dnase", "DNase"),
            ("linear_ag_vef", "VEF-only (linear)"),
            ("mlp_ag_vef", "VEF-only (MLP)"),
            ("xgb_ag_vef", "VEF-only (XGBoost)"),
            ("epicast_ag_vef", "EpiCast"),
        ],
    ),
]
# (metric, axis label, y ticks); the axis is stretched if the top tick sits above the data
metric_rows = [
    ("pearson", "PCC", [0.0, 0.2, 0.4]),
    ("spearman", "SCC", [0.0, 0.2, 0.4]),
    ("mae", "MAE", [0.0, 1.0, 2.0]),
    ("rmse", "RMSE", [0.0, 1.0, 2.0]),
]
colormap_range = (0.28, 0.92)

model_names = [name for _, _, models in model_groups for name, _ in models]


def group_colors(colormap: str, n: int) -> np.ndarray:
    return plt.get_cmap(colormap)(np.linspace(*colormap_range, n))


colors = np.concatenate(
    [group_colors(colormap, len(models)) for _, colormap, models in model_groups]
)
bar_width = 0.86
group_gap = 0.9
label_fontsize = 13
tick_fontsize = 11


def bar_positions() -> np.ndarray:
    positions = []
    offset = 0.0
    for _, _, models in model_groups:
        positions.append(offset + np.arange(len(models), dtype=float))
        offset += len(models) + group_gap
    return np.concatenate(positions)


def plot_panels(table: pd.DataFrame, save_path: Path) -> None:
    x = bar_positions()
    fig, axes = plt.subplots(
        len(metric_rows), len(test_cell_types), figsize=(6.0, 9.0), dpi=100
    )
    fig.subplots_adjust(left=0.13, bottom=0.03, right=0.98, top=0.98, hspace=0.3, wspace=0.12)

    for row, (metric, ylabel, yticks) in enumerate(metric_rows):
        values = table.pivot(index="model", columns="cell_type", values=metric).loc[model_names]
        ymax = max(float(np.nanmax(values[test_cell_types].to_numpy())) * 1.12, max(yticks))

        for col, cell_type in enumerate(test_cell_types):
            ax = axes[row, col]
            ax.bar(x, values[cell_type].to_numpy(dtype=float), color=colors, width=bar_width, linewidth=0)
            ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
            ax.set_ylim(0, ymax)
            ax.set_xticks([])
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])
            ax.tick_params(axis="y", which="major", left=True, length=3.5, labelsize=tick_fontsize)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=label_fontsize)
            else:
                ax.tick_params(axis="y", labelleft=False)

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def plot_legend(save_path: Path) -> None:
    """One block per VEF source, each headed by the source name."""
    fig, axes = plt.subplots(len(model_groups), 1, figsize=(2.8, 3.4), dpi=100)
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, hspace=0.0)

    for ax, (group, colormap, models) in zip(axes, model_groups):
        ax.set_axis_off()
        handles = [
            mpatches.Patch(facecolor=color, linewidth=0)
            for color in group_colors(colormap, len(models))
        ]
        legend = ax.legend(
            handles,
            [label for _, label in models],
            title=group,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
            fontsize=12,
            handlelength=1.0,
            handleheight=1.0,
            handletextpad=0.7,
            labelspacing=0.55,
        )
        legend._legend_box.align = "left"
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(label_fontsize)

    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_mpl_params()
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(metrics_table, sep="\t")
    print(f"[load] {metrics_table} {table.shape}")

    fig_path = figures_dir / fig_name
    plot_panels(table, fig_path)
    legend_path = figures_dir / legend_name
    plot_legend(legend_path)

    for p in [fig_path, legend_path]:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
