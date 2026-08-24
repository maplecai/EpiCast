"""Bar charts comparing models on activity (fig2c/2d), residual (fig3b) and CTS
classification (fig3c/3d) metrics. Reads results/figure_metrics, written by analysis/15."""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    build_styles,
    figure_model_names,
    figures_dir,
    figure_metrics_dir,
    test_cell_types,
)

model_names = figure_model_names
model_plot_names, colors = build_styles(model_names)

metric_plot_cfg = {
    "pearson": {"ylabel": "Pearson r", "ylim": (0.0, 0.8)},
    "spearman": {"ylabel": "Spearman ρ", "ylim": (0.0, 0.8)},
    "mae": {"ylabel": "MAE", "ylim": None},
    "rmse": {"ylabel": "RMSE", "ylim": None},
    "auroc": {"ylabel": "AUROC", "ylim": (0.0, 1.0)},
    "auprc": {"ylabel": "AUPRC", "ylim": None},
}
correlation_metrics = ["pearson", "spearman", "mae", "rmse"]
classification_metrics = ["auroc", "auprc"]

# (results/figure_metrics table, metrics, figure prefix, figure stem)
panels = [
    ("fig2c_activity_test.tsv", correlation_metrics, "fig2c", ""),
    ("fig2d_activity_cts.tsv", correlation_metrics, "fig2d", "all_cts_1_99"),
    ("fig3b_residual_test.tsv", correlation_metrics, "fig3b", ""),
    ("fig3b_residual_cts.tsv", correlation_metrics, "fig3b", "all_cts_1_99"),
    ("fig3c_cts_high.tsv", classification_metrics, "fig3c", "CTS_high"),
    ("fig3d_cts_low.tsv", classification_metrics, "fig3d", "CTS_low"),
]
legend_names = ["fig2c_legend.pdf", "fig3b_legend.pdf", "fig3c_legend.pdf"]


def plot_legend(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3, 4), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    ax.set_axis_off()

    handles = [
        mpatches.Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=label)
        for color, label in zip(colors, model_plot_names)
    ]
    ax.legend(
        handles=handles,
        loc="center",
        frameon=True,
        fontsize=12,
        handlelength=0.9,
        handleheight=0.9,
        labelspacing=0.6,
    )

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def plot_bar(values: pd.Series, save_path: Path, title: str, metric: str) -> None:
    cfg = metric_plot_cfg[metric]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    x = np.arange(len(model_names))
    bar_heights = np.asarray(values.loc[model_names], dtype=float)

    ax.bar(x, bar_heights, color=colors, edgecolor="black", linewidth=0.5, width=1)
    ax.set_title(title)
    ax.set_xlabel("Models")
    ax.set_xticks([])
    ax.set_ylabel(cfg["ylabel"])
    if cfg["ylim"] is None:
        ax.set_ylim(0.0, np.nanmax(bar_heights) * 1.12)
    else:
        ax.set_ylim(*cfg["ylim"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def main() -> None:
    set_mpl_params()
    sns.set_theme(style="whitegrid", context="notebook")
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for table_name, metrics, fig_prefix, fig_stem in panels:
        path = figure_metrics_dir / table_name
        if not path.exists():
            raise FileNotFoundError(f"未找到 {path}，请先运行 analysis/15_export_figure_metrics.py")
        table = pd.read_csv(path, sep="\t")
        print(f"[load] {path} {table.shape}")

        for metric in metrics:
            perf_df = table.pivot(index="model", columns="cell_type", values=metric)

            for cell_type in test_cell_types:
                parts = [fig_prefix, fig_stem, metric, cell_type]
                out_name = "_".join(p for p in parts if p) + ".pdf"
                out_path = figures_dir / out_name
                plot_bar(perf_df[cell_type], out_path, cell_type, metric)
                saved_paths.append(out_path)

    for legend_name in legend_names:
        legend_path = figures_dir / legend_name
        plot_legend(legend_path)
        saved_paths.append(legend_path)

    for p in saved_paths:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
