"""fig5: combined Castillo metric panel, 4 rows x 5 columns.

Adapted from the final analysis written by C.Z.; the layout is his. Reads the
tables written by analysis/12_eval_castillo.py.

Columns are the three regression settings followed by the two CTS tasks. Rows are
PCC / SCC / MAE / RMSE for the regression columns and AUROC / normalized AUPRC /
2% EF / 5% EF for the classification columns. One boxplot per model summarizes the
seven cell types; the coloured dots are the individual cell types.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    castillo_cell_colors,
    castillo_cell_types,
    castillo_cts_gap,
    castillo_min_positives,
    castillo_model_names,
    castillo_model_styles,
    figures_dir,
    results_dir,
)

metrics_dir = results_dir / "castillo"
fig_name = "fig5_castillo_combined_metrics.pdf"

regression_settings = ["All activity", "CTS-union activity", "CTS-union residual"]
regression_rows = ["pcc", "scc", "mae", "rmse"]
regression_labels = {"pcc": "PCC", "scc": "SCC", "mae": "MAE", "rmse": "RMSE"}
# (column in the classification table, axis label, screening depth to read)
classification_rows = [
    ("auroc", "AUROC", 5.0),
    ("normalized_auprc", "Normalized AUPRC", 5.0),
    ("ef", "2% EF", 2.0),
    ("ef", "5% EF", 5.0),
]
box_edge = "#666666"
box_linewidth = 0.9


def padded_limits(values, zero_floor=False, ceiling=None):
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 0.1)
    lower = 0.0 if zero_floor else min(0.0, low - 0.08 * span)
    upper = high + 0.10 * span
    if ceiling is not None:
        upper = min(ceiling, upper)
    return lower, upper


def axis_limits(regression, classification):
    """One shared y range per metric, so panels in a row stay comparable."""
    screen = classification["screen_pct"]
    return {
        "pcc": padded_limits(regression["pcc"], ceiling=1.0),
        "scc": padded_limits(regression["scc"], ceiling=1.0),
        "mae": padded_limits(regression["mae"], zero_floor=True),
        "rmse": padded_limits(regression["rmse"], zero_floor=True),
        "auroc": (0.0, 1.0),
        "normalized_auprc": padded_limits(classification["normalized_auprc"], ceiling=1.0),
        "ef2": padded_limits(classification.loc[screen.eq(2), "ef"], zero_floor=True),
        "ef5": padded_limits(classification.loc[screen.eq(5), "ef"], zero_floor=True),
    }


def draw_boxplot(ax, data, metric, ylim, cells, baseline=None):
    arrays = [
        data[data["model"].eq(model)].set_index("cell_type").reindex(cells)[metric].to_numpy(float)
        for model in castillo_model_names
    ]
    positions = np.arange(len(castillo_model_names))
    boxplot = ax.boxplot(arrays, positions=positions, widths=0.56, patch_artist=True, showfliers=False)
    for index, model in enumerate(castillo_model_names):
        boxplot["boxes"][index].set(
            facecolor=castillo_model_styles[model][1],
            edgecolor=box_edge,
            alpha=0.42,
            linewidth=box_linewidth,
        )
        boxplot["medians"][index].set(color=box_edge, linewidth=box_linewidth)
        for component in ("whiskers", "caps"):
            for line in boxplot[component][2 * index : 2 * index + 2]:
                line.set(color=box_edge, linewidth=box_linewidth)

    for cell_index, cell in enumerate(cells):
        values = np.array([array[cell_index] for array in arrays])
        valid = np.isfinite(values)
        ax.scatter(
            positions[valid],
            values[valid],
            s=21,
            color=castillo_cell_colors[cell],
            edgecolor="white",
            linewidth=0.3,
            zorder=4,
        )
    if baseline is not None:
        ax.axhline(baseline, color="#777777", linestyle="--", linewidth=0.8)
    ax.set(ylim=ylim, xticks=positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="y", labelrotation=90, labelsize=7)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    regression = pd.read_csv(metrics_dir / "castillo_regression_metrics.csv")
    classification = pd.read_csv(metrics_dir / "castillo_classification_metrics.csv")
    counts = pd.read_csv(metrics_dir / "castillo_cts_counts.csv")
    print(f"[load] {metrics_dir} {regression.shape} {classification.shape}")

    limits = axis_limits(regression, classification)
    fig, axes = plt.subplots(4, 5, figsize=(12, 12))

    for row, metric in enumerate(regression_rows):
        for column, setting in enumerate(regression_settings):
            subset = regression[regression["setting"].eq(setting)]
            draw_boxplot(
                axes[row, column],
                subset,
                metric,
                limits[metric],
                castillo_cell_types,
                baseline=0 if metric in {"pcc", "scc"} else None,
            )
            axes[row, column].set_ylabel(regression_labels[metric])
            if row == 0:
                n = int(subset["n"].iloc[0])
                title = setting.replace("CTS-union", "CTS union")
                axes[row, column].set_title(f"{title}\n(n={n:,} sequences)", fontsize=9)

        metric, label, screen_pct = classification_rows[row]
        for offset, task in enumerate(("CTS-high", "CTS-low")):
            column = 3 + offset
            subset = classification[
                classification["task"].eq(task) & classification["screen_pct"].eq(screen_pct)
            ]
            n_pos = subset[subset["model"].eq(castillo_model_names[0])].set_index("cell_type")["n_pos"]
            # too few positives makes AUROC/AUPRC/EF unstable or undefined
            cells = [cell for cell in castillo_cell_types if n_pos[cell] >= castillo_min_positives]
            baseline = {"auroc": 0.5, "normalized_auprc": 0.0, "ef": 1.0}[metric]
            scale_key = f"ef{int(screen_pct)}" if metric == "ef" else metric
            draw_boxplot(axes[row, column], subset, metric, limits[scale_key], cells, baseline=baseline)
            axes[row, column].set_ylabel(label)
            if row == 0:
                axes[row, column].set_title(
                    f"{task}\n(n+={int(n_pos[cells].sum()):,}; {len(cells)}/{len(castillo_cell_types)} cells)",
                    fontsize=9,
                )

    model_labels = [castillo_model_styles[name][0] for name in castillo_model_names]
    for axis in axes[-1]:
        axis.set_xticklabels(model_labels, rotation=38, ha="right", fontsize=7.5)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none",
            color=castillo_cell_colors[cell], markeredgecolor="white", label=cell,
        )
        for cell in castillo_cell_types
    ]
    fig.legend(
        handles=handles, loc="center left", bbox_to_anchor=(0.85, 0.5),
        frameon=False, title="Cell types",
    )
    union_n = int(counts["cts_union_n"].iloc[0])
    total_n = int(counts["total_n"].iloc[0])
    fig.suptitle(
        f"Castillo All sequences: cell-wise boxplots; CTS gap >= {castillo_cts_gap:g}; screen = 2% and 5%\n"
        f"CTS union = {union_n:,}/{total_n:,} sequences",
        x=0.43, y=0.985, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 0.84, 0.94), h_pad=1.9, w_pad=1.0)

    out_path = figures_dir / fig_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.resolve()}")


if __name__ == "__main__":
    main()
