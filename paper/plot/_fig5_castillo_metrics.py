"""DEPRECATED: the five-panel Castillo figure, kept only for the CTS-low panel.

`fig5_castillo_metrics.py` replaced this script and drops CTS-low, because four of the
seven cell types carry fewer than 12 CTS-low positives (HepG2 0, HeLa-S3 1, SK-N-SH 6,
WERI-Rb-1 11). Run this one if you need that panel back; its outputs are all prefixed
with `_` so they cannot overwrite the current figure.

One PDF per manuscript panel, plus a shared cell-type legend. The panel headers, which
also carry the n printed to stdout by this script, are added by hand.

    5A  whole CRE set, activity          1 x 2, PCC and SCC
    5B  union CTS set, activity          1 x 2
    5C  union CTS set, residual          1 x 2
    5D  CTS-high prioritization          4 x 1, AUROC / normalized AUPRC / 2% EF / 5% EF
    5E  CTS-low prioritization           4 x 1

Every box summarizes the seven evaluated cell types and the coloured dots are those
cell types individually. 5A-5C compare correlation only: Castillo activities are never
z-scored, so an MAE or RMSE would mostly measure the offset between the two scales.
PCC and SCC share a y range across 5A, 5B and 5C so the drop from whole set to CTS set
to residual can be read directly.

Normalized AUPRC is (AUPRC - prevalence) / (1 - prevalence), computed in
analysis/12_eval_castillo.py; that is why its random expectation is drawn at 0.

CTS-low drops the cell types with too few positives to score, currently HepG2 (0) and
HeLa-S3 (1), so its boxes rest on five cell types rather than seven.

The metric tables and the boxplot style come from the final analysis by C.Z.; reads
results/castillo, written by analysis/12_eval_castillo.py.
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
    castillo_min_positives,
    castillo_model_names,
    castillo_model_styles,
    figures_dir,
    results_dir,
)

metrics_dir = results_dir / "castillo"
legend_name = "_fig5_legend.pdf"

# (setting in the regression table, output figure, draw model labels)
regression_panels = [
    ("All activity", "_fig5a_castillo_activity_whole.pdf", False),
    ("CTS-union activity", "_fig5b_castillo_activity_cts.pdf", False),
    ("CTS-union residual", "_fig5c_castillo_residual_cts.pdf", True),
]
regression_rows = [("pcc", "PCC"), ("scc", "SCC")]

# (task in the classification table, output figure)
classification_panels = [
    ("CTS-high", "_fig5d_castillo_cts_high.pdf"),
    ("CTS-low", "_fig5e_castillo_cts_low.pdf"),
]
# (column in the classification table, axis label, screening depth to read, random line)
classification_rows = [
    ("auroc", "AUROC", 5.0, 0.5),
    ("normalized_auprc", "Normalized AUPRC", 5.0, 0.0),
    ("ef", "2% EF", 2.0, 1.0),
    ("ef", "5% EF", 5.0, 1.0),
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


def draw_boxplot(ax, data, metric, ylim, cells, baseline):
    arrays = [
        data[data["model"].eq(model)].set_index("cell_type").reindex(cells)[metric].to_numpy(float)
        for model in castillo_model_names
    ]
    positions = np.arange(len(castillo_model_names))
    line_style = {"color": box_edge, "linewidth": box_linewidth}
    boxplot = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops=line_style,
        whiskerprops=line_style,
        capprops=line_style,
        boxprops={"edgecolor": box_edge, "linewidth": box_linewidth, "alpha": 0.42},
    )
    for box, model in zip(boxplot["boxes"], castillo_model_names):
        box.set_facecolor(castillo_model_styles[model][1])

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
    ax.axhline(baseline, color="#777777", linestyle="--", linewidth=0.8)
    ax.set(ylim=ylim, xticks=positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)


def label_models(ax):
    ax.set_xticklabels(
        [castillo_model_styles[name][0] for name in castillo_model_names],
        rotation=38,
        ha="right",
        fontsize=7.5,
    )


def plot_regression_panel(regression, setting, limits, with_labels, save_path):
    subset = regression[regression["setting"].eq(setting)]
    fig, axes = plt.subplots(1, len(regression_rows), figsize=(4.2, 2.4))
    fig.subplots_adjust(left=0.12, bottom=0.3 if with_labels else 0.08, right=0.98, top=0.97, wspace=0.42)

    for ax, (metric, label) in zip(axes, regression_rows):
        draw_boxplot(ax, subset, metric, limits[metric], castillo_cell_types, baseline=0.0)
        ax.set_ylabel(label, fontsize=9)
        if with_labels:
            label_models(ax)

    print(f"  [panel] {setting} n={int(subset['n'].iloc[0]):,}")
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_classification_panel(classification, task, save_path):
    fig, axes = plt.subplots(len(classification_rows), 1, figsize=(2.1, 7.4))
    fig.subplots_adjust(left=0.3, bottom=0.13, right=0.97, top=0.98, hspace=0.24)

    for ax, (metric, label, screen_pct, baseline) in zip(axes, classification_rows):
        subset = classification[
            classification["task"].eq(task) & classification["screen_pct"].eq(screen_pct)
        ]
        n_pos = subset[subset["model"].eq(castillo_model_names[0])].set_index("cell_type")["n_pos"]
        # too few positives makes AUROC/AUPRC/EF unstable or undefined
        cells = [cell for cell in castillo_cell_types if n_pos[cell] >= castillo_min_positives]
        if metric == "auroc":
            ylim = (0.0, 1.0)
        else:
            ylim = padded_limits(
                subset[subset["cell_type"].isin(cells)][metric],
                zero_floor=metric == "ef",
                ceiling=1.0 if metric == "normalized_auprc" else None,
            )
        draw_boxplot(ax, subset, metric, ylim, cells, baseline=baseline)
        ax.set_ylabel(label, fontsize=9)

    label_models(axes[-1])
    print(f"  [panel] {task} n+={int(n_pos[cells].sum()):,} cells={len(cells)}")
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_legend(save_path):
    fig, ax = plt.subplots(figsize=(1.6, 2.2))
    ax.set_axis_off()
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none",
            color=castillo_cell_colors[cell], markeredgecolor="white", label=cell,
        )
        for cell in castillo_cell_types
    ]
    ax.legend(handles=handles, loc="center", frameon=False, title="Cell types", fontsize=9)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    regression = pd.read_csv(metrics_dir / "castillo_regression_metrics.csv")
    classification = pd.read_csv(metrics_dir / "castillo_classification_metrics.csv")
    print(f"[load] {metrics_dir} {regression.shape} {classification.shape}")

    # one range per correlation metric over all three settings, so 5A-5C are comparable
    limits = {
        metric: padded_limits(regression[metric], ceiling=1.0) for metric, _ in regression_rows
    }

    saved = []
    for setting, fig_name, with_labels in regression_panels:
        out_path = figures_dir / fig_name
        plot_regression_panel(regression, setting, limits, with_labels, out_path)
        saved.append(out_path)

    for task, fig_name in classification_panels:
        out_path = figures_dir / fig_name
        plot_classification_panel(classification, task, out_path)
        saved.append(out_path)

    legend_path = figures_dir / legend_name
    plot_legend(legend_path)
    saved.append(legend_path)

    for p in saved:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
