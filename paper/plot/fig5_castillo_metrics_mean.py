"""fig5A-5E, mean +- SD variant: zero-shot evaluation on the independent Castillo-Hair MPRA.

Identical to fig5_castillo_metrics.py except that the seven cell-type values are summarized
by their mean with a one-SD error bar instead of by a box. The two exist side by side so
the style can be chosen on the figure; whichever wins, the other script and its PDFs get
deleted. This one writes `..._mean.pdf` and no legend, since the legend of the main script
serves both.

One PDF per manuscript panel. The panel headers, which also carry the n printed to stdout
by this script, are added by hand.

    5A  whole CRE set, activity          1 x 2, PCC and SCC
    5B  union CTS set, activity          1 x 2
    5C  union CTS set, residual          1 x 2
    5D  CTS-high prioritization          2 x 2, AUROC / normalized AUPRC / 2% EF / 5% EF
    5E  CTS-low prioritization           2 x 2, same four metrics

The coloured dots are the seven evaluated cell types, summarized by a wide bar at their
mean and a thin vertical line one sample SD (ddof=1) in each direction, capped by two
half-width bars, the same summary the mean variants of fig1C and fig4B use. Nothing is
filled, so the points stay the most visible thing in the figure. Colour in this
figure only ever means cell type. Every panel carries the model names on its x axis, so
stacking 5A-5C leaves three copies to delete by hand; that is the cheaper mistake, since a
panel that lands on its own has to say which column is which model.

5A-5C compare correlation only: Castillo
activities are never z-scored, so an MAE or RMSE would mostly measure the offset between
the two scales. All six axes of 5A, 5B and 5C share one y range, PCC and SCC included,
so the drop from whole set to CTS set to residual can be read directly.

Normalized AUPRC is (AUPRC - prevalence) / (1 - prevalence), computed in
analysis/12_eval_castillo.py; that is why its random expectation is drawn at 0.

5E rests on three cell types only: HepG2 (0), HeLa-S3 (1), SK-N-SH (6) and WERI-Rb-1 (11)
fall below the castillo_min_positives threshold of 20 and drop out, leaving K562 (84),
GM12878 (169) and MCF-7 (281). The counts per cell type are in
results/castillo/castillo_cts_counts.csv and have to be stated in the caption.

Font sizes come from seaborn's "talk" context and are never set here: every figure in this
bundle ends up as one panel among several and gets scaled down, so the text has to start
out large. To make a figure look finer, grow its figsize; the text then reads smaller
relative to the whole. Margins are the same everywhere and do not need tuning because the
figure is saved with a tight bounding box.

The metric tables and the panel layout come from the final analysis by C.Z.; reads
results/castillo, written by analysis/12_eval_castillo.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params

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

# (task in the classification table, output figure)
classification_panels = [
    ("CTS-high", "fig5d_castillo_cts_high_mean.pdf"),
    ("CTS-low", "fig5e_castillo_cts_low_mean.pdf"),
]

# (setting in the regression table, output figure)
regression_panels = [
    ("All activity", "fig5a_castillo_activity_whole_mean.pdf"),
    ("CTS-union activity", "fig5b_castillo_activity_cts_mean.pdf"),
    ("CTS-union residual", "fig5c_castillo_residual_cts_mean.pdf"),
]
regression_rows = [("pcc", "PCC"), ("scc", "SCC")]

# (column in the classification table, axis label, screening depth to read, random line),
# filled row-major into the 2 x 2 grid of 5D
classification_cells = [
    ("auroc", "AUROC", 5.0, 0.5),
    ("normalized_auprc", "Normalized AUPRC", 5.0, 0.0),
    ("ef", "2% EF", 2.0, 1.0),
    ("ef", "5% EF", 5.0, 1.0),
]

summary_width = 0.56
summary_linewidth = 1.5
point_size = 22


def padded_limits(values, zero_floor=False, ceiling=None):
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 0.1)
    lower = 0.0 if zero_floor else min(0.0, low - 0.08 * span)
    upper = high + 0.10 * span
    if ceiling is not None:
        upper = min(ceiling, upper)
    return lower, upper


def draw_summary(ax, x, values, width=summary_width):
    """A bar at the mean and a capped line one sample SD in each direction."""
    mean, sd = np.nanmean(values), np.nanstd(values, ddof=1)
    ax.vlines(x, mean - sd, mean + sd, color="black", lw=1.0, zorder=3)
    ax.hlines(mean, x - width / 2, x + width / 2, color="black", lw=summary_linewidth, zorder=3)
    for cap in (mean - sd, mean + sd):
        ax.hlines(cap, x - width / 4, x + width / 4, color="black", lw=1.0, zorder=3)


def draw_points(ax, data, metric, ylim, cells, baseline):
    arrays = [
        data[data["model"] == model].set_index("cell_type").reindex(cells)[metric].to_numpy(float)
        for model in castillo_model_names
    ]
    positions = np.arange(len(castillo_model_names))
    for position, array in zip(positions, arrays):
        draw_summary(ax, position, array)

    for cell_index, cell in enumerate(cells):
        values = np.array([array[cell_index] for array in arrays])
        valid = np.isfinite(values)
        ax.scatter(
            positions[valid],
            values[valid],
            s=point_size,
            color=castillo_cell_colors[cell],
            zorder=4,
        )
    ax.axhline(baseline, color="gray", lw=1.0, linestyle="--")
    ax.set_ylim(ylim)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [castillo_model_styles[name][0] for name in castillo_model_names], rotation=90
    )
    ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_regression_panel(regression, setting, limits, save_path):
    subset = regression[regression["setting"] == setting]
    fig, axes = plt.subplots(1, len(regression_rows), figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.4)

    for ax, (metric, label) in zip(axes, regression_rows):
        draw_points(ax, subset, metric, limits, castillo_cell_types, baseline=0.0)
        ax.set_ylabel(label)

    print(f"  [panel] {setting} n={int(subset['n'].iloc[0]):,}")
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_classification_panel(classification, task, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(8, 12), dpi=100)
    # the vertical model names of the upper row need the whole gap to themselves, otherwise
    # the lower row's opaque background paints over them
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, hspace=0.45, wspace=0.4)

    for ax, (metric, label, screen_pct, baseline) in zip(axes.ravel(), classification_cells):
        subset = classification[
            (classification["task"] == task) & (classification["screen_pct"] == screen_pct)
        ]
        n_pos = subset[subset["model"] == castillo_model_names[0]].set_index("cell_type")["n_pos"]
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
        draw_points(ax, subset, metric, ylim, cells, baseline=baseline)
        ax.set_ylabel(label)

    print(f"  [panel] {task} n+={int(n_pos[cells].sum()):,} cells={len(cells)}")
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    regression = pd.read_csv(metrics_dir / "castillo_regression_metrics.csv")
    classification = pd.read_csv(metrics_dir / "castillo_classification_metrics.csv")
    print(f"[load] {metrics_dir} {regression.shape} {classification.shape}")

    # a single range for both metrics over all three settings, so every box in 5A-5C is
    # directly comparable, PCC against SCC included
    limits = padded_limits(
        regression[[metric for metric, _ in regression_rows]].to_numpy(), ceiling=1.0
    )

    saved = []
    for setting, fig_name in regression_panels:
        out_path = figures_dir / fig_name
        plot_regression_panel(regression, setting, limits, out_path)
        saved.append(out_path)

    for task, fig_name in classification_panels:
        out_path = figures_dir / fig_name
        plot_classification_panel(classification, task, out_path)
        saved.append(out_path)

    for p in saved:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
