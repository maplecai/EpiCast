"""Top-k retrieval curves for the CTS task. Reads results/figure_metrics, written by analysis/15."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    build_styles,
    figure_model_names,
    figures_dir,
    figure_metrics_dir,
    test_cell_types,
)

eval_split = "test"

model_names = figure_model_names
model_plot_names, colors = build_styles(model_names)

task_figs = {
    "CTS_high": "fig3e",
    "CTS_low": "fig3f",
}
curve_tables = {
    "CTS_high": "fig3e_retrieval_cts_high.tsv",
    "CTS_low": "fig3f_retrieval_cts_low.tsv",
}

metric_cfg = {
    "ef": {
        "col": "ef",
        "ylabel": "Enrichment factor@k",
        "stem": "ef",
        "baseline": "one",
    },
    "p": {
        "col": "precision",
        "ylabel": "Precision@k",
        "stem": "p",
        "baseline": "prevalence",
    },
    "nns": {
        "col": "nns",
        "ylabel": "Number needed to screen@k",
        "stem": "nns",
        "baseline": "inv_prevalence",
    },
    "r": {
        "col": "recall",
        "ylabel": "Recall@k",
        "stem": "r",
        "baseline": "fraction",
    },
}

min_frac = 0.001
max_frac = 0.10
fraction_ticks = [0.1, 1, 2, 5, 10]


def load_curves(cell_type: str, task: str) -> dict[str, pd.DataFrame]:
    path = figure_metrics_dir / curve_tables[task]
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先运行 analysis/15_export_figure_metrics.py")
    curve_df = pd.read_csv(path, sep="\t")
    curve_df = curve_df[curve_df["cell_type"] == cell_type]
    print(f"[load] {path} {cell_type} {curve_df.shape}")
    window = curve_df[(curve_df["k_frac"] >= min_frac) & (curve_df["k_frac"] <= max_frac)]
    return dict(tuple(window.groupby("model")))


def plot_legend(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3, 4), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    ax.set_axis_off()

    handles = [
        Line2D([0], [0], color=color, lw=1.8, label=label)
        for color, label in zip(colors, model_plot_names)
    ]
    handles.append(Line2D([0], [0], color="gray", lw=1.2, linestyle="--", label="Random"))
    ax.legend(
        handles=handles,
        loc="center",
        frameon=True,
        fontsize=12,
        handlelength=1.5,
        labelspacing=0.6,
    )
    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def plot_metric_curve(
    curves: dict[str, pd.DataFrame],
    save_path: Path,
    title: str,
    metric: str,
) -> None:
    cfg = metric_cfg[metric]
    y_col = cfg["col"]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)

    ymax = 0.0
    prevalence = None
    for model_name, color in zip(model_names, colors):
        curve = curves[model_name]
        y = curve[y_col].to_numpy(dtype=float)
        ax.plot(curve["k_pct"].to_numpy(), y, color=color, lw=1.8)
        if np.isfinite(y).any():
            ymax = max(ymax, float(np.nanmax(y)))
        prevalence = float(curve["prevalence"].iloc[0])

    if cfg["baseline"] == "prevalence":
        ax.axhline(prevalence, color="gray", lw=1.2, linestyle="--")
        ymax = max(ymax, prevalence)
    elif cfg["baseline"] == "one":
        ax.axhline(1.0, color="gray", lw=1.2, linestyle="--")
        ymax = max(ymax, 1.0)
    elif cfg["baseline"] == "inv_prevalence":
        inv_prev = 1.0 / prevalence if prevalence > 0 else np.nan
        ax.axhline(inv_prev, color="gray", lw=1.2, linestyle="--")
        if np.isfinite(inv_prev):
            ymax = max(ymax, inv_prev)
    else:
        xs = np.array([min_frac, max_frac]) * 100.0
        ys = np.array([min_frac, max_frac])
        ax.plot(xs, ys, color="gray", lw=1.2, linestyle="--")
        ymax = max(ymax, max_frac)

    ax.set_xlim(min_frac * 100, max_frac * 100)
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.set_xticks(fraction_ticks)
    ax.set_xlabel("Top-ranked candidate fraction (%)")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def main() -> None:
    set_mpl_params()
    sns.set_theme(style="white", context="notebook")
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for cell_type in test_cell_types:
        for task, fig_prefix in task_figs.items():
            curves = load_curves(cell_type, task)
            for metric, cfg in metric_cfg.items():
                out_path = (
                    figures_dir
                    / f"{fig_prefix}_{cell_type}_{task.lower()}_{cfg['stem']}_curve.pdf"
                )
                plot_metric_curve(curves, out_path, f"{cell_type} ({task}, test)", metric)
                saved_paths.append(out_path)

    legend_path = figures_dir / "fig3e_legend.pdf"
    plot_legend(legend_path)
    saved_paths.append(legend_path)

    for p in saved_paths:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
