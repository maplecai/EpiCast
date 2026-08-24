"""ROC / PR curves for the CTS classification task. Reads results/figure_metrics, written by
analysis/15."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

eval_split = "test"

model_names = figure_model_names
model_plot_names, colors = build_styles(model_names)

# (task, figure prefix); the results/figure_metrics curve tables are named "{prefix}_{task}_{kind}"
task_figs = [
    ("CTS_high", "fig3c"),
    ("CTS_low", "fig3d"),
]


def load_curves(cell_type: str, table_stem: str, kind: str) -> dict[str, pd.DataFrame]:
    path = figure_metrics_dir / f"{table_stem}_{kind}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先运行 analysis/15_export_figure_metrics.py")
    curve_df = pd.read_csv(path, sep="\t")
    curve_df = curve_df[curve_df["cell_type"] == cell_type]
    print(f"[load] {path} {cell_type} {curve_df.shape}")
    return dict(tuple(curve_df.groupby("model")))


def plot_roc(curves: dict[str, pd.DataFrame], save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)

    for model_name, plot_name, color in zip(model_names, model_plot_names, colors):
        curve = curves.get(model_name)
        if curve is None:
            continue
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            color=color,
            lw=1.5,
            label=f"{plot_name} (AUROC={curve['auroc'].iloc[0]:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def plot_prc(curves: dict[str, pd.DataFrame], save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)

    prevalence = None
    for model_name, plot_name, color in zip(model_names, model_plot_names, colors):
        curve = curves.get(model_name)
        if curve is None:
            continue
        prevalence = curve["prevalence"].iloc[0]
        ax.plot(
            curve["recall"],
            curve["precision"],
            color=color,
            lw=1.5,
            label=f"{plot_name} (AUPRC={curve['auprc'].iloc[0]:.3f})",
        )
    if prevalence is not None:
        ax.axhline(
            prevalence, color="gray", lw=1, linestyle="--", label=f"prevalence={prevalence:.3f}"
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
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
    for task, fig_prefix in task_figs:
        for cell_type in test_cell_types:
            title = f"{cell_type} ({task}, {eval_split})"
            task_stem = task.lower()
            table_stem = f"{fig_prefix}_{task_stem}"

            roc_path = figures_dir / f"{fig_prefix}_{cell_type}_{task_stem}_roc.pdf"
            plot_roc(load_curves(cell_type, table_stem, "roc"), roc_path, title)

            prc_path = figures_dir / f"{fig_prefix}_{cell_type}_{task_stem}_prc.pdf"
            plot_prc(load_curves(cell_type, table_stem, "pr"), prc_path, title)
            saved_paths.extend([roc_path, prc_path])

    for p in saved_paths:
        print(f"[save] {p.resolve()}")


if __name__ == "__main__":
    main()
