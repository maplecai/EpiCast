from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gosai_plot import set_plot_theme


model_names = [
    "sei_dnase",
    "enformer_dnase",
    "borzoi_dnase",
    "alphagenome_dnase",
    "linear",
    "mlp",
    "xgb",
    "lgbm",
    "seq_only",
    "epicast",
]

model_plot_names = [
    "Sei DNase",
    "Enformer DNase",
    "Borzoi DNase",
    "AlphaGenome DNase",
    "Linear",
    "MLP",
    "XGBoost",
    "LightGBM",
    "Seq only",
    "EpiCast",
]

colors = [
    "#B7D9D3",
    "#8FC2C8",
    "#6FA8C6",
    "#4F7FA8",
    "#E7D8A6",
    "#DDBE8A",
    "#D39E7A",
    "#C97F6D",
    "#B6A9CC",
    "#8E7FAF",
]

cell_types = ["K562", "HepG2", "SK-N-SH", "HCT116", "A549"]
results_dir = Path("analyze_gosai/results/loo_correlation")
figures_dir = Path("analyze_gosai/figures")
fig_prefix = "fig3.1_"


def load_mean_performance(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.loc[model_names, cell_types].mean(axis=1)


def plot_legend(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3, 4), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    ax.set_axis_off()

    handles = [
        mpatches.Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=label)
        for color, label in zip(colors, model_plot_names)
    ]
    ax.legend(handles=handles, loc="center", frameon=True, fontsize=12, handlelength=0.9, handleheight=0.9, labelspacing=0.6)

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def plot_bar(values: pd.Series, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    x = np.arange(len(model_names))

    ax.bar(
        x,
        values.loc[model_names].values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        width=1,
    )
    ax.set_xlabel("Models")
    ax.set_ylim(0.0, 0.8)
    ax.set_ylabel("Correlation")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="lightgray", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    fig.savefig(save_path, dpi=400)
    plt.close(fig)


def main() -> None:
    tsv_files = sorted(results_dir.glob("leave_one_out_*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"未找到 TSV: {results_dir}")

    set_plot_theme(style="whitegrid", context="notebook")
    figures_dir.mkdir(parents=True, exist_ok=True)

    for path in tsv_files:
        values = load_mean_performance(path)
        plot_bar(values, figures_dir / f"{fig_prefix}{path.stem}.pdf")

    plot_legend(figures_dir / f"{fig_prefix}legend.pdf")


if __name__ == "__main__":
    main()
