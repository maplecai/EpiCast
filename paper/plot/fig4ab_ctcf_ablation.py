"""fig4a: CTCF-activity scatters before/after conditioning on the other VEFs.
fig4b: the same effect summarised across cell types. Reads analysis/11 output.

fig4a is a raw point cloud (~760k variants), so it residualizes the source
columns directly rather than round-tripping them through a csv.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cell_types, figures_dir, mpra_path, results_dir, vef_paths

ablation_path = results_dir / "ctcf_ablation/ctcf_ablation.csv"
vef_sources = ["alphagenome", "sei"]
scatter_cell_type = cell_types[0]

cell_palette = dict(zip(cell_types, sns.color_palette("tab10", len(cell_types))))

# (column in the ablation table, x tick label)
beta_panels = [
    ("beta_marginal", "activity ~ CTCF"),
    ("beta_given_dnase", "activity ~ DNase + CTCF"),
    ("beta_given_all3", "activity ~ DNase + H3K4me3\n+ H3K27ac + CTCF"),
]
partial_panels = [
    ("marginal_r", "corr(CTCF, activity)"),
    ("partial_r_given_dnase", "partial corr(CTCF, activity | DNase)"),
    ("partial_r_given_all3", "partial corr(CTCF, activity |\nDNase + H3K4me3 + H3K27ac)"),
]


def residualize(y, covars):
    return sm.OLS(y, sm.add_constant(covars, has_constant="add")).fit().resid


def scatter_panel(x, y, r, xlabel, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    ax.scatter(x, y, s=8, alpha=0.45, color="gray", rasterized=True)

    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, slope * grid + intercept, color="dimgray", lw=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.03, 0.97, f"r = {r:.3f}", transform=ax.transAxes, ha="left", va="top")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(save_path, dpi=400)
    print(f"[save] {save_path}")
    plt.close(fig)


def plot_fig4a(mpra_df, vef_df, stats_row, vef_source):
    ct = scatter_cell_type
    ctcf, dnase = f"{ct}_CTCF", f"{ct}_DNase"
    other3 = [dnase, f"{ct}_H3K4me3", f"{ct}_H3K27ac"]

    df = pd.concat([mpra_df[[ct]], vef_df[[ctcf] + other3]], axis=1).dropna()
    stem = f"fig4a_{vef_source}_{ct}"

    scatter_panel(
        df[ctcf].to_numpy(),
        df[ct].to_numpy(),
        stats_row["marginal_r"],
        f"virtual CTCF ({ct})",
        f"CRE activity ({ct})",
        figures_dir / f"{stem}_activity_vs_ctcf.pdf",
    )
    scatter_panel(
        residualize(df[ctcf], df[[dnase]]).to_numpy(),
        residualize(df[ct], df[[dnase]]).to_numpy(),
        stats_row["partial_r_given_dnase"],
        f"Residual virtual CTCF | Virtual DNase ({ct})",
        f"Residual activity | Virtual DNase ({ct})",
        figures_dir / f"{stem}_residual_activity_vs_residual_ctcf_given_dnase.pdf",
    )
    scatter_panel(
        residualize(df[ctcf], df[other3]).to_numpy(),
        residualize(df[ct], df[other3]).to_numpy(),
        stats_row["partial_r_given_all3"],
        f"Residual virtual CTCF | other 3 VEFs ({ct})",
        f"Residual activity | other 3 VEFs ({ct})",
        figures_dir / f"{stem}_residual_activity_vs_residual_ctcf_given_other3vef.pdf",
    )


def plot_box(sub_df, panels, ylabel, save_path):
    positions = np.array([1.0, 2.4, 3.8])
    groups = [sub_df[col].to_numpy() for col, _ in panels]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "whitesmoke", "edgecolor": "black", "linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
    )
    for patch in bp["boxes"]:
        patch.set_alpha(0.55)

    x_offsets = np.linspace(-0.20, 0.20, len(sub_df))
    for i, (col, _) in enumerate(panels):
        for (_, row), x_off in zip(sub_df.iterrows(), x_offsets):
            ax.scatter(
                positions[i] + x_off,
                row[col],
                s=72,
                color=cell_palette[row["cell_type"]],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            label=ct,
            markerfacecolor=cell_palette[ct],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8,
        )
        for ct in cell_types
    ]
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=0)
    ax.set_xlim(0.45, 4.35)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in panels], fontsize=12)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, frameon=False, ncol=1, fontsize=12, loc="upper right")
    fig.savefig(save_path, dpi=400)
    print(f"[save] {save_path}")
    plt.close(fig)


def main():
    sns.set_theme(style="white", context="talk")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not ablation_path.exists():
        raise FileNotFoundError(f"未找到 {ablation_path}，请先运行 analysis/11_ctcf_ablation.py")
    ablation_df = pd.read_csv(ablation_path)
    print(f"[load] {ablation_path} {ablation_df.shape}")

    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")

    for vef_source in vef_sources:
        sub_df = ablation_df[ablation_df["vef_source"] == vef_source].set_index(
            "cell_type"
        ).loc[cell_types].reset_index()

        vef_df = pd.read_csv(vef_paths[vef_source], sep="\t")
        print(f"[load] {vef_paths[vef_source]} {vef_df.shape}")
        stats_row = sub_df.set_index("cell_type").loc[scatter_cell_type]
        plot_fig4a(mpra_df, vef_df, stats_row, vef_source)

        plot_box(
            sub_df,
            beta_panels,
            "Standardized β (CTCF)",
            figures_dir / f"fig4b_{vef_source}_ctcf_beta_sign_flip_boxplot.pdf",
        )
        plot_box(
            sub_df,
            partial_panels,
            "Pearson r (CTCF vs activity)",
            figures_dir / f"fig4b_{vef_source}_ctcf_partial_r_sign_flip_boxplot.pdf",
        )


if __name__ == "__main__":
    main()
