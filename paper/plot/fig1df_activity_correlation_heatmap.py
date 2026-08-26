"""fig1D / fig1F: how correlated measured CRE activity is between cell types.

Two 5 x 5 heatmaps of the same shape and the same colour scale: fig1D over the whole CRE
set, fig1F over the union CTS set. Reading them side by side is the argument the paper
rests on. On the whole library the cell types are highly correlated, so most of the
activity of a CRE is shared; inside the CTS set the correlation collapses, which is what
makes that subset the place where a cell-context-aware model can matter.

Only the lower triangle is drawn, diagonal included. The upper right is left completely
empty, without cell borders, and the triangle is outlined in black. The two panels share
one colour scale, so the colour bar is a third file of its own.

Font sizes come from seaborn's "talk" context and are never set here: every figure in this
bundle ends up as one panel among several and gets scaled down, so the text has to start
out large. To make a figure look finer, grow its figsize; the text then reads smaller
relative to the whole. Margins are the same everywhere and do not need tuning because the
figure is saved with a tight bounding box.

Correlations are pairwise complete: HCT116 and A549 are assayed on part of the library,
and their cells use whatever CREs the two cell types have in common.

Reads the MPRA table directly.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast.utils.plot_utils import set_mpl_params, warm_cmap
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cell_types, figures_dir, mpra_path, test_cell_types, train_cell_types
from utils import build_masks

# (mask key, output figure); "total" is the whole library, "all_cts_1_99" the union CTS set
panels = [
    ("total", "fig1d_activity_correlation_whole.pdf"),
    ("all_cts_1_99", "fig1f_activity_correlation_cts.pdf"),
]
colorbar_name = "fig1df_colorbar.pdf"

color_limits = (0.0, 1.0)
color_ticks = [0.0, 0.5, 1.0]
cbar_fraction = 0.6
cell_edge = "gray"
outline = "black"


def staircase(n):
    """Outline of the lower triangle of an n x n heatmap, in cell coordinates."""
    points = [(0, 0)]
    for i in range(n):
        points += [(i + 1, i), (i + 1, i + 1)]
    return points + [(0, n)]


def plot_heatmap(corr, save_path):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)

    n = len(corr)
    sns.heatmap(
        corr,
        mask=np.triu(np.ones((n, n), dtype=bool), k=1),
        cmap=warm_cmap,
        vmin=color_limits[0],
        vmax=color_limits[1],
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"color": "black"},
        cbar=False,
        linewidths=0,
        ax=ax,
    )
    ax.set_xticklabels(cell_types, rotation=90)
    ax.set_yticklabels(cell_types, rotation=0)

    # borders drawn by hand so the empty upper right stays free of grid lines
    for row in range(n):
        for col in range(row + 1):
            ax.add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor=cell_edge, lw=0.5))
    ax.add_patch(Polygon(staircase(n), closed=True, fill=False, edgecolor=outline, lw=1.0))

    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_colorbar(save_path):
    """The bar is 0.6 of the heatmap side, i.e. 0.6 of the shared panel height of 6."""
    fig = plt.figure(figsize=(2, 6), dpi=100)
    ax = fig.add_axes([0.1, 0.5 - cbar_fraction / 2, 0.12, cbar_fraction])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(*color_limits), warm_cmap),
        cax=ax,
        ticks=color_ticks,
        label="PCC",
    )
    colorbar.outline.set(edgecolor=outline, linewidth=1.0)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    set_mpl_params()
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42})
    figures_dir.mkdir(parents=True, exist_ok=True)

    mpra_df = pd.read_csv(mpra_path, sep="\t")
    print(f"[load] {mpra_path} {mpra_df.shape}")
    masks = build_masks(mpra_df, cell_types, train_cell_types, test_cell_types, verbose=False)
    masks["total"] = np.ones(len(mpra_df), dtype=bool)

    for mask_key, fig_name in panels:
        subset = mpra_df.loc[masks[mask_key], cell_types]
        print(f"[mask] {mask_key} n={len(subset):,}")
        corr = subset.corr()
        print(corr.round(3))

        out_path = figures_dir / fig_name
        plot_heatmap(corr, out_path)
        print(f"[save] {out_path.resolve()}")

    colorbar_path = figures_dir / colorbar_name
    plot_colorbar(colorbar_path)
    print(f"[save] {colorbar_path.resolve()}")


if __name__ == "__main__":
    main()
