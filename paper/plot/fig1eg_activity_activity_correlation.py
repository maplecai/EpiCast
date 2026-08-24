import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast import metrics
from epicast.utils.plot_utils import set_mpl_params, warm_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    cell_types,
    figures_dir,
    mpra_path,
    test_cell_types,
    train_cell_types,
)
from utils import (
    build_masks,
    get_mask,
)

set_mpl_params()

mpra_df = pd.read_csv(mpra_path, sep="\t")
print(f"[load] {mpra_path} {mpra_df.shape}")
masks = build_masks(
    mpra_df,
    cell_types,
    train_cell_types=train_cell_types,
    test_cell_types=test_cell_types,
    verbose=False,
)


def get_row_mask(cell_type, split):
    if split == "total":
        return np.ones(len(mpra_df), dtype=bool)
    return get_mask(split, masks, cell_type=cell_type)


def compute_activity_corr(split):
    corr = pd.DataFrame(index=cell_types, columns=cell_types, dtype=float)
    for c1 in cell_types:
        mask = get_row_mask(c1, split)
        print(f"[mask] {split} cell_type={c1} n={mask.sum()}")
        x = mpra_df[c1].to_numpy()[mask]
        for c2 in cell_types:
            y = mpra_df[c2].to_numpy()[mask]
            corr.loc[c1, c2] = metrics.pearson(x, y)
    return corr


def plot_heatmap(corr, out_file):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.3)

    sns.heatmap(
        corr,
        cmap=warm_cmap,
        vmin=0.0,
        vmax=1.0,
        square=True,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 16, "color": "black"},
        cbar=True,
        cbar_ax=cax,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)

    n = corr.shape[0]
    for i in range(n):
        ax.add_patch(
            patches.Rectangle(
                (i, i),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=2.5,
                clip_on=False,
            )
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor("black")

    outline = cax.spines["outline"]
    outline.set_linewidth(2)
    outline.set_edgecolor("black")

    plt.savefig(out_file, dpi=400)
    print(f"[save] {out_file}")
    plt.close(fig)


corr_total = compute_activity_corr("total")
print("[total]")
print(corr_total)
plot_heatmap(
    corr_total,
    figures_dir / "fig1e_total_activity_activity_correlation_heatmap.pdf",
)

corr_all_cts_1_99 = compute_activity_corr("all_cts_1_99")
print("[all_cts_1_99]")
print(corr_all_cts_1_99)
plot_heatmap(
    corr_all_cts_1_99,
    figures_dir / "fig1g_all_cts_1_99_activity_activity_correlation_heatmap.pdf",
)
