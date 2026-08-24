import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from epicast import metrics
from epicast.utils.plot_utils import set_mpl_params, warm_cmap, coolwarm_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    cell_types,
    figures_dir,
    mpra_path,
    test_cell_types,
    train_cell_types,
    vef_paths,
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

assays = ["DNase"]
splits = ["total", "all_cts_1_99"]
models = ["sei", "alphagenome"]

model_vef_paths = {name: vef_paths[name] for name in models}

train_mean_activity = mpra_df[train_cell_types].mean(axis=1)


def get_row_mask(cell_type, split):
    if split == "total":
        return np.ones(len(mpra_df), dtype=bool)
    return get_mask(split, masks, cell_type=cell_type)


def residual_activity(cell_type):
    return (mpra_df[cell_type] - train_mean_activity).to_numpy()


def residual_vef(vef_df, cell_type, assay):
    train_mean = vef_df[[f"{ct}_{assay}" for ct in train_cell_types]].mean(axis=1)
    return (vef_df[f"{cell_type}_{assay}"] - train_mean).to_numpy()


def compute_vef_specificity_corr(vef_df, mpra_df, cell_types, assay, split):
    corr = pd.DataFrame(index=[f"{c}_{assay}" for c in cell_types], columns=cell_types, dtype=float)
    for c1 in cell_types:
        mask = get_row_mask(c1, split)
        for c2 in cell_types:
            x = vef_df[f"{c1}_{assay}"].to_numpy()[mask]
            y = mpra_df[c2].to_numpy()[mask]
            corr.loc[f"{c1}_{assay}", c2] = metrics.pearson(x, y)
    return corr


def compute_residual_corr(vef_df, cell_types, assay, split):
    corr = pd.DataFrame(index=[f"{c}_{assay}" for c in cell_types], columns=cell_types, dtype=float)
    for c1 in cell_types:
        mask = get_row_mask(c1, split)
        x = residual_vef(vef_df, c1, assay)[mask]
        for c2 in cell_types:
            y = residual_activity(c2)[mask]
            corr.loc[f"{c1}_{assay}", c2] = metrics.pearson(x, y)
    return corr


def plot_heatmap(corr, out_file, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    plt.subplots_adjust(left=0.3, bottom=0.2, right=0.9, top=0.9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.3)

    sns.heatmap(
        corr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
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
    ax.set_yticklabels(
        [label.get_text().replace("_", " ") for label in ax.get_yticklabels()],
        fontsize=16,
        rotation=0,
    )

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


for split in splits:
    for cell_type in cell_types:
        n = get_row_mask(cell_type, split).sum()
        print(f"[mask] {split} cell_type={cell_type} n={n}")

for model_name in models:
    vef_path = model_vef_paths[model_name]
    vef_df = pd.read_csv(vef_path, sep="\t")
    print(f"[load] {vef_path} {vef_df.shape}")

    for assay in assays:
        for split in splits:
            corr = compute_vef_specificity_corr(vef_df, mpra_df, cell_types, assay, split)
            print(f"[{model_name} {assay} {split} absolute]")
            print(corr)
            plot_heatmap(
                corr,
                figures_dir / f"fig1f_{model_name}_{assay}_{split}_vef_specificity_heatmap.pdf",
                warm_cmap,
                0.2,
                0.8,
            )

            corr_resid = compute_residual_corr(vef_df, cell_types, assay, split)
            print(f"[{model_name} {assay} {split} residual]")
            print(corr_resid)
            plot_heatmap(
                corr_resid,
                figures_dir / f"fig1f_{model_name}_{assay}_{split}_residual_vef_activity_heatmap.pdf",
                coolwarm_cmap,
                -0.6,
                0.6,
            )
