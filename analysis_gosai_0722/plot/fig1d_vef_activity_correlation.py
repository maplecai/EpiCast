import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from epicast import metrics
from epicast.utils.plot_utils import set_mpl_params, warm_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import assays, cell_types, figures_dir, mpra_path, vef_paths

set_mpl_params()

mpra_df = pd.read_csv(mpra_path, sep="\t")
print(f"[load] {mpra_path} {mpra_df.shape}")

model_vef_paths = vef_paths


def compute_corr(vef_df, mpra_df, cell_types, assays):
    corr = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for cell_type in cell_types:
        for assay in assays:
            pred = vef_df[f"{cell_type}_{assay}"].to_numpy()
            true = mpra_df[cell_type].to_numpy()
            corr.loc[cell_type, assay] = metrics.pearson(pred, true)
    return corr


def plot_corr_heatmap(corr, out_file):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.3)

    sns.heatmap(
        corr.T,
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


for model_name, vef_path in model_vef_paths.items():
    vef_df = pd.read_csv(vef_path, sep="\t")
    print(f"[load] {vef_path} {vef_df.shape}")

    corr = compute_corr(vef_df, mpra_df, cell_types, assays)
    print(corr)

    out_file = figures_dir / f"fig1d_{model_name}_vef_activity_correlation_heatmap.pdf"
    plot_corr_heatmap(corr, out_file)
