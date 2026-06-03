import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpra_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_zscore.tsv', sep='\t')
print(mpra_df.shape)

cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']

# define masks (CTS / all_specific: same as 6_compare_leave_one_out.py)
masks = {}
masks["total"] = np.ones(len(mpra_df), dtype=bool)
masks["train"] = ~mpra_df["chr"].isin(["chr7", "chr13", "chr19", "chr21", "chrX"])
masks["val"] = mpra_df["chr"].isin(["chr19", "chr21", "chrX"])
masks["test"] = mpra_df["chr"].isin(["chr7", "chr13"])

for cell_type in cell_types:
    other_cell_types = [f"{ct}" for ct in cell_types if ct != cell_type]
    second_highest = mpra_df[other_cell_types].max(axis=1)
    gap_vs_second = mpra_df[cell_type] - second_highest
    q99 = np.percentile(gap_vs_second.dropna(), 99)
    masks[f"{cell_type}_specific"] = gap_vs_second > q99
    print(f"{cell_type}_specific:", masks[f"{cell_type}_specific"].sum())

cts_mask = np.zeros(len(mpra_df), dtype=bool)
for cell_type in cell_types:
    cts_mask |= masks[f"{cell_type}_specific"]
masks["cts"] = cts_mask
print("all_specific (cts):", masks["cts"].sum())




def plot_activity_corr_heatmap(
    mpra_df,
    cell_types,
    mask,
    out_file,
    title=None,
    cmap='coolwarm',
    vmin=0,
    vmax=1
):
    corr_df = mpra_df.loc[mask, cell_types].corr(method='pearson')

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.3)

    sns_plot = sns.heatmap(
        corr_df,
        ax=ax,
        cbar_ax=cax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt='.2f',
        square=True,
        cbar=True,
        annot_kws={'size': 16, 'color': 'black'},
        linewidths=0.5,
        linecolor="gray",
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor("black")

    cbar = sns_plot.collections[0].colorbar
    cbar.outline.set_linewidth(2)
    cbar.outline.set_edgecolor("black")

    if title is not None:
        ax.set_title(title)

    plt.savefig(out_file)
    plt.show()

    return corr_df


corr_total = plot_activity_corr_heatmap(
    mpra_df=mpra_df,
    cell_types=cell_types,
    mask=masks["total"],
    out_file='analyze_gosai/figures/fig2.4.1_activity_pairwise_pearson_heatmap.pdf',
    title='Activity pairwise Pearson correlation (all sequences)'
)

print(corr_total)



corr_cts = plot_activity_corr_heatmap(
    mpra_df=mpra_df,
    cell_types=cell_types,
    mask=masks["cts"],
    out_file='analyze_gosai/figures/fig2.4.2_cts_activity_pairwise_pearson_heatmap.pdf',
    title='Activity pairwise Pearson correlation (CTS sequences)'
)

print(corr_cts)
