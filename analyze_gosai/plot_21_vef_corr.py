import sys
import numpy as np
import pandas as pd
from epicast import models, datasets, utils, metrics
import matplotlib.pyplot as plt
import seaborn as sns

mpra_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_zscore.tsv', sep='\t')
print(mpra_df.shape)

vef_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_ag_vef_x10_log1p.tsv', sep='\t')
print(vef_df.shape)

cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']


masks = {}
masks['total'] = np.ones(len(mpra_df), dtype=bool)
masks['train'] = ~mpra_df['chr'].isin(['chr7', 'chr13', 'chr19', 'chr21', 'chrX'])
masks['val'] = mpra_df['chr'].isin(['chr19', 'chr21', 'chrX'])
masks['test'] = mpra_df['chr'].isin(['chr7', 'chr13'])

# std = mpra_df[cell_types].std(axis=1, skipna=True)
# thr = np.percentile(std, 99)
# masks['high_std'] = (std > thr)


# mpra_df = mpra_df[masks['high_std']]
# vef_df = vef_df[masks['high_std']]


# for assay in assays:
#     corr = pd.DataFrame()
#     for i, c1 in enumerate(cell_types):
#         for j, c2 in enumerate(cell_types):
#             x = vef_df[f'{c1}_{assay}']
#             y = mpra_df[f'{c2}']
#             r = metrics.pearson(x, y)
#             # r = metrics.spearman(x, y)
#             corr.loc[f'{c1}_{assay}', f'{c2}'] = r

#     sns.set_theme(style="white", context="talk")
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
#     plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)
#     sns_plot = sns.heatmap(
#         corr,
#         cmap="coolwarm",
#         vmin=0.0,
#         vmax=0.8,
#         square=True,
#         annot=True,
#         fmt=".3f",
#         annot_kws={"size": 16, "color": "black"},
#         cbar=True,
#         linewidths=0.5,
#         linecolor="gray",
#         ax=ax
#     )


#     # === 边框设置 ===
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_linewidth(2)
#         spine.set_edgecolor("black")

#     # === 色条（colorbar）设置 ===
#     cbar = sns_plot.collections[0].colorbar
#     cbar.outline.set_linewidth(2)
#     cbar.outline.set_edgecolor("black")

#     plt.savefig(f'analyze_gosai/figures/{assay}_heatmap.png', dpi=400)
#     # plt.savefig('analyze_gosai/figures/gosai_mpra_vef_activity_heatmap.pdf', dpi=400)




corr = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
for i, cell_type in enumerate(cell_types):
    for j, assay in enumerate(assays):
        pred = vef_df[f'{cell_type}_{assay}']
        true = mpra_df[cell_type]
        r = metrics.pearson(pred, true)
        corr.loc[cell_type, assay] = r


from mpl_toolkits.axes_grid1 import make_axes_locatable

# === 图形配置 ===
sns.set_theme(style="white", context="talk")
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

# === 创建与主图同高的 colorbar 轴 ===
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.3)  # pad 调整间距

# === 热图绘制 ===
sns_plot = sns.heatmap(
    corr.T,
    cmap="coolwarm",
    vmin=0.0,
    vmax=0.8,
    square=True,
    annot=True,
    fmt=".3f",
    annot_kws={"size": 16, "color": "black"},
    cbar=True,
    cbar_ax=cax,
    linewidths=0.5,
    linecolor="gray",
    ax=ax
)


# === 坐标轴美化 ===
ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=45)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)

# === 边框设置 ===
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_edgecolor("black")

# === 色条（colorbar）设置 ===
cbar = sns_plot.collections[0].colorbar
cbar.outline.set_linewidth(2)
cbar.outline.set_edgecolor("black")

# === 输出 ===
# plt.savefig('analyze_gosai/figures/gosai_mpra_vef_activity_heatmap.png', dpi=400)
plt.savefig('analyze_gosai/figures/fig2.1_vef_activity_heatmap.pdf', dpi=400)
plt.show()
