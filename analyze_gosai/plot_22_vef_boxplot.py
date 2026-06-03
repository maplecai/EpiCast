import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import epicast

mpra_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_zscore.tsv', sep='\t')
print(mpra_df.shape)

vef_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_ag_vef_x10_log1p.tsv', sep='\t')
print(vef_df.shape)

cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']
mpra_cols = cell_types
vef_cols = [f'{ct}_{assay}' for ct in cell_types for assay in assays]



from scipy.stats import mannwhitneyu

threshold = 1  # DNase阈值，可自行修改


fig, axes = plt.subplots(1, len(cell_types), figsize=(16, 4), dpi=100)
plt.subplots_adjust(wspace=0.4, bottom=0.2, top=0.85)

for i, ct in enumerate(cell_types):
    ax = axes[i]

    activity = mpra_df[ct]
    dnase_signal = vef_df[f'{ct}_DNase']

    plot_df = pd.DataFrame({
        'activity': activity,
        'DNase_group': np.where(dnase_signal > threshold, 'DNase+', 'DNase-')
    }).dropna()

    sns.boxplot(
        data=plot_df,
        x='DNase_group',
        y='activity',
        order=['DNase-', 'DNase+'],
        ax=ax,
        width=0.6,
        showfliers=False
    )

    group_neg = plot_df.loc[plot_df['DNase_group'] == 'DNase-', 'activity']
    group_pos = plot_df.loc[plot_df['DNase_group'] == 'DNase+', 'activity']

    stat, pval = mannwhitneyu(group_neg, group_pos, alternative='two-sided')

    n_neg = len(group_neg)
    n_pos = len(group_pos)
    median_neg = group_neg.median()
    median_pos = group_pos.median()
    delta = median_pos - median_neg

    y_max = plot_df['activity'].max()
    y_min = plot_df['activity'].min()
    y_range = y_max - y_min


    p_text = 'p < 1e-300'
    info_text = f'{p_text}\nn- = {n_neg}, n+ = {n_pos}\nΔmedian = {delta:.2f}'

    ax.text(
        0.05, 0.95, info_text,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=9
    )


    # line_y = y_max + 0.08 * y_range
    # text_y = y_max + 0.13 * y_range

    # ax.plot(
    #     [0, 0, 1, 1],
    #     [line_y, line_y + 0.02 * y_range, line_y + 0.02 * y_range, line_y],
    #     lw=1.2,
    #     c='black'
    # )

    # p_text = 'p < 1e-300'
    # info_text = f'{p_text}\nn- = {n_neg}, n+ = {n_pos}\nΔmedian = {delta:.2f}'

    # ax.text(
    #     0.5, text_y, info_text,
    #     ha='center', va='bottom', fontsize=9
    # )

    ax.set_title(ct)
    ax.set_xlabel('')
    if i == 0:
        ax.set_ylabel('MPRA activity')
    else:
        ax.set_ylabel('')

plt.savefig('analyze_gosai/figures/fig2.2_vef_boxplot.pdf', dpi=400, bbox_inches='tight')
plt.show()

# fig, axes = plt.subplots(1, len(cell_types), figsize=(16, 4), dpi=100)
# plt.subplots_adjust(wspace=0.4, bottom=0.2, top=0.85)

# for i, ct in enumerate(cell_types):
#     ax = axes[i]
    
#     # 取当前 cell type 的 MPRA 活性和 DNase 信号
#     activity = mpra_df[ct]
#     dnase_signal = vef_df[f'{ct}_DNase']
    
#     # 构造用于画图的数据
#     plot_df = pd.DataFrame({
#         'activity': activity,
#         'DNase_group': np.where(dnase_signal > threshold, 'DNase+', 'DNase-')
#     }).dropna()
    
#     # boxplot
#     sns.boxplot(
#         data=plot_df,
#         x='DNase_group',
#         y='activity',
#         order=['DNase-', 'DNase+'],
#         ax=ax,
#         width=0.6,
#         showfliers=False,
#     )
    
#     # 显著性检验
#     group_neg = plot_df.loc[plot_df['DNase_group'] == 'DNase-', 'activity']
#     group_pos = plot_df.loc[plot_df['DNase_group'] == 'DNase+', 'activity']
    
#     stat, pval = mannwhitneyu(group_neg, group_pos, alternative='two-sided')
    
#     # 在图上标注显著性
#     y_max = plot_df['activity'].max()
#     y_min = plot_df['activity'].min()
#     y_range = y_max - y_min
    
#     line_y = y_max + 0.08 * y_range
#     text_y = y_max + 0.12 * y_range
    
#     ax.plot([0, 0, 1, 1], [line_y, line_y + 0.02*y_range, line_y + 0.02*y_range, line_y],
#             lw=1.2, c='black')
#     ax.text(0.5, text_y, f'p = {pval:.2e}', ha='center', va='bottom', fontsize=10)
    
#     ax.set_title(ct)
#     ax.set_xlabel('')
#     if i == 0:
#         ax.set_ylabel('MPRA activity')
#     else:
#         ax.set_ylabel('')

# plt.savefig('analyze_gosai/figures/vef_act_boxplot.png', dpi=400, bbox_inches='tight')
# plt.show()

# plt.figure(figsize=(8, 6), dpi=100)
# plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)




# plt.savefig(f'analyze_gosai/figures/xx.png', dpi=400)
