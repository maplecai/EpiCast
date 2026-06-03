import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# =========================
# 读取数据
# =========================
mpra_df = pd.read_csv(
    'data/gosai_mpra/gosai_mpra_760679_zscore.tsv',
    sep='\t'
)
print('mpra_df:', mpra_df.shape)

vef_df = pd.read_csv(
    'data/gosai_mpra/gosai_mpra_760679_ag_vef_x10_log1p.tsv',
    sep='\t'
)
print('vef_df:', vef_df.shape)

cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']

# =========================
# 参数
# =========================
ct = 'K562'   # 可改
dnase_col = f'{ct}_DNase'
activity_col = ct

out_path = 'analyze_gosai/figures/fig2.3_dnase_activity_density.pdf'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# =========================
# 整理数据
# =========================
plot_df = pd.DataFrame({
    'DNase': vef_df[dnase_col].values,
    'Activity': mpra_df[activity_col].values
}).dropna()

print('plot_df:', plot_df.shape)

# 阈值
dnase_thr = plot_df['DNase'].quantile(0.8)
activity_thr = plot_df['Activity'].quantile(0.8)

print('dnase_thr =', dnase_thr)
print('activity_thr =', activity_thr)

# 正负类
plot_df['DNase_pred'] = plot_df['DNase'] > dnase_thr
plot_df['Activity_true'] = plot_df['Activity'] > activity_thr

tp = ((plot_df['DNase_pred']) & (plot_df['Activity_true'])).sum()
fp = ((plot_df['DNase_pred']) & (~plot_df['Activity_true'])).sum()
fn = ((~plot_df['DNase_pred']) & (plot_df['Activity_true'])).sum()
tn = ((~plot_df['DNase_pred']) & (~plot_df['Activity_true'])).sum()

accuracy = (tp + tn) / len(plot_df)
recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

print(f'TP={tp}, FP={fp}, FN={fn}, TN={tn}')
print(f'Accuracy={accuracy:.4f}, Recall={recall:.4f}, Precision={precision:.4f}')

# =========================
# 取坐标
# =========================
x = plot_df['DNase'].to_numpy()
y = plot_df['Activity'].to_numpy()

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

# =========================
# 画图
# =========================
fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
fig.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.92)



from matplotlib.colors import LinearSegmentedColormap
hexbin_cmap = LinearSegmentedColormap.from_list(
    'white_to_softblue',
    ['#eaf2fb', '#bcd7ee', '#8fb9dd', '#1f4e79']
)
# 1) 用 hexbin 画主密度，适合超大样本
hb = ax.hexbin(
    x, y,
    gridsize=50,
    mincnt=1,
    cmap=hexbin_cmap,
    linewidths=0,
    rasterized=True
)

cb = fig.colorbar(hb, ax=ax)
cb.set_label('Count')




# 阈值虚线
ax.axvline(dnase_thr, color='black', linestyle='--', linewidth=1.2)
ax.axhline(activity_thr, color='black', linestyle='--', linewidth=1.2)

# 四象限文字位置
x_left = (x_min + dnase_thr) / 2
x_right = (dnase_thr + x_max) / 2
y_bottom = (y_min + activity_thr) / 2
y_top = (activity_thr + y_max) / 2

# 四象限标注
ax.text(x_right, y_top, f'TP\n{tp}', ha='center', va='center', fontsize=11, color='black')
ax.text(x_right, y_bottom, f'FP\n{fp}', ha='center', va='center', fontsize=11, color='black')
ax.text(x_left, y_top, f'FN\n{fn}', ha='center', va='center', fontsize=11, color='black')
ax.text(x_left, y_bottom, f'TN\n{tn}', ha='center', va='center', fontsize=11, color='black')

# 指标文字
info_text = (
    f'Accuracy = {accuracy:.3f}\n'
    f'Recall = {recall:.3f}\n'
    f'Precision = {precision:.3f}'
)

ax.text(
    0.98, 0.98, info_text,
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=10,
    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='none')
)

ax.set_title(ct)
ax.set_xlabel('DNase signal')
ax.set_ylabel('MPRA activity')

plt.savefig(out_path, dpi=400)
plt.close()

print(f'Saved to: {out_path}')