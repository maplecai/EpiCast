import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import genoml

np.random.seed(42)

# --------------------------
# 1. 构造示例数据
# --------------------------
n_groups = 16
group_labels = [f"G{i+1}" for i in range(n_groups)]

# 每组 boxplot 的数据
data = []
means = np.linspace(0.1, -0.4, n_groups)
for i in range(n_groups):
    x = np.random.normal(loc=means[i], scale=0.7 - i*0.02, size=120)
    data.append(x)



# data = pd.read_csv("analyze_gosai/data/virtual_ccre.csv", index_col=0)



# 4个 feature × 16个组合
# 1 = 黑点, 0 = 白点
state_matrix = np.array([
    [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],  # Virtual DNase
    [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],  # Virtual H3K4me3
    [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],  # Virtual H3K27ac
    [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],  # Virtual CTCF
])

row_labels = [
    "Virtual DNase",
    "Virtual H3K4me3",
    "Virtual H3K27ac",
    "Virtual CTCF"
]

# --------------------------
# 2. 布局
# --------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    dpi=100,
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

# --------------------------
# 3. 上半部分 boxplot
# --------------------------
bp = ax1.boxplot(
    data,
    positions=np.arange(1, n_groups + 1),
    widths=0.6,
    patch_artist=True,
    showfliers=False
)

for box in bp['boxes']:
    box.set(facecolor='#c7e9f1', edgecolor='dimgray', linewidth=0.8)
for median in bp['medians']:
    median.set(color='dimgray', linewidth=0.8)
for whisker in bp['whiskers']:
    whisker.set(color='dimgray', linewidth=0.8)
for cap in bp['caps']:
    cap.set(color='dimgray', linewidth=0.8)

# 加黑点表示均值
group_means = [np.mean(x) for x in data]
ax1.scatter(np.arange(1, n_groups + 1), group_means, color='black', s=14, zorder=3)

ax1.set_ylabel("CRE activity in K562", fontsize=16)
ax1.set_xlim(0.5, n_groups + 0.5)
ax1.set_xticks([])
ax1.tick_params(axis='y', labelsize=12)

# 简化边框
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --------------------------
# 4. 下半部分 dot matrix
# --------------------------
ax2.set_ylim(4.5, 0.5)  # 让第一行显示在最上面
ax2.set_yticks([1, 2, 3, 4])
ax2.set_yticklabels(row_labels, fontsize=16)
ax2.set_xlabel("VEF state combinations", fontsize=24, labelpad=20)

# 灰色参考线
for y in [1, 2, 3, 4]:
    ax2.hlines(y, 0.5, n_groups + 0.5, color='lightgray', linewidth=1)

for x in range(1, n_groups + 1):
    ax2.vlines(x, 0.5, 4.5, color='lightgray', linewidth=1)

ax2.yaxis.grid(False)
ax2.xaxis.grid(False)

# 画黑白圆点
for r in range(4):
    for c in range(n_groups):
        filled = state_matrix[r, c] == 1
        ax2.scatter(
            c + 1, r + 1,
            s=100,
            facecolor='black' if filled else 'white',
            edgecolor='black',
            linewidth=1,
            zorder=3
        )

# 去掉多余边框和 x 刻度
ax2.set_xticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(axis='y', length=0)

plt.savefig("analyze_gosai/figures/virtual_ccre.png", dpi=400)
plt.show()