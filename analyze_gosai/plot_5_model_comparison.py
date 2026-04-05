import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# -----------------------------
# inputs
# -----------------------------
model_names = ['Sei-DNase', 'Enformer-DNase', 'Borzoi-DNase', 'EpiCast', 'EpiCast-VEF-only']
cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
train_cell_types = ['K562', 'HepG2', 'SK-N-SH']
test_cell_types = ['HCT116', 'A549']

# 例子:
result_df = pd.DataFrame(
    data=np.array(
        [[0.586, 0.660, 0.164, 0.528, 0.472],
         [0.486, 0.528, 0.238, 0.319, 0.453],
         [0.354, 0.449, 0.095, 0.348, 0.332],
         [0.636, 0.624, 0.642, 0.559, 0.750],
         [0.507, 0.620, 0.424, 0.506, 0.553],]
    ),
    columns=cell_types,
    index=model_names,
)

plot_df = pd.DataFrame({
    'Train mean': result_df[train_cell_types].mean(axis=1),
    'Test mean': result_df[test_cell_types].mean(axis=1),
}, index=result_df.index)

# -----------------------------
# plotting
# -----------------------------
sns.set_theme(style="white", context="talk")
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

y_base = np.arange(len(plot_df))

train_offset = -0.10
test_offset = 0.10

train_color = '#4C78A8'
test_color = '#E45756'

xmin = max(0, plot_df.min().min() - 0.05)
xmax = min(1.0, plot_df.max().max() + 0.05)

for y in y_base:
    ax.hlines(
        y,
        xmin=xmin,
        xmax=xmax,
        color='lightgray',
        linewidth=0.8,
        zorder=0
    )

ax.scatter(
    plot_df['Train mean'].values,
    y_base + train_offset,
    s=60,
    marker='o',
    color=train_color,
    edgecolor='black',
    linewidth=0.6,
    alpha=0.95,
    zorder=3,
    label='Training cell types (mean)'
)

ax.scatter(
    plot_df['Test mean'].values,
    y_base + test_offset,
    s=50,
    marker='D',
    color=test_color,
    edgecolor='black',
    linewidth=0.6,
    alpha=0.95,
    zorder=3,
    label='Unseen cell types (mean)'
)


for i, model in enumerate(plot_df.index):
    ax.plot(
        [plot_df.loc[model, 'Train mean'], plot_df.loc[model, 'Test mean']],
        [i + train_offset, i + test_offset],
        color='gray',
        linewidth=1.2,
        alpha=0.8,
        zorder=2
    )

ax.set_yticks(y_base)
ax.set_yticklabels(plot_df.index, fontsize=11)
ax.set_xlabel('PCC', fontsize=12)
ax.set_xlim(xmin, xmax)
ax.invert_yaxis()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='Training cell types (mean)',
           markerfacecolor=train_color, markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='D', color='w', label='Unseen cell types (mean)',
           markerfacecolor=test_color, markeredgecolor='black', markersize=8),
]

ax.legend(handles=legend_handles, loc='lower left', frameon=False, bbox_to_anchor=(1.05, 0.05),)

plt.savefig("analyze_gosai/figures/model_comparison.png", dpi=400, bbox_inches='tight')
plt.show()