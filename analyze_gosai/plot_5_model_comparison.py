import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# -----------------------------
# inputs
# -----------------------------
model_names = [
    "Sei DNase",
    "Enformer DNase",
    "Alphagenome DNase",
    "Alphagenome H3K4me3",
    "Training labels mean",
    "Seq only model: Malinois",
    "EpiCast (AG VEF)",
]

cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
train_cell_types = ['K562', 'HepG2', 'SK-N-SH']
test_cell_types = ['HCT116', 'A549']




result_df = pd.DataFrame(
    data=np.array([
        [0.519, 0.607, 0.275, 0.430, 0.320],
        [0.408, 0.443, 0.301, 0.311, 0.391],
        [0.477, 0.494, 0.383, 0.337, 0.400],
        [0.367, 0.501, 0.505, 0.335, 0.386],
        [0.555, 0.723, 0.762, 0.524, 0.759],
        [0.768, 0.803, 0.760, 0.503, 0.718],
        [0.706, 0.764, 0.725, 0.581, 0.793],
    ]),
    columns=cell_types,
    index=model_names,
)



# # specific residual pearson
# result_df = pd.DataFrame(
#     data=np.array([
#         [0.568, 0.558, 0.129, 0.429, 0.206],
#         [0.339, 0.264, 0.169, 0.185, 0.148],
#         [0.457, 0.490, -0.007, 0.138, -0.160],
#         [0.353, 0.325, 0.052, 0.204, 0.135],
#         [-0.024, -0.007, -0.002, 0.063, 0.056],
#         [0.825, 0.779, 0.720, 0.004, -0.021],
#         [0.739, 0.710, 0.629, 0.284, 0.467],
#     ]),
#     columns=cell_types,
#     index=model_names,
# )






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

xmin = -0.1
xmax = 0.9

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