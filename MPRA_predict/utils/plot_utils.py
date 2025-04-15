import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")

Set2_colors = matplotlib.colormaps.get_cmap('Set2').colors
tab10_colors = matplotlib.colormaps.get_cmap('tab10').colors
tab20_colors = matplotlib.colormaps.get_cmap('tab20').colors


mpl_params = {
    # 字体参数
    'font.family': 'Arial',
    'font.size': 12,
    
    # 数学文本参数
    'mathtext.fontset': 'stix', 
    
    # 图像参数
    'figure.dpi': 100,
    'figure.figsize': (8, 6),

    # 保存pdf字体可编辑
    'pdf.fonttype': 42,
}

plt.rcParams.update(mpl_params)

def init_fig(figsize=(8, 6), dpi=100):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    left, right, bottom, top = 0.15, 0.95, 0.15, 0.95
    ax.set_position([left, bottom, right - left, top - bottom])
    return fig, ax

# def plot_scatter(x, y, fname='0', **kwargs):
#     # 使用 scipy.stats 计算回归线系数和 R^2 值
#     slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#     line_eq = f"$y = {slope:.3f}x + {intercept:.3f}$"
#     r2 = f"$R^2 = {r_value**2:.3f}$"

#     # 绘制散点图和拟合直线
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.regplot(x=x, y=y, scatter_kws={'s': 3}, line_kws={'color': color_list[2], 'linewidth':3})
#     plt.axis('equal')

#     if 'xticks' in kwargs:
#         plt.xticks(kwargs['xticks'])
#     if 'yticks' in kwargs:
#         plt.xticks(kwargs['yticks'])
#     if 'title' in kwargs:
#         plt.title(kwargs['title'])
#     if 'xlabel' in kwargs:
#         plt.xlabel(kwargs['xlabel'])
#     if 'ylabel' in kwargs:
#         plt.ylabel(kwargs['ylabel'])
#     plt.text(0.8, 0.1, f"{r2}", transform=ax.transAxes)
#     plt.savefig(fname, bbox_inches='tight')
#     plt.show()
#     return
