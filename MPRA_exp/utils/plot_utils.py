import os
import yaml
import random
import pickle
import logging
import logging.config
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Callable
from datetime import datetime
from torch.utils.data import random_split, Subset
from collections import Counter
from scipy import stats
from scipy.stats import pearsonr

import seaborn as sns
# sns.set(style="whitegrid", context="talk")
sns.set_theme(context="talk", style="whitegrid")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

color_list = np.delete(matplotlib.colormaps.get_cmap('Set3')(np.arange(12)), 1, axis=0)

mpl_params = {
    # 字体参数
    'font.family': 'Arial',
    'font.size': 12,
    
    # 数学文本参数
    'mathtext.fontset': 'stix', 
    
    # 图像参数
    'figure.dpi': 100,
    'figure.figsize': (6, 4.5),

    # 保存字体可编辑
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}

plt.rcParams.update(mpl_params)

# 保存图像的参数
savefig_kwargs = {
    'bbox_inches': 'tight',
    #'transparent': True,
}

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
