import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def set_mpl_params():
    sns.set_theme(context="talk", style="white")
    mpl_params = {
        # 字体
        'font.family': 'Arial',
        'font.size': 12,
        # 数学字体
        'mathtext.fontset': 'stix', 
        # 图像大小和分辨率
        'figure.figsize': (8, 6),
        'figure.dpi': 400,
        # pdf字体可编辑
        'pdf.fonttype': 42,
    }
    plt.rcParams.update(mpl_params)
set_mpl_params()



def set_print_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 3)
    pd.set_option('display.float_format', '{:.3f}'.format) 
    np.set_printoptions(linewidth=1000, precision=3, formatter={'float': '{: 0.3f}'.format})
set_print_options()




from matplotlib.colors import LinearSegmentedColormap
coolwarm_cmap = plt.get_cmap("coolwarm")
warm_cmap = LinearSegmentedColormap.from_list(
    "coolwarm_warm",
    coolwarm_cmap(np.linspace(0.5, 1.0, 256))
)
cool_cmap = LinearSegmentedColormap.from_list(
    "cool_cmap",
    coolwarm_cmap(np.linspace(0.5, 0.0, 256))
)


