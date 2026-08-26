import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def set_mpl_params():
    sns.set_theme(context="talk", style="white")
    mpl_params = {
        # fonts
        'font.family': 'Arial',
        'font.size': 12,
        # math font
        'mathtext.fontset': 'stix', 
        # spines, axes and ticks are 1pt black everywhere
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        # figure size and resolution
        'figure.figsize': (8, 6),
        'figure.dpi': 400,
        # keep the text of a pdf editable
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


