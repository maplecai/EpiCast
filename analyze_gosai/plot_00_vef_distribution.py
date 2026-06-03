import sys
import numpy as np
import pandas as pd
from epicast import models, datasets, utils, metrics
import matplotlib.pyplot as plt
import seaborn as sns

# mpra_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_zs.tsv', sep='\t')
# print(mpra_df.shape)

vef_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_ag_vef_log1p.tsv', sep='\t')
print(vef_df.shape)

cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']

for j, assay in enumerate(assays):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

    for i, cell_type in enumerate(cell_types):
        x = vef_df[f'{cell_type}_{assay}']
        x = x[x<=0.5]
        sns.kdeplot(x, label=cell_type)

    # plt.xlim(0, 2)
    plt.xlabel(f'Virtual {assay}')
    plt.ylabel('Density')
    plt.legend()
    # plt.savefig(f'analyze_gosai/figures/gosai_mpra_ag_{assay}_distribution.png', dpi=400)
    plt.savefig(f'analyze_gosai/figures/gosai_mpra_ag_{assay}_distribution.pdf', dpi=400)
