import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from genoml import utils, metrics
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import h5py

if __name__ == "__main__":
    mapa_path = "data/Gosai_MPRA/Gosai_MPRA_760679.tsv"
    mpra_df = pd.read_csv(mapa_path, sep='\t')
    print(mpra_df.shape)
    print(mpra_df.describe())

    vef_path = "data/Gosai_MPRA/Gosai_MPRA_AG_VEF_raw_760k.tsv"
    vef_df = pd.read_csv(vef_path, sep='\t')
    print(vef_df.shape)
    print(vef_df.describe())
    
    cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
    assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']

    
    corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for i, cell_type in enumerate(cell_types):
        for j, assay in enumerate(assays):
            pred = vef_df[f'{cell_type}_{assay}']
            true = mpra_df[cell_type]
            r = metrics.pearson(pred, true)
            corr_df.loc[cell_type, assay] = r
    print(corr_df)


    # ##### log1p
    # vef_df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_AG_VEF_raw_1k.tsv', sep='\t')

    # cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
    # assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']
    # vef_df = np.log1p(vef_df)
    # vef_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_AG_VEF_log1p_1k.tsv', sep='\t', index=False)

    # corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    # for i, cell_type in enumerate(cell_types):
    #     for j, assay in enumerate(assays):
    #         pred = vef_df[f'{cell_type}_{assay}']
    #         true = mpra_df[cell_type]
    #         r = metrics.pearson(pred, true)
    #         corr_df.loc[cell_type, assay] = r
    # print(corr_df)
