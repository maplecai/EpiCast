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
    pred_path = "alphagenome/gosai_ag_pred_760k_not_compressed.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)

    df_pivot = pd.read_csv('alphagenome/metadata_pivot_4_vef.tsv', sep='\t', index_col=0)
    print(df_pivot)
    df_pivot['DNase'] = df_pivot['DNase'] - 713
    df_pivot['H3K4me3'] = df_pivot['H3K4me3'] - 1685
    df_pivot['H3K27ac'] = df_pivot['H3K27ac'] - 1685
    df_pivot['CTCF'] = df_pivot['CTCF'] - 2803


    cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
    assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']
    vef_df = pd.DataFrame()

    for i, cell_type in enumerate(cell_types):
        for j, assay in enumerate(assays):
            index = df_pivot.loc[cell_type, assay]
            if assay == 'DNase':
                pred = preds['dnase_1'][:, index]
            elif assay in ['H3K4me3', 'H3K27ac']:
                pred = preds['chip_histone'][:, index]
            elif assay == 'CTCF':
                pred = preds['chip_tf'][:, index]
            else:
                pred = np.nan
            vef_df[f'{cell_type}_{assay}'] = pred


    # for i, cell_type in enumerate(cell_types):
    #     for j, assay in enumerate(assays):
    #         index = df_pivot.loc[cell_type, assay]
    #         if assay == 'DNase':
    #             pred = preds['dnase'].mean((1))[:, index]
    #             # pred = outputs['dnase_1'][:, index]
    #         elif assay in ['H3K4me3', 'H3K27ac']:
    #             pred = preds['chip_histone'].mean((1))[:, index]
    #         elif assay == 'CTCF':
    #             pred = preds['chip_tf'].mean((1))[:, index]
    #         else:
    #             pred = np.nan
    #         vef_df[f'{cell_type}_{assay}'] = pred

    # print(vef_df.describe())
    # vef_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_AG_VEF_raw_pad_0.tsv', sep='\t', index=False)


    mapa_path = "data/Gosai_MPRA/Gosai_MPRA_760679.tsv"
    mpra_df = pd.read_csv(mapa_path, sep='\t')
    print(mpra_df.shape)

    corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for i, cell_type in enumerate(cell_types):
        for j, assay in enumerate(assays):
            pred = vef_df[f'{cell_type}_{assay}']
            true = mpra_df[cell_type]
            r = metrics.pearson(pred, true)
            corr_df.loc[cell_type, assay] = r
    print(corr_df)
