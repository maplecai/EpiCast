import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import h5py

import genoml
from genoml import utils, metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mpra_path', type=str, default="data/gosai_mpra/gosai_mpra_760679.tsv")
    parser.add_argument('--pred_path', type=str, default="alphagenome/gosai_ag_pred_760k_pad_0.h5")  # "alphagenome/gosai_ag_pred_760k_pad_0_25.h5"
    args = parser.parse_args()
    mpra_path = args.mpra_path
    # pred_path = args.pred_path

    mpra_df = pd.read_csv(mpra_path, sep='\t')

    pred_path = "alphagenome/gosai_ag_pred_760k_pad_0.h5"

    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)

    df_pivot = pd.read_csv('alphagenome/metadata_pivot_4_vef.tsv', sep='\t', index_col=0)
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
                pred = preds['chip_histone'][:, index] / 128
            elif assay == 'CTCF':
                pred = preds['chip_tf'][:, index] / 128
            else:
                pred = np.nan
            vef_df[f'{cell_type}_{assay}'] = pred
    print(vef_df.describe())
    vef_df.to_csv('data/gosai_mpra/gosai_mpra_760679_ag_vef_raw.tsv', sep='\t', index=False)


    corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for i, cell_type in enumerate(cell_types):
        for j, assay in enumerate(assays):
            pred = vef_df[f'{cell_type}_{assay}']
            true = mpra_df[cell_type]
            r = metrics.pearson(pred, true)
            corr_df.loc[cell_type, assay] = r
    print('pad 0, pearson')
    print(corr_df)

    corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for i, cell_type in enumerate(cell_types):
        for j, assay in enumerate(assays):
            pred = vef_df[f'{cell_type}_{assay}']
            true = mpra_df[cell_type]
            r = metrics.spearman(pred, true)
            corr_df.loc[cell_type, assay] = r
    print('pad 0, spearman')
    print(corr_df)



    vef_df = np.log1p(vef_df)
    print(vef_df.describe())
    vef_df.to_csv('data/gosai_mpra/gosai_mpra_760679_ag_vef_log1p.tsv', sep='\t', index=False)
    corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for i, cell_type in enumerate(cell_types):
        for j, assay in enumerate(assays):
            pred = vef_df[f'{cell_type}_{assay}']
            true = mpra_df[cell_type]
            r = metrics.pearson(pred, true)
            corr_df.loc[cell_type, assay] = r
    print('pad 0, log1p pearson')
    print(corr_df)


    # vef_df = pd.DataFrame()
    # for i, cell_type in enumerate(cell_types):
    #     for j, assay in enumerate(assays):
    #         index = df_pivot.loc[cell_type, assay]
    #         if assay == 'DNase':
    #             pred = preds['dnase_1'][:, index]
    #         elif assay in ['H3K4me3', 'H3K27ac']:
    #             pred = preds['chip_histone'][:, index]
    #         elif assay == 'CTCF':
    #             pred = preds['chip_tf'][:, index]
    #         else:
    #             pred = np.nan
    #         vef_df[f'{cell_type}_{assay}'] = pred

    # corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    # for i, cell_type in enumerate(cell_types):
    #     for j, assay in enumerate(assays):
    #         pred = vef_df[f'{cell_type}_{assay}']
    #         true = mpra_df[cell_type]
    #         r = metrics.pearson(pred, true)
    #         corr_df.loc[cell_type, assay] = r
    # print('pad 0, pearson')
    # print(corr_df)


    # corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    # for i, cell_type in enumerate(cell_types):
    #     for j, assay in enumerate(assays):
    #         pred = vef_df[f'{cell_type}_{assay}']
    #         true = mpra_df[cell_type]
    #         pred = np.log1p(pred)
    #         r = metrics.pearson(pred, true)
    #         corr_df.loc[cell_type, assay] = r
    # print('pad 0, log1p pearson')
    # print(corr_df)


    # corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    # for i, cell_type in enumerate(cell_types):
    #     for j, assay in enumerate(assays):
    #         pred = vef_df[f'{cell_type}_{assay}']
    #         true = mpra_df[cell_type]
    #         r = metrics.spearman(pred, true)
    #         corr_df.loc[cell_type, assay] = r
    # print('pad 0, spearman')
    # print(corr_df)
