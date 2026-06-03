import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import h5py

import epicast
from epicast import utils, metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mpra_path', type=str, default="data/gosai_mpra/gosai_mpra_760679.tsv")
    parser.add_argument('--pred_path', type=str, default="alphagenome/gosai_ag_pred_760k_pad_0.h5")  # "alphagenome/gosai_ag_pred_760k_pad_0_25.h5"
    parser.add_argument('--out_path', type=str, default="data/gosai_mpra/gosai_mpra_760679_ag_vef_raw.tsv")
    args = parser.parse_args()
    mpra_path = args.mpra_path
    pred_path = args.pred_path
    out_path = args.out_path

    mpra_df = pd.read_csv(mpra_path, sep='\t')

    # pred_path = "alphagenome/gosai_ag_pred_760k_pad_0.h5"

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
                pred = preds['dnase_128'][:, index] / 128
                vef_df[f'{cell_type}_DNase_128'] = pred
                pred = preds['dnase_1'][:, index]
                vef_df[f'{cell_type}_DNase'] = pred
            elif assay in ['H3K4me3', 'H3K27ac']:
                pred = preds['chip_histone'][:, index] / 128
            elif assay == 'CTCF':
                pred = preds['chip_tf'][:, index] / 128
            else:
                pred = np.nan
            vef_df[f'{cell_type}_{assay}'] = pred
    print(vef_df.describe())
    vef_df.to_csv(out_path, sep='\t', index=False)





    # corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    # for i, cell_type in enumerate(cell_types):
    #     for j, assay in enumerate(assays):
    #         pred = vef_df[f'{cell_type}_{assay}']
    #         true = mpra_df[cell_type]
    #         r = metrics.pearson(pred, true)
    #         corr_df.loc[cell_type, assay] = r
    # print('pearson')
    # print(corr_df)

    # corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    # for i, cell_type in enumerate(cell_types):
    #     for j, assay in enumerate(assays):
    #         pred = vef_df[f'{cell_type}_{assay}']
    #         true = mpra_df[cell_type]
    #         r = metrics.spearman(pred, true)
    #         corr_df.loc[cell_type, assay] = r
    # print('spearman')
    # print(corr_df)

