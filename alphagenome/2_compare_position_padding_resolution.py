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
    mpra_path = "data/Gosai_MPRA/Gosai_MPRA_760679.tsv"
    mpra_df = pd.read_csv(mpra_path, sep='\t')
    print(mpra_df.shape)
    mpra_df = mpra_df[:1000]

    cell_types = ['K562', 'HepG2', 'SK-N-SH']
    true = mpra_df[cell_types].mean(1)

    pred_path = "alphagenome/gosai_ag_pred.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)

    print("dnase, pad 0.25, 1bp resolution, all mean")
    pred = preds['dnase_1'].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0.25, 1bp resolution, center 200bp mean")
    pred = preds['dnase_1'][:, 1024-100:1024+100, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)


    print("dnase, pad 0.25, 1bp resolution, center 256bp mean")
    pred = preds['dnase_1'][:, 1024-128:1024+128, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0.25, 1bp resolution, 0bp-100bp mean")
    pred = preds['dnase_1'][:, 0:100, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0.25, 1bp resolution, 100bp-200bp mean")
    pred = preds['dnase_1'][:, 100:200, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0.25, 1bp resolution, 200bp-300bp mean")
    pred = preds['dnase_1'][:, 200:300, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0.25, 1bp resolution, 300-400bp mean")
    pred = preds['dnase_1'][:, 300:400, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0.25, 1bp resolution, 1000-1100bp mean")
    pred = preds['dnase_1'][:, 1000:1100, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)


    print("dnase, pad 0.25, 128bp resolution, all mean")
    pred = preds['dnase'].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0.25, 128bp resolution, center 256bp mean")
    pred = preds['dnase'][:, 7:9, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)


    print("chip, pad 0.25, histone 128bp resolution, all mean")
    pred = preds['chip_histone'][:, :, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("chip, pad 0.25, histone 128bp resolution, center 256bp mean")
    pred = preds['chip_histone'][:, 7:9, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)




    pred_path = "alphagenome/gosai_ag_pred_pad_0.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)


    print("dnase, pad 0, 1bp resolution, all mean")
    pred = preds['dnase_1'].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0, 1bp resolution, center 200bp mean")
    pred = preds['dnase_1'][:, 1024-100:1024+100, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)


    print("dnase, pad 0, 1bp resolution, center 256bp mean")
    pred = preds['dnase_1'][:, 1024-128:1024+128, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0, 128bp resolution, all mean")
    pred = preds['dnase'].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, pad 0, 128bp resolution, center 256bp mean")
    pred = preds['dnase'][:, 7:9, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)


    print("chip histone, pad 0, 128bp resolution, all mean")
    pred = preds['chip_histone'][:, :, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("chip histone, pad 0, 128bp resolution, center 256bp mean")
    pred = preds['chip_histone'][:, 7:9, :].mean((1,2))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)






    pred_path = "alphagenome/gosai_ag_pred_local_1k_test_pad_0_25.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)

    print("dnase, padding 16384bp, padding 0.25, 1bp resolution, center 200bp mean")
    pred = preds['dnase_1'].mean((1))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, padding 16384bp, padding 0.25, 128bp resolution, center 200bp mean")
    pred = preds['dnase'].mean((1))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)




    pred_path = "alphagenome/gosai_ag_pred_local_1k_test_pad_0.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)

    print("dnase, padding 16384bp, padding 0, 1bp resolution, center 200bp mean")
    pred = preds['dnase_1'].mean((1))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, padding 16384, padding 0, 128bp resolution, center 200bp mean")
    pred = preds['dnase'].mean((1))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)






    pred_path = "alphagenome/gosai_ag_pred_256bp_concat.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)
        preds[key] = preds[key][:1000]

    print("dnase, padding 256bp 2048bp, 1bp resolution, center 256bp mean")
    pred = preds['dnase_1'].mean((1))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)

    print("dnase, padding 256bp 2048bp, 1bp resolution, center 256bp mean")
    pred = preds['chip_histone'].mean((1))
    r = metrics.spearman(true, pred)
    print('spearman', r)
    r = metrics.pearson(true, pred)
    print('pearson', r)