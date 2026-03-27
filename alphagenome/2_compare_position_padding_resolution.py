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

    pred_path = "alphagenome/gosai_ag_pred.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)


    print("dnase 1bp resolution, all mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase 1bp resolution, center 200bp mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'][:, 1024-100:1024+100, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)


    print("dnase 1bp resolution, center 256bp mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'][:, 1024-128:1024+128, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase 1bp resolution, 0bp-10bp mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'][:, 0:10, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase 1bp resolution, 512bp-522bp mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'][:, 512:522, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)


    print("dnase 128bp resolution, all mean")
    true = mpra_df['K562']
    pred = preds['dnase'].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase 128bp resolution, center 256bp mean")
    true = mpra_df['K562']
    pred = preds['dnase'][:, 7:9, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)


    print("chip histone 128bp resolution, all mean")
    true = mpra_df['K562']
    pred = preds['chip_histone'][:, :, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("chip histone 128bp resolution, center 256bp mean")
    true = mpra_df['K562']
    pred = preds['chip_histone'][:, 7:9, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)




    pred_path = "alphagenome/gosai_ag_pred_pad_0.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)


    print("dnase 1bp resolution, all mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase 1bp resolution, center 200bp mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'][:, 1024-100:1024+100, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)


    print("dnase 1bp resolution, center 256bp mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'][:, 1024-128:1024+128, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase 1bp resolution, 0bp-10bp mean, code mistake but also high corr")
    true = mpra_df['K562']
    pred = preds['dnase_1'][:, 0:10, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase 1bp resolution, 512bp-522bp mean, code mistake but also high corr")
    true = mpra_df['K562']
    pred = preds['dnase_1'][:, 512:522, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)


    print("dnase 128bp resolution, all mean")
    true = mpra_df['K562']
    pred = preds['dnase'].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase 128bp resolution, center 256bp mean")
    true = mpra_df['K562']
    pred = preds['dnase'][:, 7:9, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)


    print("chip histone 128bp resolution, all mean")
    true = mpra_df['K562']
    pred = preds['chip_histone'][:, :, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)

    print("chip histone 128bp resolution, center 256bp mean")
    true = mpra_df['K562']
    pred = preds['chip_histone'][:, 7:9, :].mean((1,2))
    r = metrics.pearson(true, pred)
    print(r)






    pred_path = "alphagenome/gosai_ag_pred_local_1k_test.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)

    print("dnase, padding 16384bp, padding 0.25, 1bp resolution, center 200bp mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'].mean((1))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase, padding 16384bp, padding 0.25, 128bp resolution, center 200bp mean")
    true = mpra_df['K562']
    pred = preds['dnase'].mean((1))
    r = metrics.pearson(true, pred)
    print(r)




    pred_path = "alphagenome/gosai_ag_pred_local_1k_test_pad_0.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)

    print("dnase, padding 16384bp, padding 0, 1bp resolution, center 200bp mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'].mean((1))
    r = metrics.pearson(true, pred)
    print(r)

    print("dnase, padding 16384, padding 0, 128bp resolution, center 200bp mean")
    true = mpra_df['K562']
    pred = preds['dnase'].mean((1))
    r = metrics.pearson(true, pred)
    print(r)






    pred_path = "alphagenome/gosai_ag_pred_760k_not_compressed.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)
        preds[key] = preds[key][:1000]

    print("dnase, padding 2048bp, 1bp resolution, center 200bp mean")
    true = mpra_df['K562']
    pred = preds['dnase_1'].mean((1))
    r = metrics.pearson(true, pred)
    print(r)
