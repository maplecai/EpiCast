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
    pred_path = "alphagenome/gosai_ag_pred.h5"
    pred_path = "alphagenome/gosai_ag_pred_local_1k_test_pad_0.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)
    
    pred_dnase_1 = preds['dnase_1'][:, :305].mean((1))
    pred_histone_1 = preds['chip_histone'][:, :1116].mean((1))

    pred_path = "alphagenome/gosai_ag_pred_1k_api_test.h5"
    preds = utils.load_h5(pred_path)
    for key in preds:
        print(key, preds[key].shape)
    
    pred_dnase_2 = preds['dnase'].mean((1))
    pred_histone_2 = preds['chip_histone'].mean((1))

    print(pred_dnase_1.shape, pred_dnase_2.shape)
    print(pred_dnase_1[:10])
    print(pred_dnase_2[:10])
    print('close', np.allclose(pred_dnase_1, pred_dnase_2, rtol=0.05))
    print('pearson', metrics.pearson(pred_dnase_1, pred_dnase_2))

    print(pred_histone_1.shape, pred_histone_2.shape)
    print(pred_histone_1[:10])
    print(pred_histone_2[:10])
    print('close', np.allclose(pred_histone_1, pred_histone_2, rtol=0.05))
    print('pearson', metrics.pearson(pred_histone_1, pred_histone_2))

    # plt.figure()
    # sns.scatterplot(x=pred_dnase_1, y=pred_dnase_2, s=1)
    # plt.savefig("dnase_scatter.png")

    # plt.figure()
    # sns.scatterplot(x=pred_histone_1, y=pred_histone_2, s=1)
    # plt.savefig("histone_scatter.png")
