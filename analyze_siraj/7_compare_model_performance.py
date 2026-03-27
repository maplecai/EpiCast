import sys
import numpy as np
import pandas as pd
# from genoml.metrics import pearson, spearman
# from genoml.utils import *
from genoml import models, datasets, utils, metrics
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def define_masks(
        mpra_df: pd.DataFrame,
        cell_types: list[str],
    ) -> dict[str, np.ndarray]:

    masks = {}
    masks['total'] = np.ones(len(mpra_df), dtype=bool)
    masks['train'] = ~mpra_df['chr'].isin(['chr7', 'chr13', 'chr19', 'chr21', 'chrX'])
    masks['val'] = mpra_df['chr'].isin(['chr19', 'chr21', 'chrX'])
    masks['test']  = mpra_df['chr'].isin(['chr7', 'chr13'])

    # # 和前三种细胞类型差异top1%的序列定义为cell type specific
    # for cell_type in cell_types:
    #     ref_mean = mpra_df[cell_types[:3]].mean(axis=1)
    #     delta = (mpra_df[cell_type] - ref_mean)
    #     delta_abs = delta.abs()
    #     thr = np.percentile(delta_abs.dropna(), 99)
    #     specific = delta_abs > thr
    #     masks[f'{cell_type}_specific'] = specific
    #     print(cell_type, (delta[specific] > 0).sum(), (delta[specific] < 0).sum())

    # 上下各1%的序列定义为cell type specific
    for cell_type in cell_types:
        ref_mean = mpra_df[cell_types[:3]].mean(axis=1)
        delta = mpra_df[cell_type] - ref_mean

        d = delta.dropna()
        q05 = np.percentile(d, 1)
        q95 = np.percentile(d, 99)
        # q05 = np.percentile(d, 0.5)
        # q95 = np.percentile(d, 99.5)

        specific = (delta < q05) | (delta > q95)
        masks[f'{cell_type}_specific'] = specific

        # 一行检查：test+ct_specific 的总数，以及 delta 的正/负号（相对 mean 上/下调）
        print(f"{cell_type}_specific n={(specific).sum()}  +={(delta[specific] > 0).sum()} -={(delta[specific] < 0).sum()} {delta.mean()}, {delta.std()}")
        print(f"test+{cell_type}_specific n={(masks['test'] & specific).sum()}  +={(delta[masks['test'] & specific] > 0).sum()}  -={(delta[masks['test'] & specific] < 0).sum()}")



    std = mpra_df[cell_types].std(axis=1, skipna=True)
    thr = np.percentile(std, 95)
    masks['high_std'] = std > thr


    for key in masks:
        print(key, masks[key].sum())
    return masks



def get_mask(split: str, masks: Dict[str, np.ndarray], cell_type: Optional[str] = None) -> np.ndarray:
    """
    split:
      - 'train'/'val'/'test'/'total'
      - 'specific' (需要 cell_type)
      - 'test+specific' (需要 cell_type)
    """
    if split in masks:
        return masks[split]
    elif split == "specific":
        return masks[f"{cell_type}_specific"]
    elif '+' in split:
        split1, split2 = split.split('+')
        return get_mask(split1, masks, cell_type) & get_mask(split2, masks, cell_type)
    else:
        raise ValueError(f"Unknown split: {split}")


def compute_metrics(
    mpra_df: pd.DataFrame,
    cell_types: list[str],
    masks: dict[str, np.ndarray],
    split: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    
    pearson_df = pd.DataFrame()
    spearman_df = pd.DataFrame()
    for c1 in cell_types:
        for c2 in cell_types:
            mask = get_mask(split, masks, cell_type=c1)
            df = mpra_df[mask]
            x = df[f'{c1}']
            y = df[f'{c2}_pred']
            r = metrics.pearson(x, y)
            pearson_df.loc[f'{c1}', f'{c2}_pred'] = r
            r = metrics.spearman(x, y)
            spearman_df.loc[f'{c1}', f'{c2}_pred'] = r
    

    # print(split, len(df))
    # print('pearson')
    # print(pearson_df)
    # # print('spearman')
    # # print(spearman_df)





def compute_summary_table(
    model_dfs: Dict[str, pd.DataFrame],
    cell_types: List[str],
    masks: Dict[str, np.ndarray],
    split: str,
    metric_fn: Callable,
) -> pd.DataFrame:
    """
    “总表”：行=模型，列=5种cell type, 只算对角线 true(ct) vs pred(ct)。
    """
    summary_df = pd.DataFrame(index=model_dfs.keys(), columns=cell_types, dtype=float)
    for model_name, df_join in model_dfs.items():
        for ct in cell_types:
            mask = get_mask(split, masks, cell_type=ct)
            df = df_join[mask]
            summary_df.loc[model_name, ct] = metric_fn(df[ct], df[f"{ct}_pred"])
    return summary_df




def main():
    mpra_df = pd.read_csv('data/Siraj_MPRA/Siraj_MPRA_processed.tsv', sep='\t')
    
    print(mpra_df.shape)
    print(mpra_df.describe())
    cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']

    pred_cols = [f'{cell_type}_pred' for cell_type in cell_types]
    masks = define_masks(mpra_df, cell_types)
    print(masks.keys())

    model_dfs = {}
    
    # model_name = 'mean of 3 cell types true label'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # true_mean = mpra_df[cell_types[:3]].mean(axis=1)
    # for c in cell_types:
    #     pred_df[f'{c}_pred'] = true_mean
    # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')
    # model_dfs[model_name] = join_df


    # model_name = 'EpiCast: 0123_Gosai_ConvTransFeature_AG_VEF'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # pred = np.load('saved/0123_Gosai_ConvTransFeature_AG_VEF.yaml/0123_021357/preds.npy')
    # pred_df[pred_cols] = pred
    # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')
    # model_dfs[model_name] = join_df
    


    # model_name = 'EpiCast: 0206_gosai_ag_vef trans=0'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # pred = np.load('saved/0206_gosai_ag_vef/0206_025223/preds.npy')
    # pred_df[pred_cols] = pred
    # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')
    # model_dfs[model_name] = join_df


    # model_name = 'EpiCast: 0206_gosai_ag_vef trans=3'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # pred = np.load('saved/0206_gosai_ag_vef/0206_025400/preds.npy')
    # pred_df[pred_cols] = pred
    # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')
    # model_dfs[model_name] = join_df

    # model_name = 'EpiCast: 0207_gosai_ag_vef trans=3 huberloss'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # pred = np.load('saved/0207_gosai_ag_vef/0207_043226/preds.npy')
    # pred_df[pred_cols] = pred
    # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')
    # model_dfs[model_name] = join_df



    # model_name = 'EpiCast: 0226_gosai_ag_vef trans=3 huberloss'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # pred = np.load('saved/0226_gosai_ag_vef/0226_014611/preds.npy')
    # pred_df[pred_cols] = pred
    # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')
    # model_dfs[model_name] = join_df

    model_name = 'EpiCast: 0303_siraj_ag_vef'
    print(model_name)
    pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    pred = np.load('saved/0303_siraj_ag_vef/0303_053016/preds.npy')
    pred_df[pred_cols] = pred
    join_df = pd.concat([mpra_df, pred_df], axis=1)
    print(join_df.describe())
    # compute_metrics(join_df, cell_types, masks, split='test')
    model_dfs[model_name] = join_df

    # compute_metrics(join_df, cell_types, masks, split='train')
    # model_dfs[model_name] = join_df


    # model_name = 'EpiCast: 0206_gosai_sei_vef'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # pred = np.load('saved/0206_gosai_sei_vef/0206_103219/preds.npy')
    # pred_df[pred_cols] = pred
    # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')
    # model_dfs[model_name] = join_df

    # # model_name = 'EpiCast: 0209_gosai_sei_vef_old_version'
    # # print(model_name)
    # # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # # pred = np.load('saved/0209_gosai_sei_vef_old_version/0209_065805/preds.npy')
    # # pred_df[pred_cols] = pred
    # # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # # compute_metrics(join_df, cell_types, masks,split='test')
    # # model_dfs[model_name] = join_df





    model_name = 'Seq only: malinois official'
    print(model_name)
    pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    pred = np.load('outputs/predictions/malinois_original_pred.npy')
    pred = pred - pred.mean(axis=0)
    pred_df[pred_cols[:3]] = pred
    pred_df['HCT116_pred'] = pred.mean(axis=1)
    pred_df['A549_pred'] = pred.mean(axis=1)
    print(pred_df.describe())
    join_df = pd.concat([mpra_df, pred_df],axis=1)
    compute_metrics(join_df, cell_types, masks,split='test')
    model_dfs[model_name] = join_df


    # # model_name = 'Seq only: malinois retrain'
    # # print(model_name)
    # # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # # # pred = np.load('saved/0207_gosai_malinois_600/0207_034209/preds.npy')
    # # pred = np.load('saved/0207_gosai_malinois_600/0209_032051/preds_pos.npy')
    # # pred_df[pred_cols[:3]] = pred
    # # pred_df['HCT116_pred'] = pred.mean(axis=1)
    # # pred_df['A549_pred'] = pred.mean(axis=1)
    # # join_df = pd.concat([mpra_df, pred_df],axis=1)
    # # compute_metrics(join_df, cell_types, masks,split='test')
    # # model_dfs[model_name] = join_df

    # model_name = 'Seq only: malinois retrain rc'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # # pred = np.load('saved/0207_gosai_malinois_600/0207_034209/preds.npy')
    # pred = np.load('saved/0207_gosai_malinois_600/0209_032051/preds_rc.npy')
    # pred = pred - pred.mean(axis=0)

    # pred_df[pred_cols[:3]] = pred
    # pred_df['HCT116_pred'] = pred.mean(axis=1)
    # pred_df['A549_pred'] = pred.mean(axis=1)
    # join_df = pd.concat([mpra_df, pred_df],axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')
    # model_dfs[model_name] = join_df



    # model_name = 'Seq only: 0206_gosai_conv_200 trans=0'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # pred = np.load('saved/0206_gosai_conv_200/0206_102913/preds.npy')
    # pred_df[pred_cols[:3]] = pred
    # pred_df['HCT116_pred'] = pred.mean(axis=1)
    # pred_df['A549_pred'] = pred.mean(axis=1)
    # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')
    # model_dfs[model_name] = join_df


    # model_name = 'Seq only: 0206_gosai_convtrans_200 trans=3'
    # print(model_name)
    # pred_df = pd.DataFrame(columns=pred_cols, dtype=float)
    # pred = np.load('saved/0206_gosai_convtrans_200/0206_102827/preds.npy')
    # pred_df[pred_cols[:3]] = pred
    # pred_df['HCT116_pred'] = pred.mean(axis=1)
    # pred_df['A549_pred'] = pred.mean(axis=1)
    # join_df = pd.concat([mpra_df, pred_df], axis=1)
    # compute_metrics(join_df, cell_types, masks,split='test')

    # model_dfs[model_name] = join_df




    summary_df = compute_summary_table(model_dfs, cell_types, masks, split='test', metric_fn=metrics.pearson)
    print(summary_df)

    summary_df = compute_summary_table(model_dfs, cell_types, masks, split='test+specific', metric_fn=metrics.pearson)
    print(summary_df)

    # # summary_df = compute_summary_table(model_dfs, cell_types, masks, split='test', metric_fn=metrics.rmse)
    # # print(summary_df)

    # summary_df = compute_summary_table(model_dfs, cell_types, masks, split='test+specific', metric_fn=metrics.rmse)
    # print(summary_df)

    # # summary_df = compute_summary_table(model_dfs, cell_types, masks, split='test', metric_fn=metrics.mae)
    # # print(summary_df)

    # summary_df = compute_summary_table(model_dfs, cell_types, masks, split='test+specific', metric_fn=metrics.mae)
    # print(summary_df)




if __name__ == '__main__':
    main()
