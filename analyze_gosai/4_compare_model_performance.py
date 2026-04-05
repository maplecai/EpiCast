import sys
import numpy as np
import pandas as pd
# from genoml.metrics import pearson, spearman
# from genoml.utils import *
from genoml import models, datasets, utils, metrics
from typing import Dict, List, Optional


def define_masks(
    mpra_df: pd.DataFrame,
    cell_types: list[str],
) -> dict[str, np.ndarray]:

    masks = {}
    masks['total'] = np.ones(len(mpra_df), dtype=bool)
    masks['train'] = ~mpra_df['chr'].isin(['chr7', 'chr13', 'chr19', 'chr21', 'chrX'])
    masks['val'] = mpra_df['chr'].isin(['chr19', 'chr21', 'chrX'])
    masks['test'] = mpra_df['chr'].isin(['chr7', 'chr13'])

    # 上下各0.5%的序列定义为 cell type specific
    for cell_type in cell_types:
        ref_mean = mpra_df[cell_types[:3]].mean(axis=1)
        delta = mpra_df[cell_type] - ref_mean

        d = delta.dropna()
        q005 = np.percentile(d, 1)
        q995 = np.percentile(d, 99)

        specific = (delta < q005) | (delta > q995)
        # specific = (delta > q995)
        masks[f'{cell_type}_specific'] = specific

        print(
            f"{cell_type}_specific n={(specific).sum()}  "
            f"+={(delta[specific] > 0).sum()} "
            f"-={(delta[specific] < 0).sum()} "
            f"{delta.mean()}, {delta.std()}"
        )
        print(
            f"test+{cell_type}_specific n={(masks['test'] & specific).sum()}  "
            f"+={(delta[masks['test'] & specific] > 0).sum()}  "
            f"-={(delta[masks['test'] & specific] < 0).sum()}"
        )

    std = mpra_df[cell_types].std(axis=1, skipna=True)
    thr = np.percentile(std, 95)
    masks['high_std'] = std > thr


    for cell_type in cell_types:
        values = mpra_df[cell_type]
        q99 = np.percentile(values.dropna(), 99)
        high = values > q99
        masks[f'{cell_type}_high'] = high

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
    elif split == "specific" or split == "high":
        return masks[f"{cell_type}_{split}"]
    elif '+' in split:
        split1, split2 = split.split('+')
        return get_mask(split1, masks, cell_type) & get_mask(split2, masks, cell_type)
    else:
        raise ValueError(f"Unknown split: {split}")

def compute_metric(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cell_types: list[str],
    masks: dict[str, np.ndarray],
    split: str,
    metric_fn,
) -> pd.DataFrame:

    metric_df = pd.DataFrame(
        index=cell_types,
        columns=[f"{c}_pred" for c in cell_types],
        dtype=float
    )

    for c1 in cell_types:
        mask = get_mask(split, masks, cell_type=c1)
        for c2 in cell_types:
            x = true_df.loc[mask, c1]
            y = pred_df.loc[mask, f'{c2}_pred']

            metric_df.loc[c1, f'{c2}_pred'] = metric_fn(x, y)

    return metric_df

def append_summary_from_metric_df(
    summary_df: pd.DataFrame,
    model_name: str,
    metric_df: pd.DataFrame,
    cell_types: List[str],
) -> None:
    summary_df.loc[model_name] = {
        ct: metric_df.loc[ct, f"{ct}_pred"]
        for ct in cell_types
    }


def main():
    # mpra_df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_760679.tsv', sep='\t')
    mpra_df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_760679_norm_0209.tsv', sep='\t')

    print(mpra_df.shape)
    print(mpra_df.describe())

    cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
    pred_cols = [f'{cell_type}_pred' for cell_type in cell_types]

    masks = define_masks(mpra_df, cell_types)
    print(masks.keys())


    for split in ['test', 'test+specific']:
        for metric_fn in [metrics.pearson, metrics.spearman]:

            summary_df = pd.DataFrame(columns=cell_types)

            model_name = 'mean of 3 cell types true label'
            # print(model_name)
            true_df = mpra_df[cell_types].copy()
            pred_df = pd.DataFrame(columns=pred_cols)
            true_mean = true_df[cell_types[:3]].mean(axis=1)
            for c in cell_types:
                pred_df[f'{c}_pred'] = true_mean
            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)

            true_res_df = true_df.subtract(true_df.mean(axis=1), axis=0)
            pred_res_df = pred_df.subtract(pred_df.mean(axis=1), axis=0)
            metric_df = compute_metric(true_res_df, pred_res_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, f'{model_name} residual', metric_df, cell_types)

            # model_name = 'mean of 3 cell types true label residual'
            # # print(model_name)
            # 
            # pred_df = pd.DataFrame(columns=pred_cols)
            # true_mean = true_df[cell_types[:3]].mean(axis=1)
            # for c in cell_types:
            #     pred_df[f'{c}_pred'] = true_mean
            # pred_df = pred_df.subtract(pred_df.mean(axis=1), axis=0)
            # true_df = true_df.subtract(true_df.mean(axis=1), axis=0)

            # metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            # append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'EpiCast: 0123_Gosai_ConvTransFeature_AG_VEF'
            pred_df = pd.DataFrame(columns=pred_cols)
            pred = np.load('saved/0123_Gosai_ConvTransFeature_AG_VEF.yaml/0123_021357/preds.npy')
            pred_df[pred_cols] = pred
            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)

            true_res_df = true_df.subtract(true_df.mean(axis=1), axis=0)
            pred_res_df = pred_df.subtract(pred_df.mean(axis=1), axis=0)
            metric_df = compute_metric(true_res_df, pred_res_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, f'{model_name} residual', metric_df, cell_types)



            model_name = 'EpiCast: 0206_gosai_ag_vef trans=0'
            pred_df = pd.DataFrame(columns=pred_cols)
            pred = np.load('saved/0206_gosai_ag_vef/0206_025223/preds.npy')
            pred_df[pred_cols] = pred

            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'EpiCast: 0206_gosai_ag_vef trans=3'
            pred_df = pd.DataFrame(columns=pred_cols)
            pred = np.load('saved/0206_gosai_ag_vef/0206_025400/preds.npy')
            pred_df[pred_cols] = pred

            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            # model_name = 'EpiCast: 0207_gosai_ag_vef trans=3 huberloss'
            # pred_df = pd.DataFrame(columns=pred_cols)
            # pred = np.load('saved/0207_gosai_ag_vef/0207_043226/preds.npy')
            # pred_df[pred_cols] = pred

            # metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            # append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'EpiCast: 0225_gosai_ag_vef'
            pred_df = pd.DataFrame(columns=pred_cols)
            pred = np.load('saved/0225_gosai_ag_vef/0226_012050/preds.npy')
            pred_df[pred_cols] = pred

            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'EpiCast: 0206_gosai_sei_vef'
            pred_df = pd.DataFrame(columns=pred_cols)
            pred = np.load('saved/0206_gosai_sei_vef/0206_103219/preds.npy')
            pred_df[pred_cols] = pred

            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'EpiCast: 0327_gosai_ag_vef pad=0.25'
            pred_df = pd.DataFrame(columns=pred_cols)
            pred = np.load('saved/0327_gosai_ag_vef/0327_031759/preds.npy')
            pred_df[pred_cols] = pred

            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'EpiCast: 0330_gosai_ag_convfilmnet/1'
            pred_df = pd.DataFrame(columns=pred_cols)
            pred = np.load('saved/0330_gosai_ag_convfilmnet/0330_040003/preds.npy')
            pred_df[pred_cols] = pred

            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)







            # model_name = '0404_gosai_ag_vef_log1p/0403_032738/preds.npy'
            # pred = np.load('saved/0404_gosai_ag_vef_log1p/0403_032738/preds.npy')
            # pred_df = pd.DataFrame(pred, columns=pred_cols)
            # metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            # append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)

            # model_name = '0404_gosai_ag_vef_log1p2/0403_032843/preds.npy'
            # pred = np.load('saved/0404_gosai_ag_vef_log1p2/0403_032843/preds.npy')
            # pred_df = pd.DataFrame(pred, columns=pred_cols)
            # metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            # append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            # model_name = '0404_gosai_ag_vef_log1p/0403_032738/preds.npy residual'
            # pred = np.load('saved/0404_gosai_ag_vef_log1p/0403_032738/preds.npy')
            # pred_df = pd.DataFrame(pred, columns=pred_cols)
            # true_res_df = true_df.subtract(true_df.mean(axis=1), axis=0)
            # pred_res_df = pred_df.subtract(pred_df.mean(axis=1), axis=0)
            # metric_df = compute_metric(true_res_df, pred_res_df, cell_types, masks, split=split, metric_fn=metric_fn)
            # append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)

            # model_name = '0404_gosai_ag_vef_log1p2/0403_032843/preds.npy residual'
            # pred = np.load('saved/0404_gosai_ag_vef_log1p2/0403_032843/preds.npy')
            # pred_df = pd.DataFrame(pred, columns=pred_cols)
            # true_res_df = true_df.subtract(true_df.mean(axis=1), axis=0)
            # pred_res_df = pred_df.subtract(pred_df.mean(axis=1), axis=0)
            # metric_df = compute_metric(true_res_df, pred_res_df, cell_types, masks, split=split, metric_fn=metric_fn)
            # # metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            # append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)





            model_name = '0404_gosai_ag_vef_int/0403_035956/preds.npy'
            pred = np.load('saved/0404_gosai_ag_vef_int/0403_035956/preds.npy')
            pred_df = pd.DataFrame(pred, columns=pred_cols)
            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)

            model_name = '0404_gosai_ag_vef_log1p/0403_032738/preds.npy'
            pred = np.load('saved/0404_gosai_ag_vef_log1p/0403_032738/preds.npy')
            pred_df = pd.DataFrame(pred, columns=pred_cols)
            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)

            model_name = '0404_gosai_ag_vef_log1p2/0403_032843/preds.npy'
            pred = np.load('saved/0404_gosai_ag_vef_log1p2/0403_032843/preds.npy')
            pred_df = pd.DataFrame(pred, columns=pred_cols)
            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)





            model_name = 'saved/0405_gosai_ag_vef_log1p/0404_065142/preds.npy'
            pred = np.load('saved/0405_gosai_ag_vef_log1p/0404_065142/preds.npy')
            pred_df = pd.DataFrame(pred, columns=pred_cols)
            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'saved/0405_gosai_ag_vef_log1p_model2/0404_102739/preds.npy'
            pred = np.load('saved/0405_gosai_ag_vef_log1p_model2/0404_102739/preds.npy')
            pred_df = pd.DataFrame(pred, columns=pred_cols)
            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'saved/0405_gosai_ag_vef_log1p_ablation_CTCF/0404_120952/preds.npy'
            pred = np.load('saved/0405_gosai_ag_vef_log1p_ablation_CTCF/0404_120952/preds.npy')
            pred_df = pd.DataFrame(pred, columns=pred_cols)
            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)




            # model_name = 'Seq only: malinois retrain rc'
            # pred = np.load('saved/0207_gosai_malinois_600/0209_032051/preds_rc.npy')
            # # pred = pred - pred.mean(axis=0)
            # pred_df = pd.DataFrame(pred, columns=pred_cols[:3])
            # pred_df['HCT116_pred'] = pred.mean(axis=1)
            # pred_df['A549_pred'] = pred.mean(axis=1)

            # metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            # append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'Seq only: 0206_gosai_conv_200 trans=0'
            pred = np.load('saved/0206_gosai_conv_200/0206_102913/preds.npy')
            pred_df = pd.DataFrame(pred, columns=pred_cols[:3])
            pred_df['HCT116_pred'] = pred.mean(axis=1)
            pred_df['A549_pred'] = pred.mean(axis=1)

            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            model_name = 'Seq only: 0206_gosai_convtrans_200 trans=3'
            pred = np.load('saved/0206_gosai_convtrans_200/0206_102827/preds.npy')
            pred_df = pd.DataFrame(pred, columns=pred_cols[:3])
            pred_df[pred_cols[:3]] = pred
            pred_df['HCT116_pred'] = pred.mean(axis=1)
            pred_df['A549_pred'] = pred.mean(axis=1)

            metric_df = compute_metric(true_df, pred_df, cell_types, masks, split=split, metric_fn=metric_fn)
            append_summary_from_metric_df(summary_df, model_name, metric_df, cell_types)


            print(f"summary_df ({split}, {metric_fn.__name__})")
            print(summary_df)


if __name__ == '__main__':
    main()