import numpy as np
import pandas as pd
# from genoml.metrics import pearson, spearman
# from genoml.utils import *
from genoml import models, datasets, utils, metrics

def define_data_split(
        mpra_df: pd.DataFrame,
        cell_types: list[str],
    ) -> dict[str, np.ndarray]:

    split_masks = {}
    split_masks['total'] = np.ones(len(mpra_df), dtype=bool)
    split_masks['train'] = ~mpra_df['chr'].isin(['chr7', 'chr13', 'chr19', 'chr21', 'chrX'])
    split_masks['val'] = mpra_df['chr'].isin(['chr19', 'chr21', 'chrX'])
    split_masks['test']  = mpra_df['chr'].isin(['chr7', 'chr13'])

    # 和前三种细胞类型差异top5%的序列定义为cell type specific
    for cell_type in cell_types:
        # others = [c for c in cell_types if c != cell_type]
        ref_mean = mpra_df[cell_types[:3]].mean(axis=1)
        diff = (mpra_df[cell_type] - ref_mean).abs()
        threshold = np.percentile(diff.dropna(), 95)
        split_masks[f'{cell_type}_specific'] = diff > threshold

    # for key in split_masks:
    #     print(key, split_masks[key].sum())

    # keys = list(split_masks.keys())
    # for k1 in keys:
    #     for k2 in keys:
    #         split_masks[f'{k1}+{k2}'] = split_masks[k1] & split_masks[k2]
    
    return split_masks




def compute_metrics(
    mpra_df: pd.DataFrame,
    cell_types: list[str],
    split_masks: dict[str, np.ndarray],
    splits: tuple[str, ...],
) -> dict[str, dict[str, pd.DataFrame]]:
    
    for split in splits:
        pearson_df = pd.DataFrame()
        spearman_df = pd.DataFrame()
        for c1 in cell_types:
            for c2 in cell_types:
                if split == 'specific':
                    mask = split_masks[f'{c1}_specific']
                elif split == 'test+specific':
                    mask = split_masks[f'{c1}_specific'] & split_masks['test']
                else:
                    mask = split_masks[split]
                # mask = split_masks[split]
                df = mpra_df[mask]
                x = df[f'{c1}']
                y = df[f'{c2}_pred']
                r, p = metrics.pearson(x, y)
                pearson_df.loc[f'{c1}', f'{c2}_pred'] = r
                r, p = metrics.spearman(x, y)
                spearman_df.loc[f'{c1}', f'{c2}_pred'] = r

        print(split, len(df))
        print('pearson')
        print(pearson_df)
        # print('spearman')
        # print(spearman_df)




def main():

    mpra_df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_760679.tsv', sep='\t')
    print(mpra_df.shape)
    cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116']
    split_masks = define_data_split(mpra_df, cell_types)
    print(split_masks.keys())
    

    


    print('0123_Gosai_ConvTransFeature_AG_VEF')
    print('EpiCast')

    pred = np.load('saved/0123_Gosai_ConvTransFeature_AG_VEF.yaml/0123_021357/Gosai_pred.npy')
    cols = [f'{cell_type}_pred' for cell_type in cell_types]
    pred_df = pd.DataFrame(pred, columns=cols)
    join_df = pd.concat([mpra_df, pred_df], axis=1)
    compute_metrics(join_df, cell_types, split_masks, splits=('val', 'test', 'test+specific'))



    print('0123_Gosai_ConvTrans_AG_VEF')
    print('seq only')
    
    pred = np.load('saved/0123_Gosai_ConvTrans_AG_VEF.yaml/0127_021138/Gosai_pred.npy')
    cols = [f'{cell_type}_pred' for cell_type in cell_types[:3]]
    pred_df = pd.DataFrame(pred, columns=cols)
    pred_df['HCT116_pred'] = pred.mean(axis=1)
    join_df = pd.concat([mpra_df, pred_df], axis=1)
    compute_metrics(join_df, cell_types, split_masks, splits=('val', 'test', 'test+specific'))




    print('malinois')
    print('seq only')
    
    pred_df = pd.read_csv('outputs/predictions/gosai_malinois_pred.tsv', sep='\t')
    pred_df['HCT116_pred'] = pred.mean(axis=1)
    join_df = pd.concat([mpra_df, pred_df],axis=1)
    compute_metrics(join_df, cell_types, split_masks, splits=('val', 'test', 'test+specific'))





if __name__ == '__main__':
    main()