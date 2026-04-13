import numpy as np
import pandas as pd
from genoml import metrics
from scipy.stats import norm


if __name__ == "__main__":
    mpra_path = "data/gosai_mpra/gosai_mpra_760679.tsv"
    mpra_df = pd.read_csv(mpra_path, sep='\t')
    print(mpra_df.shape)
    print(mpra_df.describe())


    masks = {}
    masks['total'] = np.ones(len(mpra_df), dtype=bool)
    masks['train'] = ~mpra_df['chr'].isin(['chr7', 'chr13', 'chr19', 'chr21', 'chrX'])
    masks['val'] = mpra_df['chr'].isin(['chr19', 'chr21', 'chrX'])
    masks['test'] = mpra_df['chr'].isin(['chr7', 'chr13'])

    cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
    assays = ['DNase_128', 'DNase', 'H3K4me3', 'H3K27ac', 'CTCF']


    def compute_metric(mpra_df, vef_df, metric_fn):
        corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
        for cell_type in cell_types:
            for assay in assays:
                pred = vef_df[f'{cell_type}_{assay}']
                true = mpra_df[cell_type]
                r = metric_fn(pred, true)
                corr_df.loc[cell_type, assay] = r
        return corr_df
    


    # vef_path = "data/gosai_mpra/gosai_mpra_760679_ag_vef_raw.tsv"
    for vef_path in ["data/gosai_mpra/gosai_mpra_760679_ag_vef_256bp_raw.tsv"]:

        vef_df_raw = pd.read_csv(vef_path, sep='\t')
        print(vef_df_raw.shape)
        print(vef_df_raw.describe())

        corr_df = compute_metric(mpra_df, vef_df_raw, metric_fn=metrics.spearman)
        print('vef_df, spearman')
        print(corr_df)


        corr_df = compute_metric(mpra_df, vef_df_raw, metric_fn=metrics.pearson)
        print('vef_df')
        print(corr_df)


        vef_df_log1p = np.log1p(vef_df_raw.copy())
        print(vef_df_log1p.describe())
        vef_df_log1p.to_csv(f'{vef_path[:-8]}_log1p.tsv', sep='\t', index=False)
        corr_df = compute_metric(mpra_df, vef_df_log1p, metric_fn=metrics.pearson)
        print('vef_df_log1p')
        print(corr_df)


        vef_df_log1p = np.log1p(vef_df_raw.copy()*10)
        print(vef_df_log1p.describe())
        vef_df_log1p.to_csv(f'{vef_path[:-8]}_*10_log1p.tsv', sep='\t', index=False)
        corr_df = compute_metric(mpra_df, vef_df_log1p, metric_fn=metrics.pearson)
        print('vef_df_*10_log1p')
        print(corr_df)
