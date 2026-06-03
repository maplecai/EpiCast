import numpy as np
import pandas as pd
from epicast import metrics
from scipy.stats import norm


if __name__ == "__main__":
    mpra_path = "data/gosai_mpra/gosai_mpra_760679.tsv"
    mpra_df = pd.read_csv(mpra_path, sep='\t')
    print(mpra_df.shape)
    print(mpra_df.describe())

    vef_path = "data/gosai_mpra/gosai_mpra_760679_ag_vef_raw.tsv"
    vef_df_raw = pd.read_csv(vef_path, sep='\t')
    print(vef_df_raw.shape)
    print(vef_df_raw.describe())

    masks = {}
    masks['total'] = np.ones(len(mpra_df), dtype=bool)
    masks['train'] = ~mpra_df['chr'].isin(['chr7', 'chr13', 'chr19', 'chr21', 'chrX'])
    masks['val'] = mpra_df['chr'].isin(['chr19', 'chr21', 'chrX'])
    masks['test'] = mpra_df['chr'].isin(['chr7', 'chr13'])

    cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
    assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']


    def summarize_for_log1p(y):
        y = np.asarray(y)
        p50 = np.percentile(y, 50)
        p95 = np.percentile(y, 95)
        p99 = np.percentile(y, 99)
        mean = y.mean()
        print("mean:", mean)
        print("median:", p50)
        print("p95:", p95)
        print("p99:", p99)
        print("mean/median:", mean / (p50 + 1e-8))
        print("p95/median:", p95 / (p50 + 1e-8))
        print("p99/median:", p99 / (p50 + 1e-8))
    summarize_for_log1p(vef_df_raw['K562_DNase'])




    def fit_standard_zscore(train_values, eps=1e-8):
        train_values = np.asarray(train_values, dtype=float)
        mean = np.nanmean(train_values)
        std = np.nanstd(train_values)
        return mean, std + eps

    def apply_standard_zscore(values, mean, std):
        values = np.asarray(values, dtype=float)
        return (values - mean) / std

    def fit_robust_stats(train_values, eps=1e-8):
        train_values = np.asarray(train_values, dtype=float)
        med = np.nanmedian(train_values)
        q1 = np.nanpercentile(train_values, 25)
        q3 = np.nanpercentile(train_values, 75)
        iqr = q3 - q1
        return med, iqr + eps

    def apply_robust_zscore(values, med, iqr):
        values = np.asarray(values, dtype=float)
        return (values - med) / iqr

    def fit_int_reference(train_values):
        train_values = np.asarray(train_values, dtype=float)
        train_values = train_values[~np.isnan(train_values)]
        train_sorted = np.sort(train_values)
        return train_sorted

    def apply_int_with_train_reference(values, train_sorted):
        values = np.asarray(values, dtype=float)
        out = np.full_like(values, np.nan, dtype=float)

        valid_mask = ~np.isnan(values)
        x = values[valid_mask]

        n = len(train_sorted)
        ranks = np.searchsorted(train_sorted, x, side='right')

        q = (ranks + 0.5) / (n + 1.0)
        q = np.clip(q, 1e-6, 1 - 1e-6)

        out[valid_mask] = norm.ppf(q)
        return out


    def compute_metric(mpra_df, vef_df, metric_fn):
        corr_df = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
        for cell_type in cell_types:
            for assay in assays:
                pred = vef_df[f'{cell_type}_{assay}']
                true = mpra_df[cell_type]
                r = metric_fn(pred, true)
                corr_df.loc[cell_type, assay] = r
        return corr_df
        

    # =========================
    # MPRA: train z-score
    # =========================
    mpra_df_zs = mpra_df.copy()
    for cell_type in cell_types:
        train_values = mpra_df.loc[masks['train'], cell_type].values
        mean, std = fit_standard_zscore(train_values)
        mpra_df_zs[cell_type] = apply_standard_zscore(mpra_df[cell_type].values, mean, std)

    print(mpra_df_zs.describe())
    mpra_df_zs.to_csv("data/gosai_mpra/gosai_mpra_760679_zs.tsv", sep='\t', index=False)

    mpra_df = mpra_df_zs

    # =========================
    # baseline: raw
    # =========================
    corr_df = compute_metric(mpra_df, vef_df_raw, metric_fn=metrics.pearson)
    print('vef_df_raw')
    print(corr_df)




    # =========================
    # 1) log1p
    # =========================
    vef_df = vef_df_raw.copy()
    vef_df_log1p = np.log1p(vef_df)

    vef_df_log1p.to_csv(f'{vef_path[:-8]}_log1p.tsv', sep='\t', index=False)
    print(vef_df_log1p.describe())
    corr_df = compute_metric(mpra_df, vef_df_log1p, metric_fn=metrics.pearson)
    print('vef_df_log1p')
    print(corr_df)



    vef_df = vef_df_raw.copy()
    vef_df_log1p = np.log1p(vef_df*10)

    vef_df_log1p.to_csv(f'{vef_path[:-8]}_*10_log1p.tsv', sep='\t', index=False)
    print(vef_df_log1p.describe())
    corr_df = compute_metric(mpra_df, vef_df_log1p, metric_fn=metrics.pearson)
    print('vef_df_log1p')
    print(corr_df)



    vef_df = vef_df_raw.copy()
    vef_df_log1p = np.log1p(vef_df*100)

    vef_df_log1p.to_csv(f'{vef_path[:-8]}_*100_log1p.tsv', sep='\t', index=False)
    print(vef_df_log1p.describe())
    corr_df = compute_metric(mpra_df, vef_df_log1p, metric_fn=metrics.pearson)
    print('vef_df_log1p')
    print(corr_df)


    # # =========================
    # # 2) log1p + train-zscore
    # # =========================
    # vef_df = vef_df_raw.copy()
    # vef_df = np.log1p(vef_df)
    # vef_df_log1p_zs = vef_df.copy()

    # for col in vef_df.columns:
    #     train_values = vef_df.loc[masks['train'], col].values
    #     mean, std = fit_standard_zscore(train_values)
    #     vef_df_log1p_zs[col] = apply_standard_zscore(vef_df[col].values, mean, std)

    # vef_df_log1p_zs.to_csv(f'{vef_path[:-8]}_log1p_zs.tsv', sep='\t', index=False)
    # print(vef_df_log1p_zs.describe())
    # corr_df = compute_metric(mpra_df, vef_df_log1p_zs, metric_fn=metrics.pearson)
    # print('vef_df_log1p_zs')
    # print(corr_df)


    # # =========================
    # # 3) log1p+robust_zscore by train distribution
    # # =========================
    # vef_df = vef_df_raw.copy()
    # vef_df = np.log1p(vef_df)
    # vef_df_log1p_rzs = vef_df.copy()

    # for col in vef_df.columns:
    #     train_values = vef_df.loc[masks['train'], col].values
    #     med, iqr = fit_robust_stats(train_values)
    #     vef_df_log1p_rzs[col] = apply_robust_zscore(vef_df[col].values, med, iqr)

    # vef_df_log1p_rzs.to_csv(f'{vef_path[:-8]}_log1p_rzs.tsv', sep='\t', index=False)
    # print(vef_df_log1p_rzs.describe())
    # corr_df = compute_metric(mpra_df, vef_df_log1p_rzs, metric_fn=metrics.pearson)
    # print('vef_df_log1p_rzs, pearson')
    # print(corr_df)



    # # =========================
    # # 4) rank-based inverse normal transform by train distribution
    # # =========================
    # vef_df = vef_df_raw.copy()
    # # vef_df = np.log1p(vef_df)
    # vef_df_int = vef_df.copy()

    # for col in vef_df.columns:
    #     train_values = vef_df.loc[masks['train'], col].values
    #     train_sorted = fit_int_reference(train_values)
    #     vef_df_int[col] = apply_int_with_train_reference(vef_df[col].values, train_sorted)

    # vef_df_int.to_csv(f'{vef_path[:-8]}_int.tsv', sep='\t', index=False)
    # print(vef_df_int.describe())
    # corr_df = compute_metric(mpra_df, vef_df_int, metric_fn=metrics.pearson)
    # print('vef_df_int, pearson')
    # print(corr_df)
