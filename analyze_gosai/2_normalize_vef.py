import numpy as np
import pandas as pd
from genoml import metrics
from scipy.stats import norm




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









def fit_quantile_reference_matrix(train_matrix):
    """
    train_matrix: shape (n_samples_train, n_features_in_group)
    这里的 group 就是同一个 assay 下不同 cell type 的列
    返回该 group 的 reference quantile distribution
    """
    train_matrix = np.asarray(train_matrix, dtype=float)

    if np.isnan(train_matrix).any():
        raise ValueError("NaN detected in train_matrix for quantile normalization. Please handle NaNs first.")

    sorted_train = np.sort(train_matrix, axis=0)
    reference = np.mean(sorted_train, axis=1)
    return reference


def apply_quantile_normalization_with_reference(matrix, reference):
    """
    matrix: shape (n_samples, n_features_in_group)
    reference: shape (n_samples_train,)
    对每一列按列内排序后，映射到 train-derived reference quantiles
    若 n_samples != len(reference)，用线性插值把 reference 拉到对应长度
    """
    matrix = np.asarray(matrix, dtype=float)
    out = np.full(matrix.shape, np.nan, dtype=float)

    n_rows, n_cols = matrix.shape
    ref = np.asarray(reference, dtype=float)

    if len(ref) != n_rows:
        old_q = np.linspace(0.0, 1.0, len(ref))
        new_q = np.linspace(0.0, 1.0, n_rows)
        ref_used = np.interp(new_q, old_q, ref)
    else:
        ref_used = ref.copy()

    for j in range(n_cols):
        col = matrix[:, j]
        valid_mask = ~np.isnan(col)
        x = col[valid_mask]

        if len(x) == 0:
            continue

        order = np.argsort(x, kind='mergesort')
        x_sorted = x[order]

        # 处理 ties：同值取对应 reference 区间的平均值
        mapped_sorted = np.empty_like(x_sorted, dtype=float)

        start = 0
        while start < len(x_sorted):
            end = start + 1
            while end < len(x_sorted) and x_sorted[end] == x_sorted[start]:
                end += 1
            mapped_sorted[start:end] = ref_used[start:end].mean()
            start = end

        mapped = np.empty_like(x, dtype=float)
        mapped[order] = mapped_sorted
        out[valid_mask, j] = mapped

    return out













def compute_metric(mpra_df, vef_df, metric_fn):
    corr_mat = pd.DataFrame(index=cell_types, columns=assays, dtype=float)
    for cell_type in cell_types:
        for assay in assays:
            pred = vef_df[f'{cell_type}_{assay}']
            true = mpra_df[cell_type]
            r = metric_fn(pred, true)
            corr_mat.loc[cell_type, assay] = r
    return corr_mat



if __name__ == "__main__":
    mpra_path = "data/gosai_mpra/gosai_mpra_760679_raw.tsv"
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


        

    # =========================
    # MPRA: train z-score
    # =========================
    mpra_df_zs = mpra_df.copy()
    for cell_type in cell_types:
        train_values = mpra_df.loc[masks['train'], cell_type].values
        mean, std = fit_standard_zscore(train_values)
        mpra_df_zs[cell_type] = apply_standard_zscore(mpra_df[cell_type].values, mean, std)

    print(mpra_df_zs.describe())
    mpra_df_zs.to_csv("data/gosai_mpra/gosai_mpra_760679_zscore.tsv", sep='\t', index=False)



    # =========================
    # baseline: raw
    # =========================
    corr_mat = compute_metric(mpra_df, vef_df_raw, metric_fn=metrics.pearson)
    print('vef_df_raw')
    print(corr_mat)



    # # =========================
    # # 1) log1p
    # # =========================
    # vef_log1p_df = np.log1p(vef_df_raw)
    # vef_log1p_df.to_csv(f'{vef_path[:-8]}_log1p.tsv', sep='\t', index=False)
    # print(vef_log1p_df.describe())
    # corr_mat = compute_metric(mpra_df, vef_log1p_df, metric_fn=metrics.pearson)
    # print('vef_log1p_df')
    # print(corr_mat)



    # vef_log1p_df = np.log1p(vef_df_raw*10)
    # vef_log1p_df.to_csv(f'{vef_path[:-8]}_x10_log1p.tsv', sep='\t', index=False)
    # print(vef_log1p_df.describe())
    # corr_mat = compute_metric(mpra_df, vef_log1p_df, metric_fn=metrics.pearson)
    # print('vef_log1p_df')
    # print(corr_mat)



    # vef_log1p_df = np.log1p(vef_df_raw*100)
    # vef_log1p_df.to_csv(f'{vef_path[:-8]}_x100_log1p.tsv', sep='\t', index=False)
    # print(vef_log1p_df.describe())
    # corr_mat = compute_metric(mpra_df, vef_log1p_df, metric_fn=metrics.pearson)
    # print('vef_log1p_df')
    # print(corr_mat)


    # # =========================
    # # 2) log1p + train-zscore
    # # =========================
    # vef_df = vef_df_raw.copy()
    # vef_df = np.log1p(vef_df)
    # vef_log1p_df_zs = vef_df.copy()

    # for col in vef_df.columns:
    #     train_values = vef_df.loc[masks['train'], col].values
    #     mean, std = fit_standard_zscore(train_values)
    #     vef_log1p_df_zs[col] = apply_standard_zscore(vef_df[col].values, mean, std)

    # vef_log1p_df_zs.to_csv(f'{vef_path[:-8]}_log1p_zs.tsv', sep='\t', index=False)
    # print(vef_log1p_df_zs.describe())
    # corr_mat = compute_metric(mpra_df, vef_log1p_df_zs, metric_fn=metrics.pearson)
    # print('vef_log1p_df_zs')
    # print(corr_mat)


    # # =========================
    # # 3) log1p+robust_zscore by train distribution
    # # =========================
    # vef_df = vef_df_raw.copy()
    # vef_df = np.log1p(vef_df)
    # vef_log1p_df_rzs = vef_df.copy()

    # for col in vef_df.columns:
    #     train_values = vef_df.loc[masks['train'], col].values
    #     med, iqr = fit_robust_stats(train_values)
    #     vef_log1p_df_rzs[col] = apply_robust_zscore(vef_df[col].values, med, iqr)

    # vef_log1p_df_rzs.to_csv(f'{vef_path[:-8]}_log1p_rzs.tsv', sep='\t', index=False)
    # print(vef_log1p_df_rzs.describe())
    # corr_mat = compute_metric(mpra_df, vef_log1p_df_rzs, metric_fn=metrics.pearson)
    # print('vef_log1p_df_rzs, pearson')
    # print(corr_mat)



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
    # corr_mat = compute_metric(mpra_df, vef_df_int, metric_fn=metrics.pearson)
    # print('vef_df_int, pearson')
    # print(corr_mat)

    vef_log1p_df = np.log1p(vef_df_raw * 10)
    vef_qn_df = vef_log1p_df.copy()

    for assay in assays:
        assay_cols = [f'{cell_type}_{assay}' for cell_type in cell_types]

        for col in assay_cols:
            if col not in vef_log1p_df.columns:
                raise KeyError(f"Missing VEF column: {col}")

        train_matrix = vef_log1p_df.loc[masks['train'], assay_cols].values
        reference = fit_quantile_reference_matrix(train_matrix)

        full_matrix = vef_log1p_df[assay_cols].values
        normalized_matrix = apply_quantile_normalization_with_reference(full_matrix, reference)

        vef_qn_df.loc[:, assay_cols] = normalized_matrix

    vef_qn_df.to_csv(f'{vef_path[:-8]}_x10_log1p_qn.tsv', sep='\t', index=False)
    print(vef_qn_df.describe())
    corr_mat = compute_metric(mpra_df, vef_qn_df, metric_fn=metrics.pearson)
    print('vef_log1p_x10_assay_qn_df, pearson')
    print(corr_mat)
    