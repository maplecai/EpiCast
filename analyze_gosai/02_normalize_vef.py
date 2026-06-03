import numpy as np
import pandas as pd
from epicast import metrics
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





# def fit_quantile_reference_and_column_distributions(train_matrix):
#     """
#     train_matrix: shape (n_train_samples, n_features_in_group)
#     例如同一个 assay 下不同 cell type 的列

#     返回:
#         reference: 该 assay 的 train reference quantile distribution
#         train_sorted_cols: list，每一列各自训练集排序后的值
#     """
#     train_matrix = np.asarray(train_matrix, dtype=float)

#     if np.isnan(train_matrix).any():
#         raise ValueError("NaN detected in train_matrix for quantile normalization. Please handle NaNs first.")

#     # assay-level reference
#     sorted_train = np.sort(train_matrix, axis=0)   # shape (n_train, n_cols)
#     reference = np.mean(sorted_train, axis=1)      # shape (n_train,)

#     # per-column train distributions
#     train_sorted_cols = [np.sort(train_matrix[:, j]) for j in range(train_matrix.shape[1])]

#     return reference, train_sorted_cols


# def map_values_by_train_quantile(values, train_sorted_col, reference):
#     """
#     对单列 values 做严格 train-only quantile mapping：

#     1) 用该列训练集分布 train_sorted_col 估计每个值的经验分位数 q
#     2) 再把 q 映射到 assay-level train reference 上

#     values: shape (n_samples,)
#     train_sorted_col: shape (n_train_samples,)
#     reference: shape (n_train_samples,)
#     """
#     values = np.asarray(values, dtype=float)
#     out = np.full(values.shape, np.nan, dtype=float)

#     valid_mask = ~np.isnan(values)
#     x = values[valid_mask]

#     if len(x) == 0:
#         return out

#     train_sorted_col = np.asarray(train_sorted_col, dtype=float)
#     reference = np.asarray(reference, dtype=float)

#     n_train = len(train_sorted_col)
#     if n_train == 0:
#         raise ValueError("Empty training distribution.")

#     # 用训练列分布估计每个值的分位数
#     # ties 用 left/right 的平均 rank，避免全部挤到同一点
#     left = np.searchsorted(train_sorted_col, x, side='left')
#     right = np.searchsorted(train_sorted_col, x, side='right')
#     avg_rank = (left + right) / 2.0

#     # 转成 [0, 1] 分位数
#     q = (avg_rank + 0.5) / (n_train + 1.0)
#     q = np.clip(q, 1e-6, 1 - 1e-6)

#     # 用分位数映射到 assay-level train reference
#     # reference 的横轴也是 train quantile
#     ref_grid = np.linspace(1.0 / (n_train + 1.0), n_train / (n_train + 1.0), n_train)
#     mapped = np.interp(q, ref_grid, reference)

#     out[valid_mask] = mapped
#     return out


# def apply_quantile_mapping_by_train_reference(matrix, train_sorted_cols, reference):
#     """
#     matrix: shape (n_samples, n_features_in_group)
#     train_sorted_cols: list of sorted train values for each column
#     reference: assay-level train reference

#     返回:
#         对每一列分别基于该列训练分布做 quantile mapping 后的矩阵
#     """
#     matrix = np.asarray(matrix, dtype=float)
#     out = np.full(matrix.shape, np.nan, dtype=float)

#     n_rows, n_cols = matrix.shape
#     if len(train_sorted_cols) != n_cols:
#         raise ValueError("train_sorted_cols length does not match matrix n_cols.")

#     for j in range(n_cols):
#         out[:, j] = map_values_by_train_quantile(
#             matrix[:, j],
#             train_sorted_cols[j],
#             reference
#         )

#     return out







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
    vef_raw_df = pd.read_csv(vef_path, sep='\t')
    print(vef_raw_df.shape)
    print(vef_raw_df.describe())

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
    corr_mat = compute_metric(mpra_df, vef_raw_df, metric_fn=metrics.pearson)
    print('vef_raw_df')
    print(corr_mat)



    # # =========================
    # # 1) log1p
    # # =========================
    # vef_log1p_df = np.log1p(vef_raw_df)
    # vef_log1p_df.to_csv(f'{vef_path[:-8]}_log1p.tsv', sep='\t', index=False)
    # print(vef_log1p_df.describe())
    # corr_mat = compute_metric(mpra_df, vef_log1p_df, metric_fn=metrics.pearson)
    # print('vef_log1p_df')
    # print(corr_mat)



    # vef_log1p_df = np.log1p(vef_raw_df*10)
    # vef_log1p_df.to_csv(f'{vef_path[:-8]}_x10_log1p.tsv', sep='\t', index=False)
    # print(vef_log1p_df.describe())
    # corr_mat = compute_metric(mpra_df, vef_log1p_df, metric_fn=metrics.pearson)
    # print('vef_log1p_df')
    # print(corr_mat)



    # vef_log1p_df = np.log1p(vef_raw_df*100)
    # vef_log1p_df.to_csv(f'{vef_path[:-8]}_x100_log1p.tsv', sep='\t', index=False)
    # print(vef_log1p_df.describe())
    # corr_mat = compute_metric(mpra_df, vef_log1p_df, metric_fn=metrics.pearson)
    # print('vef_log1p_df')
    # print(corr_mat)


    # # =========================
    # # 2) log1p + train-zscore
    # # =========================
    # vef_df = vef_raw_df.copy()
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
    # vef_df = vef_raw_df.copy()
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
    # vef_df = vef_raw_df.copy()
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
    
    def fit_quantile_reference_matrix(train_matrix):
        """
        train_matrix: shape (n_train_samples, n_train_celltype_features_in_group)

        用 train seqs × train cell types 拟合 assay-level reference distribution
        """
        train_matrix = np.asarray(train_matrix, dtype=float)

        if np.isnan(train_matrix).any():
            raise ValueError("NaN detected in train_matrix for quantile normalization. Please handle NaNs first.")

        sorted_train = np.sort(train_matrix, axis=0)
        reference = np.mean(sorted_train, axis=1)
        return reference


    def fit_column_distribution(values):
        """
        用某一列自己的可观测值拟合 source distribution
        这里 values 现在统一只取 train seqs
        """
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            raise ValueError("Empty column distribution.")
        return np.sort(values)


    def map_values_to_reference_by_column_distribution(values, col_sorted, reference):
        """
        对单列 values:
        1) 用该列自己的 train-seqs 分布 col_sorted 估计经验分位数
        2) 再映射到 train-only reference distribution
        """
        values = np.asarray(values, dtype=float)
        out = np.full(values.shape, np.nan, dtype=float)

        valid_mask = ~np.isnan(values)
        x = values[valid_mask]
        if len(x) == 0:
            return out

        col_sorted = np.asarray(col_sorted, dtype=float)
        reference = np.asarray(reference, dtype=float)

        n_col = len(col_sorted)
        n_ref = len(reference)

        if n_col == 0 or n_ref == 0:
            raise ValueError("Empty source distribution or reference.")

        # 经验分位数，ties 用 average rank
        left = np.searchsorted(col_sorted, x, side='left')
        right = np.searchsorted(col_sorted, x, side='right')
        avg_rank = (left + right) / 2.0

        q = (avg_rank + 0.5) / (n_col + 1.0)
        q = np.clip(q, 1e-6, 1 - 1e-6)

        # 将 quantile 映射到 reference
        ref_grid = np.linspace(1.0 / (n_ref + 1.0), n_ref / (n_ref + 1.0), n_ref)
        mapped = np.interp(q, ref_grid, reference)

        out[valid_mask] = mapped
        return out


    # ##### quantile normalization

    # train_cell_types = ['K562', 'HepG2', 'SK-N-SH']
    # heldout_cell_types = ['HCT116', 'A549']
    # all_cell_types = train_cell_types + heldout_cell_types

    # vef_log1p_df = np.log1p(vef_raw_df * 10).copy()
    # vef_qn_df = vef_log1p_df.copy()

    # for assay in assays:
    #     train_assay_cols = [f'{cell_type}_{assay}' for cell_type in train_cell_types]
    #     all_assay_cols = [f'{cell_type}_{assay}' for cell_type in all_cell_types]

    #     for col in all_assay_cols:
    #         if col not in vef_log1p_df.columns:
    #             raise KeyError(f"Missing VEF column: {col}")

    #     # 1) 只用 train seqs × train cell types 拟合 reference
    #     train_matrix = vef_log1p_df.loc[masks['train'], train_assay_cols].values
    #     reference = fit_quantile_reference_matrix(train_matrix)

    #     # 2) 对每一列单独做 quantile mapping
    #     #    source distribution 也只用该列在 train seqs 上的可观测值分布
    #     for col in all_assay_cols:
    #         col_values_for_fit = vef_log1p_df.loc[masks['train'], col].values
    #         col_sorted = fit_column_distribution(col_values_for_fit)

    #         # 将该列所有值映射到 train-only reference
    #         vef_qn_df[col] = map_values_to_reference_by_column_distribution(
    #             vef_log1p_df[col].values,
    #             col_sorted,
    #             reference
    #         )

    # vef_qn_df.to_csv(f'{vef_path[:-8]}_x10_log1p_qn_trainref_trainseqcol.tsv', sep='\t', index=False)
    # print(vef_qn_df.describe())
    # corr_mat = compute_metric(mpra_df, vef_qn_df, metric_fn=metrics.pearson)
    # print('vef_log1p_x10_trainref_trainseqcol_qn_df, pearson')
    # print(corr_mat)

    # =========================
    # Enformer VEF: log1p
    # =========================
    vef_path = "data/gosai_mpra/gosai_mpra_760679_enformer_vef_raw.tsv"
    vef_raw_df = pd.read_csv(vef_path, sep="\t")
    print(vef_raw_df.shape)
    print(vef_raw_df.describe())

    corr_mat = compute_metric(mpra_df, vef_raw_df, metric_fn=metrics.pearson)
    print("vef_raw_df, pearson")
    print(corr_mat)

    enformer_log1p_df = np.log1p(vef_raw_df)
    enformer_log1p_df.to_csv(f"{vef_path[:-8]}_log1p.tsv", sep="\t", index=False)
    print(enformer_log1p_df.describe())
    corr_mat = compute_metric(mpra_df, enformer_log1p_df, metric_fn=metrics.pearson)
    print("vef_log1p_df, pearson")
    print(corr_mat)



    # =========================
    # Borzoi VEF: log1p
    # =========================
    vef_path = "data/gosai_mpra/gosai_mpra_760679_borzoi_vef_raw.tsv"
    vef_raw_df = pd.read_csv(vef_path, sep="\t")
    print(vef_raw_df.shape)
    print(vef_raw_df.describe())

    corr_mat = compute_metric(mpra_df, vef_raw_df, metric_fn=metrics.pearson)
    print("vef_raw_df, pearson")
    print(corr_mat)

    enformer_log1p_df = np.log1p(vef_raw_df)
    enformer_log1p_df.to_csv(f"{vef_path[:-8]}_log1p.tsv", sep="\t", index=False)
    print(enformer_log1p_df.describe())
    corr_mat = compute_metric(mpra_df, enformer_log1p_df, metric_fn=metrics.pearson)
    print("vef_log1p_df, pearson")
    print(corr_mat)
