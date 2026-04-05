import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

mpra_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_zs.tsv', sep='\t')
print(mpra_df.shape)

vef_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_ag_vef_log1p.tsv', sep='\t')
print(vef_df.shape)

# 避免 SK-N-SH 里的减号导致公式解析问题
mpra_df = mpra_df.rename(columns=lambda x: x.replace('-', '_'))
vef_df = vef_df.rename(columns=lambda x: x.replace('-', '_'))

cell_types = ['K562', 'HepG2', 'SK_N_SH', 'HCT116', 'A549']


def partial_corr_by_residuals(df, y_col, x_col, covars):
    # y 去掉协变量影响
    formula_y = f'{y_col} ~ ' + ' + '.join(covars)
    m_y = smf.ols(formula_y, data=df).fit()
    y_resid = m_y.resid

    # x 去掉协变量影响
    formula_x = f'{x_col} ~ ' + ' + '.join(covars)
    m_x = smf.ols(formula_x, data=df).fit()
    x_resid = m_x.resid

    # Pearson 偏相关
    r, p = pearsonr(x_resid, y_resid)

    # 残差上的 Spearman
    r_s, p_s = spearmanr(x_resid, y_resid)

    return {
        'pearson_partial_r': r,
        'pearson_p': p,
        'spearman_resid_r': r_s,
        'spearman_resid_p': p_s,
        'n': len(df)
    }


results = []

for cell_type in cell_types:
    y_col = cell_type
    ctcf = f'{cell_type}_CTCF'
    dnase = f'{cell_type}_DNase'
    h3k4me3 = f'{cell_type}_H3K4me3'
    h3k27ac = f'{cell_type}_H3K27ac'

    df = pd.concat([
        mpra_df[[y_col]],
        vef_df[[dnase, h3k4me3, h3k27ac, ctcf]]
    ], axis=1).dropna()

    # 1) 只控制 DNase
    res_dnase = partial_corr_by_residuals(
        df=df,
        y_col=y_col,
        x_col=ctcf,
        covars=[dnase]
    )

    # 2) 控制另外 3 个变量：DNase + H3K4me3 + H3K27ac
    res_all3 = partial_corr_by_residuals(
        df=df,
        y_col=y_col,
        x_col=ctcf,
        covars=[dnase, h3k4me3, h3k27ac]
    )

    # 2) 控制CTCF, 计算DNase
    res_3 = partial_corr_by_residuals(
        df=df,
        y_col=y_col,
        x_col=dnase,
        covars=[ctcf]
    )

    print(f'\n===== {cell_type} =====')
    print('partial corr(CTCF, activity | DNase)')
    print('Pearson:', res_dnase['pearson_partial_r'], 'p =', res_dnase['pearson_p'])
    # print('Spearman-like:', res_dnase['spearman_resid_r'], 'p =', res_dnase['spearman_resid_p'])

    print('partial corr(CTCF, activity | DNase, H3K4me3, H3K27ac)')
    print('Pearson:', res_all3['pearson_partial_r'], 'p =', res_all3['pearson_p'])
    # print('Spearman-like:', res_all3['spearman_resid_r'], 'p =', res_all3['spearman_resid_p'])

    print('partial corr(DNase, activity | CTCF)')
    print('Pearson:', res_3['pearson_partial_r'], 'p =', res_3['pearson_p'])
    # print('Spearman-like:', res_3['spearman_resid_r'], 'p =', res_3['spearman_resid_p'])

    results.append({
        'cell_type': cell_type,
        'n': res_dnase['n'],
        'partial_r_ctcf_given_dnase': res_dnase['pearson_partial_r'],
        'p_ctcf_given_dnase': res_dnase['pearson_p'],
        'partial_r_ctcf_given_all3': res_all3['pearson_partial_r'],
        'p_ctcf_given_all3': res_all3['pearson_p'],
        'spearman_like_r_given_dnase': res_dnase['spearman_resid_r'],
        'spearman_like_p_given_dnase': res_dnase['spearman_resid_p'],
        'spearman_like_r_given_all3': res_all3['spearman_resid_r'],
        'spearman_like_p_given_all3': res_all3['spearman_resid_p'],
    })

results_df = pd.DataFrame(results)
# print('\nSummary:')
# print(results_df)