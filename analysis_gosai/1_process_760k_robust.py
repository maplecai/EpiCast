import numpy as np
import pandas as pd

from genoml.utils import *

mpra_df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_760679.tsv', sep='\t')
print(mpra_df.shape)
print(mpra_df.describe())

cell_types = ['K562', 'HepG2', 'SK-N-SH', 'HCT116', 'A549']
norm_cols = ['K562_norm', 'HepG2_norm', 'SK-N-SH_norm', 'HCT116_norm', 'A549_norm']

# 1) 每列普通 z-score（跳过 NaN）
means = mpra_df[cell_types].mean(axis=0, skipna=True)
stds  = mpra_df[cell_types].std(axis=0, skipna=True)  # 或 ddof=0
mpra_df[norm_cols] = (mpra_df[cell_types] - means) / stds

# 2) 桥尺度：标准化后三列均值
anchor = mpra_df.loc[mpra_df['HCT116'].notna(), norm_cols[:3]].mean(axis=1)
mpra_df['HCT116_norm'] = mpra_df['HCT116_norm'] * anchor.std() + anchor.mean()

anchor = mpra_df.loc[mpra_df['A549'].notna(), norm_cols[:3]].mean(axis=1)
mpra_df['A549_norm'] = mpra_df['A549_norm'] * anchor.std() + anchor.mean()

mpra_df[cell_types] = mpra_df[norm_cols]
mpra_df = mpra_df.drop(columns=norm_cols)

print(mpra_df.describe())
mpra_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_760679_norm_0209.tsv', sep='\t', index=False)