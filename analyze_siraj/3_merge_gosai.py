import pandas as pd
import genoml
from genoml.metrics import pearson

pd.set_option('display.max_columns', None)

Siraj_MPRA_df = pd.read_csv('data/Siraj_MPRA/Siraj_MPRA_len200.tsv', sep='\t')
print(Siraj_MPRA_df.shape)
print(Siraj_MPRA_df.describe())

Gosai_MPRA_df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_nature.tsv', sep='\t')
print(Gosai_MPRA_df.shape)
print(Gosai_MPRA_df.describe())

print(Gosai_MPRA_df.duplicated(subset='seq').sum())
print(Siraj_MPRA_df.duplicated(subset='seq').sum())

cell_types = ['HepG2', 'K562', 'SK-N-SH', 'HCT116', 'A549']

merged_df = pd.merge(Gosai_MPRA_df, Siraj_MPRA_df[['seq'] + cell_types], how='left', on='seq', suffixes=('', '_y'))

for ct in cell_types[:3]:
    x = merged_df[f'{ct}']
    y = merged_df[f'{ct}_y']
    print(ct, pearson(x, y))

merged_df.drop(columns=['HepG2_y', 'K562_y', 'SK-N-SH_y'], inplace=True)

print(merged_df.shape)
print(merged_df.describe())

merged_df.to_csv('data/Gosai_MPRA/merged_gosai_siraj.tsv', sep='\t', index=False)



df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_760679.tsv', sep='\t')
print(df.shape)
print(df.describe())