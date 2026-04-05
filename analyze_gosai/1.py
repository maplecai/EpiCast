import pandas as pd
df = pd.read_csv('data/gosai_mpra/gosai_mpra_0404.tsv', sep='\t')
print(df.shape)
print(df.describe())


df = df[(df[['K562', 'HepG2', 'SK-N-SH']].notna().all(axis=1))].reset_index(drop=True)
print(f'after filter K562, HepG2, SK-N-SH not nan, df.shape = {df.shape}')

df.to_csv('data/gosai_mpra/gosai_mpra_0404.tsv', sep='\t', index=False)