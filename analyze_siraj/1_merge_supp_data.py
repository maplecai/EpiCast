import pandas as pd

df6 = pd.read_excel("data/Siraj_MPRA/raw/41586_2026_10121_MOESM6_ESM.xlsx")
print(df6.head())

df6.columns = df6.iloc[0]
df6 = df6.iloc[1:].reset_index(drop=True)
df6 = df6.rename(columns={'variant': 'Variant'})
print(df6.head())

df4 = pd.read_excel("data/Siraj_MPRA/raw/41586_2026_10121_MOESM4_ESM.xlsx")
print(df4.head())

df4.columns = df4.iloc[0]
df4 = df4.iloc[1:]
print(df4.head())

merged_df = pd.merge(df6, df4, on='Variant', how='inner')
print(merged_df.head())

merged_df.to_csv("data/Siraj_MPRA/Siraj_MPRA_raw.tsv", index=False, sep='\t')
