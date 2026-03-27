import pandas as pd

df = pd.read_csv('data/Siraj_MPRA/Siraj_MPRA_raw.tsv', sep='\t')
print(df.shape)
print(df.columns)

# 提取allele 1和allele 2的数据再合并
df_allele_1 = df[[
    'Variant', 
    'Allele 1 Oligo', 
    'allele1 log2FC activity in A549', 
    'allele1 log2FC activity in HEPG2', 
    'allele1 log2FC activity in K562', 
    'allele1 log2FC activity in SKNSH', 
    'allele1 log2FC activity in HCT116'
]]
df_allele_2 = df[[
    'Variant', 
    'Allele 2 Oligo', 
    'allele2 log2FC activity in A549', 
    'allele2 log2FC activity in HEPG2', 
    'allele2 log2FC activity in K562', 
    'allele2 log2FC activity in SKNSH', 
    'allele2 log2FC activity in HCT116'
]]

df_allele_1 = df_allele_1.rename(columns={
    'Variant': 'description', 
    'Allele 1 Oligo': 'seq', 
    'allele1 log2FC activity in A549': 'A549', 
    'allele1 log2FC activity in HEPG2': 'HepG2', 
    'allele1 log2FC activity in K562': 'K562', 
    'allele1 log2FC activity in SKNSH': 'SK-N-SH', 
    'allele1 log2FC activity in HCT116': 'HCT116'
})
df_allele_2 = df_allele_2.rename(columns={
    'Variant': 'description', 
    'Allele 2 Oligo': 'seq', 
    'allele2 log2FC activity in A549': 'A549',
    'allele2 log2FC activity in HEPG2': 'HepG2',
    'allele2 log2FC activity in K562': 'K562',
    'allele2 log2FC activity in SKNSH': 'SK-N-SH',
    'allele2 log2FC activity in HCT116': 'HCT116'
})

df_allele_1['allele'] = 'ref'
df_allele_2['allele'] = 'alt'

df = pd.concat([df_allele_1, df_allele_2], axis=0)
df = df.reset_index(drop=True)

df = df[df['seq'].str.len() == 200].reset_index(drop=True)

# print(df[df.duplicated('seq')])

df = df.groupby('seq', as_index=False).agg({
    'K562':'mean', 
    'HepG2':'mean',
    'A549':'mean',
    'SK-N-SH':'mean',
    'HCT116':'mean',
    'description':'first',
    'allele':'first',
})

# 把 description 拆成chr, pos, ref, alt
df[['chr', 'pos', 'ref', 'alt']] = df['description'].str.split(':', expand=True)
df['pos'] = df['pos'].astype(int)
df['start'] = df['pos'] - 100
df['end'] = df['pos'] + 100

df = df.sort_values(by=['chr', 'pos', 'allele']).reset_index(drop=True)
print(df.shape)

df.to_csv("data/Siraj_MPRA/Siraj_MPRA_len200.tsv", index=False, sep='\t')
