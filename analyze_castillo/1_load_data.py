import pandas as pd

df = pd.read_excel("data/Castillo_MPRA/media-10.xlsx")
print(df.shape)
print(df.columns)

print(df['source'].value_counts())
print(df['sequence'].str.len().value_counts())

cell_types = [
    'NT2_D1',
    'GM12878',
    '786_O',
    'SKNSH',
    'WERI_Rb1',
    'SJCRH30',
    'HepG2',
    'K562',
    'MCF7',
    'HeLaS3',
    # 'HEK293',
    # 'HMC3',
    # 'Retina',
]

exp_cols = [
    f'log2FC_{cell_type}'
    for cell_type in cell_types
]

base_cols = [
    'id',
    'sequence',
    'category',
    'source',
    'target',
]

df = df[df['source'].isin(['fsp', 'genome_dhs', 'den'])].reset_index(drop=True)
df = df[base_cols + exp_cols]
df = df.rename(
    columns={
        f'log2FC_{cell_type}': f'{cell_type}'
        for cell_type in cell_types
    }
)
df = df.rename(
    columns={
        'sequence': 'seq',
    }
)

df = df.drop_duplicates('sequence').reset_index(drop=True)

print(df.describe())
df.to_csv("data/Castillo_MPRA/Castillo_MPRA_processed.tsv", sep='\t', index=False)
