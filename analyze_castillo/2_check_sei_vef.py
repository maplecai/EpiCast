import pandas as pd
import genoml

mpra_df = pd.read_csv("data/Castillo_MPRA/Castillo_MPRA_processed.tsv", sep='\t')
print(mpra_df.shape)
print(mpra_df.columns)

vef_df = pd.read_csv("data/Castillo_MPRA/sei_vef.tsv", sep='\t')
print(vef_df.shape)
print(vef_df.columns)

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

sei_cell_types = [
    '',
    'GM12878_B_Lymphocyte_Blood',
    '',
    '',
    '',
    '',
    'HepG2_Hepatocellular_Carcinoma',
    'K562_Leukemia_Cell',
    'MCF-7_Epithelium_Mammary_Gland',
    'HeLa-S3_Epithelium_Cervix'
]

assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']

corr_df = pd.DataFrame(index=cell_types, columns=assays)
for i, cell_type in enumerate(cell_types):
    for j, assay in enumerate(assays):
        sei_cell_type = sei_cell_types[i]
        if f'{sei_cell_type}_{assay}' in vef_df.columns:
            corr_df.loc[cell_type, assay] = genoml.metrics.pearson(mpra_df[f'{cell_type}'], vef_df[f'{sei_cell_type}_{assay}'])

print(corr_df)
