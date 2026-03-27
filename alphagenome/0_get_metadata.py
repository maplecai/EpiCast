import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from genoml.utils import pad_seq
from alphagenome.models import dna_client, dna_output, dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

metadata = metadata_lib.load(dna_model.Organism.HOMO_SAPIENS).concatenate()
metadata.to_csv("alphagenome/metadata_padded.tsv", sep="\t", index=False)

# metadata = dna_model.output_metadata(dna_model.Organism.HOMO_SAPIENS).concatenate()
# metadata.to_csv("alphagenome/metadata.tsv", sep="\t", index=False)
metadata = metadata[metadata['name'] != "Padding"].reset_index(drop=True)
metadata.to_csv("alphagenome/metadata.tsv", sep="\t", index=False)

print(metadata['Assay title'].value_counts())

metadata = metadata[metadata['genetically_modified'] == False]

metadata['cell_type'] = metadata['biosample_name']
metadata['assay'] = pd.NA
metadata['index'] = metadata.index

mask = (metadata['Assay title'] == 'ATAC-seq')
metadata.loc[mask, 'assay'] = 'ATAC'
mask = (metadata['Assay title'] == 'DNase-seq')
metadata.loc[mask, 'assay'] = 'DNase'
mask = (metadata['Assay title'] == 'total RNA-seq')
metadata.loc[mask, 'assay'] = 'RNA-seq'
mask = (metadata['Assay title'] == 'hCAGE')
metadata.loc[mask, 'assay'] = 'CAGE'
mask = (metadata['Assay title'] == 'TF ChIP-seq')
metadata.loc[mask, 'assay'] = metadata.loc[mask, 'transcription_factor']
mask = (metadata['Assay title'] == 'Histone ChIP-seq')
metadata.loc[mask, 'assay'] = metadata.loc[mask, 'histone_mark']


df_pivot = metadata.pivot_table(
    values="index", 
    index="cell_type", 
    columns="assay", 
    aggfunc=list,
)

df_pivot.to_csv("alphagenome/metadata_pivot.tsv", sep="\t", index=False)

assays = ['DNase', 'H3K4me3', 'H3K27ac', 'CTCF']
df_pivot = df_pivot.dropna(subset=assays)
df_pivot = df_pivot[assays]
assert df_pivot.notna().all().all()
assert df_pivot.map(lambda x: len(x) == 1).all().all()
df_pivot = df_pivot.map(lambda x: x[0])
print(df_pivot.to_string())

df_pivot.to_csv("alphagenome/metadata_pivot_selected.tsv", sep="\t")
