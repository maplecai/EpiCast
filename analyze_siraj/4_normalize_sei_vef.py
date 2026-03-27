import pandas as pd
import genoml
from genoml.metrics import pearson

VEF_df = pd.read_csv("data/Siraj_MPRA/Siraj_MPRA_Sei_VEF_raw.tsv", sep="\t")
print(VEF_df.shape)
print(VEF_df.describe())

VEF_df = (VEF_df - VEF_df.mean(axis=0)) / (VEF_df.std(axis=0))
print(VEF_df.describe())
VEF_df.to_csv("data/Siraj_MPRA/Siraj_MPRA_Sei_VEF_norm.tsv", sep="\t", index=False)