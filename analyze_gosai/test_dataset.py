# import pandas as pd
# df = pd.read_csv('data/Gosai_MPRA/Gosai_MPRA_reprocessed_0203.tsv', sep='\t')
# print(df.shape)
# print(df.columns)


# print(df[df.duplicated(subset=['seq'])])

# print(df[df.duplicated(subset=['ID'])])

from epicast import datasets


val_dataset = datasets.SeqDataset(
    data_path= 'data/Gosai_MPRA/Gosai_MPRA_nature_0206.tsv',
    seq_column='seq',
    target_column=['K562'],
    apply_filter= True,
    filter_column= 'chr',
    pad=False)

print(val_dataset[0]['seq'].shape)
