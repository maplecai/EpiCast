import pandas as pd
from genoml import utils

mpra_df = pd.read_csv('data/Gosai_MPRA/41586_2024_8070_MOESM4_ESM.txt', sep='\t', low_memory=False)
print(mpra_df.shape)
print(mpra_df.columns)
print(mpra_df['data_project'].value_counts())
print(mpra_df['chr'].value_counts())
print(mpra_df['sequence'].str.len().value_counts())

mpra_df['chr'] = 'chr' + mpra_df['chr']
mpra_df = mpra_df[(mpra_df[['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']].max(axis=1) < 1.0)]
# mpra_df = mpra_df[(mpra_df['sequence'].str.len() == 200)]
mpra_df = mpra_df[['IDs', 'chr', 'sequence', 'K562_log2FC', 'HepG2_log2FC', 'SKNSH_log2FC']].copy()
mpra_df = mpra_df.rename(columns={
    'IDs': 'id',
    'sequence': 'seq',
    'K562_log2FC': 'K562',
    'HepG2_log2FC': 'HepG2',
    'SKNSH_log2FC': 'SK-N-SH'
})
mpra_df = mpra_df.reset_index(drop=True)
print(mpra_df.columns)
print("after filter lfcSE < 1:", mpra_df.shape)


# cell_cols = ['K562', 'HepG2', 'SK-N-SH']

# std_multiple_cut = 6.0
# up_cutoff_move   = 4.0

# means = mpra_df[cell_cols].mean().to_numpy()
# stds  = mpra_df[cell_cols].std().to_numpy()

# up_cut   = means + stds * std_multiple_cut + up_cutoff_move
# down_cut = means - stds * std_multiple_cut

# keep_up = (mpra_df[cell_cols] < up_cut).to_numpy().all(axis=1)
# # n_drop_up = (~keep_up).sum()
# mpra_df = mpra_df.loc[keep_up].copy()

# keep_down = (mpra_df[cell_cols] > down_cut).to_numpy().all(axis=1)
# # n_drop_down = (~keep_down).sum()
# mpra_df = mpra_df.loc[keep_down].copy()

# print("after filter extreme value:", mpra_df.shape)

mpra_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_nature_0206.tsv', sep='\t', index=False)


# # val
# val_filter = ['chr19', 'chr21', 'chrX']
# val_df = mpra_df[mpra_df['chr'].isin(val_filter)].copy()
# print('val df after chr filter:', val_df.shape)

MPRA_UPSTREAM =   'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
MPRA_DOWNSTREAM = 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'

mpra_df_200bp = mpra_df[mpra_df['seq'].str.len() == 200].reset_index(drop=True)
mpra_df_200bp.to_csv('data/Gosai_MPRA/Gosai_MPRA_nature_0206_200bp.tsv', sep='\t', index=False)
mpra_df_200bp['seq'] = MPRA_UPSTREAM[-200:] + mpra_df_200bp['seq'] + MPRA_DOWNSTREAM[:200]
mpra_df_200bp.to_csv('data/Gosai_MPRA/Gosai_MPRA_nature_0206_200bp_600bp.tsv', sep='\t', index=False)


mpra_df['seq'] = mpra_df['seq'].map(lambda x: utils.pad_seq(x, padded_len=600, pad_mode='given', left_pad_seq=MPRA_UPSTREAM, right_pad_seq=MPRA_DOWNSTREAM))
print(mpra_df['seq'].str.len().value_counts())
# MPRA_UPSTREAM[-200:] + mpra_df['seq'] + MPRA_DOWNSTREAM[:200]

mpra_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_nature_0206_600bp.tsv', sep='\t', index=False)





# # train
# train_filter = ['chr7', 'chr13', 'chr19', 'chr21', 'chrX']
# train_df = mpra_df[~mpra_df['chr'].isin(train_filter)].copy()
# print('train df after chr filter:', train_df.shape)

# # reverse comp
# rc_df = train_df.copy()
# rc_df['seq'] = rc_df['seq'].map(utils.rc_seq)
# train_df = pd.concat([train_df, rc_df], ignore_index=True)
# print('train df after reverse comp:', train_df.shape)

# # oversample
# cell_cols = ["K562", "HepG2", "SK-N-SH"]
# high_mask = (train_df[cell_cols] >= 0.5).any(axis=1)
# high_df = train_df[high_mask].copy()
# train_df = pd.concat([train_df, high_df], ignore_index=True)
# print('train df after oversample high seqs:', train_df.shape)

# train_df.to_csv('data/Gosai_MPRA/Gosai_MPRA_nature_0206_600bp_train.tsv', sep='\t', index=False)
