import pandas as pd
from genoml import utils

mpra_df = pd.read_csv("data/gosai_mpra/gosai_mpra_760679_zscore.tsv", sep='\t')
mpra_df['rc'] = 0


mpra_rc_df = mpra_df.copy()
mpra_rc_df['seq'] = mpra_df['seq'].map(lambda x: utils.rc_seq(x))
mpra_rc_df['rc'] = 1

concat_df = pd.concat([mpra_df, mpra_rc_df], ignore_index=True)
concat_df.to_csv('data/gosai_mpra/gosai_mpra_760679_zscore_with_rc.tsv', sep='\t', index=False)


# MPRA_UPSTREAM =   'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
# MPRA_DOWNSTREAM = 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'

# mpra_df = pd.read_csv('data/gosai_mpra/gosai_mpra_760679_zscore.tsv', sep='\t')
# print(mpra_df['seq'].str.len().value_counts())
# mpra_df['seq'] = mpra_df['seq'].map(lambda x: utils.pad_seq(x, padded_len=600, pad_mode='given', left_pad_seq=MPRA_UPSTREAM, right_pad_seq=MPRA_DOWNSTREAM))
# print(mpra_df['seq'].str.len().value_counts())
# mpra_df.to_csv('data/gosai_mpra/gosai_mpra_760679_zscore_600bp.tsv', sep='\t', index=False)
