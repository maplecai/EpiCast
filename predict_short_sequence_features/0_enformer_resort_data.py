import sys
sys.path.append('..')
from MPRA_predict.utils import *
from MPRA_predict.datasets import *
from MPRA_predict.datasets import GenomeInterval
from pyfaidx import Fasta

# tfr文件和sequence.bed文件顺序不同，需要手动调整顺序



# # 从基因组上截取对应的 131072bp 序列
# genome = Fasta('../../../genome/hg38.fa')

# bed_df = pd.read_csv('data/enformer_sequences.csv')
# print(bed_df.shape)

# seqs = []
# for row in tqdm(bed_df.itertuples(), total=len(bed_df)):
#     seq = genome[row.chr][row.start:row.end].seq.upper()
#     seqs.append(seq)

# count = 0
# for seq in seqs:
#     if seq == '':
#         count += 1
# print(f'{count} empty sequences')

# bed_df['seq'] = seqs
# bed_df.to_csv('data/enformer_sequences_131072.csv', index=None)








# # 从基因组上截取对应的 196608bp 序列
# genome = GenomeInterval('../../../genome/hg38.fa')

# bed_df = pd.read_csv('data/enformer_sequences.csv')
# print(bed_df.shape)

# bed_df['start'] = bed_df['start'] - 32768
# bed_df['end'] = bed_df['end'] + 32768

# seqs = []
# for row in tqdm(bed_df.itertuples(), total=len(bed_df)):
#     # seq = genome[row.chr][row.start:row.end].seq.upper()
#     seq = genome(row.chr, row.start, row.end)
#     seqs.append(seq)
#     if 'N' in seq:
#         print(row)

# bed_df['seq'] = seqs
# bed_df.to_csv('data/enformer_sequences_196608.csv', index=None)









# bed_df = pd.read_csv('data/enformer_sequences_131072.csv')
# print(bed_df.shape)

# df = bed_df[bed_df['split'] == 'test'].reset_index(drop=True)
# print(df.shape)

# seqs = load_txt('data/enformer_tfr_sequences_test.txt')
# print(len(seqs))

# dic = {seq : i for i, seq in enumerate(seqs)}
# indices = np.array([dic[row.seq] for row in df.itertuples()])
# print(indices)


# targets = np.load('data/enformer_tfr_targets_test.npy')
# targets = targets[indices]
# np.save('data/enformer_targets_test.npy', targets)





bed_df = pd.read_csv('data/enformer_sequences_196608.csv')
print(bed_df.shape)

df = bed_df[bed_df['split'] == 'test'].reset_index(drop=True)
print(df.shape)
df.to_csv('data/enformer_sequences_test.csv')

df = df[:100]
df.to_csv('data/enformer_sequences_test_100.csv')



targets = np.load('data/enformer_targets_test.npy')
targets = targets[:100]
np.save('data/enformer_targets_test_100.npy', targets)
