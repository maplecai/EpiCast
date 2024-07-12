import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *


MPRA_UPSTREAM  = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
MPRA_DOWNSTREAM= 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'

class SeqLabelDataset(Dataset):
    def __init__(
        self,
        data_path = None,
        seq_column = None,
        label_column = None,

        shuffle = False,
        subset_range = None,
        apply_filter = True,
        filter_column = None,
        filter_in_list = None,
        filter_not_in_list = None,

        seq_pad_len = None,
        N_fill_value = 0.25,
        padding = False,
        padding_mode = 'N',
        padding_upstream = MPRA_UPSTREAM,
        padding_downstream = MPRA_DOWNSTREAM,
        ) -> None:
        super().__init__()

        self.data_path = data_path
        self.seq_column = seq_column
        self.label_column = label_column

        self.shuffle = shuffle
        self.subset_range = subset_range
        self.apply_filter = self.apply_filter
        self.filter_column = filter_column
        self.filter_in_list = filter_in_list
        self.filter_not_in_list = filter_not_in_list

        self.seq_pad_len = seq_pad_len
        self.N_fill_value = N_fill_value
        self.padding = padding
        self.padding_mode = padding_mode
        self.padding_upstream = padding_upstream
        self.padding_downstream = padding_downstream
        
        self.df = pd.read_csv(data_path)

        if apply_filter is True:
            if filter_in_list is not None:
                self.df = self.df[self.df[filter_column].isin(filter_in_list)]
            if filter_not_in_list is not None:
                self.df = self.df[~self.df[filter_column].isin(filter_in_list)]

        self.df = self.df.reset_index(drop=True)

        if subset_range is not None:
            start_ratio, end_ratio = subset_range
            start_index = int(len(self.df) * start_ratio)
            end_index = int(len(self.df) * end_ratio)
            self.df = self.df.iloc[start_index: end_index].reset_index(drop=True)

        if shuffle is True:
            shuffle_index = np.random.permutation(len(self.df))
            self.df = self.df.iloc[shuffle_index].reset_index(drop=True)

        self.seqs = self.df[seq_column].to_numpy().astype(str)
        self.labels = torch.tensor(np.array(self.df[label_column]), dtype=torch.float)

        if len(self.labels) == 0:
            print('empty labels, set labels to zeros')
            self.labels = np.zeros(shape=len(self.seqs))

        # if self.seqs.dtype.kind in {'U', 'S', 'O'}:
        #     self.seqs = self.seqs.astype(str)
        #     # if self.MPRA_pad:
        #     #     self.seqs = np.array([MPRA_UPSTREAM + seq + MPRA_DOWNSTREAM for seq in self.seqs])
        #     self.seqs = strs2onehots(self.seqs, N_fill_value)

    def __getitem__(self, index) -> tuple:
        seq = self.seqs[index]
        if self.padding is True:
            if self.padding_mode == 'N':
                seq = pad_seq(seq, self.seq_pad_len)
            elif self.padding_mode == 'given':
                seq = pad_seq(seq, self.seq_pad_len, MPRA_UPSTREAM, MPRA_DOWNSTREAM)
        seq = str2onehot(seq)
        seq = torch.tensor(seq, dtype=torch.float)

        label = self.labels[index]
        input = {'seq': seq}
        return input, label
    
    def __len__(self) -> int:
        return len(self.df)




if __name__ == '__main__':
    dataset = SeqLabelDataset(
        data_path='/home/hxcai/cell_type_specific_CRE/data/GosaiMPRA/MPRA_data_len200_small.csv',
        seq_column='nt_sequence',
        seq_pad_len=210,
        )
    print(dataset[0])
    

# class SeqLabelDataset(Dataset):
#     def __init__(
#         self,
#         task_idx = None,
#         seq_exp_path = None,

#         input_column = None,
#         output_column = None,

#         shuffle = False,
#         subset_range = None,

#         filter_column = None,
#         filter_in_list = None,
#         filter_not_in_list = None,

#         seq_pad_len = None,
#         N_fill_value = 0.25,
#         pad_content = 'N',
#         ) -> None:
#         super().__init__()

#         self.task_idx = task_idx
#         self.seq_exp_path = seq_exp_path
#         self.input_column = input_column
#         self.output_column = output_column
#         self.shuffle = shuffle
#         self.subset_range = subset_range
#         self.filter_column = filter_column
#         self.filter_in_list = filter_in_list
#         self.filter_not_in_list = filter_not_in_list
#         self.seq_pad_len = seq_pad_len
#         self.N_fill_value = N_fill_value
#         self.pad_content = pad_content
        
#         if seq_exp_path is not None:
#             self.df = pd.read_csv(seq_exp_path)
        
#         if filter_column is not None:
#             self.df = self.filter_by_list(self.df, filter_column, filter_in_list, filter_not_in_list)

#         if shuffle:
#             self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

#         if subset_range is not None:
#             start_ratio, end_ratio = subset_range
#             start_index = int(len(self.df) * start_ratio)
#             end_index = int(len(self.df) * end_ratio)
#             self.df = self.df.iloc[start_index: end_index]

#         if input_column is not None:
#             self.seqs = self.df[input_column].to_numpy()
#         if output_column is not None:
#             self.labels = self.df[output_column].to_numpy()

#         if not hasattr(self, 'seqs'):
#             raise ValueError('must assign seqs by seqs or seqs_path or table_path + input_column')
#         if not hasattr(self, 'labels'):
#             self.labels = np.zeros(shape=len(self.seqs))

#         self.seqs = self.seqs.astype(str)

#         # if self.seqs.dtype.kind in {'U', 'S', 'O'}:
#         #     self.seqs = self.seqs.astype(str)
#         #     # if self.MPRA_pad:
#         #     #     self.seqs = np.array([MPRA_UPSTREAM + seq + MPRA_DOWNSTREAM for seq in self.seqs])
#         #     self.seqs = strs2onehots(self.seqs, N_fill_value)

#         # self.seqs = torch.tensor(self.seqs, dtype=torch.float)
        
#         self.labels = torch.tensor(self.labels, dtype=torch.float)
#         assert len(self.seqs) == len(self.labels)


#     def filter_by_list(self, table, filter_column, filter_in_list=None, filter_not_in_list=None):
#         if filter_column is not None:
#             if filter_in_list is not None:
#                 filtered_index = table[filter_column].isin(filter_in_list)
#                 table = table[filtered_index]
#             if filter_not_in_list is not None:
#                 filtered_index = ~table[filter_column].isin(filter_not_in_list)
#                 table = table[filtered_index]
#         return table


#     def __getitem__(self, index) -> tuple:
#         seq = self.seqs[index]
#         if self.seq_pad_len is not None:
#             if self.pad_content == 'N':
#                 seq = pad_seq(seq, self.seq_pad_len)
#             elif self.pad_content == 'given':
#                 seq = pad_seq(seq, self.seq_pad_len, MPRA_UPSTREAM, MPRA_DOWNSTREAM)

#         seq = str2onehot(seq)
#         seq = torch.tensor(seq, dtype=torch.float)
#         label = self.labels[index]

#         if self.task_idx is not None:
#             return self.task_idx, (seq, label)
#         else:
#             return seq, label
    
#     def __len__(self) -> int:
#         return len(self.labels)

