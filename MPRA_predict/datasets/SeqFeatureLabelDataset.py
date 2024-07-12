import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *


MPRA_UPSTREAM  = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
MPRA_DOWNSTREAM= 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'

class SeqFeatureLabelDataset(Dataset):
    def __init__(
        self,
        data_path = None,
        seq_column = None,
        feature_column = None,
        label_column = None,
        cell_types = None,

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
        self.feature_column = feature_column
        self.label_column = label_column
        self.cell_types = cell_types

        self.shuffle = shuffle
        self.subset_range = subset_range
        self.apply_filter = apply_filter
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
        
        if cell_types is not None:
            self.df = self.df[self.df['cell_type'].isin(cell_types)]

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
        self.features = torch.tensor(np.array(self.df[feature_column]), dtype=torch.float)
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

        feature = self.features[index]
        label = self.labels[index]
        input = {'seq': seq, 'feature': feature}
        return input, label
    
    def __len__(self) -> int:
        return len(self.df)



if __name__ == '__main__':
    dataset = SeqFeatureLabelDataset(
        data_path='/home/hxcai/cell_type_specific_CRE/MPRA_exp/pretrained_based_models/data/Sei_Siraj_features_concat.csv',
        seq_column='seq',
        feature_column=['DNase', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'H3K27me3', 'H3K27ac', 'H3K36me3', 'CTCF'],
        exp_column='exp',
        )
    print(dataset.df.shape)
    print(dataset[0][0]['seq'].shape)
    print(dataset[0][0]['feature'].shape)
    print(dataset[0][1].shape)
