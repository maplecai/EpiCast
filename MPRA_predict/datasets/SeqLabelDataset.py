import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *


# MPRA_UPSTREAM  = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
# MPRA_DOWNSTREAM= 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'

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

        padding = False,
        padding_method = 'N',
        padded_length = None,
        N_fill_value = 0.25,
        # padding_upstream = MPRA_UPSTREAM,
        # padding_downstream = MPRA_DOWNSTREAM,
        ) -> None:
        super().__init__()

        self.data_path = data_path
        self.seq_column = seq_column
        self.label_column = label_column

        self.shuffle = shuffle
        self.subset_range = subset_range
        self.apply_filter = apply_filter
        self.filter_column = filter_column
        self.filter_in_list = filter_in_list
        self.filter_not_in_list = filter_not_in_list

        self.padding = padding
        self.padding_method = padding_method
        self.padded_length = padded_length
        self.N_fill_value = N_fill_value
        # self.padding_upstream = padding_upstream
        # self.padding_downstream = padding_downstream
        
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

        if label_column is None:
            self.labels = None
        else:
            self.labels = self.df[label_column].to_numpy()
            self.labels = torch.tensor(self.labels, dtype=torch.float)


    def __getitem__(self, index) -> tuple:
        seq = self.seqs[index]
        if self.padding is True:
            seq = pad_seq(seq, self.padded_length, self.padding_method)
        seq = str2onehot(seq, N_fill_value=self.N_fill_value)
        seq = torch.tensor(seq, dtype=torch.float)
        if self.labels is None:
            return {'seq': seq}
        else:
            label = self.labels[index]
            return {'seq': seq, 'label': label}
    
    def __len__(self) -> int:
        return len(self.df)



if __name__ == '__main__':
    dataset = SeqLabelDataset(
        data_path='/home/hxcai/cell_type_specific_CRE/data/SirajMPRA/SirajMPRA_total.csv',
        seq_column='seq',
        padded_length=20000,
        )
    print(dataset[0])
