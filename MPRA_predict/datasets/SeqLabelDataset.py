import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *
from .GenomeInterval import GenomeInterval


# MPRA_UPSTREAM  = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
# MPRA_DOWNSTREAM= 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'

class SeqLabelDataset(Dataset):
    def __init__(
        self,
        data_path = None,
        data_df = None,
        data_type = None,
        genome_path = None,

        input_column = None,
        output_column = None,

        apply_filter = True,
        filter_column = None,
        filter_in_list = None,
        filter_not_in_list = None,

        shuffle = False,
        slice_range = None,

        genome_window_size = None,
        spicify_strand = False,

        crop = False,
        crop_method = 'center',
        cropped_length = None,

        padding = False,
        padding_method = 'N',
        padded_length = None,

        # aug_rc=False,
        aug_rc_prob = 0,

        N_fill_value = 0.25,
    ) -> None:
        super().__init__()

        self.data_path = data_path
        self.data_df = data_df
        self.data_type = data_type
        self.genome_path = genome_path

        self.input_column = input_column
        self.output_column = output_column

        self.apply_filter = apply_filter
        self.filter_column = filter_column
        self.filter_in_list = filter_in_list
        self.filter_not_in_list = filter_not_in_list

        self.shuffle = shuffle
        self.slice_range = slice_range

        self.genome_window_size = genome_window_size
        self.spicify_strand = spicify_strand

        self.crop = crop
        self.crop_method = crop_method
        self.cropped_length = cropped_length

        self.padding = padding
        self.padding_method = padding_method
        self.padded_length = padded_length

        self.aug_rc_prob = aug_rc_prob
        self.N_fill_value = N_fill_value
        # self.padding_upstream = padding_upstream
        # self.padding_downstream = padding_downstream

        assert (data_path is None) != (data_df is None), "data_path和data_df必须有且只有一个不是None"
        assert data_type in ['seq', 'bed'], "data_type只能是'seq'或'bed'"

        if data_path is not None:
            sep = detect_delimiter(data_path)
            self.df = pd.read_csv(data_path, sep=sep)
        else:
            self.df = data_df

        if data_type == 'bed':
            self.genome_interval = GenomeInterval(
                genome_path=genome_path,
                window_length=genome_window_size,)

        if apply_filter is True:
            if filter_in_list is not None:
                self.df = self.df[self.df[filter_column].isin(filter_in_list)]
            if filter_not_in_list is not None:
                self.df = self.df[~self.df[filter_column].isin(filter_in_list)]
        self.df = self.df.reset_index(drop=True)

        if slice_range is not None:
            start, end = slice_range
            if 0 <= start < end <= 1:
                start = int(len(self.df) * start)
                end = int(len(self.df) * end)
            self.df = self.df.iloc[start: end].reset_index(drop=True)

        if shuffle is True:
            shuffle_index = np.random.permutation(len(self.df))
            self.df = self.df.iloc[shuffle_index].reset_index(drop=True)
        
        self.df['mid'] = (self.df['start'] + self.df['end']) // 2
        if genome_window_size is not None:
            self.df['start'] = self.df['mid'] - genome_window_size // 2
            self.df['end'] = self.df['mid'] + genome_window_size // 2

        if input_column is None:
            pass
        else:
            self.seqs = self.df[input_column].to_numpy().astype(str)
        if output_column is None:
            self.labels = None
        else:
            self.labels = self.df[output_column].to_numpy()
            self.labels = torch.tensor(self.labels, dtype=torch.float)


    def get_seq_from_genome(self, index):
        row = self.df.iloc[index]
        chr, start, end = row[['chr', 'start', 'end']]
        seq = self.genome_interval(chr, start, end)
        if self.spicify_strand and row['strand'] == '-':
            seq = rc_seq(seq)
        return seq


    def __getitem__(self, index) -> tuple:
        if self.data_type == 'bed':
            seq = self.get_seq_from_genome(index)
        elif self.data_type == 'seq':
            seq = self.seqs[index]

        if self.crop is True:
            seq = crop_seq(seq, self.cropped_length, self.crop_method)
        if self.padding is True:
            seq = pad_seq(seq, self.padded_length, self.padding_method)
        if np.random.rand() < self.aug_rc_prob:
            seq = rc_seq(seq)
        
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
