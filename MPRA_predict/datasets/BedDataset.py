import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *
from pyfaidx import Fasta
from .GenomeInterval import GenomeInterval


class BedDataset(Dataset):
    def __init__(
        self,
        data_path = None,
        data_df = None,
        label_column = None,
        genome_path = None,

        shuffle = False,
        subset_range = None,

        apply_filter = True,
        filter_column = None,
        filter_in_list = None,
        filter_not_in_list = None,

        genome_window_size = None,
        spicify_strand = False,

        crop = False,
        crop_method = 'center',
        cropped_length = None,

        padding = False,
        padding_method = 'N',
        padded_length = None,

        # aug_rc=False,

        N_fill_value = 0.25,
        return_aug_info=False,
    ) -> None:
        super().__init__()

        self.bed_path = bed_path
        self.bed_df = bed_df
        self.genome_path = genome_path
        self.label_column = label_column
        self.genome_window_size = genome_window_size

        self.shuffle = shuffle
        self.subset_range = subset_range
        self.filter_column = filter_column
        self.filter_in_list = filter_in_list
        self.filter_not_in_list = filter_not_in_list
        self.use_center_pos = use_center_pos

        self.spicify_strand = spicify_strand
        self.rc_aug = rc_aug
        self.shift_aug = shift_aug
        self.shift_aug_range = shift_aug_range

        self.padding = padding
        self.padding_method = padding_method
        self.padded_length = padded_length

        self.N_fill_value = N_fill_value
        self.return_aug_info = return_aug_info

        self.seq_interval = GenomeInterval(
            genome_path=genome_path,
            window_length=genome_window_size,
            rc_aug=rc_aug,
            shift_aug=shift_aug,
            shift_aug_range=shift_aug_range,
            return_aug_info=return_aug_info,)

        assert (bed_path is None) != (bed_df is None), "bed_path和bed_df必须有且只有一个不是None"
        if bed_path is not None:
            sep = detect_delimiter(bed_path)
            self.df = pd.read_csv(bed_path, sep=sep)
        else:
            self.df = bed_df

        if shuffle:
            shuffle_index = np.random.permutation(len(self.df))
            self.df = self.df.iloc[shuffle_index].reset_index(drop=True)

        if filter_column is not None:
            self.df = filter_by_column(self.df, filter_column, filter_in_list, filter_not_in_list)

        if subset_range is not None:
            start_ratio, end_ratio = subset_range
            start_index = int(len(self.df) * start_ratio)
            end_index = int(len(self.df) * end_ratio)
            self.df = self.df.iloc[start_index:end_index]

        if use_center_pos:
            if genome_window_size is None:
                raise ValueError("window_size must be specified when use_center_pos is True.")
            self.df['start'] = self.df['pos'] - genome_window_size // 2
            self.df['end'] = self.df['pos'] + genome_window_size // 2

        if label_column is None:
            self.labels = None
        else:
            self.labels = self.df[label_column].to_numpy()
            self.labels = torch.tensor(self.labels, dtype=torch.float)

    def get_seq_from_genome(self, index):
        row = self.df.iloc[index]
        chr, start, end = row[['chr', 'start', 'end']]
        seq = self.seq_interval(chr, start, end)
        if self.spicify_strand and row['strand'] == '-':
            seq = rc_seq(seq)
        return seq


    def __getitem__(self, index) -> tuple:
        seq = self.get_seq_from_genome(index)
        if self.padding:
            seq = pad_seq(seq, self.padded_length, self.padding_method)

        seq = str2onehot(seq, N_fill_value=self.N_fill_value)
        seq = torch.tensor(seq, dtype=torch.float)

        if self.labels is None:
            return {'seq': seq}
        else:
            label = self.labels[index]
            {'seq': seq, 'label': label}

    def __len__(self) -> int:
        return len(self.df)



if __name__ == '__main__':
    dataset = BedDataset(
        bed_path = 'data/SirajMPRA/SirajMPRA_100.csv',
        genome_path = '../../genome/hg38.fa',
        output_column = 'HepG2',
        use_center_pos=True,
        genome_window_size=1000,
        padding=True,
        padded_length=10000,
    )
    print(dataset[0])
    print(dataset[0][0].shape)
    