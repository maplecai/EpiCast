import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils import GenomicInterval, rc_seq, seq2onehot, crop_seq, pad_seq, detect_delimiter


class BedDataset(Dataset):
    def __init__(
        self,

        data_path=None,
        data_df=None,

        apply_filter=True,
        filter_column=None,
        filter_in_list=None,
        filter_not_in_list=None,

        shuffle=False,
        slice_range=None,

        crop=False,
        crop_position='center',
        cropped_len=None,
        
        pad=False,
        pad_mode='N',
        padded_len=None,

        shift=False,
        shift_size=0,

        rc=False,
        
        N_fill_value=0.25,
        augmentations=[],

        ###
        genome_path=None,
        spicify_strand=False,

        random_rc=False,
        random_rc_prob=0.5,
        random_shift=False,
        random_shift_range=(0, 0),
        ###
    ) -> None:
        super().__init__()

        self.data_path = data_path
        self.data_df = data_df

        self.apply_filter = apply_filter
        self.filter_column = filter_column
        self.filter_in_list = filter_in_list
        self.filter_not_in_list = filter_not_in_list

        self.shuffle = shuffle
        self.slice_range = slice_range

        self.crop = crop
        self.crop_position = crop_position
        self.cropped_len = cropped_len

        self.pad = pad
        self.pad_mode = pad_mode
        self.padded_len = padded_len

        self.shift = shift
        self.shift_size = shift_size

        self.rc = rc

        self.N_fill_value = N_fill_value
        self.augmentations = augmentations

        self.genome_path = genome_path
        self.spicify_strand = spicify_strand
        
        self.random_rc = random_rc
        self.random_rc_prob = random_rc_prob
        self.random_shift = random_shift
        self.random_shift_range = random_shift_range

        if data_path is not None and data_df is None:
            self.df = pd.read_csv(data_path, sep=detect_delimiter(data_path))
        elif data_path is None and data_df is not None:
            self.df = data_df
        else:
            raise ValueError("data_path or data_df must be provided.")

        if apply_filter:
            if filter_in_list is not None:
                self.df = self.df[self.df[filter_column].isin(filter_in_list)]
            if filter_not_in_list is not None:
                self.df = self.df[~self.df[filter_column].isin(filter_not_in_list)]
        self.df = self.df.reset_index(drop=True)

        if slice_range is not None:
            start, end = slice_range
            self.df = self.df.iloc[start:end].reset_index(drop=True)

        if shuffle:
            shuffle_index = np.random.permutation(len(self.df))
            self.df = self.df.iloc[shuffle_index].reset_index(drop=True)

        self.seqs = None
        self.labels = None

        self.genome_interval = GenomicInterval(genome_path)


    def get_seq_from_genome(self, index):
        row = self.df.iloc[index]
        chr, start, end = row[['chr', 'start', 'end']]

        # if self.shift: # 感觉不太对，shift的应该是这条序列在输入序列位置，而不是序列内容
        #     start += self.shift_size
        #     end += self.shift_size

        # shift augmentation
        if self.random_shift:
            min_shift, max_shift = self.random_shift_range
            shift = np.random.randint(min_shift, max_shift + 1)
            start += shift
            end += shift

        # extract sequence
        seq = self.genome_interval.get(chr, start, end)

        # reverse strand
        if self.spicify_strand and row['strand'] == '-':
            seq = rc_seq(seq)

        # reverse complement augmentation
        if self.random_rc:
            if np.random.rand() < self.random_rc_prob:
                seq = rc_seq(seq)
        
        return seq


    def __len__(self) -> int:
        return len(self.df)


    def __getitem__(self, index) -> dict:
        seq = self.get_seq_from_genome(index)

        if self.crop:
            seq = crop_seq(seq, self.cropped_len, self.crop_position)
        if self.pad:
            seq = pad_seq(seq, self.padded_len, self.pad_mode)
        if self.shift:
            if self.shift_size < 0: # ABCD -> BCDN
                seq = seq[-self.shift_size:] + 'N' * (-self.shift_size)
            elif self.shift_size > 0:
                seq = 'N' * self.shift_size + seq[:-self.shift_size]
        if self.rc:
            seq = rc_seq(seq)

        seq = torch.tensor(seq2onehot(seq, N_fill_value=self.N_fill_value), dtype=torch.float)

        if self.labels is None:
            return {'seq': seq}
        else:
            label = self.labels[index]
            return {'seq': seq, 'label': label}





# if __name__ == '__main__':
#     from pathlib import Path
#     BASE_DIR = Path(__file__).resolve().parent

#     dataset = BedDataset(
#         data_path= BASE_DIR/'../predict_short_sequence_features/data/enformer_sequences_test_100.csv',
#         genome_path='/home/hxcai/genome/hg38.fa',
#         window_length=200,
#         )
#     print(dataset[0]['seq'].shape)
