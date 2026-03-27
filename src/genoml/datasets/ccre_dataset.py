import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils import GenomicInterval, rc_seq, seq2onehot, crop_seq, pad_seq, detect_delimiter


class cCREDataset(Dataset):
    def __init__(
        self,

        data_path=None,
        target_path=None,

        genome_path=None,
        window_length=None,

        # apply_filter=True,
        # filter_column=None,
        # filter_in_list=None,
        # filter_not_in_list=None,

        # shuffle=False,
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
    ) -> None:
        super().__init__()

        self.data_path = data_path
        self.target_path = target_path

        self.genome_path = genome_path
        self.window_length = window_length

        # self.apply_filter = apply_filter
        # self.filter_column = filter_column
        # self.filter_in_list = filter_in_list
        # self.filter_not_in_list = filter_not_in_list

        # self.shuffle = shuffle
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

        self.genome_interval = GenomicInterval(genome_path)

        self.df = pd.read_csv(data_path, sep=detect_delimiter(data_path))
            
        if target_path is not None:
            self.targets = torch.tensor(np.load(target_path), dtype=torch.float)
            assert len(self.targets) == len(self.df), "Targets and dataframe must have same length"
            if len(self.targets.shape) == 3:
                self.targets = self.targets.reshape(len(self.targets), -1)
        else:
            self.targets = None


        if slice_range is not None:
            start, end = slice_range
            if end <= 1.0:
                start = int(len(self.df) * start)
                end = int(len(self.df) * end)
            self.df = self.df.iloc[start:end]
            self.df = self.df.reset_index(drop=True)
            if self.targets is not None:
                self.targets = self.targets[start:end]


    def get_seq_from_genome(self, index):
        row = self.df.iloc[index]
        chr, start, end = row[['chr', 'start', 'end']]

        if self.shift:
            start += self.shift_size
            end += self.shift_size

        # adjust start and end to window_length
        if (self.window_length is not None):
            mid = (start + end) // 2
            start = mid - self.window_length // 2
            end = start + self.window_length

        seq = self.genome_interval.get(chr, start, end)
        return seq

        # # shift augmentation
        # if self.random_shift:
        #     min_shift, max_shift = self.random_shift_range
        #     shift = np.random.randint(min_shift, max_shift + 1)
        #     start += shift
        #     end += shift

        # # extract sequence
        # seq = self.genome_interval.get(chr, start, end)

        # # reverse complement augmentation
        # if self.random_rc:
        #     if np.random.rand() < self.random_rc_prob:
        #         seq = rc_seq(seq)
        
        # return seq


    def __len__(self) -> int:
        return len(self.df)


    def __getitem__(self, index) -> dict:

        seq = self.get_seq_from_genome(index)

        if self.crop:
            seq = crop_seq(seq, self.cropped_len, crop_position=self.crop_position)
        if self.pad:
            seq = pad_seq(seq, self.padded_len, pad_mode=self.pad_mode)
        # if self.shift:
        #     if self.shift_size < 0: # ABCD -> BCDN
        #         seq = seq[-self.shift_size:] + 'N' * (-self.shift_size)
        #     elif self.shift_size > 0:
        #         seq = 'N' * self.shift_size + seq[:-self.shift_size]
        if self.rc:
            seq = rc_seq(seq)

        seq = torch.tensor(seq2onehot(seq), dtype=torch.float)

        if self.targets is not None:
            target = self.targets[index]
            return {'seq': seq, 'target': target}
        else:
            return {'seq': seq}



if __name__ == '__main__':
    dataset = cCREDataset(
        data_path = './data/ccre/elements.tsv',
        target_path = './data/ccre/targets_sorted.npy', 
        genome_path = '../genome/hg38.fa',
        window_length = 350,
        )
    print(dataset[0]['seq'].shape)
    print(dataset[0]['target'].shape)
