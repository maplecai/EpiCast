import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..utils import seq2onehot, detect_delimiter, pad_seq, crop_seq

class SeqEpiDataset(Dataset):
    def __init__(
        self,
        seq_file_path=None,
        epi_file_path=None,

        apply_filter=False,
        filter_column=None,
        filter_in_list=None,
        filter_not_in_list=None,

        shuffle=False,
        slice_range=None,

        crop=False,
        crop_position='center',
        cropped_len=None,
        
        pad=False,
        pad_position='both_sides',
        pad_mode='N',
        padded_len=None,
        genome=None,
        pad_left_seq=None,
        pad_right_seq=None,

        N_fill_value=0.25,
        augmentations=[],

        ###
        seq_column='seq',
        target_column=None,
        
        cell_types=None,
        assays=None,
        ###

    ) -> None:
        
        super().__init__()
        
        self.seq_file_path = seq_file_path
        self.epi_file_path = epi_file_path

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
        self.pad_position = pad_position
        self.pad_mode = pad_mode
        self.padded_len = padded_len
        self.genome = genome
        self.pad_left_seq = pad_left_seq
        self.pad_right_seq = pad_right_seq

        self.N_fill_value = N_fill_value
        self.augmentations = augmentations

        self.seq_column = seq_column
        self.target_column = target_column

        self.cell_types = cell_types
        self.assays = assays
        
        self.seq_df = pd.read_csv(seq_file_path, sep=detect_delimiter(seq_file_path))
        self.epi_df = pd.read_csv(epi_file_path, sep=detect_delimiter(epi_file_path))

        self.df = pd.concat([self.seq_df, self.epi_df], axis=1)

        # filter data by filter_column
        if apply_filter:
            if filter_in_list is not None:
                self.df = self.df[self.df[filter_column].isin(filter_in_list)]
            if filter_not_in_list is not None:
                self.df = self.df[~self.df[filter_column].isin(filter_not_in_list)]
        self.df = self.df.reset_index(drop=True)

        if shuffle:
            self.df = self.df.sample(frac=1, random_state=42)

        if slice_range is not None:
            start, end = slice_range
            if 0 <= start < end <= 1:
                start = int(len(self.df) * start)
                end = int(len(self.df) * end)
            self.df = self.df.iloc[start:end].reset_index(drop=True)

        # set seqs, features, targets
        
        self.seqs = self.df[seq_column].to_numpy().astype(str)

        cols = [f"{cell_type}_{assay}" for cell_type in cell_types for assay in assays]
        data = self.df[cols].to_numpy().reshape(len(self.df), len(cell_types), len(assays))
        self.features = torch.from_numpy(data).float()

        if target_column is not None:
            self.targets = torch.from_numpy(self.df[target_column].to_numpy()).float()
        else:
            self.targets = None



    def __len__(self) -> int:
        return len(self.df)


    def __getitem__(self, index) -> dict:
        sample = {}
        sample['idx'] = index
        
        if self.seqs is not None:
            seq = self.seqs[index]
            if self.crop:
                seq = crop_seq(seq, self.cropped_len, self.crop_position)
            if self.pad:
                seq = pad_seq(
                    seq, self.padded_len, pad_position=self.pad_position, pad_mode=self.pad_mode, 
                    genome=self.genome, given_left_seq=self.pad_right_seq, given_right_seq=self.pad_right_seq)
            seq = torch.tensor(seq2onehot(seq, N_fill_value=self.N_fill_value), dtype=torch.float)
            sample['seq'] = seq

        if self.features is not None:
            feature = self.features[index]
            sample['feature'] = feature

        if self.targets is not None:
            target = self.targets[index]
            sample['target'] = target

        return sample



if __name__ == '__main__':
    dataset = SeqEpiDataset(
        seq_file_path = './data/Gosai_MPRA/Gosai_MPRA_760679.tsv', 
        epi_file_path = './data/Gosai_MPRA/Gosai_MPRA_AG_VEF_scalelog1p.tsv',
        seq_column = 'seq',
        target_column = ['K562', 'HepG2'],
        cell_types = ['K562', 'HepG2'],
        assays = ['DNase', 'H3K4me3'],
    )

    print(len(dataset))
    print(dataset[0])
    