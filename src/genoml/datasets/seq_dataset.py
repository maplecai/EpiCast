import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..utils import detect_delimiter, rc_seq, seq2onehot, crop_seq, pad_seq

class SeqDataset(Dataset):
    def __init__(
        self,

        data_path=None,
        data_df=None,
        data_list=None,

        seq_column=None,
        feature_column=None,
        target_column=None,

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
        left_pad_seq=None,
        right_pad_seq=None,

        N_fill_value=0.25,

        aug_rc=False,
        # drop_na=False,
        return_str=False,
    ) -> None:
        
        super().__init__()

        self.data_path = data_path
        self.data_df = data_df

        self.seq_column = seq_column
        self.feature_column = feature_column
        self.target_column = target_column

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
        self.left_pad_seq = left_pad_seq
        self.right_pad_seq = right_pad_seq

        self.N_fill_value = N_fill_value

        self.aug_rc = aug_rc
        self.return_str = return_str
        

        # read dataframe
        if data_path is not None:
            self.df = pd.read_csv(data_path, sep=detect_delimiter(data_path))
        elif data_df is not None:
            self.df = data_df
        elif data_list is not None:
            self.df = pd.DataFrame(data_list)
            self.df.columns = ['seq']
        else:
            raise ValueError("data_path or data_df must be provided.")

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
        self.seqs = None
        self.features = None
        self.targets = None

        if seq_column:
            self.seqs = self.df[seq_column].to_numpy().astype(str)
        if feature_column:
            self.features = self.df[feature_column].to_numpy()
            self.features = torch.tensor(self.features, dtype=torch.float)
        if target_column:
            self.targets = self.df[target_column].to_numpy()
            self.targets = torch.tensor(self.targets, dtype=torch.float)
        ###



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
                seq = pad_seq(seq, self.padded_len, pad_position=self.pad_position, pad_mode=self.pad_mode, 
                              genome=self.genome, left_pad_seq=self.left_pad_seq, right_pad_seq=self.right_pad_seq)

            # reverse complement augmentation
            if self.aug_rc:
                if np.random.rand() < 0.5:
                    seq = rc_seq(seq)

            if self.return_str:
                sample['seq'] = seq
            else:
                seq = torch.tensor(seq2onehot(seq, N_fill_value=self.N_fill_value), dtype=torch.float)
                sample['seq'] = seq

        if self.features is not None:
            feature = self.features[index]
            sample['feature'] = feature

        if self.targets is not None:
            target = self.targets[index]
            sample['target'] = target

        return sample




# if __name__ == '__main__':
#     dataset = SeqDataset(
#         data_path='../predict_short_sequence_features/data/enformer_sequences_test_100.csv',
#         input_column='seq',
#         crop=True,
#         cropped_len=200,
#         )
#     print(dataset[0]['seq'].shape)
