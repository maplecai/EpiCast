import torch
import numpy as np
import pandas as pd
from ..utils import *
from .BaseDataset import BaseDataset



class SeqDataset(BaseDataset):
    def __init__(
        self,
        input_column=None,
        output_column=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)


        if input_column:
            self.seqs = self.df[input_column].to_numpy().astype(str)
        if output_column:
            self.labels = self.df[output_column].to_numpy()
            self.labels = torch.tensor(self.labels, dtype=torch.float)
        else:
            self.labels = None


    def __getitem__(self, index) -> dict:
        seq = self.seqs[index]

        if self.crop:
            seq = crop_seq(seq, self.cropped_length, self.crop_method)
        if self.padding:
            seq = pad_seq(seq, self.padded_length, self.padding_method)
        # if np.random.rand() < self.aug_rc_prob:
        #     seq = rc_seq(seq)
            
        seq = torch.tensor(str2onehot(seq, N_fill_value=self.N_fill_value), dtype=torch.float)

        if self.labels is None:
            return {'seq': seq}
        else:
            label = self.labels[index]
            return {'seq': seq, 'label': label}




if __name__ == '__main__':
    dataset = SeqDataset(
        data_path='/home/hxcai/cell_type_specific_CRE/MPRA_predict/predict_short_sequence_features/data/enformer_sequences_test_100.csv',
        input_column='seq',
        crop=True,
        cropped_length=200,
        )
    print(dataset[0]['seq'].shape)
