import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# sys.path.append('..')
# sys.path.append('../..')
# print(sys.path)
from ..utils import *

class BodaMPRADataset(Dataset):
    def __init__(
            self, 
            data_dir: str=None, 
            stage: str='train',
            label_list: list = ['HepG2_mean', 'K562_mean', 'SKNSH_mean'],
            padded_length: int=None,
            reverse_complement_augmentation: bool=None,
            length_filter: bool=True, 
            variance_filter: bool=False,
            ) -> None:
        super().__init__()

        df = pd.read_csv(data_dir, sep=',')

        if length_filter == True:
            df = df[df['sequence'].str.len() == 200]
        if variance_filter == True:
            df = df[(df.loc[:, ['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']].max(axis=1) < 1.0)]

        chromatin_filter_dict = {
            'train' : ~ df['chr'].isin(['9', '21', 'X', '7', '13', 9, 21, 7, 13]),
            'valid' : df['chr'].isin(['9', '21', 'X', 9, 21]),
            'test'  : df['chr'].isin(['7', '13', 7, 13]),
        }
        df = df[chromatin_filter_dict[stage]]

        seqs = df['sequence'].to_numpy()
        labels = df[label_list].to_numpy()

        if padded_length is not None:
            seqs = np.array([pad_sequence(seq, padded_length, MPRA_UPSTREAM, MPRA_DOWNSTREAM) for seq in seqs])

        seqs = strs2onehots(seqs).transpose(0,2,1)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        self.seqs = torch.tensor(seqs, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)
        assert len(seqs) == len(labels), 'seq length and label length do not match'

        if reverse_complement_augmentation is not None:
            self.reverse_complement_augmentation = reverse_complement_augmentation
        else:
            self.reverse_complement_augmentation = (stage == 'train')


    def __getitem__(self, index) -> tuple:
        seq = self.seqs[index]
        label = self.labels[index]
        if self.reverse_complement_augmentation:
            if torch.rand(1) < 0.5:
                seq = rc_onehots(seq)
        return seq, label
    
    def __len__(self) -> int:
        return len(self.labels)


if __name__ == '__main__':
    dataset = BodaMPRADataset(data_dir='data/Malinois/filtered_MPRA_data.csv', padded_length=600)
    seq, label = dataset[0]
    print(seq.shape, label.shape)
