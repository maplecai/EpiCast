import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# sys.path.append('..')
# sys.path.append('../..')
# print(sys.path)
from ..utils import *

class LentiMPRADataset(Dataset):
    def __init__(
            self, 
            # seqs: torch.Tensor=None,
            # labels: torch.Tensor=None,
            data_dir: str=None, 
            stage: str='train',
            reverse_complement_augmentation: bool=False,
            delete_adapter: bool=False,
            # label_list: list = ['HepG2_mean', 'K562_mean', 'SKNSH_mean'],
            # padded_length: int=None,
            # length_filter: bool=True, 
            # variance_filter: bool=False,
            ) -> None:
        super().__init__()

        df = pd.read_csv(data_dir, sep=',')

        chromatin_filter_dict = {
            'train' : ~ df['chr.hg38'].isin(['chr9', 'chr21', 'chrX', 'chr7', 'chr13']),
            'valid' : df['chr.hg38'].isin(['chr9', 'chr21', 'chrX']),
            'test'  : df['chr.hg38'].isin(['chr7', 'chr13']),
            'all'   : pd.Series([True] * len(df))
        }
        df = df[chromatin_filter_dict[stage]]

        if delete_adapter == True:
            df['sequence'] = df["230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)"].str[15:-15]
        else:
            df['sequence'] = df["230nt sequence (15nt 5' adaptor - 200nt element - 15nt 3' adaptor)"]

        seqs = df['sequence'].to_numpy()
        labels = df['mean'].to_numpy()

        seqs = strs2onehots(seqs).transpose(0,2,1)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        self.seqs = torch.tensor(seqs, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)
        assert len(seqs) == len(labels), 'seqs length and labels length have to be same'

        self.reverse_complement_augmentation = (reverse_complement_augmentation and (stage == 'train'))


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
    dataset = LentiMPRADataset(data_dir='data/lentiMPRA/HepG2_table.csv')
    seq, label = dataset[0]
    print(seq.shape, label.shape)
