import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *


class SeqDataset(Dataset):
    def __init__(
            self,
            seqs=None, 
            labels=None, 
            seqs_dir: str=None, 
            labels_dir: str=None,
            transforms=None,
            reverse_complement_augmentation: bool=False,
            ) -> None:
        super().__init__()
        self.reverse_complement_augmentation = reverse_complement_augmentation
        # first consider seqs, then seqs_dir
        if seqs is None:
            if seqs_dir is None:
                raise ValueError('seqs and seqs_dir cannot be None at the same time')
            else:
                seqs = np.loadtxt(seqs_dir, dtype=str)
        if labels is None:
            if labels_dir is None:
                labels = np.zeros(len(seqs))
            else:
                labels = np.loadtxt(labels_dir, dtype=float)
        if seqs.dtype.kind in {'U', 'S'}:
            seqs = strs2onehots(seqs).transpose(0,2,1)
        if labels.ndim == 1:
            labels = labels.reshape(-1,1)
        self.seqs = torch.tensor(seqs, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)
        assert len(self.seqs) == len(self.labels)

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
    pass
    
