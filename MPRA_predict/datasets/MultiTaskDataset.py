import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *
import bisect


class MultiTaskDataset(Dataset):
    def __init__(self, datasets, task_idx_list=None):
        
        self.task_idx_list = task_idx_list
        # if task_encoding_table is None:
        #     self.task_encoding_table = np.arange(len(datasets))
        # else:
        #     self.task_encoding_table = np.array(task_encoding_table)

        self.datasets = list(datasets)
        self.sizes = [len(d) for d in datasets]
        self.cum_sizes = np.cumsum(self.sizes)
        
    def __len__(self):
        return self.cum_sizes[-1]
    
    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cum_sizes, idx)

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_sizes[dataset_idx - 1]

        if self.task_idx_list is not None:
            task_idx = self.task_idx_list[dataset_idx]
        else:
            task_idx = dataset_idx

        dataset_idx = torch.tensor(dataset_idx, dtype=torch.long)
        data = self.datasets[dataset_idx][sample_idx]

        # task_encoding = self.task_encoding_table[dataset_idx]
        # task_encoding = torch.tensor(task_encoding, dtype=torch.long)

        # 返回数据和任务标识符
        return task_idx, data
