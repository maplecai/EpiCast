import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *


class SeqLabelMultiCellTypeDataset(Dataset):
    def __init__(
        self,
        seq_exp_path = None,
        input_column = None,
        output_columns = None,
        filter_column = None,
        filter_in_list = None,
        filter_not_in_list = None,
        padded_len = None,
        N_fill_value = 0.25,
        ) -> None:
        super().__init__()

        self.cell_type_names = output_columns
        self.padded_len = padded_len
        self.N_fill_value = N_fill_value

        self.seq_exp_df = pd.read_csv(seq_exp_path, sep=',')
        self.seq_exp_df = self.filter_by_list(self.seq_exp_df, filter_column, filter_in_list, filter_not_in_list)
        # if input_column is not None:
        #     self.seqs = self.seq_exp_df[input_column].to_numpy()
        # if output_column is not None:
        #     self.labels = self.seq_exp_df[output_column].to_numpy()
        # cell_type_dic = {cell_type: i for i, cell_type in enumerate(self.cell_types)}

        self.cell_types = []
        self.seqs = []
        self.labels = []
        for i, output_column in enumerate(output_columns):
            self.cell_types.append(np.full(shape=len(self.seq_exp_df), fill_value=i))
            self.seqs.append(self.seq_exp_df[input_column].to_numpy())
            self.labels.append(self.seq_exp_df[output_column].to_numpy())
        self.cell_types = np.concatenate(self.cell_types)
        self.seqs = np.concatenate(self.seqs, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        if not hasattr(self, 'seqs'):
            raise ValueError('must assign seqs by seqs or seqs_path or table_path + input_column')
        if not hasattr(self, 'labels'):
            print('not assigned labels, initial with zeros')
            self.labels = np.zeros(shape=len(self.seqs))

        # if selected_indices is not None:
        #     self.seqs = self.seqs[selected_indices]
        #     self.labels = self.labels[selected_indices]
        
        # if self.cell_types.ndim == 1:
        #     self.cell_types = self.cell_types.reshape(-1,1)
        if self.seqs.dtype.kind in {'U', 'S', 'O'}:
            self.seqs = self.seqs.astype(str)
            self.seqs = strs2onehots(self.seqs, N_fill_value)
        if self.labels.ndim == 1:
            self.labels = self.labels.reshape(-1,1)

        self.cell_types = torch.tensor(self.cell_types, dtype=torch.long)
        self.seqs = torch.tensor(self.seqs, dtype=torch.float)
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        assert len(self.cell_types) == len(self.seqs) == len(self.labels)


    def filter_by_list(self, table, filter_column, filter_in_list=None, filter_not_in_list=None):
        if filter_column is not None:
            if filter_in_list is not None:
                filtered_index = table[filter_column].isin(filter_in_list)
                table = table[filtered_index]
            if filter_not_in_list is not None:
                filtered_index = ~table[filter_column].isin(filter_not_in_list)
                table = table[filtered_index]
        return table


    def __getitem__(self, index) -> tuple:
        cell_type = self.cell_types[index]
        seq = self.seqs[index]
        if self.padded_len is not None:
            seq = pad_onehot_N(seq, self.padded_len, self.N_fill_value)
        # seq = torch.tensor(str2onehot(seq), dtype=torch.float)
        label = self.labels[index]
        return cell_type, seq, label


    def __len__(self) -> int:
        return len(self.labels)





if __name__ == '__main__':
    dataset = SeqLabelMultiCellTypeDataset(
        seq_exp_path = '/home/hxcai/cell_type_specific_CRE/data/GosaiMPRA/GosaiMPRA_len200_100000.csv',
        input_column = 'seq',
        output_columns = ['K562', 'HepG2'])
    
    print(dataset[0])
    print(dataset[0][0].shape)