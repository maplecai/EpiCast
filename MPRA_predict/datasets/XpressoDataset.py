import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *
from pyfaidx import Fasta

# class XpressoMultiCellTypeDataset(Dataset):
#     def __init__(
#         self,
#         bed_exp_path = None,
#         genome_path = None,
#         selected_indices = None,
#         input_column = None,
#         output_columns = None,
#         filter_column = None,
#         filter_in_list = None,
#         filter_not_in_list = None,
#         padded_length = None,
#         N_fill_value = 0.25,
#         select_seq_region = None,
#         ) -> None:
#         super().__init__()

#         self.output_columns = output_columns
#         self.padded_length = padded_len
#         self.N_fill_value = N_fill_value
#         self.select_seq_region = select_seq_region
#         self.genome = Fasta(genome_path)

#         self.bed_exp_df = pd.read_csv(bed_exp_path, sep='\t', index_col=False, header=0)
#         self.bed_exp_df = self.filter_by_list(self.bed_exp_df, filter_column, filter_in_list, filter_not_in_list)
#         # if selected_indices is not None:
#         #     self.bed_exp_df = self.bed_exp_df.iloc[selected_indices]

        
        
#         for i, output_column in enumerate(output_columns):
#         self.labels = self.bed_exp_df[output_column].to_numpy()

#         # if self.seqs.dtype.kind in {'U', 'S', 'O'}:
#         #     self.seqs = self.seqs.astype(str)
#         #     self.seqs = strs2onehots(self.seqs, N_fill_value).transpose(0,2,1)
#         if self.labels.ndim == 1:
#             self.labels = self.labels.reshape(-1,1)

#         # self.seqs = torch.tensor(self.seqs, dtype=torch.float)
#         self.labels = torch.tensor(self.labels, dtype=torch.float)
#         # assert len(self.seqs) == len(self.labels)


#     def filter_by_list(self, table, filter_column, filter_in_list=None, filter_not_in_list=None):
#         if filter_column is not None:
#             if filter_in_list is not None:
#                 filtered_index = table[filter_column].isin(filter_in_list)
#                 table = table[filtered_index]
#             if filter_not_in_list is not None:
#                 filtered_index = ~table[filter_column].isin(filter_not_in_list)
#                 table = table[filtered_index]
#         return table


#     def __getitem__(self, index) -> tuple:
#         interval = self.bed_exp_df.iloc[index]
#         chr, start, end, gene_id, _, strand = interval[:6]

#         seq = self.genome[chr][start:end].seq
#         if strand == '-':
#             seq = rc_seq(seq)
        
#         if self.select_seq_region is not None:
#             start, end = self.select_seq_region
#             seq = seq[start:end]

#         seq = torch.tensor(str2onehot(seq), dtype=torch.float)
#         label = self.labels[index]
#         return seq, label
    
#     def __len__(self) -> int:
#         return len(self.labels)




class XpressoDataset(Dataset):
    def __init__(
        self,
        task_idx = None,
        bed_exp_path = None,
        genome_path = None,
        selected_indices = None,
        input_column = None,
        output_column = None,
        filter_column = None,
        filter_in_list = None,
        filter_not_in_list = None,
        padded_length = None,
        N_fill_value = 0.25,
        select_seq_region = None,
        ) -> None:
        super().__init__()

        self.task_idx = task_idx

        self.padded_length = padded_len
        self.N_fill_value = N_fill_value
        self.select_seq_region = select_seq_region

        self.genome = Fasta(genome_path)

        self.bed_exp_df = pd.read_csv(bed_exp_path, sep='\t', index_col=False, header=0)
        self.bed_exp_df = self.filter_by_list(self.bed_exp_df, filter_column, filter_in_list, filter_not_in_list)
        if selected_indices is not None:
            self.bed_exp_df = self.bed_exp_df.iloc[selected_indices]

        self.labels = self.bed_exp_df[output_column].to_numpy()

        # if self.seqs.dtype.kind in {'U', 'S', 'O'}:
        #     self.seqs = self.seqs.astype(str)
        #     self.seqs = strs2onehots(self.seqs, N_fill_value).transpose(0,2,1)
        # if self.labels.ndim == 1:
        #     self.labels = self.labels.reshape(-1,1)

        # self.seqs = torch.tensor(self.seqs, dtype=torch.float)
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        # assert len(self.seqs) == len(self.labels)


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
        interval = self.bed_exp_df.iloc[index]
        chr, start, end, gene_id, _, strand = interval[:6]

        seq = self.genome[chr][start:end].seq
        if strand == '-':
            seq = rc_seq(seq)
        
        if self.select_seq_region is not None:
            start, end = self.select_seq_region
            seq = seq[start:end]

        seq = torch.tensor(str2onehot(seq), dtype=torch.float)
        label = self.labels[index]
        if self.task_idx is not None:
            return self.task_idx, (seq, label)
        else:
            return seq, label
    
    def __len__(self) -> int:
        return len(self.labels)



if __name__ == '__main__':
    dataset = XpressoDataset(
        bed_exp_path = '/home/hxcai/cell_type_specific_CRE/data/Xpresso/merged_bed_exp.csv',
        genome_path = '/home/hxcai/genome/hg38.fa',
        output_column = 'E118',
        select_seq_region = [9000, 10000],
    )
    print(dataset[0])
    print(dataset[0][0].shape)
    
