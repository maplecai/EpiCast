import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *
from pyfaidx import Fasta
from .SeqInterval import SeqInterval


class XpressoDatasetNew(Dataset):
    def __init__(
        self,
        task_idx = None,
        bed_exp_path = None,
        genome_path = None,
        selected_ratio_range = None,
        input_column = None,
        output_column = None,
        filter_column = None,
        filter_in_list = None,
        filter_not_in_list = None,
        padded_len = None,
        N_fill_value = 0.25,
        select_seq_region = None,
        load_memory=False,
        use_strand=False,
        ) -> None:
        super().__init__()

        if task_idx is not None:
            self.task_idx = task_idx

        self.select_seq_region = select_seq_region
        self.load_memory = load_memory
        self.use_strand = use_strand

        self.seq_interval = SeqInterval(
            genome_path=genome_path,
            window_length=padded_len,
            rc_aug=False,
            shift_aug=False,
            shift_aug_range=None,
            return_onehot=False,
            return_aug_info=False,
            N_fill_value=N_fill_value,
        )

        self.bed_exp_df = pd.read_csv(bed_exp_path, sep='\t', index_col=False, header=0)
        self.bed_exp_df = filter_by_column(self.bed_exp_df, filter_column, filter_in_list, filter_not_in_list)

        if selected_ratio_range is not None:
            start, end = selected_ratio_range
            start, end = int(start * len(self.bed_exp_df)), int(end * len(self.bed_exp_df))
            self.bed_exp_df = self.bed_exp_df.iloc[start: end]

        self.labels = self.bed_exp_df[output_column].to_numpy()
        self.labels = torch.tensor(self.labels, dtype=torch.float)

        if load_memory == True:
            self.seqs = []
            for i in range(len(self.bed_exp_df)):
                interval = self.bed_exp_df.iloc[i]
                chr, start, end = interval[:3]

                seq = self.seq_interval(chr, start, end)

                if self.use_strand == True:
                    strand = interval[5]
                    if strand == '-':
                        seq = seq_reverse_complement(seq)
                if self.select_seq_region is not None:
                    start, end = self.select_seq_region
                    seq = seq[start:end]
                self.seqs.append(seq)
            self.seqs = torch.tensor(strs2onehots(self.seqs), dtype=torch.float)


    def __getitem__(self, index) -> tuple:
        if self.load_memory == True:
            seq = self.seqs[index]
            label = self.labels[index]
        else:
            interval = self.bed_exp_df.iloc[index]
            chr, start, end = interval[:3]
            seq = self.seq_interval(chr, start, end)

            if self.use_strand == True:
                strand = interval[5]
                if strand == '-':
                    seq = seq_reverse_complement(seq)
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
    dataset = XpressoDatasetNew(
        task_idx=0,
        bed_exp_path = '/home/hxcai/cell_type_specific_CRE/data/Xpresso/merged_bed_exp.csv',
        genome_path = '/home/hxcai/genome/hg38.fa',
        output_column = 'HepG2',
        select_seq_region = [9000, 10000],
    )
    print(dataset[0])
    print(dataset[0][1][0].shape)
    
