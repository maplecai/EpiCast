import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ..utils import *
from pyfaidx import Fasta
from .SeqInterval import SeqInterval


class BedDataset(Dataset):
    def __init__(
        self,
        task_idx = None,
        bed_exp_path = None,
        genome_path = None,
        # input_column = None,
        output_column = None,

        shuffle = False,
        subset_range = None,
        load_memory=False,

        filter_column = None,
        filter_in_list = None,
        filter_not_in_list = None,

        use_pos=False,
        select_seq_range=None,

        padded_len = None,
        use_strand=False,
        rc_aug=False,
        shift_aug=False,
        shift_aug_range=None,

        return_onehot=False,
        return_aug_info=False,
        N_fill_value = 0.25,
        ) -> None:
        super().__init__()

        self.task_idx = task_idx
        self.bed_exp_path = bed_exp_path
        self.genome_path = genome_path
        self.output_column = output_column
        
        self.load_memory = load_memory
        self.use_strand = use_strand

        self.seq_interval = SeqInterval(
            genome_path=genome_path,
            window_length=padded_len,
            rc_aug=rc_aug,
            shift_aug=shift_aug,
            shift_aug_range=shift_aug_range,
            return_onehot=return_onehot,
            return_aug_info=return_aug_info,
            N_fill_value=N_fill_value,
        )

        self.df = pd.read_csv(bed_exp_path, sep='\t', header=0)

        if shuffle == True:
            shuffle_index = np.random.permutation(len(self.df))
            self.df = self.df.iloc[shuffle_index].reset_index(drop=True)

        if filter_column is not None:
            self.df = filter_by_column(self.df, filter_column, filter_in_list, filter_not_in_list)

        if subset_range is not None:
            start_ratio, end_ratio = subset_range
            start_index = int(len(self.df) * start_ratio)
            end_index = int(len(self.df) * end_ratio)
            self.df = self.df.iloc[start_index: end_index]

        if use_pos == True:
            left, right = select_seq_range
            # if 'mid' not in self.df.columns:
            #     self.df['mid'] = (self.df['start'] + self.df['end']) // 2
            self.df['start'] = self.df['pos'] + left
            self.df['end'] = self.df['pos'] + right

        if load_memory == True:
            self.seqs = []
            for i in range(len(self.df)):
                seq = self.get_seq_from_genome(i)
                self.seqs.append(seq)
            self.seqs = torch.tensor(strs2onehots(self.seqs), dtype=torch.float)

        self.labels = self.df[output_column].to_numpy()
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def get_seq_from_genome(self, index):
        interval = self.df.iloc[index]
        chr, start, end = interval[['chr', 'start', 'end']]
        seq = self.seq_interval(chr, start, end)

        if self.use_strand == True:
            strand = interval['strand']
            if strand == '-':
                seq = seq_reverse_complement(seq)

        seq = torch.tensor(str2onehot(seq), dtype=torch.float)
        return seq

    def __getitem__(self, index) -> tuple:
        if self.load_memory == True:
            seq = self.seqs[index]
            label = self.labels[index]
        else:
            seq = self.get_seq_from_genome(index)
            label = self.labels[index]

        if self.task_idx is not None:
            return self.task_idx, (seq, label)
        else:
            return seq, label
    
    def __len__(self) -> int:
        return len(self.labels)





if __name__ == '__main__':
    dataset = BedDataset(
        task_idx=0,
        bed_exp_path = '/home/hxcai/cell_type_specific_CRE/data/Xpresso/merged_bed_exp.csv',
        genome_path = '/home/hxcai/genome/hg38.fa',
        output_column = 'HepG2',
        use_pos=True,
        select_seq_range=[-500, 500],
    )
    print(dataset[0])
    print(dataset[0][1][0].shape)
    