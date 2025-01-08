from ..utils import *
from .BaseDataset import BaseDataset
from .GenomeInterval import GenomeInterval

class BedDataset(BaseDataset):
    def __init__(
        self,
        genome_path,
        window_length=None,
        spicify_strand=False,
        aug_rc=False,
        aug_rc_prob=0.5,
        aug_shift=False,
        aug_shift_range=(0, 0),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.genome_path = genome_path
        self.window_length = window_length
        self.spicify_strand = spicify_strand
        self.aug_rc = aug_rc
        self.aug_rc_prob = aug_rc_prob
        self.aug_shift = aug_shift
        self.aug_shift_range = aug_shift_range

        self.genome_interval = GenomeInterval(genome_path)
        # self.df['mid'] = (self.df['start'] + self.df['end']) // 2
        # self.df['start'] = self.df['mid'] - window_length // 2
        # self.df['end'] = self.df['start'] + window_length


    def get_seq_from_genome(self, index):
        row = self.df.iloc[index]
        chr, start, end = row[['chr', 'start', 'end']]
        
        # adjust to window length
        if self.window_length is not None:
            mid = (start + end) // 2
            start = mid - self.window_length // 2
            end = start + self.window_length

        # shift augmentation
        if self.aug_shift:
            min_shift, max_shift = self.aug_shift_range
            shift = np.random.randint(min_shift, max_shift + 1)
            start += shift
            end += shift

        # extract sequence
        seq = self.genome_interval(chr, start, end)

        # reverse strand
        if self.spicify_strand and row['strand'] == '-':
            seq = rc_seq(seq)

        # reverse complement augmentation
        if self.aug_rc and np.random.rand() < self.aug_rc_prob:
            seq = rc_seq(seq)
        
        return seq



    def __getitem__(self, index) -> dict:
        seq = self.get_seq_from_genome(index)

        if self.crop:
            seq = crop_seq(seq, self.cropped_length, self.crop_method)
        if self.padding:
            seq = pad_seq(seq, self.padded_length, self.padding_method)

        seq = torch.tensor(str2onehot(seq, N_fill_value=self.N_fill_value), dtype=torch.float)

        if self.labels is None:
            return {'seq': seq}
        else:
            label = self.labels[index]
            return {'seq': seq, 'label': label}





if __name__ == '__main__':
    dataset = BedDataset(
        data_path='/home/hxcai/cell_type_specific_CRE/MPRA_predict/predict_short_sequence_features/data/enformer_sequences_test_100.csv',
        genome_path='/home/hxcai/genome/hg38.fa',
        window_length=200,
        )
    print(dataset[0]['seq'].shape)
