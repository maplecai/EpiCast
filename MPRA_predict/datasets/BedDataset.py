from ..utils import *
from torch.utils.data import Dataset
from .GenomeInterval import GenomeInterval

class BedDataset(Dataset):
    def __init__(
        self,

        data_path=None,
        data_df=None,

        apply_filter=True,
        filter_column=None,
        filter_in_list=None,
        filter_not_in_list=None,

        shuffle=False,
        slice_range=None,

        crop=False,
        crop_method='center',
        cropped_length=None,
        
        padding=False,
        padding_method='N',
        padded_length=None,

        N_fill_value=0.25,
        augmentations=[],

        ###
        genome_path=None,
        window_length=None,
        spicify_strand=False,
        random_rc=False,
        random_rc_prob=0.5,
        random_shift=False,
        random_shift_range=(0, 0),
        ###
    ) -> None:
        super().__init__()

        self.data_path = data_path
        self.data_df = data_df

        self.apply_filter = apply_filter
        self.filter_column = filter_column
        self.filter_in_list = filter_in_list
        self.filter_not_in_list = filter_not_in_list

        self.shuffle = shuffle
        self.slice_range = slice_range

        self.crop = crop
        self.crop_method = crop_method
        self.cropped_length = cropped_length

        self.padding = padding
        self.padding_method = padding_method
        self.padded_length = padded_length

        self.N_fill_value = N_fill_value
        self.augmentations = augmentations

        self.genome_path = genome_path
        self.window_length = window_length
        self.spicify_strand = spicify_strand
        
        self.random_rc = random_rc
        self.random_rc_prob = random_rc_prob
        self.random_shift = random_shift
        self.random_shift_range = random_shift_range

        if data_path is not None and data_df is None:
            self.df = pd.read_csv(data_path, sep=detect_delimiter(data_path))
        elif data_path is None and data_df is not None:
            self.df = data_df
        else:
            raise ValueError("data_path or data_df must be provided.")

        if apply_filter:
            if filter_in_list is not None:
                self.df = self.df[self.df[filter_column].isin(filter_in_list)]
            if filter_not_in_list is not None:
                self.df = self.df[~self.df[filter_column].isin(filter_not_in_list)]
        self.df = self.df.reset_index(drop=True)

        if slice_range is not None:
            start, end = slice_range
            if 0 <= start < end <= 1:
                start = int(len(self.df) * start)
                end = int(len(self.df) * end)
            self.df = self.df.iloc[start:end].reset_index(drop=True)

        if shuffle:
            shuffle_index = np.random.permutation(len(self.df))
            self.df = self.df.iloc[shuffle_index].reset_index(drop=True)

        self.seqs = None
        self.labels = None

        self.genome_interval = GenomeInterval(genome_path)


    def get_seq_from_genome(self, index):
        row = self.df.iloc[index]
        chr, start, end = row[['chr', 'start', 'end']]
        
        # adjust to window length
        if self.window_length is not None:
            mid = (start + end) // 2
            start = mid - self.window_length // 2
            end = start + self.window_length

        # shift augmentation
        if self.random_shift:
            min_shift, max_shift = self.random_shift_range
            shift = np.random.randint(min_shift, max_shift + 1)
            start += shift
            end += shift

        # extract sequence
        seq = self.genome_interval(chr, start, end)

        # reverse strand
        if self.spicify_strand and row['strand'] == '-':
            seq = rc_seq(seq)

        # reverse complement augmentation
        if self.random_rc:
            if np.random.rand() < self.random_rc_prob:
                seq = rc_seq(seq)
        
        return seq


    def __len__(self) -> int:
        return len(self.df)


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
