from ..utils import *
from torch.utils.data import Dataset


class SeqEpiDataset(Dataset):
    def __init__(
        self,
        seq_file_path=None,
        epi_file_path=None,

        apply_filter=True,
        filter_column=None,
        filter_in_list=None,
        filter_not_in_list=None,

        shuffle=False,
        slice_range=None,

        crop=False,
        crop_position='center',
        cropped_length=None,
        
        padding=False,
        padding_position='both_sides',
        padding_method='N',
        padded_length=None,
        genome=None,
        padding_left_seq=None,
        padding_right_seq=None,

        N_fill_value=0.25,
        augmentations=[],

        ###
        seq_column=None,
        label_column=None,
        
        cell_types=None,
        assays=None,
        ###

    ) -> None:
        
        super().__init__()
        
        self.seq_file_path = seq_file_path
        self.epi_file_path = epi_file_path

        self.apply_filter = apply_filter
        self.filter_column = filter_column
        self.filter_in_list = filter_in_list
        self.filter_not_in_list = filter_not_in_list

        self.shuffle = shuffle
        self.slice_range = slice_range

        self.crop = crop
        self.crop_position = crop_position
        self.cropped_length = cropped_length

        self.padding = padding
        self.padding_position = padding_position
        self.padding_method = padding_method
        self.padded_length = padded_length
        self.genome = genome
        self.padding_left_seq = padding_left_seq
        self.padding_right_seq = padding_right_seq

        self.N_fill_value = N_fill_value
        self.augmentations = augmentations

        self.seq_column = seq_column
        self.label_column = label_column

        self.cell_types = cell_types
        self.assays = assays

        self.seq_df = pd.read_csv(seq_file_path, sep=detect_delimiter(seq_file_path))
        self.epi_df = pd.read_csv(epi_file_path, sep=detect_delimiter(epi_file_path))
        self.df = pd.concat([self.seq_df, self.epi_df], axis=1)

        # filter data by filter_column
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

        # set seqs, features, labels
        self.seqs = None
        self.features = None
        self.labels = None

        if seq_column:
            self.seqs = self.df[seq_column].astype(str).to_numpy()

        cols = [f"{cell_type}_{assay}" for cell_type in cell_types for assay in assays]
        data = self.df[cols].to_numpy().reshape(len(self.df), len(cell_types), len(assays))
        self.features = torch.from_numpy(data).float()

        if label_column:
            self.labels = torch.from_numpy(self.df[label_column].to_numpy()).float()



    def __len__(self) -> int:
        return len(self.df)


    def __getitem__(self, index) -> dict:
        sample = {}
        sample['idx'] = index
        
        if self.seqs is not None:
            seq = self.seqs[index]
            if self.crop and len(seq) > self.cropped_length:
                seq = crop_seq(seq, self.cropped_length, self.crop_position)
            if self.padding and len(seq) < self.padded_length:
                seq = pad_seq(seq, self.padded_length, padding_position=self.padding_position, padding_method=self.padding_method, genome=self.genome, given_left_seq=self.padding_right_seq, given_right_seq=self.padding_right_seq)
            seq = torch.tensor(str2onehot(seq, N_fill_value=self.N_fill_value), dtype=torch.float)
            sample['seq'] = seq

        if self.features is not None:
            feature = self.features[index]
            sample['feature'] = feature

        if self.labels is not None:
            label = self.labels[index]
            sample['label'] = label

        return sample




if __name__ == '__main__':
    pass
