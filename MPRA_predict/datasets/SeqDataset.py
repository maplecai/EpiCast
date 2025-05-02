from ..utils import *
from torch.utils.data import Dataset


class SeqDataset(Dataset):
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
        feature_column=None,
        label_column=None,
        
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
        self.feature_column = feature_column
        self.label_column = label_column

        

        # read dataframe
        if data_path is not None and data_df is None:
            self.df = pd.read_csv(data_path, sep=detect_delimiter(data_path))
        elif data_path is None and data_df is not None:
            self.df = data_df
        else:
            raise ValueError("data_path or data_df must be provided.")

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
            self.seqs = self.df[seq_column].to_numpy().astype(str)
        if feature_column:
            self.features = self.df[feature_column].to_numpy()
            self.features = torch.tensor(self.features, dtype=torch.float)
        if label_column:
            self.labels = self.df[label_column].to_numpy()
            self.labels = torch.tensor(self.labels, dtype=torch.float)
        ###



    def __len__(self) -> int:
        return len(self.df)


    def __getitem__(self, index) -> dict:
        sample = {}
        sample['idx'] = index
        
        if self.seqs is not None:
            seq = self.seqs[index]
            if self.crop:
                seq = crop_seq(seq, self.cropped_length, self.crop_position)
            if self.padding:
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
    dataset = SeqDataset(
        data_path='/home/hxcai/cell_type_specific_CRE/MPRA_predict/predict_short_sequence_features/data/enformer_sequences_test_100.csv',
        input_column='seq',
        crop=True,
        cropped_length=200,
        )
    print(dataset[0]['seq'].shape)
