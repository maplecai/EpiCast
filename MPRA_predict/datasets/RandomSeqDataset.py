from ..utils import *
from torch.utils.data import Dataset


class RandomSeqDataset(Dataset):
    def __init__(
            self, 
            seq_length,
            num_samples,
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
    ):
        self.seq_length = seq_length
        self.num_samples = num_samples

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

        self.seqs = np.array([
            random_seq(seq_length)
            for _ in range(num_samples)
        ])


    def __len__(self) -> int:
        return len(self.seqs)
    
    def __getitem__(self, index) -> dict:
        seq = self.seqs[index]

        if self.crop:
            seq = crop_seq(seq, self.cropped_length, self.crop_position)
        if self.padding:
            seq = pad_seq(seq, self.padded_length, padding_position=self.padding_position, padding_method=self.padding_method, genome=self.genome, given_left_seq=self.padding_right_seq, given_right_seq=self.padding_right_seq)
        
        seq = torch.tensor(str2onehot(seq, N_fill_value=self.N_fill_value), dtype=torch.float)
        return seq