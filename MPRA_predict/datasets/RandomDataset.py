from ..utils import *
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(
            self, 
            shape,
    ):
        self.shape = shape
        self.data = torch.rand(self.shape, dtype=torch.float)
    
    def __len__(self) -> int:
        return self.shape[0]
    
    def __getitem__(self, index) -> dict:
        return self.data[index]