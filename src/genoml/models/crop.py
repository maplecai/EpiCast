import torch
import torch.nn as nn

class CenterCrop1d(nn.Module):
    def __init__(self, target_length: int, dim: int = 1):
        super().__init__()
        self.target_length = target_length
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[self.dim]
        start = (length - self.target_length) // 2
        x = x.narrow(self.dim, start, self.target_length)
        return x
