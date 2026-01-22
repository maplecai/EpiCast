import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        activation='relu',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.linear = nn.Linear(in_channels, out_channels)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f'Invalid activation:{activation}')

    def forward(self, x):
        out = self.linear(x)
        out = self.act(out)
        return out


class SqueezeLayer(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)