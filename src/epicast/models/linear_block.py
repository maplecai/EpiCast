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
            raise ValueError(f'Inval activation:{activation}')

    def forward(self, x: torch.Tensor):
        out = self.linear(x)
        out = self.act(out)
        return out


class SqueezeLayer(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(self.dim)

from collections import OrderedDict
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        linear_channels_list: list[int],
        num_linear_blocks: int,
        linear_dropout_rate: float,
        output_dim: int,
    ):
        super().__init__()

        layers = OrderedDict()
        linear_in_channels = in_features

        for i in range(num_linear_blocks):
            linear_out_channels = linear_channels_list[i]
            layers[f"linear_{i}"] = nn.Linear(linear_in_channels, linear_out_channels)
            layers[f"relu_{i}"] = nn.ReLU()
            layers[f"dropout_{i}"] = nn.Dropout(linear_dropout_rate)
            linear_in_channels = linear_out_channels

        self.layers = nn.Sequential(layers)
        self.output_layer = nn.Linear(linear_in_channels, output_dim)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)
        return x