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


class MLPBlock(nn.Module):
    """
    单个输出分支：
    多个 LinearBlock + Dropout + 最后 Linear(..., 1)
    """
    def __init__(
        self,
        num_linear_blocks: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_rate: float = 0.0,
        activation: str = 'relu',
        num_branches: int = 0,
    ):
        super().__init__()

        self.num_branches = num_branches

        if activation == 'relu':
            act_layer = nn.ReLU()
        # elif activation == 'gelu':
        #     act_layer = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        if num_branches == 0:
            self.layers = nn.Sequential()
            for j in range(num_linear_blocks):
                self.layers.add_module(
                    f'linear_{j}', 
                    nn.Linear(in_channels if j == 0 else hidden_channels, hidden_channels)
                )
                self.layers.add_module(
                    f'act_{j}', 
                    nn.ReLU(),
                )
                self.layers.add_module(
                    f'dropout_{j}', 
                    nn.Dropout(dropout_rate)
                )
            self.layers.add_module(
                f'linear_last', 
                nn.Linear(hidden_channels if num_linear_blocks > 0 else in_channels, out_channels)
            )

        else:
            self.branches = nn.ModuleList()
            for i in range(num_branches):
                self.layers = nn.Sequential()
                for j in range(num_linear_blocks):
                    self.layers.add_module(
                        f'linear_{j}', 
                        nn.Linear(in_channels if j == 0 else hidden_channels, hidden_channels)
                    )
                    self.layers.add_module(
                        f'act_{j}', 
                        nn.ReLU(),
                    )
                    self.layers.add_module(
                        f'dropout_{j}', 
                        nn.Dropout(dropout_rate)
                    )
                self.layers.add_module(
                    f'linear_last', 
                    nn.Linear(hidden_channels if num_linear_blocks > 0 else in_channels, out_channels)
                )
                self.branches.append(self.layers)



    def forward(self, x):
        if self.num_branches == 0:
            out = self.layers(x).squeeze(-1)
            return out
        else:
            out = [branch(x).squeeze(-1) for branch in self.branches]
            out = torch.stack(out, dim=-1)
            return out
