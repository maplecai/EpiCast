import torch
import torch.nn as nn
from boda.model.basset import BassetBranched


class Malinois(BassetBranched):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expect_input_sample = {
            "seq": torch.zeros(1, self.input_len, 4),
        }
    
    def get_flatten_factor(self, input_len):
        hook = input_len
        hook = hook // 3
        hook = hook // 4
        return (hook + 2) // 4
    

    def forward(self, inputs: torch.Tensor | dict | list) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            x = inputs
        elif isinstance(inputs, dict):
            x = inputs.get('seq', None)
        elif isinstance(inputs, list):
            x = inputs[0]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

        if x.dim() == 2:
            if x.shape[0] != 4:
                x = x.permute(1, 0)
        elif x.dim() == 3:
            if x.shape[1] != 4:
                x = x.permute(0, 2, 1)
        else:
            raise ValueError(f'Input must be 2D or 3D, got {x.dim()}D')
        
        out = super().forward(x)
        return out


if __name__ == '__main__':
    import torchinfo
    model = Malinois(input_len=600, n_outputs=3, n_linear_layers=1, n_branched_layers=3, branched_channels=140)
    torchinfo.summary(model, (1, 600, 4))

    model = Malinois(input_len=200, n_outputs=3, n_linear_layers=1, n_branched_layers=3, branched_channels=140)
    torchinfo.summary(model, (1, 200, 4))