import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo



class Xpresso(nn.Module):
    """
    Xpresso model architecture.
    """
    def __init__(
        self,
        input_length=10500, 
        output_dim=1, 
    ):
        super().__init__()
        self.conv_list = nn.Sequential(
            nn.Conv1d(4, 128, 6),
            nn.MaxPool1d(30, 30),
            nn.Conv1d(128, 32, 9),
            nn.MaxPool1d(10, 10),
            nn.Flatten(),
        )

        with torch.no_grad():
            test_input = torch.randn(1, 4, input_length)
            test_output = self.conv_list(test_input)
            conv_output_size = test_output.shape[-1]

        self.linear_list = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(64, 2),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(2, output_dim),
            )

    def forward(self, x, *arg, **kwargs):
        if x.shape[2] == 4:
            x = x.permute(0, 2, 1)
            # x.shape should be (batch_size, 4, input_length)
        x = self.conv_list(x)
        x = self.linear_list(x)
        x = x.squeeze(-1)

        return x


if __name__ == '__main__':
    model = Xpresso(input_length=10500, output_dim=1)
    torchinfo.summary(model, input_size=(2, 10500, 4))
