import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo


class DeepSEA(nn.Module):
    """
    DeepSEA model architecture.
    """
    def __init__(
        self,
        input_length=10500, 
        output_dim=1, 
    ):
        super().__init__()
        self.module_list = nn.Sequential( # Sequential,
            nn.Conv2d(4,320,(1, 8),(1, 1)),
            # nn.Threshold(0, 1e-06),
            nn.MaxPool2d((1, 4),(1, 4)),
            nn.Dropout(0.2),
            nn.Conv2d(320,480,(1, 8),(1, 1)),
            # nn.Threshold(0, 1e-06),
            nn.MaxPool2d((1, 4),(1, 4)),
            nn.Dropout(0.2),
            nn.Conv2d(480,960,(1, 8),(1, 1)),
            # nn.Threshold(0, 1e-06),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(50880, 925),
            # nn.Threshold(0, 1e-06),
            nn.Linear(925, output_dim), # Linear,
            nn.Sigmoid(),
            )

    def forward(self, x):
        if x.shape[2] == 4:
            x = x.permute(0, 2, 1)
            # x.shape should be (batch_size, 4, input_length)
        x = x.unsqueeze(2)
        # print(x.shape)
        x = self.module_list(x)
        # print(x.shape)
        return x






class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4, 320, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(320, 320, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(320, 480, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(480, 480, (1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4), (1, 4)),
                nn.Conv2d(480, 640, (1, 8)),
                nn.ReLU(),
                nn.Conv2d(640, 640, (1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0), -1)),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(67840, 2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2003, 2002)),
            ),
            nn.Sigmoid(),
        )

    # def forward(self, x):
    #     return self.model(x)

    # ACGT->AGCT
    def forward(self, x):
        return self.model(x.permute(0, 2, 1).contiguous().unsqueeze(2))


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
