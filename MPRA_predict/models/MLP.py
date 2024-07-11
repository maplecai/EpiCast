import torch.nn as nn
import torch
import numpy as np

class MLP(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            dropout_rate=0.5,
            ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    input = torch.rand(size=(2,10))
    target = torch.ones(size=(2,1))
    model = MLP(input_size=10, hidden_size=5, output_size=1)
    output = model(input)
    criteron = nn.MSELoss()
    loss = criteron(output, target)
    print(loss)
