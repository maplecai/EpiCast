import torch.nn as nn
import torch
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, activation=nn.ReLU):
        """
        多层感知机模块

        参数：
        - input_dim: 输入维度
        - hidden_dims: 隐藏层维度列表，例如 [128, 64]
        - output_dim: 输出维度
        - dropout: Dropout 概率
        - activation: 激活函数类（默认为 ReLU）
        """
        super(MLP, self).__init__()
        layers = []

        # 输入层到第一个隐藏层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 最后一层输出
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)





if __name__ == '__main__':
    input = torch.rand(size=(2,10))
    target = torch.ones(size=(2,1))
    model = MLP(input_size=10, hidden_size=5, output_size=1)
    output = model(input)
    criteron = nn.MSELoss()
    loss = criteron(output, target)
    print(loss)
