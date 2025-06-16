import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict
from ruamel.yaml import YAML
yaml = YAML()

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm1d(out_channels)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.relu(out)
#         out = self.bn(out)
#         return out

# class LinearBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(LinearBlock, self).__init__()
#         self.linear = nn.Linear(in_channels, out_channels)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.linear(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         return out






class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding, 
        activation='relu',
        layer_order='conv_bn_relu',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.layer_order = layer_order.replace('_add', '')

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f'Invalid activation:{self.activation}')
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.layer_order == 'conv_bn_relu':
            out = self.conv(x)
            out = self.bn(out)
            out = self.act(out)
        elif self.layer_order == 'conv_relu_bn':
            out = self.conv(x)
            out = self.act(out)
            out = self.bn(out)
        else:
            raise ValueError(f'Invalid layer_order:{self.layer_order}')
        return out



class LinearBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        return out



class ResConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding, 
        activation='relu',
        layer_order='conv_bn_add_relu', 
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.layer_order = layer_order

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        if layer_order == 'bn_relu_conv_add':
            self.bn1 = nn.BatchNorm1d(in_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if activation == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif activation == 'gelu':
            self.act1 = nn.GELU()
            self.act2 = nn.GELU()
        else:
            raise ValueError(f'Invalid activation:{self.activation}')

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):
        if self.layer_order == 'conv_bn_add_relu': # resnet original structure
            z = self.act1(self.bn1(self.conv1(x)))
            z = self.bn2(self.conv2(z))
            out = z + self.shortcut(x)
            out = self.act2(out)
        elif self.layer_order == 'conv_bn_relu_add': # 效果比较差
            z = self.act1(self.bn1(self.conv1(x)))
            z = self.act2(self.bn2(self.conv2(z)))
            out = z + self.shortcut(x)
        elif self.layer_order == 'conv_relu_bn_add': # 之前没人提过，但是我实验的效果最好
            z = self.bn1(self.act1(self.conv1(x)))
            z = self.bn2(self.act2(self.conv2(z)))
            out = z + self.shortcut(x)
        elif self.layer_order == 'bn_relu_conv_add': # resnet v2 不用
            z = self.conv1(self.act1(self.bn1(x)))
            z = self.conv2(self.act2(self.bn2(z)))
            out = z + self.shortcut(x)
        else:
            raise ValueError(f'Invalid layer_order:{self.layer_order}')
        return out

