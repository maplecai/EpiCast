import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
import yaml
from collections import OrderedDict

from .. import models, utils


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # 两种顺序需要都试一试
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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


# class SkipConnection(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if in_channels == out_channels:
#             self.layer = nn.Identity()
#         else:
#             self.layer = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

#     def forward(self, x):
#         out = self.layer(x)
#         return out


class MultiConvBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ConvBlock(in_channels, out_channels, kernel_size, stride, padding))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# class Residual(nn.Module):
#     def __init__(self, module):
#         super().__init__()
#         self.module = module
#         self.in_channels = module.in_channels
#         self.out_channels = module.out_channels
#         if self.in_channels == self.out_channels:
#             self.shortcut = nn.Identity()
#         else:
#             self.shortcut = nn.Conv1d(self.in_channels, self.out_channels, 1, 1, 0)
            
#     def forward(self, x):
#         out = self.module(x) + self.shortcut(x)
#         return out

class Residual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(self.in_channels, self.out_channels, 1, 1, 0)
            
    def forward(self, x):
        out = self.layer(x) + self.shortcut(x)
        return out


# class Shortcut(nn.Module):
#     def __init__(self, module):
#         super().__init__()
#         self.module = module
    
#     def forward(self, x, x0):
#         x = self.module(x0) + x
#         return x



class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 两种顺序需要都试一试
        # out = self.conv(x)
        # out = self.relu(out)
        # out = self.bn(out)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out



class MyBassetResidual(nn.Module):
    def __init__(
        self, 
        input_length=200,
        output_dim=1,
        # residual=True,
        # conv_per_shortcut=2,
        sigmoid=False,
        squeeze=True,

        conv_num=12,
        conv_channels=256,
        conv_kernel_size=3,
        conv_stride=1,
        conv_padding=1,

        # pool_interval=4,
        pool_kernel_size=2,
        pool_padding=0,
        conv_dropout_rate=0.2,
        gap_layer=False,

        linear_num=2,
        linear_channels=1024,
        linear_dropout_rate=0.5,

        rc_augmentation=False,
        rc_region=None,
    ):                                
        super().__init__()

        self.input_length       = input_length
        self.output_dim         = output_dim
        # self.residual           = residual
        # self.conv_per_shortcut  = conv_per_shortcut
        self.sigmoid            = sigmoid
        self.squeeze            = squeeze
        self.rc_augmentation    = rc_augmentation
        self.rc_region          = rc_region

        # assert residual == True, 'Only residual model is supported'

        self.conv_layers = nn.Sequential(OrderedDict([]))
        
        self.conv_layers.add_module(
            f'conv_block_0', ConvBlock(
                in_channels=4,
                out_channels=conv_channels, 
                kernel_size=conv_kernel_size, 
                stride=conv_stride, 
                padding=conv_padding,))

        # if residual:
            # assert conv_num % conv_per_shortcut == 0, 'conv_num should be divisible by conv_per_shortcut'
            # assert pool_interval % conv_per_shortcut == 0, 'pool_interval should be divisible by conv_per_shortcut'
            # conv_num = conv_num // conv_per_shortcut
            # pool_interval = pool_interval // conv_per_shortcut

        for i in range(1, conv_num+1):
            self.conv_layers.add_module(
                f'res_conv_block_{i}', ResidualConvBlock(
                    in_channels=conv_channels,
                    out_channels=conv_channels, 
                    kernel_size=conv_kernel_size, 
                    stride=conv_stride, 
                    padding=conv_padding,))
            
                # Residual(MultiConvBlock(
                #     num_layers=conv_per_shortcut,
                #     in_channels=conv_channels,
                #     out_channels=conv_channels,
                #     kernel_size=conv_kernel_size, 
                #     stride=conv_stride, 
                #     padding=conv_padding,)))
            
            
            self.conv_layers.add_module(
                f'max_pool_{i}', 
                nn.MaxPool1d(
                    kernel_size=pool_kernel_size, 
                    padding=pool_padding,
                    ceil_mode = True))
            
            self.conv_layers.add_module(
                f'conv_dropout_{i}', 
                nn.Dropout(p=conv_dropout_rate))
        
        if gap_layer:
            self.conv_layers.add_module(
                'gap_layer', nn.AdaptiveAvgPool1d(1))
        
        with torch.no_grad():
            x = torch.randn(1, 4, self.input_length)
            x = self.conv_layers(x)
            hidden_dim = x[0].reshape(-1).shape[0]

        self.linear_layers = nn.Sequential(OrderedDict([]))

        for i in range(linear_num):
            self.linear_layers.add_module(
                f'linear_block_{i}', 
                LinearBlock(
                    in_channels=hidden_dim if i == 0 else linear_channels, 
                    out_channels=linear_channels))
        
            self.linear_layers.add_module(
                f'linear_dropout_{i}', 
                nn.Dropout(p=linear_dropout_rate))
            
        self.last_linear_layer = nn.Linear(
            in_features=linear_channels, 
            out_features=output_dim)
        
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, inputs):
        if isinstance(inputs, dict):
            seq = inputs['seq']
        elif isinstance(inputs, (list, tuple)):
            seq = inputs[0]
        elif isinstance(inputs, torch.Tensor):
            seq = inputs
        else:
            raise ValueError('Unsupported input type')
        
        if seq.shape[2] == 4:
            seq = seq.permute(0, 2, 1)

        x = self.conv_layers(seq)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = self.last_linear_layer(x)

        if self.sigmoid:
            x = self.sigmoid_layer(x)
        if self.squeeze:
            x = x.squeeze(-1)
        return x
    


if __name__ == '__main__':

    yaml_str = '''

model:
    type: MyBassetResidual
    args:
        input_length:       200
        output_dim:         1
        sigmoid:            False
        squeeze:            True
        
        # conv_per_shortcut:  2
        conv_num:           6
        conv_channels:      256
        conv_kernel_size:   3
        conv_stride:        1
        conv_padding:       1
        conv_dropout_rate:  0.2

        pool_kernel_size:   2
        pool_padding:       0
        
        linear_num:         2
        linear_channels:    1024
        linear_dropout_rate: 0.5
        '''
    
    config = yaml.load(yaml_str, Loader=yaml.FullLoader)
    model = utils.init_obj(models, config['model'])


    model(torch.randn(2, 200, 4))
    torchinfo.summary(model, input_size=(2, 200, 4), depth=5)