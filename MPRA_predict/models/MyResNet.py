import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict

from .. import models, utils


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
        self.layer_order = layer_order.replace('add', '')


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
        # out = self.conv(x)
        # out = self.act(out)
        # out = self.bn(out)
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
        
        if layer_order == 'bn_relu_conv_add':
            self.bn1 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        if self.layer_order == 'conv_bn_add_relu':
            z = self.act1(self.bn1(self.conv1(x)))
            z = self.bn2(self.conv2(z))
            out = z + self.shortcut(x)
            out = self.act2(out)
        elif self.layer_order == 'conv_bn_relu_add': # 效果比较差
            z = self.act1(self.bn1(self.conv1(x)))
            z = self.act2(self.bn2(self.conv2(z)))
            out = z + self.shortcut(x)
        elif self.layer_order == 'conv_relu_bn_add': # 之前没人提过，但是我实验效果最好
            z = self.bn1(self.act1(self.conv1(x)))
            z = self.bn2(self.act2(self.conv2(z)))
            out = z + self.shortcut(x)
        elif self.layer_order == 'bn_relu_conv_add':
            z = self.conv1(self.act1(self.bn1(x)))
            z = self.conv2(self.act2(self.bn2(z)))
            out = z + self.shortcut(x)
        else:
            raise ValueError(f'Invalid layer_order:{self.layer_order}')
        return out






class MyResNet(nn.Module):
    def __init__(
        self, 
        input_length=200,
        input_channels=4,
        output_dim=1,
        sigmoid=False,
        squeeze=True,

        conv_first_channels=256,
        conv_first_kernel_size=7,
        conv_padding='same',
        conv_layer_order='conv_bn_add_relu',
        conv_channels_list=None,
        conv_kernel_size_list=None,
        pool_kernel_size_list=None,
        conv_dropout_rate=0.2,
        gap=False,

        linear_channels_list=None,
        linear_dropout_rate=0.5,
    ):                                
        super().__init__()

        self.input_length       = input_length
        self.input_channels     = input_channels
        self.output_dim         = output_dim
        self.sigmoid            = sigmoid
        self.squeeze            = squeeze

        if conv_channels_list is None:
            conv_channels_list = []
        if linear_channels_list is None:
            linear_channels_list = []

        self.conv_layers = nn.Sequential(OrderedDict([]))
        
        if conv_layer_order == 'bn_relu_conv_add':
            self.conv_layers.add_module(
                f'conv_first', nn.Conv1d(
                in_channels=input_channels,
                out_channels=conv_first_channels, 
                kernel_size=conv_first_kernel_size, 
                stride=1,
                padding=conv_padding,
                )
            )
        else:
            self.conv_layers.add_module(
                f'conv_block_first', ConvBlock(
                in_channels=input_channels,
                out_channels=conv_first_channels, 
                kernel_size=conv_first_kernel_size, 
                stride=1,
                padding=conv_padding,
                layer_order=conv_layer_order,
                )
            )

        for i in range(len(conv_channels_list)):
            self.conv_layers.add_module(
                f'res_conv_block_{i}', ResConvBlock(
                    in_channels=conv_first_channels if i == 0 else conv_channels_list[i-1], 
                    out_channels=conv_channels_list[i], 
                    kernel_size=conv_kernel_size_list[i], 
                    stride=1, 
                    padding=conv_padding,
                    layer_order=conv_layer_order,
                )
            )

            if pool_kernel_size_list[i] != 1:
                self.conv_layers.add_module(
                    f'max_pool_{i}', 
                    nn.MaxPool1d(
                        kernel_size=pool_kernel_size_list[i], 
                        ceil_mode=True, # keep edge information
                    )
                )
            self.conv_layers.add_module(
                f'conv_dropout_{i}', 
                nn.Dropout(conv_dropout_rate)
            )
        if gap:
            self.conv_layers.add_module(
                'gap_layer', nn.AdaptiveAvgPool1d(1)
            )
        
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.input_length)
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            hidden_dim = x.size(1)

        self.linear_layers = nn.Sequential(OrderedDict([]))
        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_block_{i}', LinearBlock(
                    in_channels=hidden_dim if i == 0 else linear_channels_list[i-1], 
                    out_channels=linear_channels_list[i],
                )
            )
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
            )
        self.linear_layers.add_module(
            f'linear_last', nn.Linear(
                in_features=hidden_dim if len(linear_channels_list) == 0 else linear_channels_list[-1], 
                out_features=output_dim,
            )
        )

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

        if self.sigmoid:
            x = self.sigmoid_layer(x)
        if self.squeeze:
            x = x.squeeze(-1)
        return x
    

















if __name__ == '__main__':

    yaml_str = '''
model:
    type: MyResNet
    args:
        input_length:       200
        input_channels:     4
        output_dim:         1
        sigmoid:            False
        squeeze:            True

        conv_channels_list: [256, 256, 256, 256, 256, 256]
        conv_kernel_size_list: [3, 3, 3, 3, 3, 3]
        conv_padding_list: [1, 1, 1, 1, 1, 1]
        pool_kernel_size_list: [2,2,2,2,2,2]
        conv_dropout_rate: 0.2
        gap: true

        linear_channels_list: [1024]
        linear_dropout_rate: 0.5
        '''
    import yaml
    config = yaml.load(yaml_str, Loader=yaml.FullLoader)
    model = utils.init_obj(models, config['model'])

    inputs = torch.zeros(size=(1,4,200))
    torchinfo.summary(
        model, 
        input_data=inputs, 
        depth=6, 
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["var_names"],
    )