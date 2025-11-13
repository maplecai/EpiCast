import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict
from ruamel.yaml import YAML
yaml = YAML()
from .ConvBlock import ConvBlock, LinearBlock


class MyBasset(nn.Module):
    """
    Basset model architecture.
    """
    def __init__(
            self, 
            input_seq_length=200,
            output_dim=1,

            sigmoid=False,
            squeeze=True,

            conv_channels_list=None,
            conv_kernel_size_list=None,
            conv_padding_list=None,
            pool_kernel_size_list=None,
            pool_padding_list=None,
            conv_dropout_rate=0.2,
            global_average_pooling=False,

            linear_channels_list=None,
            linear_dropout_rate=0.5,
        ):                                
        super().__init__()

        self.input_seq_length   = input_seq_length
        self.output_dim         = output_dim
        self.sigmoid            = sigmoid
        self.squeeze            = squeeze

        if conv_padding_list is None:
            conv_padding_list = [0] * len(conv_kernel_size_list)
        if pool_padding_list is None:
            pool_padding_list = [0] * len(pool_kernel_size_list)

        self.conv_layers = nn.Sequential(OrderedDict([]))

        for i in range(len(conv_kernel_size_list)):
            self.conv_layers.add_module(
                f'conv_block_{i}', ConvBlock(
                    in_channels=4 if i == 0 else conv_channels_list[i-1], 
                    out_channels=conv_channels_list[i], 
                    kernel_size=conv_kernel_size_list[i], 
                    stride=1, 
                    padding=conv_padding_list[i]))
                
            self.conv_layers.add_module(
                f'max_pool_{i}', nn.MaxPool1d(
                    kernel_size=pool_kernel_size_list[i], 
                    padding=pool_padding_list[i],
                    ceil_mode = True))
        
            self.conv_layers.add_module(
                f'conv_dropout_{i}', nn.Dropout(p=conv_dropout_rate))
        
        if global_average_pooling:
            self.conv_layers.add_module(
                f'gap_layer', nn.AdaptiveAvgPool1d(1))

        with torch.no_grad():
            test_input = torch.zeros(1, 4, self.input_seq_length)
            test_output = self.conv_layers(test_input)
            hidden_dim = test_output[0].reshape(-1).shape[0]
        self.linear_layers = nn.Sequential(OrderedDict([]))

        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_block_{i}', LinearBlock(
                    in_channels=hidden_dim if i == 0 else linear_channels_list[i-1], 
                    out_channels=linear_channels_list[i]))
        
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(p=linear_dropout_rate))
        
            self.linear_layers.add_module(
                f'linear_last', nn.Linear(
                    in_features=hidden_dim if len(linear_channels_list) == 0 else linear_channels_list[-1], 
                    out_features=output_dim))

        self.sigmoid_layer = nn.Sigmoid()


    def forward(self, inputs):
        if isinstance(inputs, dict):
            seq = inputs['seq']
        elif isinstance(inputs, (list, tuple)):
            seq = inputs[0]
        elif isinstance(inputs, torch.Tensor):
            seq = inputs
        else:
            raise ValueError('inputs type must be dict or list or tuple or tensor')
        
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
        type: 
            MyBassetFeatureMatrix
        args:
            input_seq_length:       200
            output_dim:             1

            conv_channels_list:     [256, 256, 256]
            conv_kernel_size_list:  [7, 7, 7]
            conv_padding_list:      [3, 3, 3]
            pool_kernel_size_list:  [2, 2, 2]
            pool_padding_list:      [0, 0, 0]
            conv_dropout_rate:      0.2

            linear_channels_list:   [256]
            linear_dropout_rate:    0.5

            sigmoid: True
        '''
    
    config = yaml.load(yaml_str)
    model = MyBasset(**config['model']['args'])

    torchinfo.summary(model, input_size=(1, 200, 4))