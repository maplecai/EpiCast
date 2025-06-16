import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict

from .. import models, utils
from .ConvBlock import ConvBlock, ResConvBlock


class MyResNet(nn.Module):
    def __init__(
        self, 
        input_seq_length=200,
        input_seq_channels=4,
        output_dim=1,
        sigmoid=False,
        squeeze=True,

        conv_first_channels=256,
        conv_first_kernel_size=7,
        pool_first_kernel_size=1,

        conv_padding='same',
        conv_activation='relu',
        conv_layer_order='conv_bn_add_relu',
        conv_channels_list=None,
        conv_kernel_size_list=None,
        conv_dropout_rate=0.2,
        pool_kernel_size_list=None,
        gap=False,

        linear_channels_list=None,
        linear_dropout_rate=0.5,
    ):                                
        super().__init__()

        self.input_seq_length   = input_seq_length
        self.input_seq_channels = input_seq_channels
        self.output_dim         = output_dim
        self.sigmoid            = sigmoid
        self.squeeze            = squeeze

        if conv_channels_list is None:
            conv_channels_list = []
        if linear_channels_list is None:
            linear_channels_list = []

        self.conv_layers = nn.Sequential(OrderedDict([]))
        
        self.conv_layers.add_module(
            f'conv_block_first', ConvBlock(
                in_channels=input_seq_channels,
                out_channels=conv_first_channels, 
                kernel_size=conv_first_kernel_size, 
                stride=1,
                padding=conv_padding,
                layer_order='conv_bn_relu',
                activation=conv_activation,
            )
        )
        
        if pool_first_kernel_size != 1:
            self.conv_layers.add_module(
                f'max_pool_first', nn.MaxPool1d(
                    kernel_size=pool_first_kernel_size, 
                    ceil_mode=True, # keep edge information
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
                    activation=conv_activation,
                )
            )

            if pool_kernel_size_list[i] != 1:
                self.conv_layers.add_module(
                    f'max_pool_{i}', nn.MaxPool1d(
                        kernel_size=pool_kernel_size_list[i], 
                        ceil_mode=True, # keep edge information
                    )
                )
            self.conv_layers.add_module(
                f'conv_dropout_{i}', nn.Dropout(conv_dropout_rate)
            )
        if gap:
            self.conv_layers.add_module(
                'gap_layer', nn.AdaptiveAvgPool1d(1)
            )

        # compute the shape
        with torch.no_grad():
            x = torch.zeros(1, self.input_seq_channels, self.input_seq_length)
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            current_dim = x.size(1)

        self.linear_layers = nn.Sequential(OrderedDict([]))

        current_dim = x.shape[1]
        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_{i}', nn.Linear(
                    in_features=current_dim, 
                    out_features=linear_channels_list[i],
                )
            )
            self.linear_layers.add_module(
                f'linear_activation_{i}', nn.ReLU()
            )
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
            )
            current_dim = linear_channels_list[i]

        self.linear_layers.add_module(
            f'linear_last', nn.Linear(
                in_features=current_dim, 
                out_features=output_dim,
            )
        )


        # for i in range(len(linear_channels_list)):
        #     self.linear_layers.add_module(
        #         f'linear_block_{i}', LinearBlock(
        #             in_channels=current_dim if i == 0 else linear_channels_list[i-1], 
        #             out_channels=linear_channels_list[i],
        #         )
        #     )
        #     self.linear_layers.add_module(
        #         f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
        #     )
        # self.linear_layers.add_module(
        #     f'linear_last', nn.Linear(
        #         in_features=current_dim if len(linear_channels_list) == 0 else linear_channels_list[-1], 
        #         out_features=output_dim,
        #     )
        # )

        self.sigmoid_layer = nn.Sigmoid()




    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            seq = inputs
        elif isinstance(inputs, dict):
            seq = inputs.get('seq')
        else:
            raise ValueError(f'inputs must be a torch.Tensor or a dict with key "seq"')

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
        input_seq_length:       200
        input_seq_channels:     4
        output_dim:             1
        sigmoid:                False
        squeeze:                True

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

    seq = torch.zeros(size=(1, 4, 200))
    inputs = {'seq': seq}
    torchinfo.summary(
        model, 
        input_data=(inputs,), 
        depth=6, 
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["var_names"],
    )