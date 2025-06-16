import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict

from .. import models, utils
from .ConvBlock import ConvBlock, ResConvBlock
from .TransformerBlock import TransformerBlock
# from .Attention import CrossAttention




class ConcatLayer(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, *inputs):
        return torch.cat(inputs, dim=-1)





class ConcatFusionLayer(nn.Module):
    def __init__(
        self,
        x_input_dim=None,
        y_input_dim=None,
        x_transform=False,
        y_transform=False,
        x_output_dim=None,
        y_output_dim=None,
    ):
        super().__init__()

        self.x_transform = x_transform
        self.y_transform = y_transform

        # 分别线性映射 x, y 再 concat
        if x_transform:
            self.x_linear_layer = nn.Linear(x_input_dim, x_output_dim)
        if y_transform:
            self.y_linear_layer = nn.Linear(y_input_dim, y_output_dim)

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        if self.x_transform:
            x = self.x_linear_layer(x)
        if self.y_transform:
            y = self.y_linear_layer(y)
        z = torch.cat([x, y], dim=1)

        return z









class MyResNetFusion(nn.Module):
    def __init__(
        self, 
        input_seq_length=200,
        input_seq_channels=4,
        input_feature_dim=0,
        input_feature_times=0,
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

        fusion_type=None,
        fusion_x_transform=False,
        fusion_y_transform=False,
        fusion_x_input_dim=None,
        fusion_y_input_dim=None,
        fusion_x_output_dim=None,
        fusion_y_output_dim=None,


        linear_channels_list=None,
        linear_dropout_rate=0.5,
    ):                                
        super().__init__()

        self.input_seq_length   = input_seq_length
        self.input_seq_channels = input_seq_channels
        self.input_feature_dim  = input_feature_dim
        self.input_feature_times= input_feature_times
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
                layer_order=conv_layer_order,
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


        if self.input_feature_dim != 0:
            if fusion_type == 'concat':
                self.fusion_layer = ConcatLayer()

            elif fusion_type == 'linear_concat':
                self.fusion_layer = ConcatFusionLayer(
                    x_input_dim=fusion_x_input_dim,
                    y_input_dim=fusion_y_input_dim,
                    x_transform=fusion_x_transform,
                    y_transform=fusion_y_transform,
                    x_output_dim=fusion_x_output_dim,
                    y_output_dim=fusion_y_output_dim,
                )

            with torch.no_grad():
                f = torch.zeros(1, input_feature_dim)
                x = self.fusion_layer(x, f)

            fusion_out_dim = x.size(1)

        self.linear_layers = nn.Sequential(OrderedDict([]))

        if len(linear_channels_list) == 0:
            self.linear_layers.add_module(
                f'linear', nn.Linear(
                    in_features=fusion_out_dim, 
                    out_features=output_dim,
                )
            )
        else:
            for i in range(len(linear_channels_list)):
                self.linear_layers.add_module(
                    f'linear_block_{i}', LinearBlock(
                        in_channels=fusion_out_dim if i == 0 else linear_channels_list[i-1], 
                        out_channels=linear_channels_list[i],
                    )
                )
                self.linear_layers.add_module(
                    f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
                )
            self.linear_layers.add_module(
                f'linear_last', nn.Linear(
                    in_features=linear_channels_list[-1], 
                    out_features=output_dim,
                )
            )

        self.sigmoid_layer = nn.Sigmoid()



    def forward_seq(self, seq):
        seq = self.conv_layers(seq)

        seq = seq.view(seq.size(0), -1)
        out = self.linear_layers(seq)

        if self.sigmoid:
            out = self.sigmoid_layer(out)
        if self.squeeze:
            out = out.squeeze(-1)
        return out



    def forward_seq_and_feature(self, seq, feature):
        seq = self.conv_layers(seq)
        seq_feature = self.fusion_layer(seq, feature)
        out = self.linear_layers(seq_feature)

        if self.sigmoid:
            out = self.sigmoid_layer(out)
        if self.squeeze:
            out = out.squeeze(-1)
        return out




    def forward_seq_and_features(self, seq, features):
        seq = self.conv_layers(seq)

        outs = []
        for i in range(self.input_feature_times):
            feature_i = features[:, i, :]  # cell type i features
            seq_feature = self.fusion_layer(seq, feature_i)
            out = self.linear_layers(seq_feature)
            if self.sigmoid:
                out = self.sigmoid_layer(out)
            if self.squeeze:
                out = out.squeeze(-1)
            outs.append(out)
        outs = torch.stack(outs, dim=1)  # (batch_size, num_cell_types)
        return outs





    def forward(self, inputs: dict):
        seq = inputs.get('seq')
        feature = inputs.get('feature')
        if seq.shape[2] == 4:
            seq = seq.permute(0, 2, 1)


        if self.input_feature_dim == 0:
            out = self.forward_seq(seq)
            return out

        elif self.input_feature_dim > 0 and self.input_feature_times == 0:
            out = self.forward_seq_and_feature(seq, feature)
            return out


        elif self.input_feature_dim > 0 and self.input_feature_times > 0:
            out = self.forward_seq_and_features(seq, feature)
            return out

        else:
            raise ValueError(f'Invalid {self.input_feature_dim=} or {self.input_feature_times=}')




if __name__ == '__main__':

#     yaml_str = '''
# model:
#     type: MyResNetFusion
#     args:
#         input_seq_length:       200
#         input_seq_channels:     4
#         input_feature_dim:      0
#         input_feature_times:    0
#         output_dim:             1
#         sigmoid:                False
#         squeeze:                True

#         conv_first_channels:    256
#         conv_first_kernel_size: 7
#         conv_layer_order:       conv_bn_add_relu
#         conv_channels_list:     [256,256,256,256,256,256]
#         conv_kernel_size_list:  [3,3,3,3,3,3]
#         pool_kernel_size_list:  [2,2,2,2,2,2]
#         conv_dropout_rate:      0.2
#         gap:                    true

#         linear_channels_list: [1024]
#         linear_dropout_rate: 0.5
#         '''
#     import yaml
#     config = yaml.load(yaml_str, Loader=yaml.FullLoader)
#     model = utils.init_obj(models, config['model'])

#     seq = torch.zeros(size=(1, 4, 200))
#     inputs = {'seq': seq}

#     torchinfo.summary(
#         model, 
#         input_data=(inputs,), 
#         depth=6, 
#         col_names=["input_size", "output_size", "num_params"],
#         row_settings=["var_names"],
#     )








#     yaml_str = '''
# model:
#     type: MyResNetFusion
#     args:
#         input_seq_length:       200
#         input_seq_channels:     4
#         input_feature_dim:      4
#         input_feature_times:    0
#         output_dim:             1
#         sigmoid:                False
#         squeeze:                True

#         conv_first_channels:    256
#         conv_first_kernel_size: 7
#         conv_layer_order:       conv_bn_add_relu
#         conv_channels_list:     [256,256,256,256,256,256]
#         conv_kernel_size_list:  [3,3,3,3,3,3]
#         pool_kernel_size_list:  [2,2,2,2,2,2]
#         conv_dropout_rate:      0.2
#         gap:                    true

#         fusion_type:            linear_concat

#         linear_channels_list: [1024]
#         linear_dropout_rate: 0.5
#         '''
#     import yaml
#     config = yaml.load(yaml_str, Loader=yaml.FullLoader)
#     model = utils.init_obj(models, config['model'])

#     seq = torch.zeros(size=(1, 4, 200))
#     feature = torch.zeros(size=(1, 4))
#     inputs = {'seq': seq, 'feature': feature}

#     torchinfo.summary(
#         model, 
#         input_data=(inputs,), 
#         depth=6, 
#         col_names=["input_size", "output_size", "num_params"],
#         row_settings=["var_names"],
#     )









    yaml_str = '''
model:
    type: MyResNetFusion
    args:
        input_seq_length:       200
        input_seq_channels:     4
        input_feature_dim:      4
        input_feature_times:    5
        output_dim:             1
        sigmoid:                False
        squeeze:                True

        conv_first_channels:    256
        conv_first_kernel_size: 7
        conv_layer_order:       conv_bn_add_relu
        conv_channels_list:     [256,256,256,256,256,256]
        conv_kernel_size_list:  [3,3,3,3,3,3]
        pool_kernel_size_list:  [2,2,2,2,2,2]
        conv_dropout_rate:      0.2
        gap:                    true

        fusion_type:            linear_concat

        linear_channels_list: [1024]
        linear_dropout_rate: 0.5
        '''
    import yaml
    config = yaml.load(yaml_str, Loader=yaml.FullLoader)
    model = utils.init_obj(models, config['model'])

    seq = torch.zeros(size=(1, 4, 200))
    feature = torch.zeros(size=(1, 5, 4))
    inputs = {'seq': seq, 'feature': feature}

    torchinfo.summary(
        model, 
        input_data=(inputs,), 
        depth=6, 
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["var_names"],
    )