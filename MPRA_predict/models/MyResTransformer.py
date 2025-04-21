import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict

from .. import models, utils
from .MyResNet import ConvBlock, LinearBlock, ResConvBlock
from .MyCNNTransformer import TransformerBlock


class MyResTransformer(nn.Module):
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
        conv_padding='same',
        conv_activation='relu',
        conv_layer_order='conv_bn_add_relu',
        conv_channels_list=None,
        conv_kernel_size_list=None,
        conv_dropout_rate=0.2,
        pool_kernel_size_list=None,
        gap=False,

        num_trans_blocks=3, 
        trans_d_embed=256, 
        trans_n_heads=8, 
        trans_d_mlp=256,
        trans_dropout_rate=0.1,
        trans_output='cls',

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
        self.trans_output       = trans_output

        if conv_channels_list is None:
            conv_channels_list = []
        if linear_channels_list is None:
            linear_channels_list = []

        self.conv_layers = nn.Sequential(OrderedDict([]))
        
        if conv_layer_order == 'bn_relu_conv_add':
            self.conv_layers.add_module(
                f'conv_first', nn.Conv1d(
                in_channels=input_seq_channels,
                out_channels=conv_first_channels, 
                kernel_size=conv_first_kernel_size, 
                stride=1,
                padding=conv_padding,
                )
            )
        else:
            self.conv_layers.add_module(
                f'conv_block_first', ConvBlock(
                in_channels=input_seq_channels,
                out_channels=conv_first_channels, 
                kernel_size=conv_first_kernel_size, 
                stride=1,
                padding=conv_padding,
                activation=conv_activation,
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

        # compute the shape
        with torch.no_grad():
            x = torch.zeros(1, self.input_seq_channels, self.input_seq_length)
            x = self.conv_layers(x)
            # x = x.view(x.size(0), -1)
            # current_dim = x.size(1)

        if input_feature_dim > 0:
            self.cls_embedding_layer = nn.Linear(input_feature_dim, trans_d_embed)
            nn.init.normal_(self.cls_embedding_layer.weight, mean=0.0, std=0.02) # 初始化可能很重要！

        self.trans_layers = nn.Sequential(OrderedDict([]))
        for i in range(num_trans_blocks):
            self.trans_layers.add_module(
                f'transformer_block_{i}', TransformerBlock(
                    d_embed=trans_d_embed, 
                    n_heads=trans_n_heads, 
                    d_mlp=trans_d_mlp, 
                    dropout_rate=trans_dropout_rate
                )
            )

        with torch.no_grad():
            x = x.permute(0, 2, 1) # (batch_size, seq_length, hidden_dim)
            x = self.trans_layers(x)
            x = x.mean(1)
            # x = x.view(x.size(0), -1)
            current_dim = x.size(1)

        self.linear_layers = nn.Sequential(OrderedDict([]))
        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_block_{i}', LinearBlock(
                    in_channels=current_dim if i == 0 else linear_channels_list[i-1], 
                    out_channels=linear_channels_list[i],
                )
            )
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
            )
        self.linear_layers.add_module(
            f'linear_last', nn.Linear(
                in_features=current_dim if len(linear_channels_list) == 0 else linear_channels_list[-1], 
                out_features=output_dim,
            )
        )

        self.sigmoid_layer = nn.Sigmoid()



    def forward_seq(self, seq):
        seq = self.conv_layers(seq)
        seq = seq.permute(0, 2, 1) # (batch_size, seq_length, hidden_dim)
        seq = self.trans_layers(seq)

        if self.trans_output == 'seq_mean':
            out = seq.mean(1)
        else:
            raise ValueError(f"Invalid {self.trans_output = }")
        
        out = self.linear_layers(out)

        if self.sigmoid:
            out = self.sigmoid_layer(out)
        if self.squeeze:
            out = out.squeeze(-1)
        return out



    def forward_seq_and_feature(self, seq, feature):
        seq = self.conv_layers(seq)
        seq = seq.permute(0, 2, 1) # (batch_size, seq_length, hidden_dim)

        cls = self.cls_embedding_layer(feature)
        cls = cls.unsqueeze(1)

        cls_seq = torch.concat([cls, seq], dim=1)
        cls_seq = self.trans_layers(cls_seq)

        if self.trans_output == 'cls':
            out = cls_seq[:, 0]
        elif self.trans_output == 'seq_mean':
            out = cls_seq[:, 1:].mean(1)

        # out = out.view(out.size(0), -1)
        out = self.linear_layers(out)

        if self.sigmoid:
            out = self.sigmoid_layer(out)
        if self.squeeze:
            out = out.squeeze(-1)
        return out




    def forward_seq_and_features(self, seq, features):
        # batch_size, num_celltypes, feature_dim = features.shape
        # # print(features.shape)
        # flat_seq, flat_feature = utils.flatten_seq_features(seq, features)
        # flat_out = self.forward_seq_and_feature(flat_seq, flat_feature)
        # out = utils.unflatten_target(flat_out, batch_size, num_celltypes)
        # return out

        # outs = []
        # for i in range(self.input_feature_times):
        #     feature_i = features[:, i, :]  # cell type i features
        #     out = self.forward_seq_and_feature(seq, feature_i)
        #     outs.append(out)
        # outs = torch.stack(outs, dim=1)  # (batch_size, num_cell_types)
        # return outs


        seq = self.conv_layers(seq)
        seq = seq.permute(0, 2, 1) # (batch_size, seq_length, hidden_dim)

        outs = []
        for i in range(self.input_feature_times):
            feature_i = features[:, i, :]  # cell type i features

            cls = self.cls_embedding_layer(feature_i)
            cls = cls.unsqueeze(1)

            cls_seq = torch.concat([cls, seq], dim=1)
            cls_seq = self.trans_layers(cls_seq)

            if self.trans_output == 'cls':
                out = cls_seq[:, 0]
            elif self.trans_output == 'seq_mean':
                out = cls_seq[:, 1:].mean(1)

            out = self.linear_layers(out)

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

    yaml_str = '''
model:
    type: MyResTransformer
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
        gap:                    false

        num_trans_blocks: 3
        trans_d_embed: 256
        trans_n_heads: 4
        trans_d_mlp: 256
        trans_dropout_rate: 0.2

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