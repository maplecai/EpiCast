import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from collections import OrderedDict
from .MyBasset import ConvBlock, LinearBlock
from .Attention import CrossAttention

# class FusionLayer(nn.Module):
#     def __init__(
#         self,
#         input_features=10,
#         output_features=10,
#         fusion_type='concat', 
#         n_heads=8,
#         d_embed=64,
#         d_cross=64,
#     ):
#         super().__init__()
#         self.input_features = input_features
#         self.output_features = output_features
#         self.fusion_type = fusion_type
#         if self.fusion_type == 'concat':
#             pass
#         elif self.fusion_type == 'cross_attention':
#             self.cross_attn = CrossAttention(
#                 n_heads=n_heads,
#                 d_embed=d_embed,
#                 d_cross=d_cross,
#             )

#     def forward(self, x, y):
#         if self.fusion_type == 'concat':
#             return torch.cat([x, y], dim=1)
#         elif self.fusion_type == 'cross_attention':
#             return self.cross_attn(x, y)


class MyBassetFusion(nn.Module):
    def __init__(
        self, 
        fusion_type='concat', 
        n_heads=1,
        input_length=200,
        input_feature_dim=8,
        output_dim=1,
        squeeze=True,

        conv_channels_list=None,
        conv_kernel_size_list=None,
        conv_padding_list=None,
        pool_kernel_size_list=None,
        pool_padding_list=None,
        conv_dropout_rate=0.2,
        gap_layer=False,

        linear_channels_list=None,
        linear_dropout_rate=0.5,
        last_linear_layer=True,
        sigmoid=False,
    ):                                
        super().__init__()

        self.fusion_type        = fusion_type
        self.input_length       = input_length
        self.input_feature_dim  = input_feature_dim
        self.output_dim         = output_dim
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
        
        if gap_layer:
            self.conv_layers.add_module(
                'gap_layer', nn.AdaptiveAvgPool1d(1))

        with torch.no_grad():
            test_input = torch.randn(1, 4, self.input_length)
            test_output = self.conv_layers(test_input)
            hidden_shape = test_output[0].shape
            hidden_dim = hidden_shape.numel()

        if self.fusion_type == 'concat':
            hidden_dim = hidden_dim + self.input_feature_dim
            pass
            # self.fusion_layer = nn.Linear(
            #     in_features=hidden_dim + self.input_feature_dim, 
            #     out_features=hidden_shape[1])
        elif self.fusion_type == 'cross_attention':
            self.fusion_layer = CrossAttention(
                n_heads=n_heads,
                d_embed=hidden_shape[0],
                d_cross=input_feature_dim,
            )

        self.linear_layers = nn.Sequential(OrderedDict([]))

        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_block_{i}', LinearBlock(
                    in_channels=hidden_dim if i == 0 else linear_channels_list[i-1], 
                    out_channels=linear_channels_list[i]))
        
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(p=linear_dropout_rate))
        
        if last_linear_layer == True:
            self.linear_layers.add_module(
                f'linear_last', nn.Linear(
                    in_features=linear_channels_list[-1], 
                    out_features=output_dim))

        if sigmoid == True:
            self.linear_layers.add_module(f'sigmoid', nn.Sigmoid())


    def forward(self, inputs):
        if isinstance(inputs, dict):
            seq, feature = inputs['seq'], inputs['feature']
        elif isinstance(inputs, (list, tuple)):
            seq, feature = inputs[0], inputs[1]
        else:
            raise ValueError('Unsupported input type')
        if seq.shape[2] == 4:
            seq = seq.permute(0, 2, 1)
        x = self.conv_layers(seq)
        if self.fusion_type == 'concat':
            x = x.view(x.size(0), -1)
            x = torch.cat([x, feature], dim=1)
        elif self.fusion_type == 'cross_attention':
            # x.shape = (batch_size, n_channels, length)
            x = x.permute(0, 2, 1)
            # x.shape = (batch_size, length, n_channels)
            feature = feature.view(feature.size(0), 1, -1)
            x = self.fusion_layer(x, feature)
            x = x.view(x.size(0), -1)

        x = self.linear_layers(x)
        if self.squeeze:
            x = x.squeeze(-1)
        return x



class MyBassetFeature(nn.Module):
    """
    Basset model with seq and feature input
    concat(seq_embedding, feature)
    """
    def __init__(
        self, 
        input_length=230,
        input_feature_dim=7,
        output_dim=1,
        squeeze=True,
        rc_augmentation=False,
        rc_region=None,

        conv_channels_list=None,
        conv_kernel_size_list=None,
        conv_padding_list=None,
        pool_kernel_size_list=None,
        pool_padding_list=None,
        conv_dropout_rate=0.2,
        gap_layer=False,

        linear_channels_list=None,
        linear_dropout_rate=0.5,
        last_linear_layer=True,
        sigmoid=False,
    ):                                
        super().__init__()

        self.input_length       = input_length
        self.input_feature_dim  = input_feature_dim
        self.output_dim         = output_dim
        self.squeeze            = squeeze
        self.rc_augmentation    = rc_augmentation
        self.rc_region          = rc_region

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
        
        if gap_layer:
            self.conv_layers.add_module(
                'gap_layer', nn.AdaptiveAvgPool1d(1))

        with torch.no_grad():
            test_input = torch.randn(1, 4, self.input_length)
            test_output = self.conv_layers(test_input)
            hidden_dim = test_output[0].reshape(-1).shape[0]

        self.linear_layers = nn.Sequential(OrderedDict([]))

        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_block_{i}', LinearBlock(
                    in_channels=hidden_dim + self.input_feature_dim if i == 0 else linear_channels_list[i-1], 
                    out_channels=linear_channels_list[i]))
        
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(p=linear_dropout_rate))
        
        if last_linear_layer == True:
            self.linear_layers.add_module(
                f'linear_last', nn.Linear(
                    in_features=hidden_dim if len(linear_channels_list) == 0 else linear_channels_list[-1], 
                    out_features=output_dim))

        if sigmoid == True:
            self.linear_layers.add_module(f'sigmoid', nn.Sigmoid())


    def forward(self, inputs):
        if isinstance(inputs, dict):
            seq, feature = inputs['seq'], inputs['feature']
        elif isinstance(inputs, (list, tuple)):
            seq, feature = inputs[0], inputs[1]
        else:
            raise ValueError('Unsupported input type')
        if seq.shape[2] == 4:
            seq = seq.permute(0, 2, 1)
        x = self.conv_layers(seq)
        x = x.view(x.size(0), -1)
        # 其实不只有这一个地方可以concat, 或许在更靠后的层concat效果更好
        x = torch.cat([x, feature], dim=1)
        x = self.linear_layers(x)
        if self.squeeze:
            x = x.squeeze(-1)
        return x

    # augmentation最好还是给dataset做, 而不是model内部做


if __name__ == '__main__':



    yaml_str = '''
    model:
        type: 
            MyBassetFeature
        args:
            input_length:           200
            input_feature_dim:      8
            output_dim:             1
            squeeze:                True

            conv_channels_list:     [256,256,256,256,256,256]
            conv_kernel_size_list:  [5,5,5,5,5,5]
            conv_padding_list:      [2,2,2,2,2,2]
            pool_kernel_size_list:  [2,2,2,2,2,2]
            pool_padding_list:      [0,0,0,0,0,0]
            conv_dropout_rate:      0.2

            linear_channels_list:   [1024]
            linear_dropout_rate:    0.5

            sigmoid: False
    '''
    
    config = yaml.load(yaml_str, Loader=yaml.FullLoader)
    model = MyBassetFeature(**config['model']['args'])

    seq = torch.randn(2, 4, 200)
    feature = torch.randn(2, 8)
    inputs = {'seq': seq, 'feature': feature}
    out = model(inputs)
    print(out.shape)
    torchinfo.summary(model, input_data=[inputs])






    yaml_str = '''
    model:
        type: 
            MyBassetFusion
        args:
            fusion_type:            'concat'
            # n_heads:                1

            input_length:           200
            input_feature_dim:      8
            output_dim:             1
            squeeze:                True

            conv_channels_list:     [256,256,256,256,256,256]
            conv_kernel_size_list:  [5,5,5,5,5,5]
            conv_padding_list:      [2,2,2,2,2,2]
            pool_kernel_size_list:  [2,2,2,2,2,2]
            pool_padding_list:      [0,0,0,0,0,0]
            conv_dropout_rate:      0.2

            linear_channels_list:   [1024]
            linear_dropout_rate:    0.5

            sigmoid: False
    '''
    
    config = yaml.load(yaml_str, Loader=yaml.FullLoader)
    model = MyBassetFusion(**config['model']['args'])

    seq = torch.randn(2, 4, 200)
    feature = torch.randn(2, 8)
    inputs = {'seq': seq, 'feature': feature}
    out = model(inputs)
    print(out.shape)
    torchinfo.summary(model, input_data=[inputs])
