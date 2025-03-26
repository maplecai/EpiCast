import numpy as np
import torch
import torch.nn as nn
import torchinfo
from collections import OrderedDict
from ruamel.yaml import YAML

from .MyBasset import ConvBlock, LinearBlock
from .Attention import CrossAttention


class FusionLayer(nn.Module):
    def __init__(
        self,
        x_features=10,
        y_features=10,
        output_features=10,

        fusion_type='concat', 

        n_heads=8,
        d_embed=64,
        d_cross=64,
    ):
        super().__init__()
        self.x_features = x_features
        self.y_features = y_features
        self.output_features = output_features
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # 直接cancat 线性映射
            self.linear = nn.Linear(in_features=(x_features + y_features), out_features=output_features)
        
        elif fusion_type == 'linear_concat':
            # 分别线性映射 x, y 再 concat
            x_out_features = output_features//2
            y_out_features = output_features - x_out_features
            self.linear_x = nn.Linear(in_features=x_features, out_features=x_out_features)
            self.linear_y = nn.Linear(in_features=y_features, out_features=y_out_features)

        elif fusion_type == 'cross_attention':
            # 注意力机制
            self.cross_attn = CrossAttention(
                n_heads=n_heads,
                d_embed=d_embed,
                d_cross=d_cross,
            )

    def forward(self, x, y):
        if self.fusion_type == 'concat':
            z = torch.cat([x, y], dim=1)
            z = self.linear(z)
        elif self.fusion_type == 'linear_concat':
            x = self.linear_x(x)
            y = self.linear_y(y)
            z = torch.cat([x, y], dim=1)
        elif self.fusion_type == 'cross_attention':
            z = self.cross_attn(x, y)
        return z



class MyBassetFeatureMatrixFusion(nn.Module):
    """
    input:  seq (batch_size, 4, seq_length)
            feature (batch_size, num_cell_types, num_features)
    output: label (batch_size, num_cell_types)
    """
    def __init__(
            self, 
            input_seq_length=200,
            input_feature_dim=4,
            output_dim=1,

            sigmoid=False,
            squeeze=True,

            fusion_type='linear_concat',

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
        self.input_feature_dim  = input_feature_dim
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

        # compute the shape
        with torch.no_grad():
            test_input = torch.zeros(1, 4, self.input_seq_length)
            test_output = self.conv_layers(test_input)
            hidden_dim = test_output[0].reshape(-1).shape[0]
        

        self.fusion_layer = FusionLayer(
            x_features=hidden_dim,
            y_features=self.input_feature_dim,
            output_features=hidden_dim,
            fusion_type=fusion_type,
        )

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
            seq, feature = inputs['seq'], inputs['feature']
        elif isinstance(inputs, (list, tuple)):
            seq, feature = inputs[0], inputs[1]
        else:
            raise ValueError('inputs type must be dict or list or tuple')
        
        if seq.shape[2] == 4:
            seq = seq.permute(0, 2, 1)

        seq_embed = self.conv_layers(seq)
        seq_embed = seq_embed.view(seq_embed.size(0), -1)

        outputs = []
        for i in range(feature.shape[1]):
            feature_i = feature[:, i, :]  # Extract features for cell type i
            x = self.fusion_layer(seq_embed, feature_i)
            # x = torch.cat([seq_embed, feature_i], dim=1)  # Concatenate sequence embedding with features
            x = self.linear_layers(x)
            if self.sigmoid:
                x = self.sigmoid_layer(x)
            if self.squeeze:
                x = x.squeeze(-1)
            outputs.append(x)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, num_cell_types)
        return outputs



if __name__ == '__main__':

    fusion = FusionLayer(x_features=10, y_features=10, fusion_type='linear_concat')
    x = torch.randn(2, 10)
    y = torch.randn(2, 10)
    z = fusion(x, y)
    print(z.shape)


    yaml_str = '''
    model:
        type: 
            MyBassetFeatureMatrix
        args:
            input_seq_length:       200
            input_feature_dim:      4
            output_dim:             1

            fusion_type:            'linear_concat'

            conv_channels_list:     [256,256,256,256,256,256]
            conv_kernel_size_list:  [5,5,5,5,5,5]
            conv_padding_list:      [2,2,2,2,2,2]
            pool_kernel_size_list:  [2,2,2,2,2,2]
            pool_padding_list:      [0,0,0,0,0,0]
            conv_dropout_rate:      0.2

            linear_channels_list:   [1024]
            linear_dropout_rate:    0.5
    '''
    yaml = YAML()
    config = yaml.load(yaml_str)
    model = MyBassetFeatureMatrixFusion(**config['model']['args'])

    seq = torch.randn(2, 4, 200)
    feature = torch.randn(2, 5, 4)
    inputs = {'seq': seq, 'feature': feature}
    
    torchinfo.summary(model, input_data=[inputs])

    out = model(inputs)
    print(out.shape)