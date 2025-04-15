import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict
from ruamel.yaml import YAML
yaml = YAML()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


# class Basset(nn.Module):
#     """
#     Basset model architecture.
#     """
#     def __init__(
#             self, 
#             input_seq_length=200, 
#             output_dim=1, 
#             conv1_channels=300, 
#             conv1_kernel_size=19, 
#             conv2_channels=200, 
#             conv2_kernel_size=11, 
#             conv3_channels=200, 
#             conv3_kernel_size=7, 
#             pool1_kernel_size=2,
#             pool1_padding=0,
#             pool2_kernel_size=2,
#             pool2_padding=0,
#             pool3_kernel_size=2,
#             pool3_padding=0,
#             linear1_channels=1000, 
#             linear2_channels=1000, 
#             conv_dropout_rate=0.1, 
#             linear_dropout_rate=0.5,
#             gap_layer=False,
#             augmentation=False,
#             augmentation_region=[0, 0],
#             ):
#         """
#         Initialize Basset model.
#         """                                         
#         super().__init__()

#         self.input_seq_length = input_seq_length
#         self.output_dim   = output_dim
#         self.gap_layer    = gap_layer
#         self.augmentation = augmentation
#         self.augmentation_region = augmentation_region

#         self.conv_block1 = ConvBlock(
#             in_channels=4, 
#             out_channels=conv1_channels, 
#             kernel_size=conv1_kernel_size, 
#             stride=1, 
#             padding=0)
#         self.conv_block2 = ConvBlock(
#             in_channels=conv1_channels, 
#             out_channels=conv2_channels, 
#             kernel_size=conv2_kernel_size, 
#             stride=1, 
#             padding=0)
#         self.conv_block3 = ConvBlock(
#             in_channels=conv2_channels, 
#             out_channels=conv3_channels, 
#             kernel_size=conv3_kernel_size, 
#             stride=1, 
#             padding=0)
#         self.maxpool1 = nn.MaxPool1d(pool1_kernel_size, padding=pool1_padding)
#         self.maxpool2 = nn.MaxPool1d(pool2_kernel_size, padding=pool2_padding)
#         self.maxpool3 = nn.MaxPool1d(pool3_kernel_size, padding=pool3_padding)
#         self.dropout_1d1 = nn.Dropout(conv_dropout_rate)
#         self.dropout_1d2 = nn.Dropout(conv_dropout_rate)
#         # self.dropout_1d1 = nn.Dropout1d(conv_dropout_rate)
#         # self.dropout_1d2 = nn.Dropout1d(conv_dropout_rate)
#         if gap_layer == True:
#             self.gap1 = nn.AdaptiveAvgPool1d(1)

#         self.dropout1 = nn.Dropout(linear_dropout_rate)
#         self.linear_block1 = LinearBlock(None, linear1_channels)
#         self.dropout2 = nn.Dropout(linear_dropout_rate)
#         self.linear_block2 = LinearBlock(linear1_channels, linear2_channels)
#         self.dropout3 = nn.Dropout(linear_dropout_rate)
#         self.linear3 = nn.Linear(linear2_channels, self.output_dim)

#         # self.dropout4 = nn.Dropout(conv_dropout_rate)
#         # self.dropout5 = nn.Dropout(conv_dropout_rate)

#     def _forward(self, x):
#         x = self.conv_block1(x)
#         x = self.maxpool1(x)
#         x = self.dropout_1d1(x)
#         x = self.conv_block2(x)
#         x = self.maxpool2(x)
#         x = self.dropout_1d2(x)
#         x = self.conv_block3(x)
#         x = self.maxpool3(x)
#         if self.gap_layer == True:
#             x = self.gap1(x)

#         x = x.view(x.size(0), -1)
#         x = self.dropout1(x)
#         x = self.linear_block1(x)
#         x = self.dropout2(x)
#         x = self.linear_block2(x)
#         x = self.dropout3(x)
#         x = self.linear3(x)

#         return x
        
        
#     def forward(self, x):
#         if self.augmentation == False:
#             x = self._forward(x)
#         else:
#             left, right = self.augmentation_region
#             x_aug = torch.flip(x[:, :, left:right], dims=[1,2])
#             x_aug = torch.cat((x[:, :, :left], x_aug, x[:, :, right:]), dim=2)
#             x = (self._forward(x) + self._forward(x_aug)) / 2
#         return x


# class MyBassetEncoder(nn.Module):
#     def __init__(
#             self, 
#             # input_seq_length=230, 
#             # output_dim=1, 
#             conv_channels_list=None,
#             conv_kernel_size_list=None,
#             conv_padding_list=None,
#             pool_kernel_size_list=None,
#             pool_padding_list=None,
#             conv_dropout_rate=0.1,
#             gap_layer=False,
#     ):
#         super().__init__()

#         # if conv_padding_list is None:
#         #     conv_padding_list = [kernel_size // 2 for kernel_size in conv_kernel_size_list]
#         if conv_padding_list is None:
#             conv_padding_list = [0] * len(conv_kernel_size_list)
#         if pool_padding_list is None:
#             pool_padding_list = [0] * len(pool_kernel_size_list)
        
#         self.conv_layers = nn.Sequential(OrderedDict([]))

#         for i in range(len(conv_kernel_size_list)):
#             self.conv_layers.add_module(
#                 f'conv_block_{i}', ConvBlock(
#                     in_channels=4 if i == 0 else conv_channels_list[i-1], 
#                     out_channels=conv_channels_list[i], 
#                     kernel_size=conv_kernel_size_list[i], 
#                     stride=1, 
#                     padding=conv_padding_list[i]))
                
#             self.conv_layers.add_module(
#                 f'max_pool_{i}', nn.MaxPool1d(
#                     kernel_size=pool_kernel_size_list[i], 
#                     padding=pool_padding_list[i],
#                     ceil_mode = True))
        
#             self.conv_layers.add_module(
#                 f'conv_dropout_{i}', nn.Dropout(p=conv_dropout_rate))
        
#         if gap_layer:
#             self.conv_layers.add_module(
#                 'gap_layer', nn.AdaptiveAvgPool1d(1))
        
        
#     def forward(self, x):
#         x = self.conv_layers(x)
#         return x



# class MyBassetDecoder(nn.Module):
#     def __init__(
#             self, 
#             input_dim,
#             output_dim=None,
#             linear_channels_list=[],
#             linear_dropout_rate=0.5,
#             last_linear_layer=False,
#             sigmoid=False,
#     ):
#         super().__init__()
        
#         self.linear_layers = nn.Sequential(OrderedDict([]))

#         for i in range(len(linear_channels_list)):
#             self.linear_layers.add_module(
#                 f'linear_block_{i}', LinearBlock(
#                     in_channels=input_dim if i == 0 else linear_channels_list[i-1], 
#                     out_channels=linear_channels_list[i]))
        
#             self.linear_layers.add_module(
#                 f'linear_dropout_{i}', nn.Dropout(p=linear_dropout_rate))
        
#         if last_linear_layer == True:
#             self.linear_layers.add_module(
#                 f'last_linear', nn.Linear(
#                     in_features=input_dim if len(linear_channels_list) == 0 else linear_channels_list[-1], 
#                     out_features=output_dim,))

#         if sigmoid == True:
#             self.linear_layers.add_module(f'sigmoid', nn.Sigmoid())

#     def forward(self, x):
#         x = self.linear_layers(x)
#         return x



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