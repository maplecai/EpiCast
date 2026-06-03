import torch
import torch.nn as nn
from collections import OrderedDict

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
            raise ValueError(f'Inval activation:{self.activation}')
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
            raise ValueError(f'Inval layer_order:{self.layer_order}')
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
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)

        if layer_order in ['bn_relu_conv_add', 'resnet_v2']:
            self.bn1 = nn.BatchNorm1d(in_channels)
        else:
            self.bn1 = nn.BatchNorm1d(out_channels)

        self.bn2 = nn.BatchNorm1d(out_channels)

        if activation == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif activation == 'gelu':
            self.act1 = nn.GELU()
            self.act2 = nn.GELU()
        else:
            raise ValueError(f'Inval activation:{self.activation}')

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):
        if self.layer_order in ['conv_bn_add_relu', 'resnet_v1']: # resnet v1
            z = self.act1(self.bn1(self.conv1(x)))
            z = self.bn2(self.conv2(z))
            out = self.act2(z + self.shortcut(x))
        elif self.layer_order in ['bn_relu_conv_add', 'resnet_v2']: # resnet v2
            z = self.conv1(self.act1(self.bn1(x)))
            z = self.conv2(self.act2(self.bn2(z)))
            out = z + self.shortcut(x)
        elif self.layer_order == 'conv_relu_bn_add': # 之前没人提过，但是我实验的效果最好
            z = self.bn1(self.act1(self.conv1(x)))
            z = self.bn2(self.act2(self.conv2(z)))
            out = z + self.shortcut(x)
        else:
            raise ValueError(f'Inval layer_order:{self.layer_order}')
        return out




class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_seq_channels: int,
        num_conv_blocks: int,
        conv_channels_list: list[int],
        conv_kernel_size_list: list[int],
        conv_stride_list: list[int],
        conv_padding_list: list[int],
        pool_kernel_size_list: list[int],
        conv_dropout_rate: float = 0.0,
        conv_layer_order=None,
        conv_activation=None,
    ):
        super().__init__()

        if not (
            num_conv_blocks == len(conv_channels_list)
            == len(conv_kernel_size_list)
            == len(conv_stride_list)
            == len(conv_padding_list)
            == len(pool_kernel_size_list)
        ):
            raise ValueError("Inconsistent number of conv layers")

        layers = OrderedDict()
        conv_in_channels = input_seq_channels

        for i in range(num_conv_blocks):
            conv_out_channels = conv_channels_list[i]
            conv_kernel_size = conv_kernel_size_list[i]
            conv_stride = conv_stride_list[i]
            conv_padding = conv_padding_list[i]
            pool_kernel_size = pool_kernel_size_list[i]

            layers[f"conv_block_{i}"] = ResConvBlock(
                in_channels=conv_in_channels,
                out_channels=conv_out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
                activation=conv_activation,
                layer_order=conv_layer_order,
            )

            if pool_kernel_size != 1:
                layers[f"max_pool_{i}"] = nn.MaxPool1d(
                    kernel_size=pool_kernel_size,
                    ceil_mode=True,
                )

            if conv_dropout_rate > 0:
                layers[f"conv_dropout_{i}"] = nn.Dropout(conv_dropout_rate)

            conv_in_channels = conv_out_channels

        self.layers = nn.Sequential(layers)
        self.out_channels = conv_in_channels

    def forward(self, x):
        return self.layers(x)
