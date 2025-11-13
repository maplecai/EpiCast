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


class MyResTransformerLateFusion(nn.Module):
    def __init__(
        self, 
        input_seq_length=200,
        input_seq_channels=4,
        input_epi=False,
        input_epi_dim=0,
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

        num_trans_blocks=3, 
        trans_d_embed=256, 
        trans_n_heads=8, 
        trans_d_mlp=256,
        trans_dropout_rate=0.1,

        trans_output='seq_mean',
        trans_add_cls=False,

        linear_channels_list=None,
        linear_dropout_rate=0.5,
    ):
        super().__init__()

        self.input_seq_length   = input_seq_length
        self.input_seq_channels = input_seq_channels
        self.input_epi          = input_epi
        self.input_epi_dim      = input_epi_dim
        self.output_dim         = output_dim
        self.sigmoid            = sigmoid
        self.squeeze            = squeeze

        self.trans_output       = trans_output
        self.trans_add_cls      = trans_add_cls

        if conv_channels_list is None:
            conv_channels_list = []
        if linear_channels_list is None:
            linear_channels_list = []

        if trans_add_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_d_embed))


        # ---------- Conv ----------
        self.conv_layers = nn.Sequential(OrderedDict([]))
        self.conv_layers.add_module(
            'conv_block_first', ConvBlock(
                in_channels=input_seq_channels,
                out_channels=conv_first_channels, 
                kernel_size=conv_first_kernel_size, 
                stride=1,
                padding=conv_padding,
                layer_order='conv_relu_bn',
                activation=conv_activation,
            )
        )
        if pool_first_kernel_size != 1:
            self.conv_layers.add_module(
                'max_pool_first', nn.MaxPool1d(
                    kernel_size=pool_first_kernel_size, 
                    ceil_mode=True,
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
                        ceil_mode=True,
                    )
                )
            self.conv_layers.add_module(f'conv_dropout_{i}', nn.Dropout(conv_dropout_rate))

        if gap:
            self.conv_layers.add_module('gap_layer', nn.AdaptiveAvgPool1d(1))

        if conv_channels_list[-1] != trans_d_embed:
            self.conv_layers.add_module(
                'conv_reshape', nn.Conv1d(
                    in_channels=conv_channels_list[-1], 
                    out_channels=trans_d_embed, 
                    kernel_size=1
                )
            )

        # ---------- Transformer ----------
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


        # ---------- 推断线性层输入维度（用一致的前向路径） ----------
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_seq_channels, self.input_seq_length)
            dummy_seq_tokens = self.conv_layers(dummy).permute(0, 2, 1)   # (1, L, H)
            dummy_seq_embed  = self.forward_trans_layers(dummy_seq_tokens) # (1, H)
        seq_embed_dim = dummy_seq_embed.shape[-1]
        total_input_dim = seq_embed_dim + (input_epi_dim if input_epi else 0)


        # ---------- Linear ----------
        self.linear_layers = nn.Sequential(OrderedDict([]))
        current_dim = total_input_dim
        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_{i}', nn.Linear(current_dim, linear_channels_list[i])
            )
            self.linear_layers.add_module(f'linear_activation_{i}', nn.ReLU())
            self.linear_layers.add_module(f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate))
            current_dim = linear_channels_list[i]
        self.linear_layers.add_module('linear_last', nn.Linear(current_dim, output_dim))

        self.sigmoid_layer = nn.Sigmoid()


    # ---------- Transformer 前向（支持 seq_mean / cls） ----------
    def forward_trans_layers(self, seq_tokens: torch.Tensor):
        """
        seq_tokens: (B, L, H)
        输出: (B, H)
        """
        batch_size = seq_tokens.shape[0]
        if self.trans_add_cls:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_token, seq_tokens], dim=1)
            cls_len = 1
        else:
            tokens = seq_tokens
            cls_len = 0

        out = self.trans_layers(tokens)

        if self.trans_output == 'cls':
            if not self.trans_add_cls:
                raise ValueError(f"Invalid {self.trans_output = } without cls token.")
            out = out[:, 0]
        elif self.trans_output == 'seq_mean':
            start = cls_len
            out = out[:, start:].mean(1)
        else:
            raise ValueError(f"Unsupported trans_output mode: {self.trans_output}")

        return out


    # ---------- 整体前向（Late Fusion） ----------
    def forward(self, inputs: dict):
        seq = inputs.get('seq', None)
        epi = inputs.get('feature', None)

        # 预处理序列: (B, L, C) -> (B, C, L)
        if seq.shape[2] == self.input_seq_channels:
            seq = seq.permute(0, 2, 1)
        assert seq.shape[1:] == (self.input_seq_channels, self.input_seq_length), f"{seq.shape = }"

        # CNN -> tokens
        seq_tokens = self.conv_layers(seq).permute(0, 2, 1)  # (B, L, H)

        # Transformer 编码 + 聚合
        seq_embed = self.forward_trans_layers(seq_tokens)     # (B, H)

        # --------- Late Fusion with EPI ---------
        if epi is None:
            fused_embed = seq_embed
            out = self.linear_layers(fused_embed)

        elif epi.ndim == 2:
            # 直接拼接原始 epi 特征（维度 D=input_epi_dim）
            fused_embed = torch.cat([seq_embed, epi], dim=-1)
            out = self.linear_layers(fused_embed)

        elif epi.ndim == 3:
            # 多条件 EPI: 对每个条件独立预测
            B, C, D = epi.shape
            out = seq_embed.new_zeros(B, C, self.output_dim)
            for c in range(C):
                epi_c = epi[:, c, :]  # (B, D)
                fused_embed_c = torch.cat([seq_embed, epi_c], dim=-1)  # (B, H + D)
                out_c = self.linear_layers(fused_embed_c)              # (B, output_dim)
                out[:, c, :] = out_c
            out = out.squeeze(-1) # (B, C)

        else:
            raise ValueError(f"Unsupported epi dimensions: {epi.shape}")

        if self.sigmoid:
            out = self.sigmoid_layer(out)
        if self.squeeze:
            out = out.squeeze(-1)
        return out

if __name__ == '__main__':
    pass
