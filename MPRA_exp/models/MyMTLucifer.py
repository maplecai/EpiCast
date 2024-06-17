import os
import numpy as np
import pdb
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

import torchinfo
import einops
from rotary_embedding_torch import RotaryEmbedding

from transformers import AutoTokenizer, AutoModel
from enformer_pytorch import Enformer as BaseEnformer

from tltorch import TRL

from torch.nn import Transformer, TransformerEncoderLayer, TransformerEncoder

from .MyBasset import ConvBlock, LinearBlock


np.random.seed(97)
torch.manual_seed(97)


class CNNBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding="same", 
            dilation=1, 
            bias=True, 
            gn_num_groups=None, 
            gn_group_size=16,
            dropout=0.1
            ):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
                             stride=stride, padding=padding, dilation=dilation, bias=bias)
        if gn_num_groups is None:
            gn_num_groups = out_channels // gn_group_size
        self.gn = nn.GroupNorm(gn_num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        seq = inputs
        x = self.gn(F.gelu(self.cnn(seq)))
        x = self.dropout(x)
        
        return x
    
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, gn_num_groups=None, gn_group_size=16):
#         super().__init__()

#         stride_for_conv1_and_shortcut = 1

#         if in_channels != out_channels:
#             stride_for_conv1_and_shortcut = 2

#         padding = kernel_size // 2

#         if gn_num_groups is None:
#             gn_num_groups = out_channels // gn_group_size

#         # modules for processing the input
#         self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride_for_conv1_and_shortcut, padding = padding, bias=False)
#         self.gn1 = nn.GroupNorm(gn_num_groups, out_channels)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv1d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = "same", bias=False)
#         self.gn2 = nn.GroupNorm(gn_num_groups, out_channels)
#         self.relu2 = nn.ReLU(inplace=True)

#         # short cut connections
#         self.shortcut = nn.Identity()
#         if in_channels != out_channels:
#             self.shortcut = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride_for_conv1_and_shortcut, bias=False)

#     def forward(self, xl):
#         input = self.shortcut(xl)

#         xl = self.relu1(self.gn1(self.conv1(xl)))
#         xl = self.conv2(xl)

#         xlp1 = input + xl

#         xlp1 = self.relu2(self.gn2(xlp1))

#         return xlp1
    
class TransformerBlock(nn.Module):
    def __init__(self, d_embed, n_heads, d_mlp, dropout=0.1, use_position_embedding=True):
        super().__init__()
        assert d_embed % n_heads == 0
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.d_mlp = d_mlp
        self.dropout = dropout
        self.use_position_embedding = use_position_embedding

        # self.in_proj = nn.Linear(d_embed, 3 * d_embed)
        # self.out_proj = nn.Linear(d_embed, d_embed)

        self.k_proj = nn.Linear(d_embed, d_embed, bias=False)
        self.q_proj = nn.Linear(d_embed, d_embed, bias=False)
        self.v_proj = nn.Linear(d_embed, d_embed, bias=False)

        if self.use_position_embedding:
            self.rotary_emb = RotaryEmbedding(dim=self.d_head)

        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.dropout1 = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=False)

        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(self.d_embed)
        
        self.fc2 = nn.Linear(d_embed, d_mlp)
        self.fc3 = nn.Linear(d_mlp, d_embed)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, input):
        input_shape = input.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        x = self.layer_norm1(input)
        xk = self.k_proj(x)
        xq = self.q_proj(x)
        xv = self.v_proj(x)

        xk = xk.reshape(interim_shape)
        xq = xq.reshape(interim_shape)
        xv = xv.reshape(interim_shape)

        if self.use_position_embedding:
            # make xq and xk have shape (batch_size, n_heads, seq_len, d_embed // n_heads)
            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
            xq = self.rotary_emb.rotate_queries_or_keys(xq, seq_dim=2)
            xk = self.rotary_emb.rotate_queries_or_keys(xk, seq_dim=2)
            # make xq and xk have shape (batch_size, seq_len, n_heads, d_embed // n_heads)
            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
        
        weight = einops.einsum(xq, xk, '... q h d, ... k h d -> ... h q k')
        weight = weight / np.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout1(weight)

        output = einops.einsum(weight, xv, '... h q k, ... k h d -> ... q h d')
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        output = self.dropout2(output)

        mlp_input = output + input
        x = self.layer_norm2(mlp_input)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = x + mlp_input

        return x


class MyMTLucifer(nn.Module):
    def __init__(
            self, 
            input_length=200, 
            output_dim=1, 
            num_conv_blocks=3, 
            conv_channels_list=[4, 256, 512], 
            d_embed=1024, 
            num_trans_blocks=5, 
            n_heads=8, 
            d_mlp=1024,
            dropout=0.1,
        ):
        super().__init__()
        self.input_length = input_length
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.d_mlp = d_mlp

        self.cls_token_embedding = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, 1, d_embed)))

        conv_channels_list.append(d_embed)
        self.promoter_cnn = nn.Sequential(OrderedDict([]))
        for i in range(num_conv_blocks):
            self.promoter_cnn.add_module(f'cnn_block_{i}', CNNBlock(
                in_channels=conv_channels_list[i], 
                out_channels=conv_channels_list[i+1], 
                kernel_size=5, 
                stride=1))

        self.promoter_transformer = nn.Sequential(OrderedDict([]))
        for i in range(num_trans_blocks):
            self.promoter_transformer.add_module(f'transformer_block_{i}', 
                TransformerBlock(d_embed=d_embed, n_heads=n_heads, d_mlp=d_mlp, dropout=dropout))

        self.output_linear = nn.Linear(d_embed, 1)
        
    def forward(self, seq, *args, **kwargs):
        if seq.shape[2] == 4:
            seq = seq.permute(0, 2, 1)
        seq = self.promoter_cnn(seq)
        seq = seq.permute(0, 2, 1)
        # seq.shape: (batch_size, seq_len, d_embed)
        seq = torch.hstack([self.cls_token_embedding.expand(seq.shape[0], -1, -1), seq])
        out = self.promoter_transformer(seq)
        out = out[:, 0]
        out = self.output_linear(out)
        out = out.squeeze(-1)
        return out





class MyMTLuciferMultiTask(nn.Module):
    def __init__(
            self, 
            input_length=200, 
            output_dim=1, 
            n_celltype = 2,
            n_output = 2,
            num_conv_blocks=3, 
            conv_channels_list=[4, 256, 512], 
            num_trans_blocks=5, 
            d_embed=1024, 
            n_heads=8, 
            d_mlp=1024,
            dropout=0.1,
        ):
        super().__init__()
        # self.input_length = input_length
        # self.output_dim = output_dim
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.d_mlp = d_mlp

        # self.cls_embedding = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, 1, d_embed)))
        self.cls_embedding_layer = nn.Embedding(1, d_embed)
        self.celltype_embedding_layer = nn.Embedding(n_celltype, d_embed)
        self.output_embedding_layer = nn.Embedding(n_output, d_embed)

        conv_channels_list.append(d_embed)
        self.promoter_cnn = nn.Sequential(OrderedDict([]))
        for i in range(num_conv_blocks):
            self.promoter_cnn.add_module(f'cnn_block_{i}', CNNBlock(
                in_channels=conv_channels_list[i], 
                out_channels=conv_channels_list[i+1], 
                kernel_size=5, 
                stride=1,
                dropout=dropout))

        self.promoter_transformer = nn.Sequential(OrderedDict([]))
        for i in range(num_trans_blocks):
            self.promoter_transformer.add_module(f'transformer_block_{i}', 
                TransformerBlock(d_embed=d_embed, n_heads=n_heads, d_mlp=d_mlp, dropout=dropout))
                # nn.TransformerEncoderLayer(d_model=d_embed, nhead=n_heads, dim_feedforward=d_mlp, dropout=dropout)) # no rotary position embedding

        self.output_linear = nn.Linear(d_embed, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, seq, cell_idx, output_idx, *args, **kwargs):
        if seq.shape[2] == 4:
            seq = seq.permute(0, 2, 1)
        seq = self.promoter_cnn(seq)
        seq = seq.permute(0, 2, 1)
        # seq.shape: (batch_size, seq_len, d_embed)

        cls_token = torch.zeros((seq.shape[0]), dtype=torch.long, device=seq.device)
        cls_embedding = self.cls_embedding_layer(cls_token).unsqueeze(1)
        celltype_embedding = self.celltype_embedding_layer(cell_idx).unsqueeze(1)
        output_embedding = self.output_embedding_layer(output_idx).unsqueeze(1)

        seq = torch.concat([cls_embedding, celltype_embedding, output_embedding, seq], dim=1)
        out = self.promoter_transformer(seq)
        out = out[:, 0]
        out = self.output_linear(out)
        out = out.squeeze(-1)
        return out



if __name__ == '__main__':
    model = MyMTLucifer(conv_channels_list=[4,256,256], d_embed=256, d_mlp=256, )

    x = torch.randn(2, 200, 4)
    torchinfo.summary(model, input_data=x)

    out = model(x)
    print(out.shape)

    model = MyMTLuciferMultiTask(conv_channels_list=[4,256,256], d_embed=256, d_mlp=256, )

    x = torch.randn(2, 200, 4)
    cell_idx = torch.randint(0, 2, (2,))
    output_idx = torch.randint(0, 2, (2,))
    torchinfo.summary(model, input_data=[x, cell_idx, output_idx])

    out = model(x, cell_idx, output_idx)
    print(out.shape)