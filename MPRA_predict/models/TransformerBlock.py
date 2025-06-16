import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from rotary_embedding_torch import RotaryEmbedding


# class ConvBlock(nn.Module):
#     def __init__(
#             self, 
#             in_channels, 
#             out_channels, 
#             kernel_size, 
#             stride, 
#             padding, 
#             gn_group_size = 16,):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
#         # 是gelu不是relu，非常重要！?
#         self.gelu = nn.GELU()
#         if gn_group_size is None:
#             self.gn = nn.BatchNorm1d(out_channels)
#         else:
#             assert out_channels % gn_group_size == 0
#             gn_num_groups = out_channels // gn_group_size
#             self.gn = nn.GroupNorm(gn_num_groups, out_channels)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.gelu(out)
#         out = self.gn(out)
#         return out


# class LinearBlock(nn.Module):
#     def __init__(
#             self, 
#             in_channels, 
#             out_channels, 
#         ):
#         super(LinearBlock, self).__init__()
#         self.linear = nn.Linear(in_channels, out_channels)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.linear(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         return out


class SelfAttention(nn.Module):
    def __init__(self, d_embed, n_heads, dropout_rate=0.1, use_position_embedding=True):
        super().__init__()
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.dropout_rate = dropout_rate
        self.use_position_embedding = use_position_embedding

        if self.use_position_embedding:
            self.rotary_emb = RotaryEmbedding(dim=self.d_head)

        self.q_linear = nn.Linear(d_embed, d_embed)
        self.k_linear = nn.Linear(d_embed, d_embed)
        self.v_linear = nn.Linear(d_embed, d_embed)
        self.out_linear = nn.Linear(d_embed, d_embed)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, d_embed = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        q = self.q_linear(x).view(interim_shape).transpose(1, 2)
        k = self.k_linear(x).view(interim_shape).transpose(1, 2)
        v = self.v_linear(x).view(interim_shape).transpose(1, 2)

        if self.use_position_embedding:
            # q.shape = k.shape = (batch_size, n_heads, seq_len, d_head)
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=2)

        attn_scores = torch.einsum('b h q d, b h k d -> b h q k', q, k) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum('b h q k, b h k d -> b h q d', attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_embed)
        output = self.out_linear(attn_output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_embed, n_heads, d_mlp, dropout_rate=0.1, bias=False, use_position_embedding=True):
        super().__init__()

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.dropout_rate = dropout_rate
        self.use_position_embedding = use_position_embedding

        self.attn = SelfAttention(
            d_embed, 
            n_heads, 
            dropout_rate, 
            use_position_embedding
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_embed),
        )

        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, input):

        input = self.layer_norm1(input)
        attn_output = self.attn(input)
        attn_output = self.dropout1(attn_output)
        mlp_input = attn_output + input

        mlp_input = self.layer_norm2(mlp_input)
        mlp_output = self.mlp(mlp_input)
        mlp_output = self.dropout2(mlp_output)
        output = mlp_output + mlp_input

        return output





if __name__ == '__main__':
    import yaml
    import torchinfo
    from . import models, utils










    model_config = '''
        model:
            type: MyCNNTransformer
            args:
                input_seq_length: 200
                input_feature_shape: null
                output_dim: 5
                sigmoid: false
                squeeze: true

                conv_channels_list: [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                256, 256, 256, 256, 256, 256]
                conv_kernel_size_list: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
                conv_padding_list: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                pool_kernel_size_list: [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
                conv_dropout_rate: 0.2
                global_average_pooling: false
                
                
                num_trans_blocks: 3
                trans_d_embed: 256
                trans_n_heads: 4
                trans_d_mlp: 256
                trans_dropout_rate: 0.2
                
                linear_channels_list: []
                linear_dropout_rate: 0.5
    '''
    model_config = yaml.load(model_config, Loader=yaml.FullLoader)
    model = utils.init_obj(models, model_config['model'])
    seq = torch.zeros(2, 200, 4)
    inputs = {'seq': seq}
    torchinfo.summary(model, input_data=(inputs,))
    out = model(inputs)
    print(out.shape)



















    model_config = '''
        model:
            type: MyCNNTransformer
            args:
                input_seq_length: 200
                input_feature_shape: [4]
                output_dim: 5
                sigmoid: false
                squeeze: true

                conv_channels_list: [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                256, 256, 256, 256, 256, 256]
                conv_kernel_size_list: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
                conv_padding_list: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                pool_kernel_size_list: [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
                conv_dropout_rate: 0.2
                global_average_pooling: false
                
                
                num_trans_blocks: 3
                trans_d_embed: 256
                trans_n_heads: 4
                trans_d_mlp: 256
                trans_dropout_rate: 0.2
                
                linear_channels_list: []
                linear_dropout_rate: 0.5
    '''
    model_config = yaml.load(model_config, Loader=yaml.FullLoader)
    model = utils.init_obj(models, model_config['model'])
    seq = torch.zeros(2, 200, 4)
    feature = torch.zeros(2, 4)
    inputs = {'seq': seq, 'feature': feature}
    torchinfo.summary(model, input_data=(inputs,))
    out = model(inputs)
    print(out.shape)





    model_config = '''
        model:
            type: MyCNNTransformer
            args:
                input_seq_length: 200
                input_feature_shape: [5,4]
                output_dim: 1
                sigmoid: false
                squeeze: true

                conv_channels_list: [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                256, 256, 256, 256, 256, 256]
                conv_kernel_size_list: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
                conv_padding_list: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                pool_kernel_size_list: [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
                conv_dropout_rate: 0.2
                global_average_pooling: false
                
                num_trans_blocks: 3
                trans_d_embed: 256
                trans_n_heads: 4
                trans_d_mlp: 256
                trans_dropout_rate: 0.2
                
                linear_channels_list: []
                linear_dropout_rate: 0.5
    '''
    model_config = yaml.load(model_config, Loader=yaml.FullLoader)
    model = utils.init_obj(models, model_config['model'])
    seq = torch.zeros(2, 200, 4)
    feature = torch.zeros(2, 5, 4)
    inputs = {'seq': seq, 'feature': feature}
    torchinfo.summary(model, input_data=(inputs,))
    out = model(inputs)
    print(out.shape)
