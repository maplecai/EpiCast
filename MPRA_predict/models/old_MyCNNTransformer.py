import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict
# from .MyBasset import ConvBlock, LinearBlock
# from .MyMTLucifer import TransformerBlock
# from .MyBassetMultiTask import TaskEncoder
import einops
from rotary_embedding_torch import RotaryEmbedding


class ConvBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            gn_group_size = 16,):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # 是gelu不是relu，非常重要！
        self.gelu = nn.GELU()
        if gn_group_size is None:
            self.gn = nn.BatchNorm1d(out_channels)
        else:
            assert out_channels % gn_group_size == 0
            gn_num_groups = out_channels // gn_group_size
            self.gn = nn.GroupNorm(gn_num_groups, out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.gelu(out)
        out = self.gn(out)
        return out

class LinearBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels,):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x



class SelfAttention(nn.Module):
    def __init__(self, d_embed, n_heads, dropout_rate=0.1, use_position_embedding=True):
        super().__init__()
        assert d_embed % n_heads == 0
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

    def forward(self, x: torch.Tensor, mask=None):
        batch_size, seq_len, d_embed = x.shape
        interim_shape = batch_size, seq_len, self.n_heads, self.d_head
        q = self.q_linear(x).view(interim_shape).transpose(1, 2)
        k = self.k_linear(x).view(interim_shape).transpose(1, 2)
        v = self.v_linear(x).view(interim_shape).transpose(1, 2)

        if self.use_position_embedding:
            # q and k have shape (batch_size, n_heads, seq_len, d_head)
            q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=2)
            k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=2)

        #weight = einops.einsum(q, k, 'b h q d, b h k d -> b h q k') / np.sqrt(self.d_head)
        weight = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        #output = einops.einsum(weight, v, 'b h q k, b h k d -> b h q d')
        output = torch.matmul(weight, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, d_embed)
        output = self.out_linear(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_embed, n_heads, d_mlp, dropout_rate=0.1, bias=False, use_position_embedding=True):
        super().__init__()

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.dropout_rate = dropout_rate
        self.use_position_embedding = use_position_embedding

        self.attn = SelfAttention(d_embed, n_heads, dropout_rate, use_position_embedding)
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


class MyCNNTransformer(nn.Module):
    def __init__(
            self, 
            input_length=230, 
            output_dim=1, 

            conv_channels_list=[256, 256, 256],
            conv_kernel_size_list=[7, 7, 7],
            conv_padding_list=[3, 3, 3],
            pool_kernel_size_list=[2, 4, 5],
            pool_padding_list=[0, 0, 0],
            conv_dropout_rate=0.1, 

            num_trans_blocks=5, 
            trans_d_embed=256, 
            trans_n_heads=8, 
            trans_d_mlp=256,
            trans_dropout_rate=0.1,

            linear_channels_list=[],
            linear_dropout_rate=0.5,
            sigmoid=False,

            ):
        super().__init__()

        self.input_length = input_length
        self.output_dim   = output_dim

        self.cls_embedding_layer = nn.Embedding(1, trans_d_embed)
        # 初始化可能很重要！
        nn.init.normal_(self.cls_embedding_layer.weight, mean=0.0, std=0.02)

        if conv_padding_list is None:
            conv_padding_list = [0] * len(conv_kernel_size_list)
        if pool_padding_list is None:
            pool_padding_list = [0] * len(pool_kernel_size_list)
        
        self.conv_layers = nn.Sequential(OrderedDict([]))
        self.linear_layers = nn.Sequential(OrderedDict([]))

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
                    padding=pool_padding_list[i]))
            
            if i != len(conv_kernel_size_list) - 1:
                self.conv_layers.add_module(
                    f'conv_dropout_{i}', nn.Dropout(conv_dropout_rate))

        self.trans_layers = nn.Sequential(OrderedDict([]))
        for i in range(num_trans_blocks):
            self.trans_layers.add_module(
                f'transformer_block_{i}', TransformerBlock(
                    d_embed=trans_d_embed, 
                    n_heads=trans_n_heads, 
                    d_mlp=trans_d_mlp, 
                    dropout_rate=trans_dropout_rate))

        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_block_{i}', LinearBlock(
                    in_channels=trans_d_embed if i == 0 else linear_channels_list[i-1], 
                    out_channels=linear_channels_list[i]))
        
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate))

        self.linear_layers.add_module(
            f'linear_last', nn.Linear(
                in_features=trans_d_embed if len(linear_channels_list) == 0 else linear_channels_list[-1], 
                out_features=output_dim))

        if sigmoid == True:
            self.linear_layers.add_module(f'sigmoid', nn.Sigmoid())


    def _forward(self, x, *args, **kwargs):
        cls_idx = torch.zeros((x.size(0), 1), dtype=torch.long, device=x.device)
        cls_embedding = self.cls_embedding_layer(cls_idx)

        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x = torch.concat([cls_embedding, x], dim=1)
        x = self.trans_layers(x)
        x = x[:, 0]
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.squeeze(-1)
        return x
    

    def forward(self, x, *args, **kwargs):
        if x.shape[2] == 4:
            x = x.permute(0, 2, 1)

        if self.augmentation == False:
            x = self._forward(x, *args, **kwargs)

        else:
            if self.augmentation_region is None:
                x_aug = torch.flip(x, dims=[1,2])
            else:
                left, right = self.augmentation_region
                x_aug = torch.flip(x[:, :, left:right], dims=[1,2])
                x_aug = torch.cat((x[:, :, :left], x_aug, x[:, :, right:]), dim=2)
            x = (self._forward(x, *args, **kwargs) + self._forward(x_aug, *args, **kwargs)) / 2
        return x




class MyCNNTransformerMultiTask(MyCNNTransformer):
    def __init__(
            self, 
            n_seqtype,
            n_celltype,
            pred_tokens = 1,
            # tokenizer_kwargs,
            *args,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)

        self.n_seqtype = n_seqtype
        self.n_celltype = n_celltype
        self.pred_tokens = pred_tokens

        d_embed = self.trans_layers[0].d_embed

        # self.token_embedding_layer = nn.Embedding(
        #     num_embeddings=(1+n_seqtype+n_celltype),
        #     embedding_dim=d_embed)
        self.cls_embedding_layer = nn.Embedding(1, d_embed)
        self.celltype_embedding_layer = nn.Embedding(n_celltype, d_embed)
        self.seqtype_embedding_layer = nn.Embedding(n_seqtype, d_embed)

        # 初始化可能很重要！
        nn.init.normal_(self.cls_embedding_layer.weight, mean=0.0, std=0.02)
        # nn.init.normal_(self.celltype_embedding_layer.weight, mean=0.0, std=0.02)
        # nn.init.normal_(self.seqtype_embedding_layer.weight, mean=0.0, std=0.02)

        # 改变用于预测的token数量
        self.linear_layers.linear_final = nn.Linear(in_features=d_embed*self.pred_tokens, out_features=self.output_dim)


    def _forward(self, x, *args, **kwargs):
        # cell_idx = kwargs.get('cell_idx', 0)
        # seq_idx  = kwargs.get('seq_idx', 0)

        cell_idx = args[0]
        seq_idx  = args[1]

        cls_idx = torch.full(size=(x.size(0), 1), fill_value=0, dtype=torch.long, device=x.device)
        # cell_idx = torch.full(size=(x.size(0), 1), value=0, dtype=torch.long, device=x.device)
        # seq_idx  = torch.full(size=(x.size(0), 1), value=0, dtype=torch.long, device=x.device)
        cell_idx = cell_idx.unsqueeze(1)
        seq_idx  = seq_idx.unsqueeze(1)
        # token_embeddings = self.token_embedding_layer(torch.cat((cls_idx, seq_idx, cell_idx), dim=0))
        token_embeddings = torch.concat([
            self.cls_embedding_layer(cls_idx), 
            self.seqtype_embedding_layer(seq_idx), 
            self.celltype_embedding_layer(cell_idx)], dim=1)

        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x = torch.concat([token_embeddings, x], dim=1)
        x = self.trans_layers(x)
        #x = x[:, 0]
        x = x[:, :self.pred_tokens]
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.squeeze(-1)
        return x





class MyCNNTransformerMultiOutput(MyCNNTransformer):
    def __init__(
            self, 
            n_seqtype,
            n_celltype,
            pred_tokens = 1,
            # tokenizer_kwargs,
            *args,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)

        self.n_seqtype = n_seqtype
        self.n_celltype = n_celltype
        self.pred_tokens = pred_tokens

        d_embed = self.trans_layers[0].d_embed

        # self.token_embedding_layer = nn.Embedding(
        #     num_embeddings=(1+n_seqtype+n_celltype),
        #     embedding_dim=d_embed)

        self.cls_embedding_layer = nn.Embedding(1, d_embed)
        self.celltype_embedding_layer = nn.Embedding(n_celltype, d_embed)
        self.seqtype_embedding_layer = nn.Embedding(n_seqtype, d_embed)

        # 初始化可能很重要！(但也可能不重要
        nn.init.normal_(self.cls_embedding_layer.weight, mean=0.0, std=0.02)
        # nn.init.normal_(self.celltype_embedding_layer.weight, mean=0.0, std=0.02)
        # nn.init.normal_(self.seqtype_embedding_layer.weight, mean=0.0, std=0.02)

        # 改变用于预测的token数量
        self.linear_layers.linear_final = nn.Linear(in_features=d_embed*self.pred_tokens, out_features=self.output_dim)


    def _forward(self, x, *args, **kwargs):
        
        cell_idx = args[0]
        seq_idx  = args[1]

        cls_idx = torch.full(size=(x.size(0), 1), fill_value=0, dtype=torch.long, device=x.device)
        # cell_idx = torch.full(size=(x.size(0), 1), value=0, dtype=torch.long, device=x.device)
        # seq_idx  = torch.full(size=(x.size(0), 1), value=0, dtype=torch.long, device=x.device)
        cell_idx = cell_idx.unsqueeze(1)
        seq_idx  = seq_idx.unsqueeze(1)
        # token_embeddings = self.token_embedding_layer(torch.cat((cls_idx, seq_idx, cell_idx), dim=0))
        token_embeddings = torch.concat([
            self.cls_embedding_layer(cls_idx), 
            self.seqtype_embedding_layer(seq_idx), 
            self.celltype_embedding_layer(cell_idx)], dim=1)

        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x = torch.concat([token_embeddings, x], dim=1)
        x = self.trans_layers(x)
        #x = x[:, 0]
        x = x[:, :self.pred_tokens]
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = x.squeeze(-1)
        return x















if __name__ == '__main__':
    model = MyCNNTransformerMultiTask(
        input_length=1000,
        output_dim=1,
        n_seqtype=2,
        n_celltype=2,
    )
    x = torch.randn(2, 1000, 4)
    cell_type = torch.randint(0, 2, (2,))
    seq_type = torch.randint(0, 2, (2,))

    torchinfo.summary(model, input_data=(x, cell_type, seq_type))