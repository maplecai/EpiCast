import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from collections import OrderedDict

from .. import models, utils
from .ConvBlock import ConvBlock, ResConvBlock #, LinearBlock
from .TransformerBlock import TransformerBlock



class MyResTransformer(nn.Module):
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
        trans_token_type_embedding=False,

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
        self.trans_token_type_embedding = trans_token_type_embedding

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
                layer_order=conv_layer_order.replace('_add', ''),
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
                f'res_conv_block_{i}', ResConvBlock( ###### ConvBlock
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
        if conv_channels_list[-1] != trans_d_embed:
            self.conv_layers.add_module(
                f'conv_reshape', nn.Conv1d(
                    in_channels=conv_channels_list[-1], 
                    out_channels=trans_d_embed, 
                    kernel_size=1
                )
            )
        # compute the shape

        if trans_add_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_d_embed))
        if input_epi:
            self.epi_embedding_layer = nn.Linear(input_epi_dim, trans_d_embed)

        if trans_token_type_embedding:
            self.token_type_embedding_layer = nn.Embedding(3, trans_d_embed)

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
            dummy = torch.zeros(1, self.input_seq_channels, self.input_seq_length)
            dummy = self.conv_layers(dummy) # (batch_size, conv_channels, seq_length)
            dummy = dummy.permute(0, 2, 1) # (batch_size, seq_length, hidden_dim)
            dummy = self.trans_layers(dummy)
            dummy = dummy.mean(1) # (batch_size, hidden_dim)

        self.linear_layers = nn.Sequential(OrderedDict([]))



        current_dim = dummy.shape[1]
        for i in range(len(linear_channels_list)):
            self.linear_layers.add_module(
                f'linear_{i}', nn.Linear(
                    in_features=current_dim, 
                    out_features=linear_channels_list[i],
                )
            )
            self.linear_layers.add_module(
                f'linear_activation_{i}', nn.ReLU()
            )
            self.linear_layers.add_module(
                f'linear_dropout_{i}', nn.Dropout(linear_dropout_rate)
            )
            current_dim = linear_channels_list[i]

        self.linear_layers.add_module(
            f'linear_last', nn.Linear(
                in_features=current_dim, 
                out_features=output_dim,
            )
        )

        self.sigmoid_layer = nn.Sigmoid()



    def forward_trans_layers(self, seq_tokens: torch.Tensor, epi_tokens: torch.Tensor=None):

        batch_size = seq_tokens.shape[0]
        hidden_dim = seq_tokens.shape[2]
        device= seq_tokens.device

        if self.trans_add_cls:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
        else:
            cls_token = torch.zeros((batch_size, 0, hidden_dim), device=device) # 维度是0的假tensor

        if not self.input_epi:
            epi_tokens = torch.zeros((batch_size, 0, hidden_dim), device=device) # 维度是0的假tensor

        seq_len = seq_tokens.shape[1]
        cls_len = cls_token.shape[1]
        epi_len = epi_tokens.shape[1]

        tokens = torch.cat([
            cls_token,
            seq_tokens,
            epi_tokens,
        ], dim=1)

        if self.trans_token_type_embedding:
            token_type_ids = torch.cat([
                torch.full((batch_size, cls_len), 2, dtype=torch.long, device=device), # CLS token
                torch.full((batch_size, seq_len), 0, dtype=torch.long, device=device), # Seq tokens
                torch.full((batch_size, epi_len), 1, dtype=torch.long, device=device), # Epi tokens
            ], dim=1)  # shape: [batch_size, cls_len + seq_len + epi_len]
            token_type_embed = self.token_type_embedding_layer(token_type_ids)
            tokens = tokens + token_type_embed
        
        out = self.trans_layers(tokens)

        if self.trans_output == 'cls':
            if not self.trans_add_cls:
                raise ValueError(f"Invalid {self.trans_output = } without cls token.")
            out = out[:, 0]
        elif self.trans_output == 'seq_mean':
            start = cls_len
            end = -epi_len if epi_len > 0 else None
            out = out[:, start:end].mean(1)
        else:
            raise ValueError(f"Unsupported trans_output mode: {self.trans_output}")

        return out


    def forward(self, inputs: dict):
        seq = inputs.get('seq', None)
        epi = inputs.get('feature', None)

        # --------- 预处理序列 ---------
        if seq.shape[2] == self.input_seq_channels:
            seq = seq.permute(0, 2, 1)
        assert seq.shape == (seq.shape[0], self.input_seq_channels, self.input_seq_length), f"{seq.shape = }"

        # --------- 没有表观特征 ---------
        if epi is None:
            seq_tokens = self.conv_layers(seq).permute(0, 2, 1)      # (B, L, H)
            out = self.forward_trans_layers(seq_tokens)

        # --------- epi 维度 = 2 ---------
        elif epi.ndim == 2:
            seq_tokens = self.conv_layers(seq).permute(0, 2, 1)      # (B, L, H)
            epi_tokens = self.epi_embedding_layer(epi).unsqueeze(1)  # (B, 1, H)
            out = self.forward_trans_layers(seq_tokens, epi_tokens)

        # --------- epi 维度 = 3：循环计算 ↓ ---------
        elif epi.ndim == 3:
            seq_tokens = self.conv_layers(seq).permute(0, 2, 1)      # (B, L, H)
            B, L, H = seq_tokens.shape
            B, C, D = epi.shape

            # 预分配输出张量，节省反复 cat 时的显存 & 时间
            out = seq_tokens.new_zeros(B, C, H)

            for c in range(C):
                epi_c = epi[:, c, :]                                 # (B, D)
                epi_tokens = self.epi_embedding_layer(epi_c).unsqueeze(1)  # (B, 1, H)
                out_c = self.forward_trans_layers(seq_tokens, epi_tokens)  # (B, H)
                out[:, c, :] = out_c                                 # 写入 (B, C, H)

        else:
            raise ValueError(f"Unsupported epi dimensions: {epi.shape}")

        # --------- 后续全连接 & 激活 ---------
        out = self.linear_layers(out)
        if self.sigmoid:
            out = self.sigmoid_layer(out)
        if self.squeeze:
            out = out.squeeze(-1)

        return out


# alternative name
MyCNNTransformer = MyResTransformer




if __name__ == '__main__':

    yaml_str = '''
model:
    type: MyResTransformer
    args:
        input_seq_length:       200
        input_seq_channels:     4
        input_epi_dim:      4
        input_epi_times:    5
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
    epi = torch.zeros(size=(1, 5, 4))
    inputs = {'seq': seq, 'epi': epi}

    torchinfo.summary(
        model, 
        input_data=(inputs,), 
        depth=6, 
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["var_names"],
    )