import torch
import torch.nn as nn
import math

from .conv_block import ConvBlock, ResConvBlock
from .linear_block import LinearBlock, MLPBlock
from .trans_block import TransBlock
from .film import FiLM

class ConvTransformerBranch(nn.Module):
    def __init__(
        self, 
        input_len=200,
        input_channels=4,
        output_dim=1,
        target_length=None,
        last_activation=None,
        squeeze=True,

        num_conv_blocks=6,
        conv_channels=256,
        conv_kernel_size=3,
        conv_padding='same',
        conv_dropout_rate=0.2,
        pool_kernel_size=2,
        conv_layer_order='resnet_v2',
        conv_activation='relu',

        num_trans_blocks=3, 
        trans_d_embed=256, 
        trans_n_heads=8, 
        trans_d_mlp=256,
        trans_dropout_rate=0.1,

        trans_add_cls=False,
        trans_output_mode='seq_all',

        num_branches=1,

        num_linear_blocks=1,
        linear_channels=1024,
        linear_dropout_rate=0.5,
    ):
        super().__init__()

        self.input_len          = input_len
        self.input_channels     = input_channels
        self.output_dim         = output_dim

        self.target_length      = target_length
        self.squeeze            = squeeze

        self.trans_output_mode  = trans_output_mode
        self.trans_add_cls      = trans_add_cls
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_d_embed))

        current_shape = (input_channels, input_len)

        self.conv_layers = nn.Sequential()
        for i in range(num_conv_blocks):
            self.conv_layers.add_module(
                f'conv_block_{i}', ResConvBlock(
                    in_channels=input_channels if i == 0 else conv_channels, 
                    out_channels=conv_channels, 
                    kernel_size=conv_kernel_size, 
                    stride=1, 
                    padding=conv_padding,
                    activation=conv_activation,
                    layer_order=conv_layer_order,
                )
            )

            if pool_kernel_size != 1:
                self.conv_layers.add_module(
                    f'max_pool_{i}', nn.MaxPool1d(
                        kernel_size=pool_kernel_size, 
                        ceil_mode=True, # keep edge information
                    )
                )
            self.conv_layers.add_module(
                f'conv_dropout_{i}', nn.Dropout(conv_dropout_rate)
            )

        L = input_len
        for _ in range(num_conv_blocks):
            if pool_kernel_size != 1:
                L = math.ceil(L / pool_kernel_size)
        current_shape = (conv_channels, L)

        self.trans_layers = nn.Sequential()
        for i in range(num_trans_blocks):
            self.trans_layers.add_module(
                f'transformer_block_{i}', TransBlock(
                    d_embed=trans_d_embed, 
                    n_heads=trans_n_heads, 
                    d_mlp=trans_d_mlp, 
                    dropout_rate=trans_dropout_rate
                )
            )
        
        current_shape = (L, trans_d_embed)
        if trans_output_mode == 'seq_flatten':
            linear_in_channels = trans_d_embed * L
        else:
            linear_in_channels = trans_d_embed
        
        self.linear_layers = nn.Sequential()
        
        if num_branches == 1:
            self.linear_layers.add_module(
                f'mlp_block', 
                MLPBlock(
                    num_linear_blocks=num_linear_blocks,
                    in_channels=linear_in_channels,
                    hidden_channels=linear_channels,
                    out_channels=output_dim,
                    dropout_rate=linear_dropout_rate,
                    num_branches=1,
                )
            )
        else:
            self.linear_layers.add_module(
                f'mlp_block', 
                MLPBlock(
                    num_linear_blocks=num_linear_blocks,
                    in_channels=linear_in_channels,
                    hidden_channels=linear_channels,
                    out_channels=1,
                    dropout_rate=linear_dropout_rate,
                    num_branches=num_branches,
                )
            )

        if last_activation is None:
            self.last_activation_layer = nn.Identity()
        elif last_activation == 'sigmoid':
            self.last_activation_layer = nn.Sigmoid()
        elif last_activation == 'softmax':
            self.last_activation_layer = nn.Softmax()
        elif last_activation == 'softplus':
            self.last_activation_layer = nn.Softplus()
        else:
            raise ValueError(f"Unsupported last_activation mode: {last_activation}")


    def forward_conv(self, seq: torch.Tensor):
        seq = seq.permute(0, 2, 1)
        seq_emb = self.conv_layers(seq)
        seq_emb = seq_emb.permute(0, 2, 1)
        return seq_emb

    def forward_trans(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L, H = tokens.shape
        if self.trans_add_cls:
            tokens = torch.cat([self.cls_token.expand(B, 1, -1), tokens], 1)
        cls_len = int(self.trans_add_cls)

        out = self.trans_layers(tokens)

        if self.trans_output_mode == 'cls':
            out = out[:, 0]
        elif self.trans_output_mode == 'seq_avg':
            out = out[:, cls_len:].mean(1)
        elif self.trans_output_mode == 'seq_all':
            out = out[:, cls_len:]
        elif self.trans_output_mode == 'seq_flatten':
            out = out[:, cls_len:].reshape(B, -1)
        elif self.trans_output_mode == 'all':
            out = out
        else:
            raise ValueError(f"Unsupported trans_output_mode mode: {self.trans_output_mode}")

        return out

    def forward_linear(self, emb: torch.Tensor):
        out = self.linear_layers(emb)
        if self.squeeze:
            out = out.squeeze(-1)
        return out

    def forward(self, inputs: torch.Tensor | dict | list) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            seq = inputs
        elif isinstance(inputs, dict):
            seq = inputs.get('seq', None)
        elif isinstance(inputs, list):
            seq = inputs[0]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")
        
        batch_size = seq.shape[0]
        expected_shape = (batch_size, self.input_len, self.input_channels)
        assert seq.shape == expected_shape, f"{seq.shape = }, {expected_shape = }"

        emb = self.forward_conv(seq)
        emb = self.forward_trans(emb)
        out = self.forward_linear(emb)

        # if self.target_length is not None:
        #     start = (self.total_token_length - self.target_length) // 2
        #     end = start + self.target_length
        #     out = out[:, start:end]
        out = self.last_activation_layer(out)
        return out
