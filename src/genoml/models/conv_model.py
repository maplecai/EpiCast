import torch
import torch.nn as nn

from .conv_block import ConvBlock, ResConvBlock
from .linear_block import LinearBlock
from .trans_block import TransBlock
from .film import FiLM
from .crop import CenterCrop1d

class ConvModel(nn.Module):
    def __init__(
        self, 
        input_len=200,
        input_channels=4,
        output_dim=1,

        num_conv_blocks=6,
        conv_channels=256,
        conv_kernel_size=3,
        conv_padding='same',
        conv_dropout_rate=0.2,
        pool_kernel_size=2,
        conv_layer_order='resnet_v2',
        conv_activation='relu',

        global_avg_pooling=False,
        flatten=True,

        num_linear_blocks=1,
        linear_channels=1024,
        linear_dropout_rate=0.5,

        target_length=None,
        output_activation=None,
        squeeze=True,
    ):
        super().__init__()

        self.input_len          = input_len
        self.input_channels     = input_channels
        self.output_dim         = output_dim
        self.target_length      = target_length
        self.output_activation  = output_activation
        self.squeeze            = squeeze

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

        if global_avg_pooling:
            self.conv_layers.add_module(
                f'global_avg_pooling_0', nn.AdaptiveAvgPool1d(1)
            )

        if flatten:
            self.conv_layers.add_module(
                f'flatten_0', nn.Flatten()
            )

        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, input_len)
            current_shape = self.conv_layers(dummy_input).shape


        self.linear_layers = nn.Sequential()
        for i in range(num_linear_blocks):
            self.linear_layers.add_module(
                f'linear_{i}', 
                nn.Linear(
                    in_features=current_shape[-1] if i == 0 else linear_channels, 
                    out_features=linear_channels,
                )
            )
            self.linear_layers.add_module(
                f'act_{i}', 
                nn.ReLU(),
            )
            self.linear_layers.add_module(
                f'dropout_{i}', 
                nn.Dropout(linear_dropout_rate)
            )
        self.linear_layers.add_module(
            f'linear_{num_linear_blocks}', 
            nn.Linear(
                in_features=linear_channels if num_linear_blocks > 0 else current_shape[-1],
                out_features=output_dim,
            )
        )

            # f'mlp_block_0', 
            # MLPBlock(
            #     num_linear_blocks=num_linear_blocks,
            #     in_channels=current_shape[-1],
            #     hidden_channels=linear_channels,
            #     out_channels=output_dim,
            #     dropout_rate=linear_dropout_rate,
            # )


        if self.target_length is not None:
            self.linear_layers.add_module(
                f'center_crop_0', CenterCrop1d(target_length=self.target_length)
            )

        if output_activation is None:
            self.output_activation_layer = nn.Identity()
        elif output_activation == 'sigmoid':
            self.output_activation_layer = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.output_activation_layer = nn.Softmax(dim=-1)
        elif output_activation == 'softplus':
            self.output_activation_layer = nn.Softplus()
        else:
            raise ValueError(f"Unsupported output_activation mode: {output_activation}")



    def forward(self, inputs: torch.Tensor | dict | list) -> torch.Tensor:
        seq = self._parse_inputs(inputs)
        seq = seq.permute(0, 2, 1)
        emb = self.conv_layers(seq)
        out = self.linear_layers(emb)
        out = self.output_activation_layer(out)
        if self.squeeze:
            out = out.squeeze(-1)
        return out


    def _parse_inputs(self, inputs):
        if isinstance(inputs, torch.Tensor):
            seq = inputs
        elif isinstance(inputs, dict):
            seq = inputs.get('seq', None)
        elif isinstance(inputs, list):
            seq = inputs[0]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

        expected_shape = (seq.shape[0], self.input_len, self.input_channels)
        if seq.shape != expected_shape:
            raise ValueError(f"{seq.shape = }, {expected_shape = }")

        return seq