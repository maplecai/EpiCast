import torch
import torch.nn as nn

from .conv_block import ResConvBlock
from .crop import CenterCrop1d

class ConvNet(nn.Module):
    def __init__(
        self, 
        input_seq_length=200,
        input_seq_channels=4,
        output_dim=1,

        global_avg_pooling=False,
        flatten=True,
        output_activation=None,

        num_conv_blocks=6,
        conv_channels_list=None,
        conv_kernel_size_list=None,
        conv_stride_list=None,
        conv_padding_list=None,
        conv_dropout_rate=0.2,
        conv_layer_order=None,
        conv_activation=None,
        pool_kernel_size_list=None,

        num_linear_blocks=2,
        linear_channels_list=None,
        linear_dropout_rate=0.5,
    ):
        super().__init__()

        self.input_seq_length = input_seq_length
        self.input_seq_channels = input_seq_channels
        self.output_dim = output_dim
        self.global_avg_pooling = global_avg_pooling
        self.flatten = flatten
        self.output_activation = output_activation

        assert len(conv_channels_list) == num_conv_blocks
        assert len(conv_kernel_size_list) == num_conv_blocks
        assert len(conv_stride_list) == num_conv_blocks
        assert len(conv_padding_list) == num_conv_blocks
        assert len(pool_kernel_size_list) == num_conv_blocks
        assert len(linear_channels_list) == num_linear_blocks

        self.expect_input_sample = {
            "seq": torch.zeros(1, input_seq_length, input_seq_channels), 
        }

        self.conv_layers = nn.Sequential()
        conv_in_channels = input_seq_channels

        for i in range(num_conv_blocks):
            conv_out_channels = conv_channels_list[i]
            conv_kernel_size = conv_kernel_size_list[i]
            conv_stride = conv_stride_list[i]
            conv_padding = conv_padding_list[i]
            pool_kernel_size = pool_kernel_size_list[i]

            self.conv_layers.add_module(
                f"conv_block_{i}",
                ResConvBlock(
                    in_channels=conv_in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=conv_padding,
                    activation=conv_activation,
                    layer_order=conv_layer_order,
                ),
            )

            if pool_kernel_size != 1:
                self.conv_layers.add_module(
                    f"max_pool_{i}",
                    nn.MaxPool1d(
                        kernel_size=pool_kernel_size,
                        ceil_mode=True,
                    ),
                )

            if conv_dropout_rate > 0:
                self.conv_layers.add_module(
                    f"conv_dropout_{i}",
                    nn.Dropout(conv_dropout_rate),
                )

            conv_in_channels = conv_out_channels

        # infer shapes
        with torch.no_grad():
            dummy = self.expect_input_sample["seq"].permute(0, 2, 1)
            dummy = self.conv_layers(dummy)

        self.gap_layer = nn.AdaptiveAvgPool1d(1) if global_avg_pooling else nn.Identity()
        self.flatten_layer = nn.Flatten() if flatten else nn.Identity()

        with torch.no_grad():
            dummy = self.gap_layer(dummy)
            dummy = self.flatten_layer(dummy)
        linear_in_channels = dummy.shape[-1]
        
        self.linear_layers = nn.Sequential()
        for i in range(num_linear_blocks):
            linear_out_channels = linear_channels_list[i]
            self.linear_layers.add_module(
                f"linear_{i}",
                nn.Linear(
                    in_features=linear_in_channels,
                    out_features=linear_out_channels,
                ),
            )
            self.linear_layers.add_module(f"act_{i}", nn.ReLU())
            self.linear_layers.add_module(f"dropout_{i}", nn.Dropout(linear_dropout_rate))
            linear_in_channels = linear_out_channels

        self.output_layer = nn.Linear(
            in_features=linear_in_channels,
            out_features=output_dim,
        )

        if output_activation is None:
            self.output_activation_layer = nn.Identity()
        elif output_activation == "sigmoid":
            self.output_activation_layer = nn.Sigmoid()
        elif output_activation == "softmax":
            self.output_activation_layer = nn.Softmax(dim=-1)
        elif output_activation == "softplus":
            self.output_activation_layer = nn.Softplus()
        else:
            raise ValueError(f"Unsupported output_activation mode: {output_activation}")


    def forward(self, inputs):
        seq = self._parse_inputs(inputs)
        seq = seq.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        emb = self.conv_layers(seq)
        emb = self.gap_layer(emb)
        emb = self.flatten_layer(emb)
        out = self.linear_layers(emb)
        out = self.output_layer(out)
        out = self.output_activation_layer(out)
        out = out.squeeze(-1)
        return out

    def _parse_inputs(self, inputs):
        if isinstance(inputs, torch.Tensor):
            seq = inputs
        elif isinstance(inputs, dict):
            seq = inputs.get('seq', None)
        elif isinstance(inputs, (list, tuple)):
            seq = inputs[0]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")
        return seq
