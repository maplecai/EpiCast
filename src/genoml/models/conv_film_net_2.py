import torch
import torch.nn as nn

from .conv_block import ResConvBlock
from .film import FiLM


class ConvFiLMNet2(nn.Module):
    def __init__(
        self,
        input_seq_length=200,
        input_seq_channels=4,
        input_feature_dim=0,
        output_dim=1,

        global_avg_pooling=False,
        flatten=True,
        output_activation=None,
        add_film_after_each_conv=False,

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
        self.input_feature_dim = input_feature_dim
        self.output_dim = output_dim

        self.global_avg_pooling = global_avg_pooling
        self.flatten = flatten
        self.output_activation = output_activation
        self.add_film_after_each_conv = add_film_after_each_conv

        if conv_channels_list is None:
            raise ValueError("conv_channels_list must be provided")
        if conv_kernel_size_list is None:
            raise ValueError("conv_kernel_size_list must be provided")
        if conv_stride_list is None:
            raise ValueError("conv_stride_list must be provided")
        if conv_padding_list is None:
            raise ValueError("conv_padding_list must be provided")
        if pool_kernel_size_list is None:
            raise ValueError("pool_kernel_size_list must be provided")
        if linear_channels_list is None:
            raise ValueError("linear_channels_list must be provided")

        assert len(conv_channels_list) == num_conv_blocks
        assert len(conv_kernel_size_list) == num_conv_blocks
        assert len(conv_stride_list) == num_conv_blocks
        assert len(conv_padding_list) == num_conv_blocks
        assert len(pool_kernel_size_list) == num_conv_blocks
        assert len(linear_channels_list) == num_linear_blocks

        # -------------------------
        # conv layers
        # -------------------------
        self.conv_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        conv_in_channels = input_seq_channels

        for i in range(num_conv_blocks):
            conv_out_channels = conv_channels_list[i]
            conv_kernel_size = conv_kernel_size_list[i]
            conv_stride = conv_stride_list[i]
            conv_padding = conv_padding_list[i]
            pool_kernel_size = pool_kernel_size_list[i]

            conv_block = ResConvBlock(
                in_channels=conv_in_channels,
                out_channels=conv_out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
                activation=conv_activation,
                layer_order=conv_layer_order,
            )
            self.conv_blocks.append(conv_block)

            if pool_kernel_size != 1:
                pool_layer = nn.MaxPool1d(
                    kernel_size=pool_kernel_size,
                    ceil_mode=True,
                )
            else:
                pool_layer = nn.Identity()
            self.pool_layers.append(pool_layer)

            if conv_dropout_rate > 0:
                dropout_layer = nn.Dropout(conv_dropout_rate)
            else:
                dropout_layer = nn.Identity()
            self.dropout_layers.append(dropout_layer)

            if self.add_film_after_each_conv:
                self.film_layers.append(
                    FiLM(
                        num_features=conv_out_channels,
                        cond_dim=self.input_feature_dim,
                        hidden_dim=0,
                        modulated_dim=1,
                    )
                )
            conv_in_channels = conv_out_channels

        with torch.no_grad():
            dummy = torch.empty(1, input_seq_channels, input_seq_length)
            for i in range(num_conv_blocks):
                dummy = self.conv_blocks[i](dummy)
                dummy = self.pool_layers[i](dummy)

        conv_out_channels = dummy.shape[1]

        if self.add_film_after_each_conv is False:
            self.film_layer = FiLM(
                num_features=conv_out_channels,
                cond_dim=self.input_feature_dim,
                hidden_dim=0,
                modulated_dim=1,
            )

        self.gap_layer = nn.AdaptiveAvgPool1d(1) if global_avg_pooling else nn.Identity()
        self.flatten_layer = nn.Flatten() if flatten else nn.Identity()

        with torch.no_grad():
            dummy = self.gap_layer(dummy)
            dummy = self.flatten_layer(dummy)
        linear_in_channels = dummy.shape[-1]

        # -------------------------
        # linear layers
        # -------------------------
        self.linear_blocks = nn.ModuleList()
        self.linear_activation_layers = nn.ModuleList()
        self.linear_dropout_layers = nn.ModuleList()

        for i in range(num_linear_blocks):
            linear_out_channels = linear_channels_list[i]

            self.linear_blocks.append(
                nn.Linear(
                    in_features=linear_in_channels,
                    out_features=linear_out_channels,
                )
            )
            self.linear_activation_layers.append(nn.ReLU())

            if linear_dropout_rate > 0:
                self.linear_dropout_layers.append(nn.Dropout(linear_dropout_rate))
            else:
                self.linear_dropout_layers.append(nn.Identity())

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

    def _forward_conv(self, seq, feature):
        for i in range(len(self.conv_blocks)):
            seq = self.conv_blocks[i](seq)
            seq = self.pool_layers[i](seq)
            seq = self.dropout_layers[i](seq)

            if self.add_film_after_each_conv:
                seq = self.film_layers[i](seq, feature)

        if self.add_film_after_each_conv is False:
            seq = self.film_layer(seq, feature)

        return seq

    def _forward_head(self, emb):
        emb = self.gap_layer(emb)
        emb = self.flatten_layer(emb)

        for i in range(len(self.linear_blocks)):
            emb = self.linear_blocks[i](emb)
            emb = self.linear_activation_layers[i](emb)
            emb = self.linear_dropout_layers[i](emb)

        out = self.output_layer(emb)
        return out

    def _forward_single(self, seq, feature):
        emb = self._forward_conv(seq, feature)
        out = self._forward_head(emb)
        return out

    def forward(self, inputs):
        seq, feature = self._parse_inputs(inputs)

        seq = seq.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        if feature is None:
            raise ValueError("feature must be provided")
        
        elif feature.ndim == 2:
            out = self._forward_single(seq, feature)
            out = out.squeeze(-1)

        elif feature.ndim == 3:
            _, num_conditions, _ = feature.shape
            outs = []
            for i in range(num_conditions):
                feature_i = feature[:, i, :]  # (B, D)
                out_i = self._forward_single(seq, feature=feature_i)  # (B, output_dim)
                outs.append(out_i)

            out = torch.stack(outs, dim=1)  # (B, C, output_dim)
            out = out.squeeze(-1)
        else:
            raise ValueError(f"Unsupported feature shape: {feature.shape}")

        out = self.output_activation_layer(out)
        return out

    def _parse_inputs(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs, None

        if isinstance(inputs, dict):
            return inputs["seq"], inputs.get("feature", None)

        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 1:
                return inputs[0], None
            if len(inputs) == 2:
                return inputs[0], inputs[1]

        raise TypeError(f"Unsupported inputs type: {type(inputs)}")