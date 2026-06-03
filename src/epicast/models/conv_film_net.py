import torch
import torch.nn as nn

from .conv_block import ResConvBlock
from .crop import CenterCrop1d
from .film import FiLM


class ConvFiLMNet(nn.Module):
    def __init__(
        self,
        input_seq_length=200,
        input_seq_channels=4,
        input_feature_dim=0,
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
        self.input_feature_dim = input_feature_dim
        self.output_dim = output_dim
        self.global_avg_pooling = global_avg_pooling
        self.flatten = flatten
        self.output_activation = output_activation
        
        self.expect_input_sample = {
            "seq": torch.zeros(1, input_seq_length, input_seq_channels), 
            "feature": torch.zeros(1, 2, input_feature_dim)
        }
        self.dummy_input = {
            "seq": torch.zeros(1, input_seq_length, input_seq_channels), 
            "feature": torch.zeros(1, 2, input_feature_dim)
        }
        

        assert len(conv_channels_list) == num_conv_blocks
        assert len(conv_kernel_size_list) == num_conv_blocks
        assert len(conv_stride_list) == num_conv_blocks
        assert len(conv_padding_list) == num_conv_blocks
        assert len(pool_kernel_size_list) == num_conv_blocks
        assert len(linear_channels_list) == num_linear_blocks

        # -------------------------
        # conv layers
        # -------------------------
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
        conv_out_channels = dummy.shape[1]
        conv_out_length = dummy.shape[2]

        # FiLM before flatten: modulate channel dimension on (B, H, L')
        self.film_layer = FiLM(
            num_features=conv_out_channels,
            cond_dim=input_feature_dim,
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
        seq, feature = self._parse_inputs(inputs)
        seq = seq.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        if feature is None:
            raise ValueError("Feature is required for ConvFiLMNet")

        elif feature.ndim == 2:
            emb = self.conv_layers(seq)
            emb = self.film_layer(emb, feature)
            emb = self.gap_layer(emb)
            emb = self.flatten_layer(emb)
            out = self.linear_layers(emb)
            out = self.output_layer(out)
            out = out.squeeze(-1)

        elif feature.ndim == 3:
            B, C, D = feature.shape
            emb = self.conv_layers(seq)   # (B, H, L)

            outs = []
            for i in range(C):
                feature_i = feature[:, i, :]      # (B, D)
                emb_i = self.film_layer(emb, feature_i)
                emb_i = self.gap_layer(emb_i)
                emb_i = self.flatten_layer(emb_i)
                out_i = self.linear_layers(emb_i)
                out_i = self.output_layer(out_i)  # (B, output_dim)
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

        elif isinstance(inputs, dict):
            return inputs["seq"], inputs.get("feature", None)

        elif isinstance(inputs, (list, tuple)):
            if len(inputs) == 1:
                return inputs[0], None
            if len(inputs) == 2:
                return inputs[0], inputs[1]
        raise TypeError(f"Unsupported inputs type: {type(inputs)}")
