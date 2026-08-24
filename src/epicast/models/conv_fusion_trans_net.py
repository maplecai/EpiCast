from collections import OrderedDict

import torch
import torch.nn as nn

from .linear_block import MLPBlock
from .trans_block import TransEncoder
from .conv_block import ConvEncoder
from .film import FiLM

class ConvFusionTransNet(nn.Module):
    def __init__(
        self,
        input_seq_length=200,
        input_seq_channels=4,
        input_feature_dim=0,
        output_dim=1,
        fusion_mode="FiLM",
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
        num_trans_blocks=3,
        trans_d_embed=256,
        trans_n_heads=8,
        trans_d_mlp=256,
        trans_dropout_rate=0.1,
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
            "feature": torch.zeros(1, 2, input_feature_dim),
        }

        if not (
            num_conv_blocks == len(conv_channels_list) == len(conv_kernel_size_list)
            == len(conv_stride_list) == len(conv_padding_list) == len(pool_kernel_size_list)
        ):
            raise ValueError("Inconsistent number of conv blocks")

        if not (num_linear_blocks == len(linear_channels_list)):
            raise ValueError("Inconsistent number of linear blocks")

        if conv_channels_list[-1] != trans_d_embed:
            raise ValueError(
                f"conv output channels ({conv_channels_list[-1]}) must equal trans_d_embed ({trans_d_embed})"
            )

        # -------------------------
        # conv layers
        # -------------------------

        self.conv_encoder = ConvEncoder(
            input_seq_channels=input_seq_channels,
            num_conv_blocks=num_conv_blocks,
            conv_channels_list=conv_channels_list,
            conv_kernel_size_list=conv_kernel_size_list,
            conv_stride_list=conv_stride_list,
            conv_padding_list=conv_padding_list,
            pool_kernel_size_list=pool_kernel_size_list,
            conv_dropout_rate=conv_dropout_rate,
            conv_layer_order=conv_layer_order,
            conv_activation=conv_activation,
        )

        if fusion_mode == "FiLM":
            # FiLM before flatten: modulate channel dimension on (B, H, L')
            self.fusion_layer = FiLM(
                num_features=conv_channels_list[-1],
                cond_dim=input_feature_dim,
                hidden_dim=0,
                modulated_dim=1,
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.trans_encoder = TransEncoder(
            num_trans_blocks=num_trans_blocks,
            trans_d_embed=trans_d_embed,
            trans_n_heads=trans_n_heads,
            trans_d_mlp=trans_d_mlp,
            trans_dropout_rate=trans_dropout_rate,
        )

        self.gap_layer = nn.AdaptiveAvgPool1d(1) if global_avg_pooling else nn.Identity()
        self.flatten_layer = nn.Flatten() if flatten else nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, input_seq_channels, input_seq_length)
            dummy = self.conv_encoder(dummy)
            dummy = dummy.permute(0, 2, 1)
            dummy = self.trans_encoder(dummy)
            dummy = dummy.permute(0, 2, 1)
            dummy = self.gap_layer(dummy)
            dummy = self.flatten_layer(dummy)
        linear_in_channels = dummy.shape[-1]

        self.output_head = MLPBlock(
            in_features=linear_in_channels,
            linear_channels_list=linear_channels_list,
            num_linear_blocks=num_linear_blocks,
            linear_dropout_rate=linear_dropout_rate,
            output_dim=output_dim,
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

    def _forward_only_seq(self, seq):
        emb = self.conv_encoder(seq)
        emb = emb.permute(0, 2, 1)
        emb = self.trans_encoder(emb)
        emb = emb.permute(0, 2, 1)
        emb = self.gap_layer(emb)
        emb = self.flatten_layer(emb)
        emb = self.output_head(emb)
        out = self.output_activation_layer(emb)
        out = out.squeeze(-1)
        return out

    def _forward_after_fusion(self, emb):
        emb = emb.permute(0, 2, 1)
        emb = self.trans_encoder(emb)
        emb = emb.permute(0, 2, 1)
        emb = self.gap_layer(emb)
        emb = self.flatten_layer(emb)
        emb = self.output_head(emb)
        out = self.output_activation_layer(emb)
        out = out.squeeze(-1)
        return out

    def forward(self, inputs):
        seq, feature = self._parse_inputs(inputs)
        seq = seq.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        if feature is None:
            out = self._forward_only_seq(seq)
            return out
        elif feature.ndim == 2:
            seq_emb = self.conv_encoder(seq)
            emb = self.fusion_layer(seq_emb, feature)
            out = self._forward_after_fusion(emb)
            return out
        elif feature.ndim == 3:
            B, C, D = feature.shape
            seq_emb = self.conv_encoder(seq)  # (B, H, L)

            outs = []
            for i in range(C):
                feature_i = feature[:, i, :]  # (B, D)
                emb_i = self.fusion_layer(seq_emb, feature_i)
                out_i = self._forward_after_fusion(emb_i)
                outs.append(out_i)
            outs = torch.stack(outs, dim=1)  # (B, C, output_dim)
            outs = outs.squeeze(-1)
            return outs
        else:
            raise ValueError(f"Unsupported feature shape: {feature.shape}")

    def _parse_inputs(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs, None
        elif isinstance(inputs, dict):
            return inputs.get("seq"), inputs.get("feature")
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) == 1:
                return inputs[0], None
            if len(inputs) == 2:
                return inputs[0], inputs[1]
        raise TypeError(f"Unsupported inputs type: {type(inputs)}")